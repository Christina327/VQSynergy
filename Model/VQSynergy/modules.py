from functools import partial
from scipy.cluster.vq import kmeans2
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from utils import reset, LambdaLayer


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, kmeans, decay, lambda_, nu, tau):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._commitment_cost = commitment_cost
        self._kmeans = kmeans
        self._decay = decay
        self._lambda = lambda_
        self._nu = nu
        self._tau = tau

        # Buffers for EMA algorithm
        self.register_buffer('_initted', torch.zeros(1))
        self.register_buffer('_ema_count', torch.empty(self._num_embeddings))
        self.register_buffer('_ema_weight', torch.empty(
            self._num_embeddings, self._embedding_dim))

        # Create a non-trainable embedding layer
        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        self._embedding.requires_grad_(False)

        # Initialize weights randomly if not using kmeans for initialization
        if not self._kmeans:
            uniform_range = 1 / self._num_embeddings
            self.ema_weight.data.uniform_(-uniform_range, uniform_range)
            # self._embedding.weight.data.normal_()

    def forward(self, inputs):
        # Convert inputs from BCHW -> BHWC
        inputs = rearrange(inputs, 'b d h w -> b h w d').contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = rearrange(inputs, 'b h w d -> (b h w) d')
        _num_vectors = flat_input.size(0)

        if self.training and self._initted.item() == 0:
            self._initted.fill_(1)

            self._ema_count.fill_(_num_vectors / self._num_embeddings)

            if self._kmeans:
                # Batch size is small in RIM, up sampling here for clustering
                rp = torch.randint(0, _num_vectors, (20000,))
                kd, _ = kmeans2(flat_input[rp].data.cpu(
                ).numpy(), self._num_embeddings, minit='points')
                self._embedding.weight.data.copy_(torch.from_numpy(kd))

                self._ema_weight = torch.einsum(
                    'ij,i->ij', self._embedding.weight.data, self._ema_count)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.einsum('bd,nd->bn', flat_input, self._embedding.weight))

        # Encoding
        soft_encodings = torch.softmax(-distances/self._tau, -1)
        with torch.no_grad():
            encodings = F.gumbel_softmax(-distances, tau=self._tau, hard=True)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                # Laplace smoothing of the cluster size
                _ema_count = self._ema_count * self._decay + \
                    torch.sum(soft_encodings, 0) * (1 - self._decay)
                self._ema_count = (
                    (_ema_count + self._lambda)
                    / (_num_vectors + self._num_embeddings * self._lambda) * _num_vectors)

                # Update the weight sums
                d_weight = torch.matmul(soft_encodings.t(), flat_input)
                self._ema_weight = self._ema_weight * \
                    self._decay + d_weight * (1 - self._decay)

                self._embedding.weight.data = torch.einsum(
                    'ij,i->ij', self._ema_weight, 1 / self._ema_count)

        # Loss
        # 2. modify the latent loss
        # loss_latents = F.mse_loss(quantized.detach(), inputs, reduction='mean') * _num_vectors

        probs = soft_encodings
        loss_latents = (
            torch.sum(probs * torch.log(torch.clamp(probs, min=1e-8)), dim=-1).mean(dim=0)
            # + torch.log(torch.tensor(self._num_embeddings, dtype=probs.dtype))
        )
        loss = self._commitment_cost * loss_latents

        # Straight Through Estimator
        # 3. modify the quantized
        # quantized = inputs + (quantized - inputs).detach() + \
        #     self._nu*(quantized - quantized.detach())
        quantized = quantized + self._nu*(quantized - quantized.detach())
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class Initializer(nn.Module):
    def __init__(self, drug_dim, cline_dim, output_dim, num_hidden_layers, drug_heads, maxpooling):
        super(Initializer, self).__init__()

        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.drug_heads = drug_heads
        self.use_maxpooling = maxpooling

        # Drug branch
        # Initial TransformerConv layer for drugs
        self.drug_initial_conv = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(
                drug_dim, self.output_dim // self.drug_heads,
                heads=self.drug_heads, root_weight=False), 'x, edge_index -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.BatchNorm1d(self.output_dim), 'x -> x'),
        ])

        # Subsequent TransformerConv layers for drugs
        self.drug_hidden_convs = nn.ModuleList([
            gnn.Sequential('x, edge_index', [
                (gnn.TransformerConv(
                    self.output_dim, self.output_dim // self.drug_heads,
                    heads=self.drug_heads, root_weight=False), 'x, edge_index -> x'),
                (nn.ReLU(), 'x -> x'),
                (nn.BatchNorm1d(self.output_dim), 'x -> x'),
            ]) for _ in range(self.num_hidden_layers)
        ])

        # Cell line branch
        # Initial linear layer for cell lines
        self.cline_initial_fc = nn.Sequential(
            nn.Linear(cline_dim, self.output_dim),
            nn.Tanh(),
        )

        # Subsequent layers for cell lines
        self.cline_hidden_layers = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(self.output_dim),
            nn.Linear(self.output_dim, self.output_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
        ) for _ in range(self.num_hidden_layers)])

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, drug_features, drug_adj, ibatch, cline_features, *args):
        # Drug branch forward pass
        drug_features = self.drug_initial_conv(drug_features, drug_adj)

        for i in range(self.num_hidden_layers):
            drug_features = drug_features + \
                self.drug_hidden_convs[i](drug_features, drug_adj)

        # Drug pooling
        if self.use_maxpooling:
            drug_features = gnn.global_max_pool(drug_features, ibatch)
        else:
            drug_features = gnn.global_mean_pool(drug_features, ibatch)

        # Cell line branch forward pass
        cline_features = self.cline_initial_fc(cline_features)
        for i in range(self.num_hidden_layers):
            cline_features = cline_features + \
                self.cline_hidden_layers[i](cline_features)

        return drug_features, cline_features


class Refiner(torch.nn.Module):
    def __init__(self, input_dim, output_dim, multiplier, num_hidden_layers, quantized, num_embeddings, commitment_cost, kmeans, decay, lambda_, nu, tau):
        super(Refiner, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.multiplier = multiplier
        hidden_dim = int(output_dim * self.multiplier)
        self.num_hidden_layers = num_hidden_layers

        # Vector Quantizer
        self.quantized = quantized
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.kmeans = kmeans
        self.decay = decay
        self.lambda_ = lambda_
        self.nu = nu
        self.tau = tau

        # Up projection
        if input_dim == hidden_dim:
            self.linear_up = nn.Identity()
        else:
            self.linear_up = nn.Sequential(
                nn.Linear(input_dim, hidden_dim, bias=True),
                nn.ReLU(),
            )

        # Bottleneck
        self.vector_quantizers = nn.ModuleList([
            VectorQuantizerEMA(self.num_embeddings, hidden_dim, self.commitment_cost,
                               self.kmeans, self.decay, self.lambda_, self.nu, self.tau)
            for _ in range(self.num_hidden_layers)])

        self.hypergraph_convs = nn.ModuleList([gnn.Sequential('x, hyperedge_index', [
            (nn.BatchNorm1d(hidden_dim), 'x -> x'),
            (gnn.HypergraphConv(hidden_dim, hidden_dim), 'x, hyperedge_index -> x'),
            (nn.ReLU(), 'x -> x'),
        ]) for _ in range(self.num_hidden_layers)])
        # self.ffns = nn.ModuleList([nn.Sequential(
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Linear(hidden_dim, hidden_dim*3),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(hidden_dim*3, hidden_dim),
        # ) for _ in range(self.hgnn_hdn_layers)])

        init_bias = 5.0
        self.gate_layers = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid(),
        ) for _ in range(self.num_hidden_layers)])
        # self.ffn_w_out = nn.ModuleList([nn.Sequential(
        #     nn.BatchNorm1d(hidden_dim),
        #     nn.Linear(hidden_dim, 1, bias=True),
        #     nn.Sigmoid(),
        # ) for _ in range(self.hgnn_hdn_layers)])
        for layer in self.gate_layers:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.constant_(sublayer.bias, init_bias)
        # for layer in self.ffn_w_out:
        #     for sublayer in layer:
        #         if isinstance(sublayer, nn.Linear):
        #             nn.init.constant_(sublayer.bias, init_bias)

        if hidden_dim == output_dim:
            self.conv_down = gnn.Sequential('x, hyperedge_index', [
                (nn.Identity(), 'x -> x'),
            ])
        else:
            self.conv_down = gnn.Sequential('x, hyperedge_index', [
                (nn.BatchNorm1d(hidden_dim), 'x -> x'),
                (gnn.HypergraphConv(hidden_dim, output_dim),
                 'x, hyperedge_index -> x'),
                (nn.ReLU(), 'x -> x'),
            ])

    def forward(self, X, H):
        # Up projection
        X = self.linear_up(X)
        identity = X

        # vector quantization
        # loss_latent, quantized, perplexity, _ = self._vq_vae(
        #     rearrange(X, 'n d -> n d 1 1'))

        # if self.quantized:
        #     quantized = rearrange(quantized, 'n d 1 1 -> n d')
        #     X = X + self.q_gate(X).mean() * (quantized - X)  # + shared
        # X = X + self.q_gate(X) * (quantized - X)       # initial

        loss_latents = .0
        for i in range(self.num_hidden_layers):
            initial_message = self.hypergraph_convs[i](X, H)
            message_gate = self.gate_layers[i](X)
            message = initial_message * message_gate

            loss_latent, quantized_message, perplexity, _ = self.vector_quantizers[i](
                rearrange(message, 'n d -> n d 1 1'))
            loss_latents += loss_latent
            quantized_message = rearrange(quantized_message, 'n d 1 1 -> n d')
            if self.quantized:
                message = quantized_message

            X = X + message
            # X = X + self.ffns[i](X) * self.ffn_w_out[i](X)

        # down projection
        # X = self.lin_down(X)
        # X = self.lin_down(X) + self.conv_down(X, H)
        X = self.conv_down(X, H)  # conv down
        # X = self.lin_down(X + identity) + self.conv_down(X, H)  # + bypass

        return X, loss_latents, perplexity


class AttnConsolidation(nn.Module):
    def __init__(self, in_dim, hdn_dim, num_heads):
        super(AttnConsolidation, self).__init__()

        self.in_dim = in_dim
        self.hdn_dim = hdn_dim
        self.num_heads = num_heads

        val_dim = self.hdn_dim // self.num_heads
        self.attn_scalar = 1 / (val_dim ** 0.5)

        self.w_q = nn.Linear(self.in_dim, self.hdn_dim, bias=True)
        self.w_k = nn.Linear(self.in_dim, self.hdn_dim, bias=True)
        self.w_v = nn.Linear(self.in_dim, self.hdn_dim, bias=True)
        self.w_o = nn.Linear(self.hdn_dim, self.hdn_dim, bias=True)

    def forward(self, druga, drugb, cline):
        bsz = druga.size(0)

        identity = torch.cat([druga, drugb, cline], dim=-1)
        # drug_pairs = torch.stack([druga, drugb], dim=-2)
        drug_pairs = torch.stack([druga, drugb, cline], dim=-2)

        query = self.w_q(cline).view(bsz, self.num_heads, -1)
        # keys = self.w_k(drug_pairs).view(bsz, 2, self.num_heads, -1)
        # vals = self.w_v(drug_pairs).view(bsz, 2, self.num_heads, -1)
        keys = self.w_k(drug_pairs).view(bsz, 3, self.num_heads, -1)
        vals = self.w_v(drug_pairs).view(bsz, 3, self.num_heads, -1)

        scores = torch.einsum('nhd,nkhd->nhk', query, keys) * self.attn_scalar
        # TODO: gumbel-softmax
        attns = torch.softmax(scores, dim=-1)
        cand_h = torch.einsum('nhk,nkhd->nhd', attns, vals).view(bsz, -1)
        cand_h = identity + self.w_o(cand_h)
        return cand_h


class Consolidator(torch.nn.Module):
    def __init__(self, in_dim):
        super(Consolidator, self).__init__()

        # self.act = nn.Tanh()
        self.act = nn.ReLU()

        hidden_dim = in_dim * 3
        num_heads = 12
        self.attn_consolidation = AttnConsolidation(
            in_dim, hidden_dim, num_heads)

        # == cat style

        # == undirected maxmin style
        # in_dim = in_dim
        # maxmin
        # maxmin + norm
        # maxmin + norm * beta
        # self.beta = nn.Parameter(torch.tensor(0.05).log(), requires_grad=False)
        # cat(maxmin, norm)
        # in_channels = in_channels * 2

        # mlp decoder
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            self.act,
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            self.act,
            nn.Dropout(0.4),
            nn.Linear(hidden_dim // 4, 1),
            LambdaLayer(lambda x: x.squeeze()),
        )

        # == trilinear style
        # self.latents = nn.Parameter(torch.empty(node_dim, node_dim, node_dim), requires_grad=True)
        # nn.init.xavier_normal_(self.latents.data)

        self.reset_parameters()

    def forward(self, graph_embed, druga_id, drugb_id, cline_id):
        druga = graph_embed[druga_id, :]
        drugb = graph_embed[drugb_id, :]
        cline = graph_embed[cline_id, :]
        if self.training:
            # preds_original = self.forward_once(druga, drugb, cline)
            # preds_augmented = self.forward_once(drugb, druga, cline)
            # preds = torch.cat((preds_original, preds_augmented), dim=0)
            preds = self.forward_once(druga, drugb, cline)
        else:
            # preds_original = self.forward_once(druga, drugb, cline)
            # preds_augmented = self.forward_once(drugb, druga, cline)
            # preds = (preds_original + preds_augmented) * 0.5
            preds = self.forward_once(druga, drugb, cline)
        return preds

    def forward_once(self, druga, drugb, cline):

        cand_h = self.attn_consolidation(druga, drugb, cline)

        # == cat style
        # cand_h = torch.cat((druga, drugb, cline), -1)

        # == maxmin style
        # cand_v = torch.stack((druga, drugb, cline), -1)
        # maxmin_h = cand_v.amax(dim=-1) + (-cand_v).amax(dim=-1)
        # norm_h = cand_v.square().mean(dim=-1).sqrt()

        # maxmin
        # cand_h = maxmin_h

        # maxmin + norm
        # cand_h = maxmin_h + norm_h

        # maxmin + norm * beta
        # coef = torch.exp(self.beta)
        # if torch.rand(1).item() < 7e-2:
        #     print(f"Coefficient of norm: {coef.item():.3f}")
        # cand_h = maxmin_h + norm_h * coef

        # cat(maxmin, norm)
        # cand_h = torch.cat((maxmin_h, norm_h), dim=-1)

        # 2-layer decoder
        logits = self.mlp(cand_h)

        # == trilinear style

        return logits

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class VQSynergy(torch.nn.Module):
    def __init__(self, initializer, refiner, consolidator, noise):
        super(VQSynergy, self).__init__()
        self.initializer: Initializer = initializer
        self.refiner: Refiner = refiner
        self.consolidator: Consolidator = consolidator

        self.noise = noise

        hgnn_out_dim = self.refiner.output_dim
        # reconstruction
        self.drug_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))
        self.cline_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))

        self.sim_proj = nn.Identity()

        # linear_proj = nn.Linear(out_dim, out_dim, bias=True)
        # nn.init.eye_(linear_proj.weight)
        # self.sim_proj = nn.Sequential(
        #     linear_proj,
        #     # nn.ReLU(),
        # )

        self.reset_parameters()
        self.register_buffer('initted', torch.tensor(False))

    def reset_parameters(self):
        reset(self.initializer)
        reset(self.refiner)
        reset(self.consolidator)

    def initialize(self, H_syn):
        self.register_buffer('H', H_syn)

    def forward(self, drug_x, drug_adj, ibatch, cline_x, druga_id, drugb_id, cline_id, *args):
        if not self.initted:
            self.num_drug = max(ibatch).item() + 1
            self.num_cline = len(cline_x)
            self.initted.fill_(True)
            print("number of drugs: ", self.num_drug)
            print("number of cell lines:", self.num_cline)

        # extra Gaussian noise
        if not self.training and self.noise:
            # drug_x += torch.rand_like(drug_x) * self.noise
            cline_x += torch.rand_like(cline_x) * self.noise

        # initial embedding
        drug_embed, cline_embed = self.initializer(
            drug_x, drug_adj, ibatch, cline_x)

        merge_embed = torch.cat(
            (drug_embed, cline_embed), dim=0)

        graph_embed, loss_latent, perplexity = self.refiner(
            merge_embed, self.H)

        # reconstruction
        graph_embed = self.sim_proj(graph_embed)
        # drug
        drug_emb = graph_embed[:self.num_drug]
        rec_drug = torch.sigmoid(
            drug_emb @ self.drug_rec_weight @ drug_emb.t())
        # rec_drug = torch.sigmoid(drug_emb @ drug_emb.t())
        # cline
        cline_emb = graph_embed[self.num_drug: self.num_drug+self.num_cline]
        rec_cline = torch.sigmoid(
            cline_emb @ self.cline_rec_weight @ cline_emb.t())
        # rec_cline = torch.sigmoid(cline_emb @ cline_emb.t())
        rec_s = rec_drug, rec_cline

        # decode and predict
        pred = self.consolidator(graph_embed, druga_id, drugb_id, cline_id)
        return pred, rec_s, loss_latent, perplexity
