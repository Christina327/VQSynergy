from scipy.cluster.vq import kmeans2
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn

from utils import reset, LambdaLayer


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=1.5, nu=0.0, decay=0.99, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self.register_buffer('initted', torch.zeros(1))

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(
            self._num_embeddings, self._embedding_dim)
        # self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(
            num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon
        self._nu = nu

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = rearrange(inputs, 'b h w d -> (b h w) d')

        if self.training and self.initted.item() == 0:
            # batch size is small in RIM, up sampling here for clustering
            rp = torch.randint(0, flat_input.size(0), (20000,))
            kd, _ = kmeans2(flat_input[rp].data.cpu(
            ).numpy(), self._num_embeddings, minit='points')
            self._embedding.weight.data.copy_(torch.from_numpy(kd))
            self.initted.fill_(1)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        encodings = F.gumbel_softmax(-distances, tau=1, hard=True)

        # Quantize and unflatten
        quantized = torch.matmul(
            encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            with torch.no_grad():
                self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                    (1 - self._decay) * torch.sum(encodings, 0)

                # Laplace smoothing of the cluster size
                n = torch.sum(self._ema_cluster_size.data)
                self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w = nn.Parameter(
                    self._ema_w * self._decay + (1 - self._decay) * dw)

                self._embedding.weight = nn.Parameter(
                    self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        # 2. modify the latent loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        probs = torch.sigmoid(-distances)
        kld_discrete = torch.sum(
            probs * torch.log(torch.clamp(probs, min=1e-8)), dim=-1).mean(dim=0)
        loss = self._commitment_cost * (kld_discrete + e_latent_loss * (
            kld_discrete / torch.clamp(e_latent_loss, min=1e-8)).detach())

        # Straight Through Estimator
        # 3. modify the quantized
        quantized = quantized + self._nu*(quantized - quantized.detach())
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs *
                               torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


class BioEncoder(nn.Module):
    def __init__(self, drug_dim, cline_dim, out_dim, use_GMP=False):
        super(BioEncoder, self).__init__()

        self.out_dim = out_dim
        self.enc_hdn_layers = 1

        # drug
        drug_heads = 4
        self.drug_same_layers = self.enc_hdn_layers
        self.drug_first = gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(drug_dim, out_dim//drug_heads,
             heads=drug_heads, root_weight=False), 'x, edge_index -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.BatchNorm1d(self.out_dim), 'x -> x'),
            # (nn.ReLU(), 'x -> x'),
        ])
        self.drug_conv_same = nn.ModuleList([gnn.Sequential('x, edge_index', [
            (gnn.TransformerConv(self.out_dim, self.out_dim//drug_heads,
             heads=drug_heads, root_weight=False), 'x, edge_index -> x'),
            (nn.ReLU(), 'x -> x'),
            (nn.BatchNorm1d(self.out_dim), 'x -> x'),
        ]) for _ in range(self.drug_same_layers)])
        self.use_GMP = use_GMP

        # cell
        self.cline_same_layers = self.enc_hdn_layers
        self.cline_first = nn.Sequential(
            nn.Linear(cline_dim, self.out_dim),
            nn.Tanh(),
        )
        self.cline_same = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(self.out_dim),
            nn.Linear(self.out_dim, self.out_dim, bias=True),
            nn.ReLU(),
        ) for _ in range(self.cline_same_layers)])

        # weight initialization
        self.reset_para()

    def forward(self, drug_x, drug_adj, ibatch, cline_x, *args):
        # -----drug
        drug_x = self.drug_first(drug_x, drug_adj)
        for i in range(self.drug_same_layers):
            drug_x = drug_x + self.drug_conv_same[i](drug_x, drug_adj)
        # drug: pooling
        if self.use_GMP:
            drug_x = gnn.global_max_pool(drug_x, ibatch)
        else:
            drug_x = gnn.global_mean_pool(drug_x, ibatch)

        # -----cell
        cline_x = self.cline_first(cline_x)
        for i in range(self.cline_same_layers):
            cline_x = cline_x + self.cline_same[i](cline_x)

        return drug_x, cline_x

    # def reset_parameters(self):
    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HgnnEncoder(torch.nn.Module):
    def __init__(self, in_dim, out_dim, quantized=True):
        super(HgnnEncoder, self).__init__()
        self.out_dim = out_dim
        self.quantized = quantized

        # up projection
        self.scalar = 3.0
        hidden_dim = int(out_dim * self.scalar)
        if self.scalar == 1:
            self.lin_up = nn.Identity()
        else:
            self.lin_up = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=True),
                nn.ReLU(),
            )

        self.hgnn_hdn_layers = 3

        # bottleneck
        num_embeddings = 1024
        commitment_cost = 0.5
        nu = 2
        self._vq_vae_s = nn.ModuleList([
            VectorQuantizerEMA(num_embeddings, hidden_dim, commitment_cost, nu)
            for _ in range(self.hgnn_hdn_layers)])

        self.conv_same = nn.ModuleList([gnn.Sequential('x, hyperedge_index', [
            (nn.BatchNorm1d(hidden_dim), 'x -> x'),
            (gnn.HypergraphConv(hidden_dim, hidden_dim), 'x, hyperedge_index -> x'),
            (nn.ReLU(), 'x -> x'),
        ]) for _ in range(self.hgnn_hdn_layers)])

        init_bias = -5.0
        self.conv_w_out = nn.ModuleList([nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid(),
        ) for _ in range(self.hgnn_hdn_layers)])
        for layer in self.conv_w_out:
            for sublayer in layer:
                if isinstance(sublayer, nn.Linear):
                    nn.init.constant_(sublayer.bias, init_bias)

        if self.scalar == 1:
            self.proj_down = nn.Identity()
            self.conv_down = gnn.Sequential('x, hyperedge_index', [
                (nn.Identity(), 'x -> x'),
            ])
        else:
            self.lin_down = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Linear(hidden_dim, out_dim, bias=True),
            )
            self.conv_down = gnn.Sequential('x, hyperedge_index', [
                (nn.BatchNorm1d(hidden_dim), 'x -> x'),
                (gnn.HypergraphConv(hidden_dim, out_dim), 'x, hyperedge_index -> x'),
                (nn.ReLU(), 'x -> x'),
            ])

    def forward(self, X, H):
        # up projection
        X = self.lin_up(X)

        identity = X
        for i in range(self.hgnn_hdn_layers):
            message = self.conv_same[i](X, H)
            message_gate = self.conv_w_out[i](X)

            loss_latent, quantized, perplexity, _ = self._vq_vae_s[i](
                rearrange(message * message_gate, 'n d -> n d 1 1'))
            if self.quantized:
                quantized = rearrange(quantized, 'n d 1 1 -> n d')
                message = quantized

            X = X + message
        X = self.lin_down(X + identity) + self.conv_down(X, H)

        return X, loss_latent, perplexity


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
        drug_pairs = torch.stack([druga, drugb], dim=-2)

        query = self.w_q(cline).view(bsz, self.num_heads, -1)
        keys = self.w_k(drug_pairs).view(bsz, 2, self.num_heads, -1)
        vals = self.w_v(drug_pairs).view(bsz, 2, self.num_heads, -1)

        scores = torch.einsum('nhd,nkhd->nhk', query, keys) * self.attn_scalar
        attns = torch.softmax(scores, dim=-1)
        cand_h = torch.einsum('nhk,nkhd->nhd', attns, vals).view(bsz, -1)
        cand_h = identity + self.w_o(cand_h)
        return cand_h


class Decoder(torch.nn.Module):
    def __init__(self, in_dim):
        super(Decoder, self).__init__()
        self.act = nn.ReLU()

        hidden_dim = in_dim * 3
        num_heads = 12
        self.attn_consolidation = AttnConsolidation(
            in_dim, hidden_dim, num_heads)

        # mlp decoder
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            self.act,
            nn.Dropout(0.50),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            self.act,
            nn.Dropout(0.50),
            nn.Linear(hidden_dim // 4, 1),
            LambdaLayer(lambda x: x.squeeze()),
        )

        self.reset_parameters()

    def forward(self, graph_embed, druga_id, drugb_id, cline_id):
        druga = graph_embed[druga_id, :]
        drugb = graph_embed[drugb_id, :]
        cline = graph_embed[cline_id, :]
        if self.training:
            preds = self.forward_once(druga, drugb, cline)
        else:
            preds = self.forward_once(druga, drugb, cline)
        return preds

    def forward_once(self, druga, drugb, cline):

        cand_h = self.attn_consolidation(druga, drugb, cline)
        logits = self.mlp(cand_h)
        return torch.sigmoid(logits)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class HypergraphSynergy(torch.nn.Module):
    def __init__(self, bio_encoder, graph_encoder, decoder):
        super(HypergraphSynergy, self).__init__()
        self.bio_encoder: BioEncoder = bio_encoder
        self.hgnn_encoder: HgnnEncoder = graph_encoder
        self.decoder: Decoder = decoder

        hgnn_out_dim = self.hgnn_encoder.out_dim
        # reconstruction
        self.drug_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))
        self.cline_rec_weight = nn.Parameter(
            torch.rand(hgnn_out_dim, hgnn_out_dim))

        self.sim_proj = nn.Identity()

        self.reset_parameters()
        self.register_buffer('initted', torch.tensor(False))

    def reset_parameters(self):
        reset(self.bio_encoder)
        reset(self.hgnn_encoder)
        reset(self.decoder)

    def initialize(self, H_syn, num_synergy=0):
        self.register_buffer('H', H_syn)

    def forward(self, drug_x, drug_adj, ibatch, cline_x, druga_id, drugb_id, cline_id, *args):
        if not self.initted:
            self.num_drug = max(ibatch).item() + 1
            self.num_cline = len(cline_x)
            self.initted.fill_(True)

        # extra Gaussian noise
        noise_scale = 0.0
        if not self.training and noise_scale:
            drug_x += torch.rand_like(drug_x) * noise_scale
            cline_x += torch.rand_like(cline_x) * noise_scale

        # initial embedding
        drug_embed, cline_embed = self.bio_encoder(
            drug_x, drug_adj, ibatch, cline_x)

        merge_embed = torch.cat(
            (drug_embed, cline_embed), dim=0)

        graph_embed, loss_latent, perplexity = self.hgnn_encoder(
            merge_embed, self.H)

        # reconstruction
        graph_embed = self.sim_proj(graph_embed)
        # drug
        drug_emb = graph_embed[:self.num_drug]
        rec_drug = torch.sigmoid(
            drug_emb @ self.drug_rec_weight @ drug_emb.t())
        # cline
        cline_emb = graph_embed[self.num_drug: self.num_drug+self.num_cline]
        rec_cline = torch.sigmoid(
            cline_emb @ self.cline_rec_weight @ cline_emb.t())
        rec_s = rec_drug, rec_cline

        # decode and predict
        pred = self.decoder(graph_embed, druga_id, drugb_id, cline_id)
        return pred, rec_s, loss_latent, perplexity
