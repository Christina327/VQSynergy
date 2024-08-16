import os
import time
import glob
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from einops import rearrange, repeat

from VQSynergy.config import ARGS, DEVICE, CHECKPOINT_PATH, DRUG_DIM, CE_CRITERION
from VQSynergy.modules import Initializer, Refiner, Consolidator, VQSynergy
from utils import set_seed_all, get_metrics, TruncatedExponentialLR, IdenticalLR
from data_utils import load_data, create_data_loaders, data_split, k_fold_cross_validation


def train(model, optimizer, batch_tensor, batch_label, alpha, loaders):
    model.train()
    if ARGS.swap:
        batch_label = torch.cat((batch_label, batch_label), dim=0)

    # initialization
    druga_id, drugb_id, cline_id, _ = batch_tensor.unbind(1)

    optimizer.zero_grad()
    loss_train = 0
    batch_label_ls, batch_pred_ls = [], []
    for item in zip(*loaders):
        drug, (cline, ) = item
        pred, rec_s, loss_latent, perplexity = model(
            # atom-feature, atom-adjacency,
            drug.x, drug.edge_index, drug.batch,
            # cell line expression
            cline,
            druga_id, drugb_id, cline_id,
        )
        rec_drug, rec_cline = rec_s

        # calculate loss
        # ce
        loss_target = CE_CRITERION(torch.sigmoid(pred), batch_label)

        # dypu
#         neg_scores_mean = pred[batch_label == 0].mean()
#         loss_target = torch.log(
#             1 + torch.exp(neg_scores_mean - pred[batch_label == 1])).mean()

        # ce + dypu
#         neg_scores_mean = pred[batch_label == 0].mean()
#         loss_target_dypu = torch.log(
#             1 + torch.exp(neg_scores_mean - pred[batch_label == 1])).mean()
#         loss_target_ce = ce_loss_fn(pred, batch_label)
#         loss_target = loss_target_ce + loss_target_dypu * (loss_target_ce / (loss_target_dypu + 1e-8)).detach()

        # reconstruct loss
        loss_aux = CE_CRITERION(rec_drug, drug_similarity_matrix) + \
            CE_CRITERION(rec_cline, cline_similarity_matrix)

        # 0.005
        # if np.random.rand() < 1e-2:
        #     print("train:", (loss_target / loss_aux).detach())
        # loss = loss_target + alpha * loss_aux * (loss_target / loss_aux).detach()
        loss = loss_target + alpha * loss_aux

        # calculate the derivative of loss
        # add to gradients to according tensor
        loss.backward()

        # optimizer has traced each parameter of the model
        # update parameters and set theirs gradients to zero
        optimizer.step()

        loss_train += loss.item()
        # labels
        batch_label_ls += batch_label.cpu().detach().numpy().tolist()
        # predictions
        batch_pred_ls += torch.sigmoid(pred).cpu().detach().numpy().tolist()

    return loss_train, batch_label_ls, batch_pred_ls


def test(model, batch_tensor, batch_label, alpha, loaders):
    model.eval()

    druga_id, drugb_id, cline_id, _ = batch_tensor.unbind(1)
    with torch.no_grad():
        for item in zip(*loaders):
            drug, (cline, ) = item
            batch_pred, rec_s, loss_latent, perplexity = model(
                drug.x, drug.edge_index, drug.batch, cline,
                druga_id, drugb_id, cline_id)
            rec_drug, rec_cline = rec_s

        # target loss
        # ce
        loss_target = CE_CRITERION(torch.sigmoid(batch_pred), batch_label)

        # dypu
        # neg_scores_mean = pred[label == 0].mean()
        # loss_target = torch.log(
        #     1 + torch.exp(neg_scores_mean - pred[label == 1])).mean()

        # ce + dypu
        # loss_target_dypu = torch.log(
        #     1 + torch.exp(neg_scores_mean - pred[label == 1])).mean()
        # loss_target_ce = ce_loss_fn(pred, label)
        # loss_target = loss_target_ce + loss_target_dypu * (loss_target_ce / (loss_target_dypu + 1e-8)).detach()
        # loss_target = loss_target_ce

        # reconstruct loss
        loss_aux = CE_CRITERION(rec_drug, drug_similarity_matrix) + \
            CE_CRITERION(rec_cline, cline_similarity_matrix)
        # if np.random.rand() < 1e-2:
        #     print("eval:", (loss_target / loss_aux).detach())
        # loss = loss_target + alpha * loss_aux * (loss_target / loss_aux).detach()
        loss = loss_target + alpha * loss_aux

        batch_label = batch_label.cpu().detach().numpy()
        batch_pred = torch.sigmoid(batch_pred).cpu().detach().numpy()
        return loss.item(), batch_label, batch_pred


def train_loop():
    # Iterate over each cross-validation mode
    for cv_mode in ARGS.cv_mode_ls:
        set_seed_all(ARGS.rd_seed)

        # Initialize metrics
        final_valid_metric = np.zeros(2)
        final_test_metric = np.zeros(2)
        final_valid_metrics, final_test_metrics = [], []

        # K-Fold Cross-Validation
        for fold, synergy_train, synergy_valid in k_fold_cross_validation(data_cv_raw, cv_mode, ARGS.num_split, ARGS.rd_seed):

            print(f"Fold {fold}")
            print("    Number of training samples:  ", len(synergy_train))
            print("    Number of validation samples:", len(synergy_valid))
            fold_path = os.path.join(CHECKPOINT_PATH, str(cv_mode), str(fold))
            if not os.path.exists(fold_path):
                os.makedirs(fold_path)

            # Convert data to tensors
            tensor_train = torch.from_numpy(synergy_train).to(DEVICE)
            tensor_valid = torch.from_numpy(synergy_valid).to(DEVICE)
            label_train = torch.from_numpy(
                np.array(synergy_train[:, 3], dtype='float32')).to(DEVICE)
            label_valid = torch.from_numpy(
                np.array(synergy_valid[:, 3], dtype='float32')).to(DEVICE)

            # Construct synergy hypergraph
            pos_node = synergy_train[synergy_train[:, 3] == 1, 0:3]
            num_synergy = len(pos_node)
            H_syn_node = rearrange(pos_node, 'n triplet -> (n triplet)')
            H_syn_edge = repeat(np.arange(num_synergy), 'n -> (n 3)')
            H_syn = np.stack((H_syn_node, H_syn_edge), axis=0)
            H_syn = torch.from_numpy(H_syn).long().to(DEVICE)

            # Construct model
            model = VQSynergy(
                Initializer(
                    drug_dim=DRUG_DIM,
                    cline_dim=cline_features.shape[-1],
                    output_dim=ARGS.refiner_in_dim,
                    num_hidden_layers=ARGS.initializer_hidden_layers,
                    drug_heads=ARGS.drug_heads,
                    maxpooling=ARGS.graph_maxpooling
                ),
                Refiner(
                    input_dim=ARGS.refiner_in_dim,
                    output_dim=ARGS.refiner_out_dim,
                    multiplier=ARGS.multiplier,
                    num_hidden_layers=ARGS.refiner_hidden_layers,
                    quantized=ARGS.quantized,
                    num_embeddings=ARGS.num_embeddings,
                    commitment_cost=ARGS.commitment_cost,
                    kmeans=ARGS.kmeans,
                    decay=ARGS.decay,
                    lambda_=ARGS.lambda_,
                    nu=ARGS.nu,
                    tau=ARGS.tau,
                ),
                Consolidator(
                    in_dim=ARGS.refiner_out_dim
                ),
                noise=ARGS.noise,
            ).to(DEVICE)
            model.initialize(H_syn=H_syn)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=ARGS.learning_rate, betas=(ARGS.beta1, ARGS.beta2), weight_decay=ARGS.weight_decay)
            scheduler = TruncatedExponentialLR(
                optimizer, gamma=ARGS.lr_decay, min_lr=ARGS.min_lr)
            # scheduler = IdenticalLR(optimizer)

            # 2.4 train
            best_metric = [0, 0]
            best_epoch = 0
            for epoch in range(ARGS.max_epoch):
                train_label_ls, train_pred_ls = [], []
                for _ in range(1):
                    train_loss, batch_label_ls, batch_pred_ls = train(
                        model, optimizer, tensor_train, label_train, ARGS.alpha, loaders)
                    train_label_ls.extend(batch_label_ls)
                    train_pred_ls.extend(batch_pred_ls)
                valid_label_ls, valid_pred_ls = [], []
                for _ in range(1):
                    valid_loss, batch_label_ls, batch_pred_ls = test(
                        model, tensor_valid, label_valid, ARGS.alpha, loaders)
                    valid_label_ls.extend(batch_label_ls)
                    valid_pred_ls.extend(batch_pred_ls)
                scheduler.step()

                if not os.path.exists(fold_path):
                    os.makedirs(fold_path)
                if epoch >= ARGS.start_update_epoch:
                    train_metric = get_metrics(train_label_ls, train_pred_ls)
                    valid_metric = get_metrics(valid_label_ls, valid_pred_ls)
                    valid_auc, best_auc = valid_metric[0], best_metric[0]

                    if valid_auc > best_auc:
                        best_metric = valid_metric
                        best_epoch = epoch
                        torch.save(model.state_dict(), os.path.join(
                            fold_path, f'{epoch}.pth'))

                    files = glob.glob(os.path.join(fold_path, '*.pth'))
                    for f in files:
                        epoch_nb = int(os.path.basename(f).split('.')[0])
                        if epoch_nb < best_epoch:
                            os.remove(f)

                    if (epoch + 1) % ARGS.print_interval == 0:
                        print(f'Epoch: {epoch:04d},',
                              f'loss_train: {train_loss:.4f},',
                              f'AUC: {train_metric[0]:.4f},',
                              f'AUPR: {train_metric[1]:.4f},',
                              )
                        print(f'Epoch: {epoch:04d},',
                              f'loss_valid: {valid_loss:.4f},',
                              f'AUC: {valid_metric[0]:.4f},',
                              f'AUPR: {valid_metric[1]:.4f},',
                              )
                        print("-" * 71)

            files = glob.glob(os.path.join(fold_path, '*.pth'))
            for f in files:
                epoch_nb = int(os.path.basename(f).split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(f)
            print('The best results on valid set,',
                  f'Epoch: {best_epoch:04d},',
                  f'AUC: {best_metric[0]:.4f},',
                  f'AUPR: {best_metric[1]:.4f},',
                  )

            model.load_state_dict(torch.load(
                os.path.join(fold_path, f'{best_epoch}.pth')))
            valid_label_ls, valid_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    model, tensor_valid, label_valid, ARGS.alpha, loaders)
                valid_label_ls.extend(batch_label_ls)
                valid_pred_ls.extend(batch_pred_ls)
            valid_metric = get_metrics(valid_label_ls, valid_pred_ls)
            test_label_ls, test_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    model, tensor_test, label_test, ARGS.alpha, loaders)
                test_label_ls.extend(batch_label_ls)
                test_pred_ls.extend(batch_pred_ls)
            test_metric = get_metrics(test_label_ls, test_pred_ls)

            final_valid_metric += valid_metric
            final_test_metric += test_metric

            final_valid_metrics.append(valid_metric)
            final_test_metrics.append(test_metric)

        summary = pd.DataFrame(final_valid_metrics, columns=['AUC', 'AUPR'])
        summary *= 100
        summary.loc['mean'], summary.loc['std'] = summary.mean(), summary.std()
        summary.to_csv(os.path.join(CHECKPOINT_PATH, str(cv_mode),
                       'summary.csv'), index_label='Index')

        print("-" * 71)
        for i, metrics in enumerate(final_valid_metrics, 1):
            print(f'{i}-th valid results,',
                  f'AUC: {metrics[0]:.4f},',
                  f'AUPR: {metrics[1]:.4f},',
                  )
        print("-" * 71)

        final_valid_metric /= fold
        print('Final 5-cv valid results,',
              f'AUC: {final_valid_metric[0]:.4f},',
              f'AUPR: {final_valid_metric[1]:.4f},',
              )

        final_test_metric /= fold
        print('Final 5-cv test results,',
              f'AUC: {final_test_metric[0]:.4f},',
              f'AUPR: {final_test_metric[1]:.4f},',
              )

        end_time = time.time()

        elapsed_time = end_time - start_time

        elapsed_time = int(elapsed_time)
        print(
            f"Program execution time: {elapsed_time//60} min {elapsed_time%60} sec")
        print("=" * 71)
        print()


if __name__ == '__main__':
    # Parse command-line arguments
    start_time = time.time()

    # Load data
    drug_features, cline_features, synergy, similarity_matrices = load_data(
        ARGS.dataset_name)
    # For reconstructed similarity matrix
    drug_similarity_matrix, cline_similarity_matrix = similarity_matrices

    # Create data loaders
    drug_loader, cline_loader = create_data_loaders(
        drug_features, cline_features)
    loaders = [drug_loader, cline_loader]

    # Split data for cross-validation
    data_cv_raw, tensor_test, label_test = data_split(
        synergy, ARGS.cv_ratio, ARGS.rd_seed)

    if ARGS.use_checkpoint is False:
        train_loop()

    all_means = list()
    for mode in os.listdir(CHECKPOINT_PATH):
        mode_path = os.path.join(CHECKPOINT_PATH, mode)
        if os.path.isdir(mode_path):
            file_path = os.path.join(mode_path, 'summary.csv')
            summary = pd.read_csv(file_path, index_col=0)
            mean_summary = summary.loc['mean']
            mean_summary.name = mode
            all_means.append(mean_summary)
    all_means_df = pd.DataFrame(all_means)
    all_means_df.index.name = 'mode'
    all_means_df.to_csv(os.path.join(CHECKPOINT_PATH, 'all_means.csv'))
