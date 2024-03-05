import time
import os
import glob
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import torch.utils.data as Data
from einops import rearrange, repeat

from utils import set_seed_all, metrics_graph, TruncatedExponentialLR
from process_data import getData
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from drug_util import GraphDataset, collate
from model import BioEncoder, HypergraphSynergy, HgnnEncoder, Decoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_data(dataset):
    drug_fea, drug_smiles_fea, cline_fea, gene_data, synergy = \
        getData(dataset)
    cline_fea = torch.from_numpy(cline_fea).to(device)

    # update the synergy scores to be binary classification according to threshold
    THRESHOLD = 30
    for row in synergy:
        row[3] = 1 if row[3] >= THRESHOLD else 0

    # get similarity matrix used for reconstruction loss
    sim_matrices = get_sim_mat(
        drug_smiles_fea, np.array(gene_data, dtype='float32'))

    # Dict{pubchemid: (Feat, SparseAdj)}, Tensor, List[drug_a, drug_b, cline, synergy], Tensor, Tensor
    ret = drug_fea, cline_fea, synergy, sim_matrices
    return ret


def _split_dataframe_to_arrays(df, ratio, rd_seed):
    shuffled_df = df.sample(frac=1, random_state=rd_seed)
    shuffled_arr = np.array(shuffled_df)
    arr_train, arr_test = np.split(
        shuffled_arr, [int(ratio*len(shuffled_arr))])
    return arr_train, arr_test


def data_split(synergy, cv_mode, rd_seed):
    cv_ratio = 0.9

    set_seed_all(rd_seed)
    # random, but ensure pos and neg samples percentage are identity in train and test
    print("balanced random")

    # pos + neg
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])

    # cv + test
    synergy_cv_pos, synergy_test_pos = _split_dataframe_to_arrays(
        synergy_pos, cv_ratio, rd_seed)
    synergy_cv_neg, synergy_test_neg = _split_dataframe_to_arrays(
        synergy_neg, cv_ratio, rd_seed)

    # cross validation dataset
    data_cv_raw = np.concatenate(
        (np.array(synergy_cv_neg), np.array(synergy_cv_pos)), axis=0)
    # test dataset
    data_test = np.concatenate(
        (np.array(synergy_test_neg), np.array(synergy_test_pos)), axis=0)

    np.random.seed(rd_seed)
    np.random.shuffle(data_cv_raw)
    np.random.seed(rd_seed)
    np.random.shuffle(data_test)
    print("    number of cross:", len(data_cv_raw))
    print("    number of test: ", len(data_test))
    tensor_test = torch.from_numpy(data_test).to(device)
    label_test = torch.from_numpy(
        np.array(data_test[:, 3], dtype='float32')).to(device)
    # np.savetxt(path + 'test_y_true.txt', data_test[:, 3])

    # Ndarray[Tuple4], Tensor[Tuple4], Tensor[Tuple1]
    return data_cv_raw, tensor_test, label_test

def get_sim_mat(drug_smiles_fea, cline_fea):
    # drug_smiles_fea: zero-one vector
    drug_sim_matrix = get_Cosin_Similarity(drug_smiles_fea)
    cline_sim_matrix = get_pvalue_matrix(cline_fea)
    matrices_np = [drug_sim_matrix, cline_sim_matrix]
    matries = []
    for matrix in matrices_np:
        matrix = torch.from_numpy(matrix).float().to(device)
        matries.append(matrix)
    return matries


def train(batch_tensor, batch_label, alpha, loaders):
    model.train()
    if swap:
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

        # target loss
        loss_target = ce_loss_fn(pred, batch_label)
        # reconstruct loss
        loss_aux = ce_loss_fn(rec_drug, drug_sim_mat) + \
            ce_loss_fn(rec_cline, cline_sim_mat)
        loss = loss_target + alpha * loss_aux

        loss.backward()
        optimizer.step()

        loss_train += loss.item()
        # labels
        batch_label_ls += batch_label.cpu().detach().numpy().tolist()
        # predictions
        batch_pred_ls += pred.cpu().detach().numpy().tolist()
    
    return loss_train, batch_label_ls, batch_pred_ls


def test(batch_tensor, batch_label, alpha, loaders):
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
        loss_target = ce_loss_fn(batch_pred, batch_label)
        # reconstruct loss
        loss_aux = ce_loss_fn(rec_drug, drug_sim_mat) + \
            ce_loss_fn(rec_cline, cline_sim_mat)
        loss = loss_target + alpha * loss_aux

        batch_label = batch_label.cpu().detach().numpy()
        batch_pred = batch_pred.cpu().detach().numpy()
        return loss.item(), batch_label, batch_pred


if __name__ == '__main__':
    start_time = time.time()

    swap = False
    max_epoch = 1500
    start_update_epoch = 499
    print_interval = 200
    num_split = 5

    rd_seed = 1
    dataset_name = 'ALMANAC'  # or ONEIL
    cv_mode_ls = [1, 2, 3, 4, 5]
    hgnn_in_dim, hgnn_out_dim = 256, 256

    lr_decay = 1 - 3e-4
    min_lr = 1e-5
    learning_rate = 1e-4
    alpha = 1e-2
    weight_decay = 2e-2

    # ****** 1. first loop layer: 3 scenarios' cv ******
    for cv_mode, in itertools.product(cv_mode_ls):
        # ******* get data
        # create directories if not exist
        path = 'result_cls/' + dataset_name + '_' + str(cv_mode) + '_'
        if not os.path.isdir(path):
            os.makedirs(path)

        set_seed_all(rd_seed)

        drug_feature, cline_feature, synergy_data, sim_matrices = \
            load_data(dataset_name)
        drug_sim_mat, cline_sim_mat = sim_matrices
        drug_loader = Data.DataLoader(
            dataset=GraphDataset(graphs_dict=drug_feature),
            collate_fn=collate,
            batch_size=len(drug_feature),
            shuffle=False
        )
        cline_loader = Data.DataLoader(
            dataset=Data.TensorDataset(cline_feature),
            batch_size=len(cline_feature),
            shuffle=False
        )
        loaders = [drug_loader, cline_loader]

        # split [all] -> [cv, test]
        data_cv_raw, tensor_test, label_test = data_split(
            synergy_data, cv_mode, rd_seed)

        # split [cv] -> [train, val]
        if cv_mode == 1:
            # random
            data_cv = data_cv_raw
        elif cv_mode == 2:
            # cline level
            data_cv = np.unique(data_cv_raw[:, 2])
        elif cv_mode == 3:
            # drug pairs level
            drugcomb = np.column_stack((data_cv_raw[:, 0], data_cv_raw[:, 1]))
            drugcomb = [tuple(sorted(pair)) for pair in drugcomb]
            data_cv = np.unique(drugcomb, axis=0)
        elif cv_mode in [4, 5]:
            # drug pair: only both
            data_cv = np.unique(np.concatenate([data_cv_raw[:, 0], data_cv_raw[:, 1]]))
        else:
            raise NotImplementedError

        # fold_num = 0
        final_valid_metric = np.zeros(4)
        final_test_metric = np.zeros(4)
        kf = KFold(n_splits=num_split, shuffle=True, random_state=rd_seed)
        for idx, (train_index, valid_index) in enumerate(kf.split(data_cv), start=1):
            if cv_mode == 1:
                synergy_train, synergy_valid = data_cv[train_index], data_cv[valid_index]
            elif cv_mode == 2:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    cline = row[2]
                    if cline in train_name:
                        synergy_train.append(row)
                    elif cline in test_name:
                        synergy_valid.append(row)
                    else:
                        raise ValueError
            elif cv_mode == 3:
                pair_train, pair_validation = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    drugcomb = sorted((row[0], row[1]))
                    if any(all(x == y for x, y in zip(drugcomb, pair)) for pair in pair_train):
                        synergy_train.append(row)
                    else:
                        synergy_valid.append(row)
            elif cv_mode == 4:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    druga, drugb = row[0], row[1]
                    if (druga in train_name) and (drugb in train_name):
                        synergy_train.append(row)
                    elif (druga in test_name) and (drugb in test_name):
                        # just discard it
                        pass
                    else:
                        synergy_valid.append(row)
            elif cv_mode == 5:
                train_name, test_name = data_cv[train_index], data_cv[valid_index]
                synergy_train, synergy_valid = [], []
                for row in data_cv_raw:
                    druga, drugb = row[0], row[1]
                    if (druga in train_name) and (drugb in train_name):
                        synergy_train.append(row)
                    elif (druga in test_name) and (drugb in test_name):
                        synergy_valid.append(row)
                    else:
                        # just discard it
                        pass
            else:
                raise NotImplementedError

            synergy_train = np.array(synergy_train)
            synergy_valid = np.array(synergy_valid)
            print(f"split {idx}")
            print("    number of train:", len(synergy_train))
            print("    number of valid:", len(synergy_valid))

            # data
            tensor_train = torch.from_numpy(synergy_train).to(device)
            tensor_valid = torch.from_numpy(synergy_valid).to(device)
            # label
            label_train = torch.from_numpy(
                np.array(synergy_train[:, 3], dtype='float32')).to(device)
            label_valid = torch.from_numpy(
                np.array(synergy_valid[:, 3], dtype='float32')).to(device)

            # -----construct hyper_synergy_graph_set
            pos_node = synergy_train[synergy_train[:, 3] == 1, 0:3]
            num_synergy = len(pos_node)
            H_syn_node = rearrange(pos_node, 'n triplet -> (n triplet)')
            H_syn_edge = repeat(np.arange(num_synergy), 'n -> (n 3)')
            H_syn = np.stack((H_syn_node, H_syn_edge), axis=0)
            H_syn = torch.from_numpy(H_syn).long().to(device)

            # 2.3 construct model
            # model_build
            set_seed_all(rd_seed)
            model = HypergraphSynergy(
                BioEncoder(drug_dim=75, cline_dim=cline_feature.shape[-1], out_dim=hgnn_in_dim),
                HgnnEncoder(in_dim=hgnn_in_dim, out_dim=hgnn_out_dim),
                Decoder(in_dim=hgnn_out_dim),
            ).to(device)
            model.initialize(num_synergy=num_synergy, H_syn=H_syn)

            ce_loss_fn = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            scheduler = TruncatedExponentialLR(optimizer, gamma=lr_decay, min_lr=min_lr)

            # 2.4 train
            best_metric = [0, 0, 0, 0]
            best_epoch = 0
            for epoch in range(max_epoch):
                # train
                train_label_ls, train_pred_ls = [], []
                for _ in range(1):
                    train_loss, batch_label_ls, batch_pred_ls = train(
                        tensor_train, label_train, alpha, loaders)
                    train_label_ls.extend(batch_label_ls)
                    train_pred_ls.extend(batch_pred_ls)
                # validate
                valid_label_ls, valid_pred_ls = [], []
                for _ in range(1):
                    valid_loss, batch_label_ls, batch_pred_ls = test(
                        tensor_valid, label_valid, alpha, loaders)
                    valid_label_ls.extend(batch_label_ls)
                    valid_pred_ls.extend(batch_pred_ls)
                # learning rate scheduler
                scheduler.step()

                if epoch >= start_update_epoch:
                    # evaluate
                    train_metric = metrics_graph(train_label_ls, train_pred_ls)
                    valid_metric = metrics_graph(valid_label_ls, valid_pred_ls)

                    # evaluate model according to evaluation
                    torch.save(model.state_dict(), '{}.pth'.format(epoch))
                    valid_auc, best_auc = valid_metric[0], best_metric[0]
                    if valid_auc > best_auc:
                        # if True:
                        best_metric = valid_metric
                        best_epoch = epoch
                    files = glob.glob('*.pth')
                    # delete all model records (i.e. ".pth" files) before best model
                    for f in files:
                        epoch_nb = int(f.split('.')[0])
                        if epoch_nb < best_epoch:
                            os.remove(f)

                    # verbose
                    if (epoch + 1) % print_interval == 0:
                        print('Epoch: {:04d},'.format(epoch),
                                'loss_train: {:.4f},'.format(train_loss),
                                'AUC: {:.4f},'.format(train_metric[0]),
                                'AUPR: {:.4f},'.format(train_metric[1]),
                                'F1: {:.4f},'.format(train_metric[2]),
                                'ACC: {:.4f},'.format(train_metric[3]),
                                )
                        print('Epoch: {:04d},'.format(epoch),
                                'loss_valid: {:.4f},'.format(valid_loss),
                                'AUC: {:.4f},'.format(valid_metric[0]),
                                'AUPR: {:.4f},'.format(valid_metric[1]),
                                'F1: {:.4f},'.format(valid_metric[2]),
                                'ACC: {:.4f},'.format(valid_metric[3])
                                )
                        print("-" * 83)

            # after training
            files = glob.glob('*.pth')
            for f in files:
                epoch_nb = int(f.split('.')[0])
                if epoch_nb > best_epoch:
                    os.remove(f)
            print('The best results on valid set, Epoch: {:04d},'.format(best_epoch),
                  'AUC: {:.4f},'.format(best_metric[0]),
                  'AUPR: {:.4f},'.format(best_metric[1]),
                  'F1: {:.4f},'.format(best_metric[2]),
                  'ACC: {:.4f}\n'.format(best_metric[3]),
                )
            
            # load best model
            model.load_state_dict(torch.load('{}.pth'.format(best_epoch)))
            # valid
            valid_label_ls, valid_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    tensor_valid, label_valid, alpha, loaders)
                valid_label_ls.extend(batch_label_ls)
                valid_pred_ls.extend(batch_pred_ls)
            valid_metric = metrics_graph(valid_label_ls, valid_pred_ls)
            # test
            test_label_ls, test_pred_ls = [], []
            for _ in range(1):
                _, batch_label_ls, batch_pred_ls = test(
                    tensor_test, label_test, alpha, loaders)
                test_label_ls.extend(batch_label_ls)
                test_pred_ls.extend(batch_pred_ls)
            test_metric = metrics_graph(test_label_ls, test_pred_ls)
            final_valid_metric += valid_metric
            final_test_metric += test_metric

            # if idx == 1:
            #     break

        final_valid_metric /= idx
        print('Final 5-cv valid results, AUC: {:.4f},'.format(final_valid_metric[0]),
              'AUPR: {:.4f},'.format(final_valid_metric[1]),
              'F1: {:.4f},'.format(final_valid_metric[2]),
              'ACC: {:.4f},'.format(final_valid_metric[3]))

        final_test_metric /= idx
        print('Final 5-cv test results, AUC: {:.4f},'.format(final_test_metric[0]),
              'AUPR: {:.4f},'.format(final_test_metric[1]),
              'F1: {:.4f},'.format(final_test_metric[2]),
              'ACC: {:.4f},'.format(final_test_metric[3]))

        # Place your program code to be timed here
        # Record the end time of the program
        end_time = time.time()

        # Calculate the program's execution time
        elapsed_time = end_time - start_time

        # Print the program's execution time
        elapsed_time = int(elapsed_time)
        print(
            f"Program execution time: {elapsed_time//60} min {elapsed_time%60} sec")
        print("=" * 94)
        print()

        # file must close()
        # reason: texts are recordes in the buffer, instead of file
        # file.close()
