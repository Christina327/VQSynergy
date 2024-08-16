import os
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem
from sklearn.model_selection import KFold
import torch
import torch.utils.data as Data

from VQSynergy.config import DATASET_DIR, DEVICE, THRESHOLD
from utils import get_MACCS, set_seed_all
from drug_util import drug_feature_extract
from similarity import get_Cosin_Similarity, get_pvalue_matrix
from drug_util import GraphDataset, collate


def _get_data(dataset):
    # Determine the dataset name based on the input
    dataset_mapping = {
        'ONEIL': 'ONEIL-COSMIC',
        'ALMANAC': 'ALMANAC-COSMIC'
    }
    if dataset not in dataset_mapping:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
    dataset_name = dataset_mapping[dataset]

    drug_smiles_file = os.path.join(
        DATASET_DIR, dataset_name, 'drug_smiles.csv')
    cline_feature_file = os.path.join(
        DATASET_DIR, dataset_name, 'cell line_gene_expression.csv')
    drug_synergy_file = os.path.join(
        DATASET_DIR, dataset_name, 'drug_synergy.csv')

    # 1. Load Drug Data
    # drug_smile (SMILE) ==> MACCS and Graph
    # Prepare feature converter
    featurizer = dc.feat.ConvMolFeaturizer()
    # Load data in "SMILES" format
    drug_smiles_df = pd.read_csv(
        drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data, drug_smiles_features = pd.DataFrame(), list()

    for pubchemid, isosmiles in zip(drug_smiles_df['pubchemid'], drug_smiles_df['isosmiles']):
        # 1.1 Used for initial drug feature
        # "SMILES" string => Molecule object
        mol = Chem.MolFromSmiles(isosmiles)
        # Molecule object => 2D ndarray
        mol_features = featurizer.featurize(mol)
        # drug_data: {pubchemid: [atom-features, dense-adjacency]}
        drug_data[str(pubchemid)] = [
            mol_features[0].get_atom_features(),
            mol_features[0].get_adjacency_list()
        ]

        # 1.2 Used for reconstruction
        # Convert "SMILES" to zero-one vector (MACCS keys)
        # The public version contains 166 keys, each corresponding to a specific molecular feature (e.g., presence of a carbonyl group)
        drug_smiles_features.append(get_MACCS(isosmiles))

    # drug_features: {pubchemid: [atom-features, sparse-adjancy]} -> List[atom-features, sparse-adjancy]
    drug_features = drug_feature_extract(drug_data)

    # 2. Load Cell Line Data
    # gene expression ==> dataframe and Ndarray
    # gene_data: Cell line names and their features
    gene_data_df = pd.read_csv(
        cline_feature_file, sep=',', header=0, index_col=[0])
    # cline_features: Initial cell line features
    cline_features = np.array(gene_data_df, dtype='float32')

    # 3. SYNERGY
    synergy_df = pd.read_csv(drug_synergy_file, sep=',', header=0)

    # Assign unique IDs to drugs and cell lines
    # {pubchemid: drug_id}
    drug_map = {str(pubchemid): idx for idx,
                pubchemid in enumerate(drug_data.keys())}
    # {cname: cline_id}
    cline_map = {cname: idx + len(drug_map)
                 for idx, cname in enumerate(gene_data_df.index)}
    # synergy: [drug_a_id, drug_b_id, cell_line_id, synergy_score]
    synergy = [
        [
            drug_map[str(row[0])],  # drug_a_id: pubchemid => drug_id
            drug_map[str(row[1])],  # drug_b_id: pubchemid => drug_id
            cline_map[row[2]],      # cell_line_id => encoding numbe
            float(row[3])           # synergy_score
        ]
        for _, row in synergy_df.iterrows()
        if (str(row[0]) in drug_data.keys() and  # Check if drug_a_id is in drug_data
            # Check if drug_b_id is in drug_data
            str(row[1]) in drug_data.keys() and
            str(row[2]) in gene_data_df.index)   # Check if cell_line_id is in gene_data
    ]

    # drug_features         List[atom-features, sparse-adjancy], feat_dim=75
    # drug_smiles_features  List[zero-one vector for each drug]
    # cline_features        DataFrame[cname, feats], feat_dim=651
    # gene_data_df          DataFrame[feats]
    # synergy               List[drug_a_id, drug_b_id, cline_id, synergy]
    return drug_features, drug_smiles_features, cline_features, gene_data_df, synergy


def _get_similarity_matrics(drug_smiles_features, cline_features):
    # drug_smiles_fea: zero-one vector
    drug_sim_matrix = get_Cosin_Similarity(drug_smiles_features)
    cline_sim_matrix = get_pvalue_matrix(cline_features)
    matrices_np = [drug_sim_matrix, cline_sim_matrix]
    matries = [torch.from_numpy(matrix_np).float().to(DEVICE)
               for matrix_np in matrices_np]
    return matries


def load_data(dataset):
    drug_features, drug_smiles_features, cline_features, gene_data_df, synergy = _get_data(
        dataset)
    cline_features = torch.from_numpy(cline_features).to(DEVICE)

    # Update the synergy scores to be binary classification according to threshold
    for row in synergy:
        row[3] = 1 if row[3] >= THRESHOLD else 0

    # Get similarity matrix used for reconstruction loss
    similarity_matrices = _get_similarity_matrics(
        drug_smiles_features, np.array(gene_data_df, dtype='float32'))

    # Dict{pubchemid: (Feat, SparseAdj)}, Tensor, List[drug_a, drug_b, cline, synergy], Tensor, Tensor
    return drug_features, cline_features, synergy, similarity_matrices


def create_data_loaders(drug_feature, cline_feature):
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
    return drug_loader, cline_loader


def _split_dataframe_to_arrays(df, ratio, rd_seed):
    shuffled_df = df.sample(frac=1, random_state=rd_seed)
    shuffled_arr = np.array(shuffled_df)
    arr_train, arr_test = np.split(
        shuffled_arr, [int(ratio * len(shuffled_arr))])
    return arr_train, arr_test


def _split_synergy_data(synergy, cv_ratio, rd_seed):
    # Split synergy into positive and negative sets
    synergy_pos = pd.DataFrame([i for i in synergy if i[3] == 1])
    synergy_neg = pd.DataFrame([i for i in synergy if i[3] == 0])

    # Split each set into cross-validation and test subsets
    synergy_cv_pos, synergy_test_pos = _split_dataframe_to_arrays(
        synergy_pos, cv_ratio, rd_seed)
    synergy_cv_neg, synergy_test_neg = _split_dataframe_to_arrays(
        synergy_neg, cv_ratio, rd_seed)

    # Combine positive and negative cross-validation data
    data_cv_raw = np.concatenate((synergy_cv_neg, synergy_cv_pos), axis=0)

    # Combine positive and negative test data
    data_test = np.concatenate((synergy_test_neg, synergy_test_pos), axis=0)

    # Shuffle the datasets
    np.random.seed(rd_seed)
    np.random.shuffle(data_cv_raw)
    np.random.seed(rd_seed)
    np.random.shuffle(data_test)

    return data_cv_raw, data_test


def data_split(synergy, cv_ratio, rd_seed):
    set_seed_all(rd_seed)
    print("Balanced random split")

    # Split the synergy data
    data_cv_raw, data_test = _split_synergy_data(synergy, cv_ratio, rd_seed)

    # Convert test data to tensors
    tensor_test = torch.from_numpy(data_test).to(DEVICE)
    label_test = torch.from_numpy(
        np.array(data_test[:, 3], dtype='float32')).to(DEVICE)

    # Print dataset sizes
    print("    number of cross-validation samples:", len(data_cv_raw))
    print("    number of test samples: ", len(data_test))

    return data_cv_raw, tensor_test, label_test


def _process_cv_mode(data_cv_raw, cv_mode):
    if cv_mode == 1:
        # Random split
        return data_cv_raw
    elif cv_mode == 2:
        # Cell line level split
        return np.unique(data_cv_raw[:, 2])
    elif cv_mode == 3:
        # Drug pairs level split
        drugcomb = np.column_stack((data_cv_raw[:, 0], data_cv_raw[:, 1]))
        drugcomb = [tuple(sorted(pair)) for pair in drugcomb]
        return np.unique(drugcomb, axis=0)
    else:
        raise NotImplementedError("CV mode not implemented.")


def _split_data_cv(cv_mode, train_index, valid_index, data_cv, data_cv_raw):
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
                raise ValueError("Unexpected data.")
    elif cv_mode == 3:
        pair_train, pair_validation = data_cv[train_index], data_cv[valid_index]
        synergy_train, synergy_valid = [], []
        for row in data_cv_raw:
            drugcomb = sorted((row[0], row[1]))
            if any(all(x == y for x, y in zip(drugcomb, pair)) for pair in pair_train):
                synergy_train.append(row)
            else:
                synergy_valid.append(row)
    else:
        raise NotImplementedError("CV mode not implemented.")

    return np.array(synergy_train), np.array(synergy_valid)


def k_fold_cross_validation(data_cv_raw, cv_mode, num_splits, random_seed):
    # Process data based on cross-validation mode
    data_cv = _process_cv_mode(data_cv_raw, cv_mode)
    
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=random_seed)
    for idx, (train_index, valid_index) in enumerate(kf.split(data_cv), start=1):
        synergy_train, synergy_valid = _split_data_cv(
            cv_mode, train_index, valid_index, data_cv, data_cv_raw)
        yield idx, synergy_train, synergy_valid
