import os
import numpy as np
import pandas as pd
import deepchem as dc
from rdkit import Chem

from utils import get_MACCS
from drug_util import drug_feature_extract


def getData(dataset):
    if dataset == 'ONEIL':
        dataset_name = 'ONEIL-COSMIC'
    elif dataset == "ALMANAC":
        dataset_name = 'ALMANAC-COSMIC'
    else:
        raise NotImplementedError
    DATASET_DIR = '../Data'
    drug_smiles_file = os.path.join(DATASET_DIR, dataset_name, 'drug_smiles.csv')
    cline_feature_file = os.path.join(DATASET_DIR, dataset_name, 'cell line_gene_expression.csv')
    drug_synergy_file = os.path.join(DATASET_DIR, dataset_name, 'drug_synergy.csv')

    # 1. DRUG
    # drug_smile (SMILE) ==> MACCS and Graph
    # prepare feature converter
    featurizer = dc.feat.ConvMolFeaturizer()
    # load data in "SMILES" format
    drug = pd.read_csv(drug_smiles_file, sep=',', header=0, index_col=[0])
    drug_data, drug_smiles_fea = pd.DataFrame(), list()
    for pubchemid, isosmiles in zip(drug['pubchemid'], drug['isosmiles']):

        # 1.1 used for initial drug feature
        # "smiles" string => Molecule object
        mol = Chem.MolFromSmiles(isosmiles)
        # Molecule object => 2-d nd_array
        mol_f = featurizer.featurize(mol)
        # drug_data: {pubchemid: [atom-features, dense-adjacency]}
        drug_data[str(pubchemid)] = [mol_f[0].get_atom_features(), mol_f[0].get_adjacency_list()]

        # 1.2 used for reconstruction
        # convert "smiles" to zero-one vector
        drug_smiles_fea.append(get_MACCS(isosmiles))
    
    # drug_fea: {pubchemid: [atom-features, sparse-adjancy]} -> List[atom-features, sparse-adjancy]
    drug_fea = drug_feature_extract(drug_data)


    # 2. CELL LINE
    # gene expression ==> dataframe and ndarray
    # gene_data: cline name and its feature
    gene_data = pd.read_csv(cline_feature_file, sep=',', header=0, index_col=[0])
    # cline_fea: initial cell line feature
    cline_fea = np.array(gene_data, dtype='float32')


    # 3. SYNERGY
    synergy_load = pd.read_csv(drug_synergy_file, sep=',', header=0)

    # assign unique id to drugs and cline
    # {pubchemid: drug_id}
    drug_num = len(drug_data.keys())
    drug_map = dict(zip(drug_data.keys(), range(drug_num)))
    # {cname: cline_id}
    cline_num = len(gene_data.index)
    cline_map = dict(zip(gene_data.index, range(drug_num, drug_num + cline_num)))
    # synergy: [drug_a_id, drug_b_id, cell_line_id, synergy_score]
    synergy = [
        [
            drug_map[str(row[0])],  # drug_a: pubchemid => my_drug_id
            drug_map[str(row[1])],  # drug_b: pubchemid => my_drug_id
            cline_map[row[2]],  # cell line => encoding num
            float(row[3])  # synergy
        ]
        for _, row in synergy_load.iterrows()
        if (str(row[0]) in drug_data.keys()  # drug 1
            and str(row[1]) in drug_data.keys()  # drug 2
            and str(row[2]) in gene_data.index)  # cell line
    ]
    
    # drug_data         List[atom-features, sparse-adjancy], feat_dim=75
    # drug_smiles_fea   List[zero-one vector for each drug]
    # cline_fea         DataFrame[cname, feats], feat_dim=651
    # gene_data         Array[feats]
    # synergy           List[drug_a_id, drug_b_id, cline_id, synergy] 
    return drug_fea, drug_smiles_fea, cline_fea, gene_data, synergy