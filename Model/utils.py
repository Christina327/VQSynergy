import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_recall_curve, mean_squared_error, r2_score
from scipy.stats import pearsonr
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import random


def set_seed_all(rd_seed):
    random.seed(rd_seed)
    np.random.seed(rd_seed)
    torch.manual_seed(rd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rd_seed)
        torch.cuda.manual_seed_all(rd_seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class TruncatedExponentialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, min_lr=0, last_epoch=-1):
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs]


class IdenticalLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return self.base_lrs


class LambdaLayer(torch.nn.Module):
    def __init__(self, func):
        super(LambdaLayer, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)


def get_metrics(yt, yp):
    # 使用内置函数计算AUC和AUPR
    auc = roc_auc_score(yt, yp)
    precision, recall, _ = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    return auc, aupr


class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df
    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """

    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)


def get_cfps(mol, radius=2, nBits=256, useFeatures=False, counts=False, dtype=np.float32):
    """Calculates circural (Morgan) fingerprint.
    http://rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn
    Returns
    -------
    ML.FP
        Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures,
                                                   bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array(
            [len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))


def get_fingerprint_from_smiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp


def get_MACCS(smiles: str):
    # "smiles" string => Molecule object
    m = Chem.MolFromSmiles(smiles)
    arr = np.zeros((1,), np.float32)
    # fingerprint
    fp = MACCSkeys.GenMACCSKeys(m)  # ExplicitBitVect
    DataStructs.ConvertToNumpyArray(fp, arr)  # 1-d nd_array
    return arr


def regression_metric(ytrue, ypred):
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, p = pearsonr(ytrue, ypred)
    return rmse, r2, r
