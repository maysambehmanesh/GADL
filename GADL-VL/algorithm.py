import numpy as np
import torch
from munkres import Munkres
from tqdm import tqdm
from scipy.sparse import csr_matrix

def greedy_match(X):
    X = X.cpu().numpy()
    m, n = X.shape
    minSize = min(m, n)
    usedRows = np.zeros(m, dtype=bool)
    usedCols = np.zeros(n, dtype=bool)
    maxList = np.zeros(minSize)
    row = np.zeros(minSize, dtype=int)
    col = np.zeros(minSize, dtype=int)
    x = X.flatten()
    ix = np.argsort(-x)
    matched = 0
    index = 0
    while matched < minSize:
        ipos = ix[index]
        jc = ipos // m
        ic = ipos % m
        if not usedRows[ic] and not usedCols[jc]:
            row[matched] = ic
            col[matched] = jc
            maxList[matched] = x[ipos]
            usedRows[ic] = True
            usedCols[jc] = True
            matched += 1
        index += 1
    data = np.ones(minSize)
    M = csr_matrix((data, (row, col)), shape=(m, n))
    return M

def get_match(D,device):
    P = torch.zeros_like(D)
    size = D.shape[0]
    index_S = [i for i in range(size)]
    index_S_hat = [i for i in range(size)]
    
    # D = D.clone().detach().cpu()  # avoid modifying computation graph

    for i in range(size):
        cur_size = D.shape[0]
        argmin = torch.argmin(D.to(device)).item()
        r = argmin // cur_size
        c = argmin % cur_size
        P[index_S[r]][index_S_hat[c]] = 1
        index_S.remove(index_S[r])
        index_S_hat.remove(index_S_hat[c])
        D = D[torch.arange(D.size(0)) != r]
        D = D.t()[torch.arange(D.t().size(0)) != c].t()
    return P.t()

def hungarian(D):
    P = torch.zeros_like(D)
    matrix = D.tolist()
    m = Munkres()
    indexes = m.compute(matrix)
    total = 0
    for r,c in indexes:
        P[r][c] = 1
        total += matrix[r][c]
    return P.t()



