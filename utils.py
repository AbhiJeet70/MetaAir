import torch
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.datasets import Airports

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_airports_data(country):
    dataset = Airports(root='/tmp/Airports', name=country)
    data = dataset[0]
    adj = to_scipy_sparse_matrix(data.edge_index)
    features = data.x.to(device)
    labels = data.y.to(device)
    return adj, features, labels

def preprocess(adj, features, labels, preprocess_adj=True, preprocess_feature=True, sparse=False):
    if preprocess_adj:
        adj = adj + sp.eye(adj.shape[0])
    if preprocess_feature:
        features = features / features.sum(1, keepdims=True)
    if sparse:
        adj = sp.coo_matrix(adj)
        adj = torch.sparse.FloatTensor(
            torch.LongTensor([adj.row, adj.col]),
            torch.FloatTensor(adj.data),
            torch.Size(adj.shape)
        ).to(device)
    else:
        adj = torch.FloatTensor(np.array(adj.todense())).to(device)
    features = torch.FloatTensor(np.array(features.cpu())).to(device)
    labels = torch.LongTensor(labels.cpu()).to(device)
    return adj, features, labels

def get_train_val_test(idx, train_size, val_size, test_size, stratify=None):
    idx_train, idx_test = train_test_split(idx, train_size=train_size, test_size=test_size, stratify=stratify)
    idx_train, idx_val = train_test_split(idx_train, train_size=(train_size / (train_size + val_size)), test_size=(val_size / (train_size + val_size)), stratify=stratify[idx_train] if stratify is not None else None)
    return idx_train, idx_val, idx_test

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
