import torch
import numpy as np

class Meta:
    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test, nfeat, nhid, nclass, dropout, lr, weight_decay, patience, epochs, device):
        self.adj = adj.to(device)
        self.features = features.to(device)
        self.labels = labels.to(device)
        self.idx_train = torch.LongTensor(idx_train).to(device)
        self.idx_val = torch.LongTensor(idx_val).to(device)
        self.idx_test = torch.LongTensor(idx_test).to(device)
        self.nfeat = nfeat
        self.nhid = nhid
        self.nclass = nclass
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.patience = patience
        self.epochs = epochs
        self.device = device
        self.gcn = GCN(nfeat=nfeat, nhid=nhid, nclass=nclass, dropout=dropout).to(device)
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = torch.nn.NLLLoss().to(device)

    def train_gcn(self):
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0

        for epoch in range(self.epochs):
            self.gcn.train()
            self.optimizer.zero_grad()
            output = self.gcn(self.features, self.adj)
            loss_train = self.criterion(output[self.idx_train], self.labels[self.idx_train])
            loss_train.backward()
            self.optimizer.step()

            self.gcn.eval()
            output = self.gcn(self.features, self.adj)
            loss_val = self.criterion(output[self.idx_val], self.labels[self.idx_val])

            if loss_val < best_val_loss:
                best_val_loss = loss_val
                best_model = self.gcn.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.gcn.load_state_dict(best_model)
        return self.gcn

    def evaluate_gcn(self, model):
        model.eval()
        output = model(self.features, self.adj)
        loss_test = self.criterion(output[self.idx_test], self.labels[self.idx_test])
        acc_test = accuracy(output[self.idx_test], self.labels[self.idx_test])
        return loss_test.item(), acc_test

    def perturb_adj(self, perturbation):
        np.random.seed(42)
        num_edges = int(self.adj.sum() / 2)
        num_perturb = int(num_edges * perturbation)

        modified_adj = self.adj.clone()
        all_edges = np.array(np.triu_indices(self.adj.shape[0], k=1)).T

        np.random.shuffle(all_edges)
        perturbed_edges = all_edges[:num_perturb]

        for edge in perturbed_edges:
            modified_adj[edge[0], edge[1]] = 1 - modified_adj[edge[0], edge[1]]
            modified_adj[edge[1], edge[0]] = 1 - modified_adj[edge[1], edge[0]]

        return modified_adj.to(self.device)
