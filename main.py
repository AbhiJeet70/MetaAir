import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import Airports
from torch_geometric.utils import to_scipy_sparse_matrix
from gcn_model import GCN
from meta_attack import Meta
from utils import load_airports_data, preprocess, get_train_val_test, accuracy, save_results_to_csv, plot_accuracies

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_attack_and_evaluate(countries, perturbation_levels, seed=42):
    results = {}

    for country in countries:
        print(f"Processing country: {country}")
        adj, features, labels = load_airports_data(country)
        n_nodes, n_features = features.shape
        n_classes = len(set(labels.cpu().numpy()))
        idx = np.arange(n_nodes)
        idx_train, idx_val, idx_test = get_train_val_test(idx, 0.7, 0.1, 0.15, stratify=labels.cpu().numpy())

        adj, features, labels = preprocess(adj, features, labels)

        meta = Meta(adj, features, labels, idx_train, idx_val, idx_test, nfeat=n_features, nhid=64, nclass=n_classes, dropout=0.5,
                    lr=0.01, weight_decay=5e-4, patience=100, epochs=500, device=device)

        results[country] = {'perturbation_results': {}}

        for perturbation in perturbation_levels:
            print(f"Starting attack and evaluation for perturbation level {perturbation * 100:.1f}%")
            perturbed_adj = meta.perturb_adj(perturbation)
            meta.adj = perturbed_adj  # Update the adjacency matrix in the meta object
            model = meta.train_gcn()
            loss_test, acc_test = meta.evaluate_gcn(model)

            print(f"Perturbation level: {perturbation * 100:.1f}%")
            print(f"Test Accuracy: {acc_test.cpu().item():.4f}")
            print("---")

            results[country]['perturbation_results'][perturbation] = acc_test.cpu().item()

    return results

if __name__ == "__main__":
    countries = ['USA', 'Brazil', 'Europe']
    perturbation_levels = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
    results = run_attack_and_evaluate(countries, perturbation_levels, seed=42)
    save_results_to_csv(results, 'results.csv')
    plot_accuracies(results)
