# MetaAir

This repository contains code for implementing and evaluating Graph Convolutional Networks (GCNs) on the Airports dataset, focusing on meta-attacks with various perturbation levels.

## Files

- **main.py**: Main script to run the GCN training, evaluation, and perturbation experiments.
- **gcn_model.py**: Definition of GCN model architecture (`GraphConvolution`, `GCN`).
- **utils.py**: Utility functions for data loading, preprocessing, evaluation metrics, etc.
- **meta_attack.py**: Meta class for implementing perturbations and conducting evaluations.
- **results.csv**: CSV file containing accuracy results for different countries and perturbation levels.

## Requirements

- Python 3.x
- PyTorch
- torch_geometric
- NumPy
- SciPy
- scikit-learn
- Pandas
- Matplotlib


