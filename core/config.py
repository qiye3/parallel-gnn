# core/config.py

class Config:
    # Model
    hidden_dim = 64
    num_layers = 2
    num_neighbors = [15, 10]

    # Training
    lr = 1e-3
    batch_size = 1024
    epochs = 4

    # Evaluation
    K = 10
    num_neg = 100
