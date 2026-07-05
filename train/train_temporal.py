import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from models.gcn import TwoGCNLayer

def train_temporal(A_norm_list, X_list, labels_list,
                   in_features=12, hidden=32, seq_len=8, epochs=20, lr=0.001):

    split_idx = int(0.8 * len(X_list))

    # build GCN embeddings
    gcn_model = TwoGCNLayer(in_features=in_features, hidden=hidden, out_features=16)
    embeddings = []
    with torch.no_grad():
        for t in range(len(A_norm_list)):
            X_t = torch.tensor(X_list[t], dtype=torch.float)
            A_t = torch.tensor(A_norm_list[t], dtype=torch.float)
            H1 = gcn_model.layer1(X_t, A_t)
            embeddings.append(H1)

    # GRU + final layer
    gru = nn.GRU(input_size=32, hidden_size=32, batch_first=True)
    final_layer = nn.Linear(32, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        list(gcn_model.parameters()) +
        list(gru.parameters()) +
        list(final_layer.parameters()),
        lr=lr
    )

    # training loop
    for epoch in range(epochs):
        total_loss = 0
        for t in range(seq_len, split_idx):
            seq = torch.stack(embeddings[t-seq_len:t]).permute(1, 0, 2)
            gru_out, _ = gru(seq)
            last = gru_out[:, -1, :]
            pred = final_layer(last)
            label_t = torch.tensor(labels_list[t], dtype=torch.long)
            loss = criterion(pred, label_t)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

        print(f"epoch {epoch}, loss = {total_loss.item()/split_idx:.4f}")

    # evaluation
    gcn_model.eval()
    gru.eval()
    final_layer.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for t in range(max(seq_len, split_idx), len(embeddings) - 1):
            seq = torch.stack(embeddings[t-seq_len:t]).permute(1, 0, 2)
            gru_out, _ = gru(seq)
            last = gru_out[:, -1, :]
            pred = final_layer(last)
            predicted = torch.argmax(pred, dim=1)
            label_t = torch.tensor(labels_list[t], dtype=torch.long)
            correct += (predicted == label_t).sum().item()
            total += len(label_t)

    accuracy = correct / total
    print(f"\nTemporal GCN Test Accuracy: {accuracy:.4f}")
    return gcn_model, gru, final_layer, accuracy
