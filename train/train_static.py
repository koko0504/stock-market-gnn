import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append('..')
from models.gcn import TwoGCNLayer

def train_static(X_train, labels_train, X_test, labels_test, A_norm,
                 in_features=20, hidden=32, epochs=100, lr=0.01):

    model = TwoGCNLayer(in_features=in_features, hidden=hidden, out_features=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    A_torch = torch.tensor(A_norm, dtype=torch.float)

    # training
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_train)):
            X_t = torch.tensor(X_train[i], dtype=torch.float)
            label_t = torch.tensor(labels_train[i], dtype=torch.long)

            out = model(X_t, A_torch)
            loss = criterion(out, label_t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"epoch {epoch}, avg train loss = {total_loss/len(X_train):.4f}")

    # evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(len(X_test)):
            X_t = torch.tensor(X_test[i], dtype=torch.float)
            label_t = torch.tensor(labels_test[i], dtype=torch.long)
            out = model(X_t, A_torch)
            predicted = torch.argmax(out, dim=1)
            correct += (predicted == label_t).sum().item()
            total += len(label_t)

    accuracy = correct / total
    print(f"\nStatic GCN Test Accuracy: {accuracy:.4f}")
    return model, accuracy
