import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# PETNN Cell (as in paper/demos)
# ------------------------------
class PETNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        def init_weight(shape): return nn.Parameter(torch.randn(*shape) * 0.01)
        #def init_bias(shape): return nn.Parameter(torch.zeros(*shape))
        def init_bias(shape): return nn.Parameter(torch.zeros(*shape).squeeze())

        # Weights and biases
        self.W_Zt = init_weight((1, input_dim + hidden_dim))
        self.W_Zc = init_weight((cell_dim, input_dim + hidden_dim))
        self.W_Zw = init_weight((hidden_dim, input_dim + hidden_dim))
        self.W_It = init_weight((cell_dim, input_dim))
        self.W_Rt = init_weight((1, input_dim))
        self.W_h = init_weight((hidden_dim, input_dim + hidden_dim))

        self.b_Zt = init_bias((1, 1))
        self.b_Zc = init_bias((cell_dim, 1))
        self.b_Zw = init_bias((hidden_dim, 1))
        self.b_It = init_bias((cell_dim, 1))
        self.b_Rt = init_bias((1, 1))
        self.b_h = init_bias((hidden_dim, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, X_t, S_prev, C_prev, T_prev):
        batch_size = X_t.shape[0]

        concat_input = torch.cat((X_t, S_prev), dim=1)  # (B, I+H)

        Z_t = (F.linear(concat_input, self.W_Zt, self.b_Zt))  # (B, 1)
        Z_c = F.linear(concat_input, self.W_Zc, self.b_Zc)    # (B, C)
        Z_w = self.sigmoid(F.linear(concat_input, self.W_Zw, self.b_Zw))  # (B, H)
        I_t = F.linear(X_t, self.W_It, self.b_It)  # (B, C)
        R_t = self.sigmoid(F.linear(X_t, self.W_Rt, self.b_Rt))  # (B, 1)

        T_t_new = R_t * self.sigmoid(T_prev + Z_t) - 1
        m = (T_t_new <= 0).float()
        T_t_new = torch.clamp(T_t_new, min=0)

        C_t_new = (1 - m) * C_prev + m * I_t + Z_c
        C_for_h = (1 - m) * C_prev

        scalar_C = C_for_h.mean(dim=1, keepdim=True)  # (B, 1)
        S_prev_scaled = S_prev * scalar_C  # (B, H)

        h_input = torch.cat((X_t, S_prev_scaled), dim=1)  # (B, I+H)
        h_t = self.sigmoid(F.linear(h_input, self.W_h, self.b_h))  # (B, H)

        S_t_new = self.sigmoid((1 - Z_w) * S_prev + Z_w * h_t)
        return S_t_new, C_t_new, T_t_new

# ------------------------------
# PETNN Sequential Model
# ------------------------------
class PETNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_dim, output_dim, num_layers=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim

        self.layers = nn.ModuleList([PETNNCell(input_dim, hidden_dim, cell_dim)] +
                                    [PETNNCell(hidden_dim, hidden_dim, cell_dim) for _ in range(num_layers - 1)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        device = x.device

        S = torch.zeros(batch_size, self.hidden_dim).to(device)
        C = torch.zeros(batch_size, self.cell_dim).to(device)
        T = torch.ones(batch_size, 1).to(device) * 5

        for t in range(seq_len):
            X_t = x[:, t, :]
            for layer in self.layers:
                S, C, T = layer(X_t, S, C, T)

        return self.fc(S)

# ------------------------------
# CLI Arguments
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--cell_dim", type=int, default=1)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output", type=str, default="confusion_matrix.png")
    return parser.parse_args()

# ------------------------------
# Main Training and Evaluation
# ------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
        transforms.Lambda(lambda x: x.squeeze().T)  # (28, 28) → each column is a timestep
    ])

    train_set = datasets.MNIST(root=args.data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=args.data_dir, train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000)

    model = PETNNModel(28, args.hidden_dim, args.cell_dim, 10, args.layers).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # (B, 28, 28)
            out = model(x)
            loss = criterion(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            y_true.extend(y.numpy())
            y_pred.extend(preds)

    print("\n--- Evaluation Report ---")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(args.output)
    print(f"Saved confusion matrix to {args.output}")

if __name__ == "__main__":
    main()

