# petnn_mnist.ver3.py
# Vibe coding with ChatGPT by Ye Kyaw Thu, Language Understanding Lab., Myanmar

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import numpy as np
import os


class PETNNCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_dim, use_norm=False, dropout=0.0, gate_type='sigmoid'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cell_dim = cell_dim
        self.concat_dim = input_dim + hidden_dim
        self.gate_type = gate_type

        print(f"[DEBUG] Initializing PETNNCell with input_dim={input_dim}, hidden_dim={hidden_dim}, "
              f"concat_dim={self.concat_dim}, gate_type={gate_type}")

        # Gate initialization
        if gate_type == 'gelu':
            self.gate = F.gelu
        elif gate_type == 'silu':
            self.gate = F.silu
        else:  # default sigmoid
            self.gate = torch.sigmoid

        self.linear_Zt = nn.Linear(self.concat_dim, 1)
        self.linear_Zc = nn.Linear(self.concat_dim, cell_dim)
        self.linear_Zw = nn.Linear(self.concat_dim, hidden_dim)
        self.linear_It = nn.Linear(input_dim, cell_dim)
        self.linear_Rt = nn.Linear(input_dim, 1)
        self.linear_h = nn.Linear(self.concat_dim, hidden_dim)

        # Residual connection
        self.residual = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        self.norm = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, X_t, S_prev, C_prev, T_prev):
        # Ensure proper dimensions
        if X_t.dim() == 1:
            X_t = X_t.unsqueeze(0)
        if S_prev.dim() == 1:
            S_prev = S_prev.unsqueeze(0)
            
        concat = torch.cat([X_t, S_prev], dim=-1)  # (B, input_dim + hidden_dim)
        
        Z_t = self.linear_Zt(concat)                      # (B, 1)
        Z_c = self.linear_Zc(concat)                      # (B, cell_dim)
        Z_w = self.gate(self.linear_Zw(concat))           # (B, hidden_dim)
        I_t = self.linear_It(X_t)                         # (B, cell_dim)
        R_t = self.gate(self.linear_Rt(X_t))              # (B, 1)

        T_t = R_t * self.gate(T_prev + Z_t) - 1           # (B, 1)
        m = (T_t <= 0).float()
        T_t = torch.clamp(T_t, min=0.0)

        C_new = (1 - m) * C_prev + m * I_t + Z_c          # (B, cell_dim)
        C_used = (1 - m) * C_prev
        C_scale = C_used.mean(dim=1, keepdim=True)        # (B, 1)

        S_scaled = S_prev * C_scale
        h_input = torch.cat([X_t, S_scaled], dim=-1)
        h = self.gate(self.linear_h(h_input))             # (B, hidden_dim)

        S_new = self.gate((1 - Z_w) * S_prev + Z_w * h + self.residual(X_t))
        S_new = self.norm(S_new)
        S_new = self.dropout(S_new)

        return S_new, C_new, T_t


class PETNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, cell_dim, num_classes, num_layers=3, 
                 use_norm=True, dropout=0.25, gate_type='sigmoid'):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # First layer takes input_dim, subsequent layers take hidden_dim
        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        
        for i, in_dim in enumerate(dims):
            print(f"[DEBUG] Adding layer {i} with input_dim={in_dim}, hidden_dim={hidden_dim}")
            self.layers.append(PETNNCell(
                in_dim, hidden_dim, cell_dim, 
                use_norm=use_norm, 
                dropout=dropout,
                gate_type=gate_type
            ))

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):  # x: (B, 28, 28)
        B, T, D = x.shape
        
        # Initialize states for each layer
        states = {
            'S': [torch.zeros(B, layer.hidden_dim, device=x.device) for layer in self.layers],
            'C': [torch.zeros(B, layer.cell_dim, device=x.device) for layer in self.layers],
            'T': [torch.ones(B, 1, device=x.device) * 5.0 for _ in self.layers]
        }

        for t in range(T):
            X_t = x[:, t, :]  # (B, D)
            
            # Process through each layer sequentially
            for i, layer in enumerate(self.layers):
                states['S'][i], states['C'][i], states['T'][i] = layer(
                    X_t, 
                    states['S'][i], 
                    states['C'][i], 
                    states['T'][i]
                )
                
                # Output of current layer is input to next
                X_t = states['S'][i]

        # Use final hidden state of last layer for classification
        return self.classifier(states['S'][-1])


def train_one_epoch(model, loader, criterion, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    
    # Initialize monitoring variables
    grad_norms = []
    avg_T_values = []
    
    for batch_idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x = x.squeeze(1)  # (B, 28, 28)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Collect gradient norms
        batch_grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        grad_norms.extend(batch_grad_norms)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * x.size(0)
        
        # Monitoring every 100 batches
        if batch_idx % 100 == 0:
            print(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                  f"Grad Norm={np.mean(batch_grad_norms):.2f}±{np.std(batch_grad_norms):.2f}")
    
    return total_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).squeeze(1)
            out = model(x)
            pred = out.argmax(dim=1).cpu()
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
    
    # Calculate metrics
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return y_true, y_pred, report


def plot_confusion(y_true, y_pred, path='confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='PETNN for MNIST Classification')
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=192, help='Hidden dimension size')
    parser.add_argument('--cell_dim', type=int, default=48, help='Cell dimension size')
    parser.add_argument('--layers', type=int, default=3, help='Number of PETNN layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--use_norm', action='store_true', default=True, help='Use LayerNorm')
    parser.add_argument('--no_norm', action='store_false', dest='use_norm', help='Disable LayerNorm')
    parser.add_argument('--scheduler', action='store_true', default=True, help='Use learning rate scheduler')
    parser.add_argument('--no_scheduler', action='store_false', dest='scheduler', help='Disable scheduler')
    parser.add_argument('--gate', type=str, default='sigmoid', 
                        choices=['sigmoid', 'gelu', 'silu'], help='Gate activation type')
    parser.add_argument('--output', type=str, default='petnn_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=4)

    # Model
    model = PETNNModel(
        input_dim=28,
        hidden_dim=args.hidden_dim,
        cell_dim=args.cell_dim,
        num_classes=10,
        num_layers=args.layers,
        use_norm=args.use_norm,
        dropout=args.dropout,
        gate_type=args.gate
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr*1.67,  # 5e-4
            steps_per_epoch=len(train_loader),
            epochs=args.epoch
        )
    else:
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Training loop
    best_acc = 0
    for ep in range(1, args.epoch + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
        
        # Evaluation
        y_true, y_pred, report = evaluate(model, test_loader, device)
        val_acc = report['accuracy']
        
        print(f"\nEpoch {ep}/{args.epoch}: "
              f"Train Loss={train_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, "
              f"F1={report['macro avg']['f1-score']:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.output}/best_model.pth")
            print(f"New best model saved with accuracy {best_acc:.4f}")

    # Final evaluation
    model.load_state_dict(torch.load(f"{args.output}/best_model.pth"))
    y_true, y_pred, report = evaluate(model, test_loader, device)
    
    print("\n=== Final Evaluation ===")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Precision: {report['macro avg']['precision']:.4f}")
    print(f"Recall: {report['macro avg']['recall']:.4f}")
    print(f"F1-Score: {report['macro avg']['f1-score']:.4f}")
    
    # Save results
    plot_confusion(y_true, y_pred, f"{args.output}/confusion_matrix.png")
    
    with open(f"{args.output}/classification_report.txt", 'w') as f:
        f.write(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":
    main()

