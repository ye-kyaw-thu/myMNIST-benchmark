# lstm_mnist_baseline.py
# Vibe coding with ChatGPT by Ye Kyaw Thu, Language Understanding Lab., Myanmar

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, num_layers=2, num_classes=10, 
                 dropout=0.25, use_norm=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Normalization
        self.norm = nn.LayerNorm(hidden_dim) if use_norm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        
        # Classifier
        self.fc = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1)
        
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0)
    
    def forward(self, x):
        # Input shape: (batch_size, 1, 28, 28)
        # Reshape to: (batch_size, 28, 28) - treating each row as a time step
        x = x.squeeze(1)  # Remove channel dimension
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, seq_length, hidden_dim)
        
        # Decode the hidden state of the last time step
        out = out[:, -1, :]
        out = self.norm(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    return total_loss / total, correct / total

def evaluate(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return y_true, y_pred, report

def plot_confusion(y_true, y_pred, path):
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
    parser = argparse.ArgumentParser(description='LSTM Baseline for MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=192, help='Hidden dimension size')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--use_norm', action='store_true', default=True, help='Use LayerNorm')
    parser.add_argument('--no_norm', action='store_false', dest='use_norm', help='Disable LayerNorm')
    parser.add_argument('--scheduler', action='store_true', default=True, help='Use learning rate scheduler')
    parser.add_argument('--no_scheduler', action='store_false', dest='scheduler', help='Disable scheduler')
    parser.add_argument('--output', type=str, default='lstm_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Data loading (same as PETNN)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=4)

    # Model
    model = LSTMClassifier(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        use_norm=args.use_norm
    ).to(device)

    # Optimizer and scheduler (same as PETNN)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    if args.scheduler:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr*1.67,
            steps_per_epoch=len(train_loader),
            epochs=args.epochs
        )
    else:
        scheduler = None
    
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluation
        y_true, y_pred, report = evaluate(model, test_loader, device)
        val_acc = report['accuracy']
        
        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Acc={val_acc:.4f}, F1={report['macro avg']['f1-score']:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.output}/best_model.pth")
            print(f"New best model saved with accuracy {best_acc:.4f}")

        if scheduler:
            scheduler.step()

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

