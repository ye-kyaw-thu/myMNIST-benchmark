# transformer_mnist_baseline.py
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

class MNISTTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(28, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, 28, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 10)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x = x.squeeze(1)  # [B, 1, 28, 28] → [B, 28, 28]
        x = self.embedding(x)  # [B, 28, d_model]
        x = x + self.pos_encoder
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
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
    parser = argparse.ArgumentParser(description='Transformer Baseline for MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--d_model', type=int, default=64, help='Embedding dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Transformer layers')
    parser.add_argument('--scheduler', action='store_true', default=True, help='Use LR scheduler')
    parser.add_argument('--no_scheduler', action='store_false', dest='scheduler', help='Disable scheduler')
    parser.add_argument('--output', type=str, default='transformer_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Data loading (identical to CNN baseline)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=4)

    # Model and optimizer (same config as CNN)
    model = MNISTTransformer(d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr*1.67, steps_per_epoch=len(train_loader), epochs=args.epochs) if args.scheduler else None
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        y_true, y_pred, report = evaluate(model, test_loader, device)
        val_acc = report['accuracy']
        
        print(f"Epoch {epoch+1}/{args.epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"{args.output}/best_model.pth")
            print(f"New best model saved with accuracy {best_acc:.4f}")
        
        if scheduler:
            scheduler.step()

    # Final evaluation and plots
    model.load_state_dict(torch.load(f"{args.output}/best_model.pth"))
    y_true, y_pred, report = evaluate(model, test_loader, device)
    plot_confusion(y_true, y_pred, f"{args.output}/confusion_matrix.png")  # <-- Added this line
    
    with open(f"{args.output}/classification_report.txt", 'w') as f:
        f.write(classification_report(y_true, y_pred, digits=4))

if __name__ == "__main__":
    main()


