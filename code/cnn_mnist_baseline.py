# cnn_mnist_baseline.py
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

class CNNClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(CNNClassifier, self).__init__()
        # 1-layer CNN as specified in the paper (Conv2d(3,3))
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (28x28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # (14x14)
            
            # Additional conv layer to match capacity (not in paper spec but needed for decent performance)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (14x14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (7x7)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )
        
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

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
        
        # Gradient clipping for consistency
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
    parser = argparse.ArgumentParser(description='CNN Baseline for MNIST')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--scheduler', action='store_true', default=True, help='Use learning rate scheduler')
    parser.add_argument('--no_scheduler', action='store_false', dest='scheduler', help='Disable scheduler')
    parser.add_argument('--output', type=str, default='cnn_results', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # Data loading (consistent with other baselines)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=1000, num_workers=4)

    # Model
    model = CNNClassifier().to(device)

    # Optimizer and scheduler (consistent with PETNN)
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

