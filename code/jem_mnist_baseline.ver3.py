# jem_mnist_baseline.ver3.py
# Vibe coding with ChatGPT by Ye Kyaw Thu, Language Understanding Lab., Myanmar

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
import random
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# -----------------------------
# Architecture
# -----------------------------
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.swish = Swish()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.swish(self.conv1(x))
        x = self.swish(self.conv2(x))
        x = self.swish(self.conv3(x))
        x = self.flatten(x)
        x = self.swish(self.fc1(x))
        return self.fc2(x)

# -----------------------------
# Sampler with Langevin Dynamics
# -----------------------------
class Sampler:
    def __init__(self, model, img_shape, sample_size, max_len=8192):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(sample_size)]

    def sample(self, steps=60, step_size=10):
        device = next(self.model.parameters()).device
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape, device=device) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size - n_new), dim=0).to(device)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach()
        inp_imgs.requires_grad = True

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        for _ in range(steps):
            noise = torch.randn_like(inp_imgs) * 0.005
            inp_imgs.data.add_(noise)
            inp_imgs.data.clamp_(-1, 1)
            energy = -self.model(inp_imgs).logsumexp(dim=1).mean()
            energy.backward()
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(-1, 1)

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

        self.examples = list(inp_imgs.cpu().chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[:self.max_len]
        return inp_imgs.detach()

    def sample_new_exmps(self, steps=60, step_size=10):
        device = next(self.model.parameters()).device
        inp_imgs = torch.rand((self.sample_size,) + self.img_shape, device=device) * 2 - 1
        inp_imgs.requires_grad = True

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        for _ in range(steps):
            noise = torch.randn_like(inp_imgs) * 0.005
            inp_imgs.data.add_(noise)
            inp_imgs.data.clamp_(-1, 1)
            energy = -self.model(inp_imgs).logsumexp(dim=1).mean()
            energy.backward()
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(-1, 1)

        for p in self.model.parameters():
            p.requires_grad = True
        self.model.train()

        return inp_imgs.detach()

# -----------------------------
# Utility Functions
# -----------------------------
def calculate_ece(confidences, predictions, labels, num_bins=15):
    # Convert inputs to numpy arrays if they're tensors
    if isinstance(confidences, torch.Tensor):
        confidences = confidences.numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Create bins with equal sample sizes (quantile bins)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)
    
    for i in range(num_bins):
        in_bin = (confidences > bin_lowers[i]) & (confidences <= bin_uppers[i])
        bin_sizes[i] = np.sum(in_bin)
        
        if bin_sizes[i] > 0:
            # Convert boolean to float before mean calculation
            bin_accs[i] = np.mean((predictions[in_bin] == labels[in_bin]).astype(float))
            bin_confs[i] = np.mean(confidences[in_bin])
    
    # Filter out empty bins
    non_empty = bin_sizes > 0
    bin_accs = bin_accs[non_empty]
    bin_confs = bin_confs[non_empty]
    bin_sizes = bin_sizes[non_empty]
    bin_centers = (bin_lowers + bin_uppers)[non_empty] / 2
    
    ece = np.sum((bin_sizes / np.sum(bin_sizes)) * np.abs(bin_accs - bin_confs))
    return ece, bin_centers, bin_accs, bin_confs

def plot_reliability_diagram(ece, bin_centers, bin_accs, bin_confs, accuracy, title, path):
    plt.figure(figsize=(10, 10))
    
    # Plot accuracy bars
    bar_width = 0.05  # Fixed width for better visibility
    plt.bar(bin_centers, bin_accs, width=bar_width, alpha=0.7, 
            color='royalblue', edgecolor='black', linewidth=1, label='Accuracy')
    
    # Plot confidence line
    plt.plot(bin_centers, bin_confs, 'o-', color='crimson', 
             linewidth=3, markersize=8, label='Confidence')
    
    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, 
             label='Perfect Calibration')
    
    # Add ECE annotation
    plt.text(0.02, 0.95, f"ECE: {ece*100:.2f}%", fontsize=16,
             bbox=dict(facecolor='white', edgecolor='black', alpha=0.9))
    
    # Formatting
    plt.title(f"{title}\nAccuracy: {accuracy*100:.2f}%", fontsize=18, pad=20)
    plt.xlabel("Confidence", fontsize=16)
    plt.ylabel("Accuracy", fontsize=16)
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=14)
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    plt.legend(loc='upper left', fontsize=14, framealpha=1)
    
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_images_with_sampler(model, sampler, num_images=16, steps=2000, step_size=0.1):
    model.eval()
    generated_imgs = sampler.sample_new_exmps(steps=steps, step_size=step_size)
    generated_imgs = generated_imgs[:num_images].detach()
    with torch.no_grad():
        logits = model(generated_imgs.to(next(model.parameters()).device))
        preds = logits.argmax(1).cpu().numpy()
    return generated_imgs.cpu(), preds

def plot_generated_images(images, preds, save_path, num_cols=4):
    num_images = images.size(0)
    num_rows = (num_images + num_cols - 1) // num_cols
    images = (images + 1) / 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i].squeeze(0).numpy()
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Pred: {preds[i]}", fontsize=10)
            ax.axis("off")
        else:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

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

# -----------------------------
# Training
# -----------------------------
def train_one_epoch(model, loader, optimizer, sampler, alpha, device):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    total_loss = total_acc = 0

    for real_imgs, labels in loader:
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        fake_imgs = sampler.sample()

        logits_real = model(real_imgs)
        logits_fake = model(fake_imgs)

        ce_loss = ce_loss_fn(logits_real, labels)
        energy_real = logits_real.logsumexp(dim=1).mean()
        energy_fake = logits_fake.logsumexp(dim=1).mean()
        
        # Add gradient clipping and loss scaling
        cd_loss = torch.clamp(energy_fake - energy_real, -10, 10)
        reg_loss = alpha * torch.clamp((logits_real**2).mean() + (logits_fake**2).mean(), 0, 10)
        
        loss = ce_loss + cd_loss + reg_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()

        preds = logits_real.argmax(1)
        total_loss += loss.item() * real_imgs.size(0)
        total_acc += (preds == labels).sum().item()

    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred, confidences = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(y.cpu().numpy())
            # Use maximum probability as confidence
            confidences.extend(probs.max(dim=1).values.cpu().numpy())

    # Convert to numpy arrays and ensure proper types
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidences = np.asarray(confidences)
    
    # Clip confidences to avoid numerical issues
    confidences = np.clip(confidences, 0.001, 0.999)
    
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return y_true, y_pred, confidences, report

def main():
    parser = argparse.ArgumentParser(description='Train JEM on MNIST')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--alpha', type=float, default=1e-5, help='Regularization weight')
    parser.add_argument('--output', type=str, default='jem_results')
    parser.add_argument('--visuals', type=str, default='jem_visuals')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.visuals, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=1000)

    model = EnergyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sampler = Sampler(model, (1, 28, 28), args.batch_size)

    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, sampler, args.alpha, device)
        y_true, y_pred, confidences, report = evaluate(model, test_loader, device)
        val_acc = report['accuracy']

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output, 'best_model.pth'))
            print(f"[Saved best model] Accuracy: {best_acc:.4f}")

    model.load_state_dict(torch.load(os.path.join(args.output, 'best_model.pth')))
    y_true, y_pred, confidences, report = evaluate(model, test_loader, device)

    print("\n=== Final Evaluation ===")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Precision: {report['macro avg']['precision']:.4f}")
    print(f"Recall: {report['macro avg']['recall']:.4f}")
    print(f"F1: {report['macro avg']['f1-score']:.4f}")

    # Save confusion matrix
    plot_confusion(y_true, y_pred, os.path.join(args.output, 'confusion_matrix.png'))

    # Save classification report
    with open(os.path.join(args.output, 'classification_report.txt'), 'w') as f:
        f.write(classification_report(y_true, y_pred, digits=4))

    # Save calibration diagram
    ece, bin_centers, bin_accs, bin_confs = calculate_ece(torch.tensor(confidences), torch.tensor(y_pred), torch.tensor(y_true))
    plot_reliability_diagram(ece, bin_centers, bin_accs, bin_confs, best_acc, "JEM Classifier", os.path.join(args.visuals, 'reliability_diagram.png'))

    # Save generated images
    #generated_images = generate_images_with_sampler(model, sampler, num_images=16, steps=2000, step_size=0.1)
    #plot_generated_images(generated_images, os.path.join(args.visuals, 'generated_images.png'))

    generated_images, generated_preds = generate_images_with_sampler(model, sampler, num_images=16, steps=2000, step_size=0.1)
    plot_generated_images(generated_images, generated_preds, os.path.join(args.visuals, 'generated_images.png'))

if __name__ == '__main__':
    main()

