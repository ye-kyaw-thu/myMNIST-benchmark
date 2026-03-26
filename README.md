# myMNIST Benchmark

This repository provides a comprehensive benchmark of **PETNN**, **KAN**, and classical deep learning models for **Burmese Handwritten Digit Recognition** using the BHDD dataset.

This work is based on our paper:

> **myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition**  
> Accepted to ICNLP 2026

---

## 🔍 Overview

While MNIST has long served as a standard benchmark for handwritten digit recognition, similar standardized benchmarks for **Myanmar (Burmese) digits** remain limited.

In this work, we introduce **myMNIST Benchmark**, a systematic evaluation of **11 neural architectures** on the BHDD dataset, including:

- Classical deep learning models:
  - MLP, CNN, LSTM, GRU, Transformer
- Emerging architectures:
  - FastKAN, EfficientKAN
- Energy-based and physics-inspired models:
  - JEM
  - PETNN (Sigmoid, GELU, SiLU)

All models are implemented under a unified training pipeline to ensure **fair and reproducible comparison**.

---

## 📂 Repository Structure

```
.
├── code/ # Model implementations
├── data/ # Dataset description 
├── results/ # Experimental outputs
```

## 📊 Data Information

We used the **BHDD (Burmese Handwritten Digit Dataset)** for all experiments.

- 60,000 training images  
- 27,561 testing images  
- 10 classes (digits 0–9)  
- Image size: 28×28 grayscale  

To ensure fair comparison, we used the **original training/test split** provided by the dataset authors.

⚠️ The dataset is **not redistributed** in this repository.

👉 Please refer to the data documentation:  
https://github.com/ye-kyaw-thu/myMNIST-benchmark/tree/main/data  

---

## ⚙️ Experimental Settings

All models were implemented in **PyTorch** and trained under consistent conditions.

### Training Configuration

- Optimizer: AdamW  
- Learning rate: 3 × 10⁻⁴  
- Scheduler: OneCycleLR (max LR = 5 × 10⁻⁴)  
- Batch size:
  - Training: 128  
  - Evaluation: 1000  
- Loss: Cross-Entropy  
- Regularization:
  - Dropout (0.1–0.3)
  - Layer normalization
  - Gradient clipping (max norm = 1.0)  
- Training epochs: 30–100 (with early stopping)  
- Random seed: 42  

### Hardware

- GPU: NVIDIA RTX 3090 Ti (24GB VRAM)

---

## 📈 Results

### Table III: Performance comparison on BHDD dataset

| Model                | Precision | Recall | F1-Score | Accuracy |
|---------------------|----------|--------|----------|----------|
| MLP                 | 0.9810   | 0.9895 | 0.9852   | 0.9907   |
| CNN                 | 0.9955   | 0.9963 | 0.9959   | **0.9970** |
| LSTM                | 0.9907   | 0.9942 | 0.9924   | 0.9951   |
| GRU                 | 0.9886   | 0.9934 | 0.9910   | 0.9937   |
| Transformer         | 0.9898   | 0.9944 | 0.9921   | 0.9946   |
| JEM                 | 0.9931   | 0.9957 | 0.9944   | 0.9958   |
| FastKAN             | 0.9844   | 0.9914 | 0.9879   | 0.9922   |
| EfficientKAN        | 0.9841   | 0.9898 | 0.9869   | 0.9918   |
| PETNN (Sigmoid)     | 0.9897   | 0.9940 | 0.9918   | 0.9943   |
| PETNN (GELU)        | 0.9947   | 0.9963 | 0.9955   | 0.9966   |
| PETNN (SiLU)        | 0.9944   | 0.9961 | 0.9952   | 0.9964   |

### Key Findings

- **CNN** remains the strongest baseline (Accuracy = 0.9970)
- **PETNN (GELU)** achieves near state-of-the-art performance
- **JEM** performs competitively with strong calibration properties
- **KAN models** provide meaningful alternatives but lag behind top models

---

## 🔎 Error Analysis & Insights

Our analysis reveals that errors are highly **structured and script-dependent**.

### 1. Strong Confusion Between Digit 1 (၁) and 0 (၀)

- Most frequent error across almost all models
- Caused by:
  - Open vs. closed circular shapes
  - Handwriting variation closing or breaking strokes

---

### 2. Confusion Between Digit 1 (၁) and 5 (၅)

- Occurs when:
  - The tail of digit 5 is shortened or unclear
- Particularly observed in PETNN variants

---

### 3. Directional Confusion: Digit 3 (၃) → 4 (၄)

- More prominent in **KAN models**
- Caused by:
  - Subtle differences in stroke direction
  - Left-open vs. right-open shapes

---

## 💡 Conclusion

- CNN remains a strong and reliable baseline for Myanmar digit recognition  
- PETNN demonstrates **highly competitive performance**, especially with GELU/SiLU  
- Energy-based and physics-inspired models are promising directions  
- KAN models provide interpretability but require further improvement  

## 📌 Citation

If you use this work, please cite:  

```
@misc{thu2026mymnistbenchmarkpetnnkan,
      title={myMNIST: Benchmark of PETNN, KAN, and Classical Deep Learning Models for Burmese Handwritten Digit Recognition}, 
      author={Ye Kyaw Thu and Thazin Myint Oo and Thepchai Supnithi},
      year={2026},
      eprint={2603.18597},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.18597}, 
}
```
