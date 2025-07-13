# 🎨 Photo-to-Monet Image Translation using CycleGAN & Contrastive Learning

This repository contains our final project for the course  
**"Deep Learning and Its Application to Signal and Image Processing and Analysis"**,  
focused on unpaired image translation from photographs to Monet-style paintings.

---

## 📌 Project Overview

### 🎯 Objective

Develop a model that translates real-world **photos** into the artistic style of **Claude Monet**,  
while preserving the original image structure.

We implemented and evaluated two models:
- **Model 1**: Vanilla CycleGAN - https://arxiv.org/abs/1703.10593
- **Model 2**: CycleGAN & PatchNCE-based model

---

## 🧠 Key Challenges

- Small Monet dataset (only 300 images)
- No paired data between Photo and Monet domains
- Achieving both realism and semantic preservation

---

## 🗂️ Dataset

| Class  | Count | Type | Size          |
|--------|-------|------|----------------|
| Photo  | 7,038 | RGB  | 256×256        |
| Monet  | 300   | RGB  | 256×256        |

- Training set: 200 random images from each class (with seeds 42, 123, 2025)
- Test set: 893 Monet paintings from [TFDS Monet Dataset](https://www.kaggle.com/datasets/dimitreoliveira/monet-paintings-jpg-berkeley)

---

## 🏗️ Model Architectures

### 🌀 Model 1 – Vanilla CycleGAN

- Two generators (G_AB, G_BA) and two discriminators (D_A, D_B)
- Uses:
  - **Cycle-consistency loss**
  - **Identity loss**
  - **Adversarial loss (LSGAN)**

#### Generator Loss:











---

### ✨ Model 2 – PatchNCE (Contrastive Learning)

- Removed cycle and identity losses
- Introduced **PatchNCE loss** for contrastive feature alignment
- Generator encourages semantic similarity between input & output patches

#### Generator Loss:




Default: `λ_NCE = 1.0`

---

## 📏 Evaluation – MiFID

We used the **MiFID** (Memorization-informed FID) metric, which penalizes overfitting.  
It is computed between generated Monet-style images and a separate Monet test set.

---

## 📊 Results Summary

| Model        | Mean MiFID | Std. Deviation |
|--------------|------------|----------------|
| Vanilla      | 91.537     | 3.456          |
| PatchNCE     | 78.448     | 2.987          |

---

## 🧪 Ablation Study – PatchNCE Weight (λ)

To evaluate the effect of the PatchNCE loss on training, we varied `λ_NCE`:

| λ_NCE | MiFID   | Feature Distance |
|--------|---------|------------------|
| 1.0    | 76.711  | 0.254            |
| 0.35   | 80.272  | 0.257            |
| 0.01   | 293.404 | 0.353            |

**Conclusion**:  
High contrastive supervision (λ = 1.0) is crucial for semantic alignment and stable training.  
Low λ leads to poor generalization and unstable convergence.

---

## 🖼️ Sample Outputs

All qualitative results can be found in the `outputs/` folder:

- `positive_performance_model2.png`
- `both_good.png`
- `model1_good_model2_bad.png`
- `model2_good_model1_bad.png`
- `exapmle_lamba035.png`
- `exapmle_lamba001.png`

---

## ⚙️ Training Details

| Hyperparameter       | Value                     |
|----------------------|---------------------------|
| Epochs               | 100 (with linear decay)   |
| Batch size           | 4                         |
| Learning rate        | 0.0002                    |
| Optimizer            | Adam (β1=0.5, β2=0.999)   |
| GAN Loss Type        | Least Squares GAN (LSGAN) |
| PatchNCE Temperature | 0.07                      |
| Contrastive Patches  | 256                       |
| Feature Layers       | 0, 4, 8, 12, 16           |

---

## 📂 References

- [Kaggle GAN Competition](https://www.kaggle.com/competitions/gan-getting-started/data)
- [TFDS Monet Dataset](https://www.kaggle.com/datasets/dimitreoliveira/monet-paintings-jpg-berkeley)
- [MiFID Evaluation Metric](https://www.kaggle.com/competitions/gan-getting-started/overview/evaluation)

---

## 👥 Authors

- **Dolev Dahan** – 209406768
- **Ronel Davidov** – 209405216  


📫 Contact us at:  
`dahandol@post.bgu.ac.il`, `davidovr@post.bgu.ac.il`.

