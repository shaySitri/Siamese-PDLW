
# Siamese Neural Network for Similarity Learning 🔍🤖

[![Jupyter Notebook](https://img.shields.io/badge/Notebook-Siamese_ass3.ipynb-orange?logo=jupyter)](https://github.com/shaySitri/Siamese-PDLW/blob/main/Siamese-ass3.ipynb)

This repository contains a Jupyter Notebook that implements a Siamese Neural Network for similarity learning, designed to evaluate the similarity between pairs of inputs. This project focuses on leveraging a Siamese model to improve pairwise similarity tasks, such as identifying matching patterns across data samples.

---

## 📄 Project Overview
The **Siamese Neural Network** model aims to learn the similarity between pairs of inputs by using two identical subnetworks to process each input independently. The resulting embeddings are then compared to calculate the similarity score. This project applies the Siamese architecture to a dataset for similarity-based learning tasks.

---

## 🧩 Model Architecture and Hyperparameters
- **Architecture**: Twin neural networks with shared weights, where each network learns embeddings independently and then compares them.
- **Embedding Layers**: Dense layers to capture and encode input features.
- **Distance Metric**: Uses Euclidean or cosine similarity between embeddings to measure similarity.
- **Hyperparameters**:
  - **Learning Rate**: 1e-3
  - **Batch Size**: 32
  - **Epochs**: Up to 50 (early stopping applied)

### Training and Evaluation
- **Loss Function**: Mean Squared Error (MSE), optimized to minimize similarity prediction error.
- **Early Stopping**: Training stopped at optimal points to prevent overfitting, often around epoch 22 for larger datasets.

---

## 📊 Performance Metrics
- **Mean Squared Error (MSE)**:
  - **Test Set**: MSE = 0.24647 on the Random Forest (RF) baseline for comparison.
  - Siamese network achieves improved similarity prediction across training and test sets, demonstrating its effectiveness in identifying closely related samples.

---

## 🔍 Key Findings
- **Improved Similarity Learning**: Siamese network effectively captures similarity between pairs, outperforming baseline methods like Random Forest.
- **Generalization**: Early stopping improved generalization on unseen data by preventing overfitting.
- **Robustness**: Siamese architecture is highly adaptable to various similarity-based tasks.
