# 🚨 Financial Fraud Detection and Transaction Clustering

This repository contains an R-based data science pipeline designed to detect fraudulent financial activities and identify anomalous transactions. It tackles the problem using two distinct machine learning approaches: a Supervised Neural Network for predicting known fraud patterns, and an Unsupervised Clustering model to isolate suspicious, out-of-the-ordinary behaviors.

The goal is to demonstrate how different algorithmic strategies can be combined to enhance financial security and monitor transaction integrity.

## 🛠️ Tools & Libraries Used
* **Language:** R
* **Deep Learning:** `torch`, `luz` (for building and training neural networks)
* **Clustering:** `dbscan` (Density-Based Spatial Clustering of Applications with Noise)
* **Data Manipulation & Processing:** `dplyr`, `scale`

---

## ⚙️ Project Workflow & Methodology

The project is divided into two main analytical phases, each addressing a different aspect of fraud detection.

### Phase 1: Supervised Fraud Classification (Neural Networks)
This phase focuses on identifying fraudulent credit card transactions using historical, labeled data.

* **Dataset:** Credit Card Fraud Detection Dataset (containing anonymized transaction features `V1` to `V28`, `Amount`, and a binary `Class` label).
* **Data Preparation:** The `Amount` feature was scaled to ensure uniformity. The dataset was split into an 80/20 train-test ratio and converted into PyTorch tensors using the R `torch` library.
* **Model Architecture:** We built a Multi-Layer Perceptron (MLP) with the following structure:
  * Input layer handling 29 features.
  * Two hidden layers (64 and 32 nodes respectively) utilizing `ReLU` activation functions to capture non-linear relationships.
  * An output layer with a `Sigmoid` activation function to output a probability score between 0 and 1.
* **Training:** The model was optimized using the Adam optimizer and Binary Cross-Entropy Loss (BCELoss), making it well-suited for binary classification tasks.

### Phase 2: Unsupervised Anomaly Detection (Clustering)
Since not all fraud is historically labeled, this phase attempts to discover suspicious banking behaviors by finding transactions that do not fit standard patterns.

* **Dataset:** Bank Transaction Dataset (containing user behaviors such as transaction amounts, durations, login attempts, and account balances).
* **Feature Engineering:** We extracted key behavioral metrics (`TransactionAmount`, `TransactionDuration`, `LoginAttempts`, `AccountBalance`) and normalized them using standard scaling to ensure features with larger numeric ranges didn't dominate the clustering process.
* **DBSCAN Implementation:** We applied the DBSCAN algorithm (`eps = 0.5`, `minPts = 5`). Unlike K-Means, DBSCAN is highly effective at identifying arbitrarily shaped clusters and, most importantly, isolating data points that do not belong to any cluster.
* **Anomaly Extraction:** Transactions classified as "Cluster 0" (Noise) were flagged as structural outliers. These represent highly unusual banking behaviors requiring further investigation.

---

## 📈 Key Insights & Results

* **Neural Network Performance:** The deep learning model successfully learned the latent patterns of fraudulent credit card transactions, outputting high-confidence predictions based on the anonymized PCA features.
* **Clustering Anomalies:** The DBSCAN approach successfully isolated edge-case transactions. By analyzing the "Noise" cluster, we were able to profile the outliers against normal transactions (e.g., comparing the proportion of outliers occurring via Branch, ATM, vs. Online channels). This provides a actionable starting point for security teams to investigate suspicious account activities.
