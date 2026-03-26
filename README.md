# Fraud Detection and Transaction Clustering

## 📌 Overview
This repository contains my academic project for the *Statistical Learning for Data Scientists II* course at KMUTT. The project explores the application of Deep Learning and clustering algorithms to identify anomalous financial behaviors and detect fraudulent transactions.

## 🎯 Objective
To develop a robust financial security system capable of detecting fraudulent transactions and clustering anomalous financial activities to support risk management strategies.

## 📊 Dataset
* **Source:** Real-world anonymized credit card/bank transaction dataset from Kaggle.
* **Size:** Over 550,000 transaction records.

## 🛠️ Methodology & Tech Stack
* **Language:** R
* **Classification:** Implemented a **Neural Network** model using the `torch` package in R for binary classification (Fraud vs. Non-Fraud).
* **Clustering:** Applied the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm to cluster data points and identify transaction outliers/anomalies.

## 📈 Key Results
* **Detection Performance:** The Neural Network model achieved a high detection **accuracy of 95.53%** and an **F1-Score of 0.95**.
* **Risk Management Application:** Successfully demonstrated how combining supervised learning (Neural Networks) with unsupervised learning (DBSCAN) can effectively detect complex fraud patterns in high-volume financial data.

## 📁 Repository Structure
* `/code`: Contains the R scripts for data scaling, DBSCAN clustering, and Neural Network modeling.
* `/report`: Contains the project presentation/report (PDF).
