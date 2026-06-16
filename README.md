# 🏏 Cricket Player Classification

## Overview
This repository contains a Multiclass Classification machine learning project that categorizes cricket players into three distinct roles: **Batsman, Bowler, or All-Rounder**. The classification is based on players' career statistics across all three major formats (Test, ODI, and T20I).

**Author:** Muhammad Maaz Wali Khan  
🔗 [GitHub: @mmaazwk27](https://github.com/mmaazwk27)

---

## 📊 Dataset
The model trains on a dataset of cricket career statistics (`cricket_players.csv`). 
* **Size:** 3651 clean rows
* **Features:** 14 statistical features (e.g., runs scored, batting/bowling averages, strike rates, wickets, etc.) + 1 format encoding = **15 total features**.
* **Target Classes:** `Batsman` (1), `Bowler` (2), `All-Rounder` (0)
* **Balancing:** The `All-Rounder` class naturally has fewer instances. **SMOTE** (Synthetic Minority Over-sampling Technique) is applied to correct class imbalance, expanding the training data to ensure unbiased predictions.

## 🛠️ Project Structure
The implementation is structured into the following key sections in the notebook:
1. **Setup & Data Loading:** Importing libraries and loading data from Google Drive.
2. **Load Preprocessed Splits:** Applying `StandardScaler` and `SMOTE` (80/20 train/test split initially).
3. **Algorithm 1 — K-Nearest Neighbours (Non-parametric):** Testing 3 variants (uniform/distance weights, euclidean/manhattan metrics).
4. **Algorithm 2 — Logistic Regression (Parametric Shallow):** Testing 3 variants (L1/L2 penalties, different C values).
5. **Algorithm 3 — Neural Networks:** Deep learning models built with Keras/TensorFlow.
6. **Comparative Analysis:** Evaluating all 9 models on a held-out test set.
7. **Data Split Ratio Investigation:** Testing model robustness across `70/15/15`, `80/10/10`, and `60/20/20` splits.
8. **Interactive Prediction Interface:** A UI widget to input custom stats and predict a player's role interactively.
9. **Error Analysis:** Analyzing misclassifications (especially All-Rounders).
10. **Summary**

## 🚀 Models & Performance
A total of 9 models were trained and compared:
* **3 KNN Models**
* **3 Logistic Regression Models**
* **3 Neural Network Architectures**

### 🏆 Best Model
* **Algorithm:** Neural Network (NN-1: 64 → 32 Dense Architecture)
* **Test Accuracy:** **97.81%**
* **Test F1-Score:** **0.96**

**Key Observations:**
- **KNN** performs well because Batsmen (high runs) and Bowlers (high wickets) form natural clusters in the feature space.
- **Logistic Regression** does well but is limited by its linear decision boundaries.
- **Neural Networks** capture non-linear patterns effectively and achieve the highest overall accuracy.
- **All-Rounders** are consistently the hardest class to predict across all models due to their statistical overlap with both pure batsmen and pure bowlers.

## ⚙️ Dependencies & Installation
To run this notebook locally or on Google Colab, you will need the following Python libraries installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn ipywidgets tensorflow
```
### 💻 How to Use
* Clone this repository.
* Ensure you have the required dependencies installed.
* Open Cricket Player Classification.ipynb in Jupyter Notebook, JupyterLab, or Google Colab.
* Mount your dataset (cricket_players.csv) in the specified directory.
* Run all cells to process the data, train the models, and view the comparative evaluation.
* Use the Interactive Prediction Interface (Section 8) to test the model on custom player statistics using the UI sliders!

📄 License
This project is licensed under the Apache License 2.0. See the LICENSE file for details.
