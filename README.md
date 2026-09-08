# credit-card-fraud-detection-using-VAEGAN
Built a hybrid VAE-GAN in PyTorch to counter severe credit card fraud imbalance (&lt;0.2%). Synthesized high-fidelity fraud samples using feature-level adversarial loss, boosting PR-AUC and recall by 20% over SMOTE. Deployed an XGBoost classifier with a REST API for sub-100ms inference.

# Credit Card Fraud Detection Using VAE-GAN

An end-to-end machine learning project addressing extreme class imbalance in credit card transaction data using a hybrid Variational Autoencoder – Generative Adversarial Network (VAE-GAN) architecture, coupled with gradient-boosted decision trees (XGBoost and CatBoost) and an interactive web application.

---

## 📌 Project Overview

Credit card fraud datasets typically exhibit extreme class imbalance (often less than 0.2% fraudulent transactions). Traditional oversampling techniques like SMOTE can lead to noisy or unrealistic samples in high-dimensional feature spaces. 

This repository leverages:
- **VAE-GAN Architecture:** Encodes genuine and rare fraud patterns into a continuous latent space and synthesizes realistic, high-fidelity minority-class samples using adversarial training.
- **Ensemble Classifiers:** Evaluates balanced datasets using pre-trained **XGBoost** and **RandomForsest** models to achieve high precision, recall, and PR-AUC.
- **Inference Application:** A lightweight application (`app.py`) for testing and classifying sample transactions in real-time.

---

## 📂 Repository Structure

```plaintext
├── Credit_Card_GANs.ipynb   # Model training, VAE-GAN synthesis, and evaluation notebook
├── app.py                   # Web application / inference script for fraud prediction
├── catboost_model.joblib    # Serialized pre-trained CatBoost classification model
├── xgb_model.joblib         # Serialized pre-trained XGBoost classification model
├── check.csv                # Sample/test transaction records for validation and demo
├── code.pdf                 # Documentation / exported code report
└── README.md                # Project documentation


STEPS TO RUN FILES:
1. CREATE A VIRTUAL ENVIRONMENT
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
2. INSTALL DEPENDECIES
    pip install numpy pandas scikit-learn xgboost catboost joblib flask streamlit torch
3. RUN THE APPLICATION
    python app.py
4. TESTING WITH SAMPLE DATA
    CHECK.CSV FILE
