import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler # Make sure this is imported
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
import random
import math
import pickle
import joblib

from model import FireflyFeatureSelection, load_data, preprocess_data, train_naive_bayes # For saving/loading models

# --- Your existing functions (load_data, preprocess_data, FireflyFeatureSelection, etc.) go here ---
# ... (Make sure these functions are defined before they are called) ...


# --- Main Execution Flow (Corrected) ---
if __name__ == "__main__":
    print("--- Starting Breast Cancer Prediction System ---")

    # 1. Data Acquisition
    X, y = load_data()
    print(f"Original Dataset Shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")

    # 2. Data Preprocessing
    # Ensure this line is called and its outputs are assigned
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    # --- The code block for saving model components needs to come AFTER
    #     the model has been fully trained and features selected.
    #     This block should be adjusted from what I previously provided
    #     to fit into your existing main execution flow.
    #     Here's how it would typically fit to save the final model components
    #     AFTER the Firefly Feature Selection has run and the Naive Bayes model
    #     is trained on the selected features.

    # --- Optional: Evaluate Naive Bayes without Feature Selection for comparison ---
    print("\n--- Evaluating Naive Bayes without Feature Selection ---")
    nb_full_features = GaussianNB()
    nb_full_features.fit(X_train, y_train)
    # evaluate_model(nb_full_features, X_test, y_test) # You can re-enable this if needed

    # 3. Feature Selection using Firefly Algorithm
    print("\n--- Running Firefly Feature Selection ---")
    # Instantiate the FA selector with appropriate parameters
    fa_selector = FireflyFeatureSelection(n_fireflies=30, max_iter=70, alpha=0.1, beta_min=0.2, gamma=1.0)
    
    # Run feature selection; this will return the indices of selected features
    selected_feature_indices, best_fa_fitness = fa_selector.select_features(X_train, y_train)

    selected_feature_names = X.columns[selected_feature_indices].tolist()
    print(f"\nFirefly Algorithm selected {len(selected_feature_indices)} features:")
    print(selected_feature_names)
    print(f"Best cross-validation accuracy during FA: {best_fa_fitness:.4f}")

    # Prepare data with selected features
    X_train_selected = X_train.iloc[:, selected_feature_indices]
    X_test_selected = X_test.iloc[:, selected_feature_indices]

    # 4. Model Training (Naive Bayes with selected features)
    print("\n--- Training Naive Bayes with Selected Features ---")
    nb_model = train_naive_bayes(X_train_selected, y_train)

    # 5. Model Evaluation
    # evaluate_model(nb_model, X_test_selected, y_test) # You can re-enable this if needed

    # --- Save the final trained model components for the Streamlit app ---
    print("\n--- Saving Model Components ---")
    # The scaler needs to be fitted on the X_train (full features) data *before* scaling
    # that is sent into the FA for fitness calculation or final model training.
    # So, we'll re-initialize and fit a scaler to ensure it's ready for new data.
    # This scaler will be used to transform *new* raw input data in the Streamlit app.
    final_scaler = StandardScaler()
    final_scaler.fit(X_train) # Fit scaler on the FULL X_train data

    # Save the scaler
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(final_scaler, f)

    # Save the selected feature indices
    with open('selected_feature_indices.pkl', 'wb') as f:
        pickle.dump(selected_feature_indices, f)

    # Save the Naive Bayes model trained on selected features
    joblib.dump(nb_model, 'naive_bayes_model.pkl')

    print("Model components saved successfully: scaler.pkl, selected_feature_indices.pkl, naive_bayes_model.pkl")
    print("\n--- System Execution Complete ---")