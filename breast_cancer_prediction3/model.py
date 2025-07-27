import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.datasets import load_breast_cancer # To get the dataset easily
import random
import math

# --- 1. Data Acquisition ---
def load_data():
    """Loads the Breast Cancer Wisconsin (Diagnostic) dataset."""
    breast_cancer = load_breast_cancer()
    X = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y = breast_cancer.target
    return X, y

# --- 2. Data Preprocessing ---
def preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Preprocesses the data: splits into train/test, and scales features.
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y) # Stratify to maintain class distribution

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert back to DataFrame to maintain feature names for selection
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

    return X_train_scaled, X_test_scaled, y_train, y_test

# --- 3. Feature Selection (Firefly Algorithm) ---
class FireflyFeatureSelection:
    def __init__(self, n_fireflies=20, max_iter=50, alpha=0.5, beta_min=0.2, gamma=1.0):
        self.n_fireflies = n_fireflies
        self.max_iter = max_iter
        self.alpha = alpha  # Randomization parameter
        self.beta_min = beta_min # Minimum attractiveness
        self.gamma = gamma  # Light absorption coefficient

        self.best_features = None
        self.best_fitness = -np.inf # Initialize with negative infinity for maximization

    def _calculate_fitness(self, features_mask, X, y):
        """
        Calculates the fitness of a feature subset using Naive Bayes accuracy.
        features_mask: a binary array where 1 indicates selected feature.
        """
        selected_indices = np.where(features_mask == 1)[0]
        if len(selected_indices) == 0:
            return -1.0 # Penalize empty feature sets

        X_selected = X.iloc[:, selected_indices]

        # Use cross-validation for a more robust fitness evaluation
        classifier = GaussianNB()
        scores = cross_val_score(classifier, X_selected, y, cv=5, scoring='accuracy')
        return np.mean(scores)

    def _euclidean_distance(self, f1, f2):
        """Calculates Euclidean distance between two fireflies (feature masks)."""
        return np.sqrt(np.sum((f1 - f2)**2))

    def _update_alpha(self, iteration):
        """Decreases alpha over iterations for better convergence."""
        return self.alpha * (1 - iteration / self.max_iter)

    def select_features(self, X_train, y_train):
        n_features = X_train.shape[1]

        # Initialize fireflies (random binary feature subsets)
        # Each firefly is a binary array representing feature selection
        fireflies = np.random.randint(0, 2, size=(self.n_fireflies, n_features))
        
        # Calculate initial brightness (fitness) for each firefly
        brightness = np.array([self._calculate_fitness(f, X_train, y_train) for f in fireflies])

        self.best_fitness = np.max(brightness)
        self.best_features = fireflies[np.argmax(brightness)].copy()

        for iteration in range(self.max_iter):
            alpha_t = self._update_alpha(iteration)

            for i in range(self.n_fireflies):
                for j in range(self.n_fireflies):
                    if brightness[j] > brightness[i]: # Firefly j is brighter than firefly i
                        r = self._euclidean_distance(fireflies[i], fireflies[j])
                        beta = self.beta_min * np.exp(-self.gamma * r**2) # Attractiveness

                        # Move firefly i towards firefly j
                        movement = beta * (fireflies[j] - fireflies[i]) + alpha_t * (np.random.rand(n_features) - 0.5)
                        fireflies[i] = fireflies[i] + movement
                        
                        # Binarize the firefly's position after movement
                        # A simple thresholding or sigmoid function can be used
                        fireflies[i] = (fireflies[i] > 0.5).astype(int) 
                        fireflies[i][fireflies[i] > 1] = 1 # Ensure values are 0 or 1
                        fireflies[i][fireflies[i] < 0] = 0 # Ensure values are 0 or 1

                        # Recalculate brightness for firefly i
                        brightness[i] = self._calculate_fitness(fireflies[i], X_train, y_train)

                        # Update global best if this firefly is brighter
                        if brightness[i] > self.best_fitness:
                            self.best_fitness = brightness[i]
                            self.best_features = fireflies[i].copy()
            
            # (Optional) Random walk for fireflies that aren't attracted to brighter ones, or if all are equally bright
            # For simplicity, this is often incorporated into the alpha term's random component

            print(f"Iteration {iteration+1}/{self.max_iter}, Best Fitness: {self.best_fitness:.4f}")

        selected_feature_indices = np.where(self.best_features == 1)[0]
        return selected_feature_indices, self.best_fitness

# --- 4. Model Training (Naive Bayes Classifier) ---
def train_naive_bayes(X_train, y_train):
    """Trains a Gaussian Naive Bayes classifier."""
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# --- 5. Model Evaluation ---
def evaluate_model(model, X_test, y_test):
    """Evaluates the trained model and prints performance metrics."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary') # 'binary' for 2 classes
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    # ROC AUC requires probability estimates
    y_pred_proba = model.predict_proba(X_test)[:, 1] # Probability of the positive class
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Sensitivity): {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("------------------------")

# --- Main Execution Flow ---
if __name__ == "__main__":
    print("--- Starting Breast Cancer Prediction System ---")

    # 1. Data Acquisition
    X, y = load_data()
    print(f"Original Dataset Shape: X={X.shape}, y={y.shape}")
    print(f"Features: {list(X.columns)}")

    # 2. Data Preprocessing
    X_train, X_test, y_train, y_test = preprocess_data(X, y)
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    
    # --- Optional: Evaluate Naive Bayes without Feature Selection for comparison ---
    print("\n--- Evaluating Naive Bayes without Feature Selection ---")
    nb_full_features = GaussianNB()
    nb_full_features.fit(X_train, y_train)
    evaluate_model(nb_full_features, X_test, y_test)

    # 3. Feature Selection using Firefly Algorithm
    print("\n--- Running Firefly Feature Selection ---")
    fa_selector = FireflyFeatureSelection(n_fireflies=30, max_iter=70, alpha=0.1, beta_min=0.2, gamma=1.0)
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
    evaluate_model(nb_model, X_test_selected, y_test)

    print("\n--- System Execution Complete ---")