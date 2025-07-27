import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

# --- SET PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

# --- Load Pre-trained Model Components ---
@st.cache_resource
def load_model_components():
    """Loads the pre-trained scaler, selected features, and Naive Bayes model."""
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('selected_feature_indices.pkl', 'rb') as f:
            selected_feature_indices = pickle.load(f)
        
        model = joblib.load('naive_bayes_model.pkl')
        
        return scaler, selected_feature_indices, model
    except FileNotFoundError:
        st.error("Error: Model components not found. Please ensure 'scaler.pkl', 'selected_feature_indices.pkl', and 'naive_bayes_model.pkl' are in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading model components: {e}")
        st.stop()

scaler, selected_feature_indices, model = load_model_components()

# Define the original feature names (as per Breast Cancer Wisconsin dataset)
# This order is crucial for matching user inputs to model features
original_feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Get the names of the features selected by the Firefly Algorithm
selected_feature_names = [original_feature_names[i] for i in selected_feature_indices]

# --- Streamlit App Layout ---
st.title("🔬 Breast Cancer Prediction System")
st.markdown("""
This application predicts whether a breast mass is **Benign** or **Malignant** using a **Naive Bayes Classifier** optimized with **Firefly Feature Selection**.
""")

st.markdown("---")

# Initialize input_df as None or an empty DataFrame outside the conditional blocks
# This ensures it always exists, even if no input method has provided data yet.
input_df = None 

input_method = st.radio(
    "Choose input method:",
    ("Manual Input", "Upload CSV File"),
    horizontal=True,
    key="input_method_radio"
)

if input_method == "Manual Input":
    st.header("Manual Input: Patient Characteristics")
    st.write("Please enter the patient's cell nucleus measurements:")

    manual_input_data = {} # Dictionary to store manual input features
    
    # Organize inputs into columns for better UI
    num_cols = 3
    cols = st.columns(num_cols)
    
    for i, feature_name in enumerate(original_feature_names):
        col_idx = i % num_cols
        with cols[col_idx]:
            min_val = 0.0 # Default minimum
            max_val = 100.0 # Default maximum
            
            # Common ranges for some features (adjust based on your specific WBCD data)
            # You might need to run X.describe() on your full training data to get precise ranges
            if 'radius' in feature_name:
                min_val, max_val = 6.0, 30.0
            elif 'texture' in feature_name:
                min_val, max_val = 9.0, 40.0
            elif 'perimeter' in feature_name:
                min_val, max_val = 40.0, 200.0
            elif 'area' in feature_name:
                min_val, max_val = 100.0, 2500.0
            elif 'smoothness' in feature_name or 'fractal dimension' in feature_name:
                min_val, max_val = 0.05, 0.15
            elif 'compactness' in feature_name or 'concavity' in feature_name:
                min_val, max_val = 0.0, 0.5
            elif 'concave points' in feature_name:
                min_val, max_val = 0.0, 0.2
            elif 'error' in feature_name: # For error features, ranges are typically smaller
                min_val, max_val = 0.0, 1.0 # Adjust as needed
                
            manual_input_data[feature_name] = st.number_input(
                f"{feature_name.replace('mean ', '').replace('worst ', '').replace(' error', ' SE').title()}", # Cleaner labels
                min_value=float(min_val),
                max_value=float(max_val),
                value=float(min_val + (max_val - min_val) / 2), # Default to middle value
                step=0.001, # More granular step
                format="%.4f",
                key=f"input_{feature_name}"
            )
    
    # Assign the manually created DataFrame to input_df
    input_df = pd.DataFrame([manual_input_data])

elif input_method == "Upload CSV File":
    st.header("Upload CSV File")
    st.write("Upload a CSV file containing patient data. Ensure column names match the original dataset features.")
    st.info(f"Expected columns (order doesn't strictly matter but names must match): {', '.join(original_feature_names)}")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Data Preview:")
            st.dataframe(temp_df.head())

            # Validate columns
            missing_cols = [col for col in original_feature_names if col not in temp_df.columns]
            if missing_cols:
                st.warning(f"Warning: The following expected columns are missing: {', '.join(missing_cols)}. Please ensure your CSV has all 30 features.")
                # If columns are missing, we stop processing this file to prevent errors
                input_df = None # Reset input_df if validation fails
            else:
                # Reorder columns to match the training data and assign to input_df
                input_df = temp_df[original_feature_names]

        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
            input_df = None # Reset input_df on read error
    # If no file is uploaded yet, input_df remains None from its initialization

st.markdown("---")

# The Prediction Button now checks if input_df is ready before proceeding
if st.button("Predict Diagnosis 🚀", type="primary"):
    if input_df is None or input_df.empty:
        st.warning("Please provide input data first (either manually or by uploading a valid CSV).")
    else:
        try:
            # 1. Scale the input data using the PRE-TRAINED scaler
            # Ensure the input_df has columns in the correct order for the scaler
            scaled_input_data = scaler.transform(input_df)
            scaled_input_df = pd.DataFrame(scaled_input_data, columns=original_feature_names)

            # 2. Select features using the PRE-DETERMINED indices
            final_input_for_model = scaled_input_df.iloc[:, selected_feature_indices]

            # 3. Make prediction using the PRE-TRAINED Naive Bayes model
            prediction_proba = model.predict_proba(final_input_for_model)
            prediction_class = model.predict(final_input_for_model)

            st.header("Prediction Results")

            # Display results for each input row (whether single manual input or multiple CSV rows)
            for i in range(len(input_df)):
                # Get the original data for display (optional, but helpful for CSV)
                original_record_display = f"Record {i+1}"
                if input_method == "CSV Upload":
                    st.subheader(f"Result for {original_record_display}:")
                else:
                     st.subheader(f"Diagnosis Result:")


                prob_benign = prediction_proba[i, 0]
                prob_malignant = prediction_proba[i, 1]
                diagnosis = "Benign" if prediction_class[i] == 0 else "Malignant"

                if diagnosis == "Malignant":
                    st.error(f"**Diagnosis: MALIGNANT** 🚨")
                    st.write(f"Confidence (Malignant): **{prob_malignant:.2%}**")
                    st.write(f"Confidence (Benign): {prob_benign:.2%}")
                    st.warning("Immediate medical consultation is highly recommended.")
                else:
                    st.success(f"**Diagnosis: BENIGN** ✅")
                    st.write(f"Confidence (Benign): **{prob_benign:.2%}**")
                    st.write(f"Confidence (Malignant): {prob_malignant:.2%}")
                    st.info("While the prediction is benign, regular check-ups are always advisable.")
                
                if len(input_df) > 1: # Only add separator if there are multiple records
                    st.markdown("---") # Visual separator between multiple record results

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check your input values. They might be outside the expected range or format.")

st.markdown("---")
st.sidebar.header("About the System")
st.sidebar.info("""
This application utilizes a **Gaussian Naive Bayes** classifier for breast cancer prediction. 
Crucially, **Firefly Feature Selection** was employed during model training to identify 
the most impactful features, enhancing model accuracy and efficiency.

**Key Technologies Used:**
- **Python**
- **Streamlit** (for the web interface)
- **Scikit-learn** (for Naive Bayes, data preprocessing)
- **Numpy & Pandas** (for data handling)

**Disclaimer:** This tool is for educational and demonstrative purposes only and should **not** be used for actual medical diagnosis. Always consult with a qualified healthcare professional.
""")