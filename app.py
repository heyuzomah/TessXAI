import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Load the saved model and metadata
model_dir = Path("saved_model")
model_pipeline = joblib.load(model_dir / "best_pipeline.pkl")
with open(model_dir / "metadata.json", "r") as f:
    metadata = json.load(f)

# Light theme styling
st.markdown("""
<style>
    body { background-color: #f8f9fa; }
    .stSlider, .stSelectbox { margin-bottom: 0.5rem !important; }
    .stButton>button { 
        width: 100%; 
        background-color: #007bff; 
        color: white !important; 
        border-radius: 5px; 
        border: none; /* Removes any border */
    }
    .stButton>button:hover { 
        background-color: #0056b3; 
        color: white !important; 
        border: none; /* Ensures no red border on hover */
    }
    .stButton>button:active { 
        background-color: #004494 !important; 
        color: white !important; /* Ensures text stays white when clicked */
        border: none !important;
    }
    .prediction-box { 
        background-color: #f1f3f5; 
        padding: 1rem; 
        border-radius: 8px; 
        border: 1px solid #dee2e6; 
    }
    .prediction-value { font-size: 2rem; font-weight: bold; color: #007bff; }
</style>
""", unsafe_allow_html=True)

# App header
st.title("🚗 TessX PriceSense")
st.caption("Estimate your car’s market value instantly.")

# Sidebar input form
with st.sidebar:
    st.header("Vehicle Specs ⚙️")
    
    horsepower = st.slider("Engine Power (HP)", 122, 831, 345)
    
    selected_make = st.selectbox("Manufacturer", 
        ["Ford", "Mercedes-Benz", "Audi", "Nissan", "BMW", "Bentley", "Aston Martin"], 
        index=0
    )
    
    selected_cylinder = st.selectbox("Cylinders", 
        ["I3", "I4", "I5", "I6", "V6", "V8", "V10", "V12", "W12"], 
        index=3
    )

    # Prediction button
    predict_btn = st.button("Predict Market Value")

# Prediction display in main area
if predict_btn:
    try:
        input_data = pd.DataFrame({
            "Horsepower": [horsepower],
            "Cylinders": [selected_cylinder],
            "Make": [selected_make]
        })
        prediction_log = model_pipeline.predict(input_data)
        predicted_price = np.expm1(prediction_log)[0]

        st.markdown(f"""
        <div class="prediction-box">
            <h3 style="color: #007bff;">Estimated Value</h3>
            <p class="prediction-value">${predicted_price:,.2f}</p>
            <p style="color: #6c757d;">Based on market trends & vehicle specs.</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Model details
with st.expander("🔍 Model Details"):
    st.write("### Key Features")
    st.write("- Horsepower\n- Cylinder Configuration\n- Manufacturer")
    
    st.write("### Processing Steps")
    st.write("- Feature Scaling\n- One-Hot Encoding\n- Linear Regression Model")
