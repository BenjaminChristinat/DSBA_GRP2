import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path

# --- SETUP PATHS ---
# Add the project root to python path so we can import 'src'
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# Import your custom logic
# (Ensure src.features.build_features exists as per your files)
from src.features import build_features 

st.set_page_config(page_title="Swiss Restaurant Predictor", layout="wide", page_icon="🇨🇭")

# --- CONFIGURATION ---
MODEL_PATH = PROJECT_ROOT / "models" / "model1_rf_best.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "restaurant_data_output.csv"

# --- LOAD RESOURCES ---
@st.cache_resource
def load_model():
    """Loads the pre-trained machine learning model."""
    if not MODEL_PATH.exists():
        st.error(f"Model not found at {MODEL_PATH}. Did you move the .pkl files?")
        return None
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_dataset():
    """Loads the historical data for visualization."""
    if not DATA_PATH.exists():
        st.warning(f"Data file not found at {DATA_PATH}. Visualization disabled.")
        return None
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_dataset()

# --- APP INTERFACE ---
st.title("🇨🇭 Restaurant Success Predictor")
st.markdown("""
This tool predicts whether a new restaurant concept will be a **Winner** (High Rating + Popularity) 
in Switzerland based on location features and competition analysis.
""")

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📍 New Location Details")
    
    # User Inputs mimicking your model features
    # These specific inputs match the 'review_dynamics' snippet you showed me
    reviews_m = st.number_input("Estimated Monthly Reviews", min_value=0, value=10)
    competitors_200m = st.slider("Competitors within 200m", 0, 50, 5)
    competitors_500m = st.slider("Competitors within 500m", 0, 100, 15)
    
    attractions_500m = st.number_input("Attractions within 500m", 0, 50, 2)
    busstop_200m = st.checkbox("Bus stop within 200m?", value=True)
    
    # Calculate derived features (Simulating your src/features logic)
    # In a full version, we would calculate these from Lat/Lon automatically
    input_features = pd.DataFrame({
        'reviews_m': [reviews_m],
        'comp_count_200m': [competitors_200m],
        'comp_count_500m': [competitors_500m],
        'attractions_500m': [attractions_500m],
        'busstop_200m': [1 if busstop_200m else 0],
        # Add placeholders for other columns your model expects
        'dist_station_m': [500],  # Default
        'dist_hotel_m': [200],    # Default
        'avg_rating_trailing3': [4.5], # Assumption for new venue
    })

    predict_btn = st.button("🔮 Predict Success", type="primary")

with col2:
    # --- MAP VISUALIZATION ---
    st.subheader("🗺️ Market Analysis")
    if df is not None:
        # Simple map of existing successful venues
        winners = df[df['rating'] >= 4.5]  # Assuming 4.5+ is a winner
        st.map(winners[['latitude', 'longitude']].dropna(), size=20, color='#00ff00')
        st.caption("Green dots show existing highly-rated restaurants.")

    # --- PREDICTION RESULT ---
    if predict_btn and model:
        try:
            # Ensure input columns match model expectation
            # (We might need to align columns exactly with model.feature_names_in_)
            missing_cols = set(model.feature_names_in_) - set(input_features.columns)
            for c in missing_cols:
                input_features[c] = 0 # Fill missing with 0 for demo
            
            # Reorder columns
            input_features = input_features[model.feature_names_in_]
            
            prediction = model.predict(input_features)[0]
            
            st.divider()
            if prediction >= 0.5: # Assuming 1/0 output or probability
                st.success(f"## 🏆 Prediction: WINNER! (Score: {prediction:.2f})")
                st.write("This location shows strong signals for high engagement.")
            else:
                st.error(f"## ⚠️ Prediction: RISK (Score: {prediction:.2f})")
                st.write("Competition might be too high or foot traffic too low.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.info("Tip: Ensure all 11 model files are in the 'models/' folder.")