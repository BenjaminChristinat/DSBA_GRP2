import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from pyproj import Transformer

st.set_page_config(page_title="Concept Optimizer", layout="wide")
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

st.title("💡 Concept Suitability Engine")

# --- EXTENSIVE CATEGORY LIST ---
CATEGORIES = [
    "Afghan", "African", "American", "Asian", "Bagel", "Bakery", "Bar", "Barbecue", 
    "Bistro", "Brazilian", "Breakfast", "Brunch", "Bubble Tea", "Burger", "Cafe", 
    "Cajun", "Cantonese", "Caribbean", "Chinese", "Coffee", "Creperie", "Deli", 
    "Dessert", "Dim Sum", "Diner", "Donut", "Dumpling", "Ethiopian", "European", 
    "Falafel", "Fast Food", "Filipino", "Fine Dining", "Fish & Chips", "Fondue", 
    "French", "Fried Chicken", "Gastropub", "German", "Greek", "Grill", "Halal", 
    "Hawaiian", "Healthy", "Ice Cream", "Indian", "Indonesian", "Irish", "Italian", 
    "Japanese", "Juice Bar", "Kebab", "Korean", "Latin American", "Lebanese", 
    "Malay", "Mediterranean", "Mexican", "Middle Eastern", "Moroccan", "Noodle", 
    "Pasta", "Persian", "Peruvian", "Pizza", "Poke", "Portuguese", "Pub", "Ramen", 
    "Salad", "Sandwich", "Seafood", "Snack", "Soup", "Spanish", "Steak", "Sushi", 
    "Swiss", "Tacos", "Taiwanese", "Tapas", "Tea", "Thai", "Turkish", "Vegan", 
    "Vegetarian", "Vietnamese", "Waffle", "Wine Bar", "Wings"
]

# --- DEFAULTS ---
# We default to a balanced profile
DEFAULT_WEIGHTS = {"transit": 5, "traffic": 5, "comp": 5}

# 1. INPUTS
c1, c2 = st.columns(2)
with c1:
    city = st.selectbox("Target City", ["Lausanne", "Zürich", "Köniz", "Vernier", "Emmen", "Dübendorf",
                                        "Geneva","Basel", "Bern", "Luzern", "Lugano", "Rolle",
                                        "Montreux", "Nyon", "Renens", "Vevey"
                                        ])
with c2:
    concept_type = st.selectbox("Restaurant Category", CATEGORIES, index=CATEGORIES.index("Italian") if "Italian" in CATEGORIES else 0)

st.write("---")
st.subheader("👇 Define your Deal-Breakers")
st.caption("If you set a weight to **10**, that factor becomes a **Requirement**. If it's missing, the score will be 0.")

c_w1, c_w2, c_w3 = st.columns(3)
with c_w1: 
    w_transit = st.slider("🚇 Train Station Proximity", 0, 10, 7)
with c_w2: 
    w_traffic = st.slider("📸 Tourist / Foot Traffic", 0, 10, 8)
with c_w3: 
    w_comp = st.slider("🏪 Avoid General Competition", 0, 10, 5, 
                       help="High value = I want to be in a quiet area. Low value = I like busy restaurant streets.")

if st.button("Find Spots", type="primary"):
    slug = city.lower().replace(" ", "_")
    path = DATA_DIR / f"grid_features_{slug}.parquet"
    
    if not path.exists():
        st.error(f"No data for {city}. Please run 'python scripts/run_grid_scan.py --city {city}'")
        st.stop()
        
    df = pd.read_parquet(path)
    
    # --- 1. CALCULATE RAW SCORES (0.0 to 1.0) ---
    
    # Transit: Exponential decay. 
    # 0m = 1.0, 500m = 0.36, 1km = 0.13
    df['s_transit'] = np.exp(-df['dist_station_m'] / 500.0)
    
    # Traffic: Min-Max normalization
    if 'log_attractions_500m' in df.columns:
        col = df['log_attractions_500m']
        if col.max() > col.min():
            df['s_traffic'] = (col - col.min()) / (col.max() - col.min())
        else:
            df['s_traffic'] = 0.0
    else:
        df['s_traffic'] = 0.0

    # Competition: 
    # Logic: 0 competitors = 1.0 (Perfect score for 'Avoid Comp')
    # 20 competitors = 0.0 (Bad score)
    if 'comp_count_500m' in df.columns:
        # We use a soft decay: score drops by half every 10 competitors
        df['s_comp'] = np.exp(-df['comp_count_500m'] / 10.0)
    else:
        df['s_comp'] = 0.0

    # --- 2. THE FIX: WEIGHTED GEOMETRIC MEAN ---
    # Formula: Score = (Feature1 ^ Weight1) * (Feature2 ^ Weight2) ...
    # Effect: If any feature is 0 and has a high weight, the Total Score becomes 0.
    
    # Add epsilon to avoid log(0) errors
    epsilon = 1e-6
    
    # Normalize weights so they sum to 1 (for the geometric mean power)
    total_weight = w_transit + w_traffic + w_comp + epsilon
    p_transit = w_transit / total_weight
    p_traffic = w_traffic / total_weight
    p_comp = w_comp / total_weight
    
    # Calculate Geometric Mean
    # We use exp(sum(log)) for stability
    df['log_score'] = (p_transit * np.log(df['s_transit'] + epsilon)) + \
                      (p_traffic * np.log(df['s_traffic'] + epsilon)) + \
                      (p_comp * np.log(df['s_comp'] + epsilon))
                      
    df['final_score'] = np.exp(df['log_score'])
    
    # Scale to 0-100
    df['Suitability'] = (df['final_score'] / df['final_score'].max() * 100).fillna(0).round(1)

    # --- 3. OUTPUTS ---
    transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)
    lons, lats = transformer.transform(df['x_lv95'].values, df['y_lv95'].values)
    df['longitude'] = lons
    df['latitude'] = lats
    
    top_spots = df.sort_values('final_score', ascending=False).head(50).copy()
    
    st.success(f"Identified top locations for '{concept_type}' in {city}.")
    
    # MAP
    fig = px.scatter_mapbox(
        top_spots, lat="latitude", lon="longitude", color="Suitability", size="Suitability",
        color_continuous_scale="RdYlGn", hover_name="Suitability", 
        hover_data={"latitude": False, "longitude": False, "dist_station_m": ":.0f", "comp_count_500m": True},
        zoom=13, mapbox_style="carto-positron", height=600
    )
    st.plotly_chart(fig, use_container_width=True)

    # TABLE
    st.subheader("🏆 Top Candidates")
    display_df = top_spots.head(10).copy()
    display_df['Maps'] = [f"http://maps.google.com/?q={lat},{lon}" for lat, lon in zip(display_df['latitude'], display_df['longitude'])]
    
    st.data_editor(
        display_df[['Suitability', 'dist_station_m', 'comp_count_500m', 'Maps']],
        column_config={
            "Maps": st.column_config.LinkColumn("Link"),
            "dist_station_m": st.column_config.NumberColumn("Dist. Train (m)", format="%.0f"),
            "comp_count_500m": st.column_config.NumberColumn("Competitors (500m)"),
            "Suitability": st.column_config.ProgressColumn("Suitability", format="%.1f%%", min_value=0, max_value=100)
        },
        hide_index=True
    )