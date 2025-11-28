import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Market Analysis", layout="wide")

# --- CONFIG ---
DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "restaurant_data_output.csv"

CUISINE_KEYWORDS = [
    "pizza", "italian", "chinese", "japanese", "sushi", "thai", "indian", 
    "mexican", "tacos", "burger", "french", "vietnamese", "kebab", 
    "steak", "seafood", "vegan", "vegetarian", "bakery", "cafe", "bar", "pub"
]

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    
    def get_category(row):
        text = str(row.get('types', '')).lower() + " " + str(row.get('primary_amenity', '')).lower() + " " + str(row.get('display_name', '')).lower()
        for keyword in CUISINE_KEYWORDS:
            if keyword in text:
                return keyword.title()
        return "Other"

    df['Category'] = df.apply(get_category, axis=1)
    return df

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0 
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- APP ---
st.title("🍕 Market Landscape")
st.markdown("Analyze competitors within a radius.")

df = load_data()
if df is None:
    st.error("Data not found.")
    st.stop()

# 1. CONTROLS
c1, c2 = st.columns([2, 1])
with c1:
    available_cities = sorted(df['__SelectedCity__'].dropna().unique())
    selected_city = st.selectbox("Select City", available_cities)
with c2:
    radius_km = st.slider("Radius (km)", 1, 20, 5)

# 2. FILTER DATA
city_data = df[df['__SelectedCity__'] == selected_city]
if city_data.empty:
    st.warning("No data.")
    st.stop()

center_lat = city_data['latitude'].mean()
center_lon = city_data['longitude'].mean()

df['dist_km'] = haversine_distance(center_lat, center_lon, df['latitude'], df['longitude'])
perimeter_df = df[df['dist_km'] <= radius_km].copy()

# 3. DISPLAY
col_map, col_list = st.columns([3, 1])

with col_list:
    st.subheader(f"Stats ({radius_km}km)")
    st.metric("Total Venues", len(perimeter_df))
    
    # Prepare counts for the list
    counts = perimeter_df['Category'].value_counts().reset_index()
    counts.columns = ["Category", "Count"]
    counts = counts[counts["Count"] > 1] # Filter small ones
    
    # DYNAMIC HEIGHT CALCULATION (approx 35px per row + header)
    table_height = (len(counts) + 1) * 35 
    if table_height > 600: table_height = 600 # Max cap
    
    # INTERACTIVE TABLE
    st.markdown("Select a category to highlight:")
    selection = st.dataframe(
        counts, 
        height=table_height, 
        use_container_width=True,
        hide_index=True,
        on_select="rerun",             # <--- Triggers rerun on click
        selection_mode="single-row"    # <--- Only one allowed
    )

    # Determine Highlighted Category from Selection
    highlight = "All"
    if selection and selection.get("selection") and selection["selection"].get("rows"):
        row_idx = selection["selection"]["rows"][0]
        highlight = counts.iloc[row_idx]["Category"]

with col_map:
    if not perimeter_df.empty:
        # HIGHLIGHT LOGIC
        if highlight != "All":
            perimeter_df['ColorGroup'] = np.where(
                perimeter_df['Category'] == highlight, 
                highlight, 
                "Other"
            )
            color_map = {"Other": "#d3d3d3", highlight: "#FF4B4B"} # Grey vs Red
            perimeter_df = perimeter_df.sort_values('ColorGroup', ascending=False) # Highlight on top
        else:
            perimeter_df['ColorGroup'] = perimeter_df['Category']
            color_map = None 

        fig = px.scatter_mapbox(
            perimeter_df,
            lat="latitude",
            lon="longitude",
            color="ColorGroup",
            color_discrete_map=color_map,
            hover_name="display_name",
            hover_data={"Category": True, "formatted_address": True, "ColorGroup": False, "latitude": False, "longitude": False, "dist_km": False},
            zoom=11,
            center={"lat": center_lat, "lon": center_lon},
            mapbox_style="carto-positron",
            height=600
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No restaurants found in this radius.")