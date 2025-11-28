import streamlit as st

# --- SETUP: Set Page Config ---
# This configuration applies to the whole app, including the main page
st.set_page_config(
    page_title="Swiss Restaurant Predictor", 
    layout="wide", 
    page_icon="🇨🇭"
)

# --- APP INTERFACE: Simplified Landing Page ---
st.title("Swiss Restaurant Success Predictor")
st.markdown("""
Welcome to the predictive tool for launching successful restaurant concepts in Switzerland.
\n
This application offers two main services to guide your decision-making:

1.  **Market Analysis:** Review historical data and market trends to understand the landscape.
2.  **Location Finder:** Input specific details about a potential location to receive a success prediction from our model.

**Choose a feature to begin your analysis:**
""")

# --- NAVIGATION BUTTONS ---
# Streamlit pages automatically appear in the sidebar based on files in the 'pages/' folder.
# We will use st.page_link to create prominent buttons in the main view for better visibility.

# Use st.columns for better button layout
col1, col2 = st.columns([1, 1])

with col1:
    # Link to the first page, assuming the file is now named '1_Market_Analysis.py'
    if st.button("📍 LOCATION FINDER", use_container_width=True):
        # Use st.switch_page() to manually redirect when clicked
        st.switch_page("pages/2_Location_Finder.py")
    st.caption("Get a machine learning prediction for a new site.")
with col2:
    # Link to the second page, assuming the file is now named '2_Location_Finder.py'
    if st.button("📍 LOCATION FINDER", use_container_width=True):
        # Use st.switch_page() to manually redirect when clicked
        st.switch_page("pages/2_Location_Finder.py")
    st.caption("Get a machine learning prediction for a new site.")

st.divider()
st.caption("You can also navigate using the sidebar on the left.")
