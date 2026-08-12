# app.py
import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="KaStack ML Assignment", layout="wide")

# --- Helper to load data ---
@st.cache_data
def load_data():
    def read_json(filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
        return []
    
    return (
        read_json("outputs/classification_report.json"),
        read_json("outputs/extraction_report.json"),
        read_json("outputs/sensitive_report.json")
    )

classifications, extractions, sensitive = load_data()

# --- UI Header ---
st.title("🛡️ AI/ML Message Processing Pipeline")
st.markdown("Developed for KaStack Labs - AI/ML Engineer Intern Assignment")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Mandatory Demo IDs", 
    "Part 1: Classifications", 
    "Part 2: Task & Event Extractions", 
    "Part 3: Sensitive Info (Masked)"
])

with tab1:
    st.header("🎯 Mandatory Message IDs Demo")
    st.info("This tab isolates the 15 mandatory IDs required for the Loom video demonstration.")
    
    # The 15 IDs requested in the prompt
    mandatory_ids = [
        "MSG_0001", "MSG_0002", "MSG_0003", "MSG_0004", "MSG_0005", 
        "MSG_0006", "MSG_0007", "MSG_0009", "MSG_0012", "MSG_0013", 
        "MSG_0014", "MSG_0015", "MSG_0016", "MSG_0024", "MSG_0037"
    ]
    
    if classifications:
        df_class = pd.DataFrame(classifications)
        demo_data = df_class[df_class['message_id'].isin(mandatory_ids)]
        st.dataframe(demo_data, use_container_width=True)
    else:
        st.warning("Please run the backend scripts first.")

with tab2:
    st.header("Part 1: Zero-Shot Classification Results")
    if classifications:
        df_class = pd.DataFrame(classifications)
        
        # Filters
        category_filter = st.selectbox("Filter by Category", ["All"] + list(df_class['category'].unique()))
        if category_filter != "All":
            df_class = df_class[df_class['category'] == category_filter]
            
        st.dataframe(df_class, use_container_width=True)
        
        # Example with explanation
        st.subheader("Classification Decision Example")
        sample = df_class.iloc[0]
        st.json({
            "Message ID": sample['message_id'],
            "Assigned Category": sample['category'],
            "Confidence": sample['confidence'],
            "Reasoning": sample['reason']
        })

with tab3:
    st.header("Part 2: Information Extraction")
    if extractions:
        df_ext = pd.DataFrame(extractions)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Extracted Tasks")
            st.dataframe(df_ext[df_ext['type'] == 'task'].dropna(axis=1, how='all'), use_container_width=True)
            
        with col2:
            st.subheader("Extracted Events")
            st.dataframe(df_ext[df_ext['type'] == 'event'].dropna(axis=1, how='all'), use_container_width=True)

with tab4:
    st.header("Part 3: Privacy Firewall & Sensitive Data")
    if sensitive:
        df_sens = pd.DataFrame(sensitive)
        
        # Highlight risks
        def highlight_risk(val):
            color = '#ff4b4b' if val == 'high' else '#ffa421' if val == 'medium' else ''
            return f'background-color: {color}'
            
        # FIXED: Changed 'applymap' to 'map' for newer versions of Pandas
        st.dataframe(df_sens.style.map(highlight_risk, subset=['risk']), use_container_width=True)