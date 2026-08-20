# app.py
import streamlit as st
import pandas as pd
import json
import os

st.set_page_config(page_title="KaStack L2 System", layout="wide")

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
        read_json("outputs/sensitive_report.json"),
        read_json("outputs/grouping_report.json"),
        read_json("outputs/priority_report.json"),
        read_json("outputs/assistant_responses.json")
    )

classifications, extractions, sensitive, groups, priorities, assistant = load_data()

# --- UI Header ---
st.title("🧠 AI/ML Message Processing Pipeline (L2 System)")
st.markdown("Developed for KaStack Labs - Extended from L1 Architecture")

# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🤖 Intelligent Assistant (Part 3)", 
    "🗂️ Message Groups (Part 2)", 
    "⚡ Priority Engine (Part 1)", 
    "🛡️ Privacy Routing",
    "📊 L1 vs L2 Benchmarks"
])

with tab1:
    st.header("Semantic RAG Assistant")
    st.info("Demonstrating responses to the supplied L2 Demo Queries.")
    if assistant:
        for item in assistant:
            with st.expander(f"🗣️ Query: {item['query']}"):
                st.markdown(f"**Answer:** {item['answer']}")
                st.markdown(f"**Reason:** {item['reason']}")
                st.markdown(f"**Supporting Message IDs:** {', '.join(item['supporting_message_ids'])}")
                if item['group_id']:
                    st.markdown(f"**Group ID:** {item['group_id']}")

with tab2:
    st.header("Related-Message Grouping")
    st.info("Messages grouped by semantic meaning and chronological follow-ups.")
    if groups:
        df_groups = pd.DataFrame(groups)
        
        # Color code statuses
        def highlight_status(val):
            color = ''
            if val == 'completed': color = '#28a745'
            elif val == 'cancelled': color = '#dc3545'
            elif val == 'rescheduled': color = '#ffc107'
            return f'background-color: {color}'
            
        st.dataframe(df_groups[['group_id', 'title', 'status', 'latest_deadline', 'confidence', 'summary']].style.map(highlight_status, subset=['status']), use_container_width=True, hide_index=True)

with tab3:
    st.header("Dynamic Priority Engine")
    if priorities:
        df_priorities = pd.DataFrame(priorities)
        
        # Highlight critical priorities
        def highlight_priority(val):
            return 'background-color: #ff4b4b' if val == 'critical' else ''
            
        st.dataframe(df_priorities.style.map(highlight_priority, subset=['priority']), use_container_width=True, hide_index=True)
        
        st.subheader("Priority Decision Example")
        critical_items = df_priorities[df_priorities['priority'] == 'critical']
        if not critical_items.empty:
            st.json(critical_items.iloc[0].to_dict())

with tab4:
    st.header("Privacy-Aware Routing")
    st.info("Demonstrating local processing, blocked external requests, and confirmation routing.")
    if sensitive:
        df_sens = pd.DataFrame(sensitive)
        st.dataframe(df_sens[['message_id', 'sensitivity_type', 'risk', 'recommended_action', 'masked_text']], use_container_width=True, hide_index=True)

with tab5:
    st.header("Optimization & Benchmarking (L1 vs L2)")
    st.markdown("""
    ### 🚀 System Evolution
    * **L1 System:** Processed items iteratively in isolation. Fast, but lacked historical context.
    * **L2 System:** Introduces state management and semantic embeddings (`all-MiniLM-L6-v2`).
    
    ### ⏱️ Performance Comparison
    | Metric | L1 Architecture | L2 Architecture |
    |--------|----------------|----------------|
    | **Data Processed** | 900 isolated messages | 1104 chronologically linked messages |
    | **Model Size** | 1.6 GB (`bart-large-mnli`) | 1.6 GB + 90 MB (`all-MiniLM`) |
    | **Processing Time** | ~3-4 Minutes | ~4-5 Minutes (Added embedding latency) |
    | **Result Quality** | Static rules. | Dynamic state. Statuses and priorities update chronologically. |
    
    ### 🛠️ Key Optimization
    To keep response times low, the L2 system **pre-computes semantic embeddings** into memory during the grouping phase. When the Intelligent Assistant receives a query, it performs an $O(1)$ lookup for exact IDs and lightning-fast cosine similarity for semantic questions, rather than re-evaluating the transformer network.
    """)