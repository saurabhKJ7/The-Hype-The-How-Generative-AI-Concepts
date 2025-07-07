import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import json
from streamlit_option_menu import option_menu

# Constants
API_URL = "http://localhost:8000"
DOC_TYPES = ["income_tax", "gst", "court_judgment", "property"]

# Page config
st.set_page_config(
    page_title="Indian Legal Document Search System",
    page_icon="⚖️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .search-container {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .results-container {
        margin-top: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        "Navigation",
        ["Upload Documents", "Search", "Document List"],
        icons=["cloud-upload", "search", "list-ul"],
        menu_icon="cast",
        default_index=0
    )

def upload_document():
    st.header("📄 Upload Legal Documents")
    
    with st.form("upload_form"):
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])
        doc_type = st.selectbox("Document Type", DOC_TYPES)
        submit = st.form_submit_button("Upload")
        
        if submit and uploaded_file:
            files = {"file": uploaded_file}
            response = requests.post(
                f"{API_URL}/upload",
                files=files,
                data={"doc_type": doc_type}
            )
            
            if response.status_code == 200:
                st.success("Document uploaded successfully!")
                st.json(response.json())
            else:
                st.error(f"Error: {response.text}")

def display_search_results(results):
    """Display search results in a 4-column layout with metrics."""
    # Display metrics
    st.subheader("📊 Search Performance Metrics")
    metrics = results.get("metrics", {})
    cols = st.columns(4)
    
    for idx, (method, metric) in enumerate(metrics.items()):
        with cols[idx]:
            st.markdown(f"**{method.title()} Method**")
            st.metric("Precision@5", f"{metric['precision@5']:.2f}")
            st.metric("Recall", f"{metric['recall']:.2f}")
            st.metric("Diversity", f"{metric['diversity']:.2f}")
    
    # Display results
    st.subheader("🔍 Search Results")
    method_results = {
        "Cosine": results.get("cosine_results", []),
        "Euclidean": results.get("euclidean_results", []),
        "MMR": results.get("mmr_results", []),
        "Hybrid": results.get("hybrid_results", [])
    }
    
    cols = st.columns(4)
    for idx, (method, results_list) in enumerate(method_results.items()):
        with cols[idx]:
            st.markdown(f"**{method} Similarity**")
            for result in results_list:
                with st.expander(f"Score: {result['score']:.3f}"):
                    st.markdown(result["chunk"]["text"])
                    if result["chunk"].get("legal_entities"):
                        st.markdown("**Legal Entities:**")
                        st.write(", ".join(result["chunk"]["legal_entities"]))

def search_documents():
    st.header("🔍 Search Legal Documents")
    
    with st.form("search_form"):
        query = st.text_input("Enter your search query")
        doc_type = st.selectbox("Filter by Document Type", [""] + DOC_TYPES)
        top_k = st.slider("Number of results per method", 1, 10, 5)
        submit = st.form_submit_button("Search")
        
        if submit and query:
            payload = {
                "query": query,
                "doc_type": doc_type if doc_type else None,
                "top_k": top_k
            }
            response = requests.post(f"{API_URL}/search", json=payload)
            
            if response.status_code == 200:
                display_search_results(response.json())
            else:
                st.error(f"Error: {response.text}")

def list_documents():
    st.header("📚 Document List")
    
    response = requests.get(f"{API_URL}/documents")
    if response.status_code == 200:
        documents = response.json()["documents"]
        if documents:
            df = pd.DataFrame(documents)
            df["upload_date"] = pd.to_datetime(df["upload_date"])
            df["upload_date"] = df["upload_date"].dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(
                df[["title", "doc_type", "upload_date"]],
                column_config={
                    "title": "Document Title",
                    "doc_type": "Document Type",
                    "upload_date": "Upload Date"
                },
                hide_index=True
            )
        else:
            st.info("No documents uploaded yet.")
    else:
        st.error(f"Error: {response.text}")

# Main content based on navigation
if selected == "Upload Documents":
    upload_document()
elif selected == "Search":
    search_documents()
else:
    list_documents() 