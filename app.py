import streamlit as st
import joblib
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Class labels
class_names = ['Fake', 'Real']

# Function for LIME prediction
def predict_proba(texts):
    X = vectorizer.transform(texts)
    return model.predict_proba(X)

# ------------------------------
# Streamlit Page Config & Styling
# ------------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem;
        margin: 2rem auto;
        max-width: 800px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
        line-height: 1.2;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #475569;
        font-weight: 500;
        margin-bottom: 0;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        border: 2px solid #374151;
        border-radius: 15px;
        padding: 1.5rem;
        font-size: 1rem;
        transition: all 0.3s ease;
        background: #1f2937 !important;
        color: #f9fafb !important;
        font-family: 'Inter', sans-serif;
        line-height: 1.6;
        resize: vertical;
        min-height: 200px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2), 0 4px 20px rgba(0, 0, 0, 0.1);
        background: #1f2937 !important;
        outline: none;
    }
    
    .stTextArea textarea::placeholder {
        color: #9ca3af !important;
        font-style: italic;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Result cards */
    .result-card {
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        backdrop-filter: blur(10px);
        animation: fadeIn 0.5s ease-in;
    }
    
    .real-news {
        background: linear-gradient(135deg, #10b981, #059669);
        border-left-color: #047857;
        color: white;
    }
    
    .fake-news {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        border-left-color: #b91c1c;
        color: white;
    }
    
    .result-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Explanation section */
    .explanation-container {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid #e2e8f0;
    }
    
    .explanation-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Footer */
    .footer-container {
        text-align: center;
        margin-top: 3rem;
        padding-top: 2rem;
        border-top: 1px solid #e2e8f0;
    }
    
    .footer-text {
        color: #000000;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .footer-name {
        color: #667eea;
        font-weight: 600;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.5rem;
        }
        
        .main-container {
            margin: 1rem;
            padding: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
    <h1 class="main-title">🔍 Fake News Detector</h1>
    <p class="subtitle">Analyze news articles with AI-powered accuracy and explainable insights</p>
</div>
""", unsafe_allow_html=True)

# Input section
st.markdown('<h3 style="color: #1e293b; font-weight: 600; margin-bottom: 1rem;">📝 Enter News Article</h3>', unsafe_allow_html=True)
user_input = st.text_area(
    "",
    height=200,
    placeholder="Paste your news article here... We'll analyze it for authenticity and show you exactly why our AI made its decision.",
    help="Enter the full text of the news article you want to analyze"
)

# Predict button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_clicked = st.button("🚀 Analyze Article", use_container_width=True)

if predict_clicked:
    if user_input.strip() == "":
        st.error("⚠️ Please enter some text to analyze.")
    else:
        with st.spinner("🔄 Analyzing article..."):
            # Predict
            X = vectorizer.transform([user_input])
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X)[0]
            confidence = max(prob) * 100
            
            # Display result
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card real-news">
                    <div class="result-title">
                        ✅ Real News
                    </div>
                    <div class="confidence-text">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card fake-news">
                    <div class="result-title">
                        🚨 Fake News
                    </div>
                    <div class="confidence-text">
                        Confidence: {confidence:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # LIME Explanation
            st.markdown("""
            <div class="explanation-container">
                <div class="explanation-title">
                    🧠 AI Explanation
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.spinner("🔍 Generating detailed explanation..."):
                explainer = LimeTextExplainer(class_names=class_names)
                exp = explainer.explain_instance(user_input, predict_proba, num_features=10)
                
                # Save and customize LIME HTML
                exp.save_to_file("lime_explanation.html")
                
                with open("lime_explanation.html", "r", encoding="utf-8") as f:
                    lime_html = f.read()
                
                # Enhanced custom styles for LIME
                custom_css = """
                <style>
                    body {
                        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
                        color: #1e293b !important;
                        font-family: 'Inter', sans-serif !important;
                        padding: 1.5rem;
                        margin: 0;
                    }
                    
                    .lime-top-label {
                        color: #1e293b !important;
                        font-weight: 600 !important;
                        font-size: 1.1rem !important;
                        margin-bottom: 1rem !important;
                    }
                    
                    .highlighted {
                        border-radius: 6px !important;
                        padding: 3px 6px !important;
                        font-weight: 500 !important;
                        margin: 1px !important;
                        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
                    }
                    
                    .lime-prediction-bar {
                        border-radius: 10px !important;
                        overflow: hidden !important;
                        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1) !important;
                        margin: 1rem 0 !important;
                    }
                    
                    table {
                        border-radius: 12px !important;
                        overflow: hidden !important;
                        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
                        border: none !important;
                        width: 100% !important;
                        margin: 1rem 0 !important;
                        background: white !important;
                    }
                    
                    th {
                        background: linear-gradient(135deg, #667eea, #764ba2) !important;
                        color: white !important;
                        font-weight: 600 !important;
                        padding: 1rem !important;
                        font-size: 0.95rem !important;
                        text-transform: uppercase !important;
                        letter-spacing: 0.5px !important;
                    }
                    
                    td {
                        padding: 0.8rem 1rem !important;
                        border-bottom: 1px solid #e2e8f0 !important;
                        font-size: 0.9rem !important;
                        color: #374151 !important;
                    }
                    
                    tr:hover {
                        background-color: #f8fafc !important;
                    }
                    
                    .lime-explanation {
                        background: white !important;
                        border-radius: 12px !important;
                        padding: 1.5rem !important;
                        margin: 1rem 0 !important;
                        box-shadow: 0 2px 15px rgba(0, 0, 0, 0.05) !important;
                        border: 1px solid #e2e8f0 !important;
                    }
                    
                    /* Probability bars */
                    .lime-prediction-bar div {
                        height: 30px !important;
                        line-height: 30px !important;
                        font-weight: 600 !important;
                        text-align: center !important;
                        font-size: 0.9rem !important;
                    }
                    
                    /* Text highlighting improvements */
                    .lime-text-with-highlighted-words {
                        font-size: 1rem !important;
                        line-height: 1.8 !important;
                        padding: 1rem !important;
                        background: #f8fafc !important;
                        border-radius: 8px !important;
                        border: 1px solid #e2e8f0 !important;
                    }
                </style>
                """
                lime_html = custom_css + lime_html
                
                st.markdown('<h4 style="color: #1e293b; font-weight: 600; margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem;"><span style="font-size: 1.2em;">📊</span> Word Importance Analysis</h4>', unsafe_allow_html=True)
                st.markdown('<p style="color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; padding: 0.75rem; background: #f1f5f9; border-radius: 8px; border-left: 4px solid #667eea;"><strong>How to read:</strong> Green highlights show words that support the prediction, while red highlights show words that oppose it. The intensity of the color indicates the strength of influence.</p>', unsafe_allow_html=True)
                
                components.html(lime_html, height=600, scrolling=True)

# Footer
st.markdown("""
<div class="footer-container">
    <p class="footer-text">
        Built with ❤️ by <span class="footer-name">Chinmayee Bharadwaj</span>
    </p>
    <p class="footer-text" style="font-size: 0.8rem; margin-top: 0.5rem;">
        Powered by Machine Learning & LIME Explainability
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

import pandas as pd
import os

# ✅ Feedback Prompt (only show if prediction exists)
if 'prediction' in locals():
    st.markdown("### 📝 Was this prediction correct?")
    col1, col2 = st.columns(2)

    feedback_file = "user_feedback.csv"

    def save_feedback(text, label):
        feedback = pd.DataFrame([[text, label]], columns=["text", "label"])
        if os.path.exists(feedback_file):
            feedback.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback.to_csv(feedback_file, index=False)
        st.success("✅ Thanks! Your feedback has been saved.")

    with col1:
        if st.button("👍 Yes, it was correct"):
            save_feedback(user_input, prediction)

    with col2:
        new_label = 1 - prediction  # flip the label if user says "No"
        if st.button(f"👎 No, it was {'Real' if prediction == 0 else 'Fake'}"):
            save_feedback(user_input, new_label)
