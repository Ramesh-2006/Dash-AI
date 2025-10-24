import streamlit as st
st.set_page_config(
    page_title="AutoDash AI", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Now import other libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import BytesIO
import base64
import os
from datetime import datetime
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from dotenv import load_dotenv
load_dotenv()
from groq import Groq
import joblib
from rapidfuzz import fuzz
import statsmodels.api as sm
import json
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')
import pickle
import os
import pandas as pd

# Add these global variables at the top of your script
model = None
vectorizer = None
label_encoder = None
# === INITIALIZATION ===
# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load classification model artifacts
@st.cache_resource
def load_classification_model():
    try:
        model = joblib.load("hybrid_dataset_type_classifier_groq.pkl")
        vectorizer = joblib.load("hybrid_column_vectorizer_groq.pkl")
        label_encoder = joblib.load("dataset_type_label_encoder_groq.pkl")
        return model, vectorizer, label_encoder
    except Exception as e:
        st.error(f"Failed to load classification model: {str(e)}")
        return None, None, None

model, vectorizer, label_encoder = load_classification_model()

# === THEME SYSTEM ===
def apply_theme(theme_name):
    if theme_name == "Light":
        return {
            "bg": "#FFFFFF",
            "text": "#31333F",
            "primary": "#4A90E2",
            "secondary": "#F0F2F6",
            "chart_bg": "#FFFFFF",
            "grid": "#E5E5E5"
        }
    else:  # Dark
        return {
            "bg": "#1E1E1E",
            "text": "#FFFFFF",
            "primary": "#90CAF9",
            "secondary": "#2D2D2D",
            "chart_bg": "#2D2D2D",
            "grid": "#424242"
        }

# === GROQ ANALYZER CLASS ===
class GroqAnalyzer:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)
    
    def analyze_dataset_and_suggest_functions(self, df: pd.DataFrame, available_analyses: List[str]) -> Dict[str, Any]:
        """
        Analyze the dataset using Groq LLM to determine the most relevant predefined analyses.
        """
        # Sample the first few rows for analysis
        sample_data = df.head(5).to_string()
        
        # Create a string of available analysis functions for the prompt
        available_analyses_str = ", ".join(available_analyses)
        
        prompt = f"""
        You are a senior data analysis expert. Analyze the following dataset sample and recommend 3-5 of the most relevant analyses to perform. 
        You must choose analysis names ONLY from the following list of available functions: {available_analyses_str}
        
        For each recommended analysis, provide a brief description and suggest which columns to use based on the dataset sample.
        The given analysis functions are predefined and can be called directly. Do NOT suggest any analyses outside this list.
        
        Dataset sample:
        {sample_data}
        
        Respond in JSON format with these keys:
        - "analysis_recommendations": [
            {{
                "name": "analysis_function_name",
                "description": "brief description",
                "columns": ["list", "of", "columns"]
            }}
        ]
        - "confidence_score": (0-1 confidence in recommendation)
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                response_format={"type": "json_object"},
                temperature=0.6
            )
            
            response = json.loads(chat_completion.choices[0].message.content)
            
            # Post-process to ensure analysis names are valid functions
            valid_recommendations = []
            for rec in response.get('analysis_recommendations', []):
                if rec['name'] in available_analyses:
                    valid_recommendations.append(rec)
            
            response['analysis_recommendations'] = valid_recommendations
            return response
        except Exception as e:
            st.error(f"Error analyzing dataset with Groq: {str(e)}")
            return {
                "analysis_recommendations": [],
                "confidence_score": 0
            }

# === HELPER FUNCTIONS ===
def is_year_or_time_column(series):
    """Check if a column represents years, time values, or phone numbers that shouldn't be averaged"""
    if is_numeric_dtype(series):
        # Check for year columns (e.g., values between 1900-2100)
        if series.dropna().between(1900, 2100).all():
            return True
        # Check for time values (e.g., values between 0-2400 representing hours)
        if series.dropna().between(0, 2400).all():
            return True
        # Check for phone numbers (10-15 digit numbers)
        if series.dropna().between(1000000000, 999999999999999).all():
            return True
    elif series.dtype == 'object':
        # Check for string-formatted phone numbers
        if series.str.match(r'^\+?[\d\s-]{10,15}$').all():
            return True
    return False

def classify_dataset(df):
    """Classify the dataset type using the hybrid model"""
    if model is None or vectorizer is None or label_encoder is None:
        return "unknown", 0.0
    
    column_names = df.columns.tolist()
    column_text = " ".join(column_names)
    
    # Try classifier first - with error handling
    try:
        X_unseen = vectorizer.transform([column_text])
        proba = model.predict_proba(X_unseen)
        confidence = proba.max()
        predicted_encoded = model.predict(X_unseen)[0]
        
        if confidence >= 0.8:
            predicted_label = label_encoder.inverse_transform([predicted_encoded])[0]
            return predicted_label, confidence
    except Exception as e:
        st.warning(f"Classifier failed, using fallback: {str(e)}")
    
    # Fallback to fuzzy matching
    training_df = pd.read_csv("dataset_type_training_data.csv")
    best_score = 0
    best_label = None

    for i, row in training_df.iterrows():
        score = fuzz.token_set_ratio(column_text, row["column_text"])
        if score > best_score:
            best_score = score
            best_label = row["dataset_type"]

    if best_score >= 80:
        return best_label, best_score/100
    
    # Final fallback to Groq LLM
    possible_labels = label_encoder.classes_.tolist()
    prompt = f"""
You are a highly intelligent AI model trained to classify datasets into specific types based on their column names alone.

Your task is to choose the single most appropriate dataset type from the list below:

Dataset Types:
sales, marketing, student, employee, finance, healthcare, ecommerce, banking, manufacturing, real_estate, transport, agriculture, education, retail, logistics

Rules:
1. You must analyze only the column names and infer the dataset type.
2. Do NOT include explanations, thoughts, reasoning, or any extra words in the output.
3. Your answer must exactly match one of the dataset types listed above (case-sensitive).
4. Your decision must be based on similarity to the known keywords and patterns below.

Now, analyze the following column names and return only the dataset type from the list above.

Column Names:
{column_text}

Answer:
""".strip()

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        llm_output = response.choices[0].message.content.strip()
        if llm_output in possible_labels:
            return llm_output, 0.9  # High confidence for LLM fallback
        else:
            return "unknown", 0.0
    except Exception as e:
        st.error(f"Groq fallback failed: {e}")
        return "unknown", 0.0

def get_ai_insights(df, dataset_type):
    """Get AI insights about the dataset including model recommendations"""
    data_summary = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    Dataset Type: {dataset_type}
    
    Numeric Columns Summary:
    {df.describe().to_string()}
    
    Categorical Columns Summary:
    {df.select_dtypes(include=['object', 'category']).describe().to_string()}
    
    Sample Data:
    {df.head(3).to_string()}
    """
    
    prompt = f"""
You are a senior data scientist with extensive experience in data analysis and machine learning. 
Analyze this dataset and provide detailed insights in the following structure:

1. **Dataset Classification Summary**:
   - Confirm the dataset type classification ({dataset_type}) and explain why this classification makes sense based on the columns
   - Identify any potential misclassifications or alternative classifications that might be possible

2. **Data Quality Assessment**:
   - Evaluate the overall data quality including missing values, data types, and potential anomalies
   - Highlight any data cleaning or preprocessing steps that should be performed

3. **Key Findings**:
   - Identify 3-5 most interesting patterns, trends, or correlations in the data
   - Point out any unexpected or surprising findings that warrant further investigation

4. **Model Recommendations**:
   - Recommend appropriate machine learning models for this type of data (include both supervised and unsupervised options)
   - For each recommended model, explain why it would be suitable and what business questions it could answer
   - Include any special preprocessing needed for each model

5. **Business Applications**:
   - Suggest 3-5 potential business use cases for this data
   - For each use case, explain how the data could drive value and what metrics would be important
   - Recommend visualization types that would best communicate insights for each use case

Data Summary:
{data_summary}

Provide the response in markdown format with clear section headings and bullet points where appropriate.
The insights should be professional, detailed, and actionable for business stakeholders.
"""
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=3000,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insights: {str(e)}"

def chat_with_data(df, question):
    """Generate a response to a user question about the data"""

    # ✅ Safeguard: handle empty datasets
    if df.empty or df.shape[1] == 0:
        return "⚠️ The dataset has no valid columns after cleaning. Please upload a file with proper headers."

    # ✅ Handle missing numeric or categorical columns gracefully
    numeric_summary = (
        df.describe().to_string()
        if not df.select_dtypes(include=np.number).empty
        else "No numeric columns"
    )

    categorical_summary = (
        df.select_dtypes(include=['object', 'category']).describe().to_string()
        if not df.select_dtypes(include=['object', 'category']).empty
        else "No categorical columns"
    )

    data_summary = f"""
    Dataset Shape: {df.shape}
    Columns: {list(df.columns)}
    
    Numeric Columns Summary:
    {numeric_summary}
    
    Categorical Columns Summary:
    {categorical_summary}
    
    Sample Data:
    {df.head(3).to_string()}
    """

    prompt = f"""
You are a helpful data analyst assistant. The user has uploaded a dataset and is asking questions about it.
Your task is to answer their question based on the dataset summary below. If the question requires specific calculations
or analysis that isn't in the summary, you can describe how to perform that analysis.

Rules:
1. Be concise but thorough in your answers
2. If the question requires calculations not in the summary, explain how to perform them
3. For visualization questions, recommend chart types and how to interpret them
4. If the question is unclear or can't be answered from the data, politely explain why

Dataset Summary:
{data_summary}

User Question:
{question}

Provide your response in clear, plain language. If appropriate, use bullet points for multiple items.
"""

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1500,
            top_p=1,
            stream=False
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating chat response: {str(e)}"


def show_key_metrics(df):
    """Display key metrics about the dataset"""
    st.header("📊 Key Metrics")
    
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", total_records)
    
    with col2:
        st.metric("Total Features", len(df.columns))
    
    with col3:
        st.metric("Numeric Features", len(numeric_cols))
    
    with col4:
        st.metric("Categorical Features", len(categorical_cols))

def show_general_insights(df, title="General Insights"):
    """Show general data visualizations"""
    st.header(f"📊 {title}")
    
    show_key_metrics(df)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        st.subheader("Numeric Features Analysis")
        selected_num_col = st.selectbox("Select numeric feature to analyze", numeric_cols)
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.histogram(df, x=selected_num_col, 
                               title=f"Distribution of {selected_num_col}")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.box(df, y=selected_num_col, 
                           title=f"Box Plot of {selected_num_col}")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("No numeric columns found for analysis.")
    
    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        st.subheader("Feature Correlations")
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", 
                         title="Correlation Between Numeric Features")
        st.plotly_chart(fig3, use_container_width=True)
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        st.subheader("Categorical Features Analysis")
        selected_cat_col = st.selectbox("Select categorical feature to analyze", categorical_cols)
        
        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        
        fig4 = px.bar(value_counts.head(10), x='Value', y='Count', 
                       title=f"Distribution of {selected_cat_col}")
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("No categorical columns found for analysis.")

# === MAIN APP ===
# Sidebar for controls
with st.sidebar:
    st.title("Dashboard Controls")
    theme_choice = st.radio("Choose Theme", ("Light", "Dark"), index=0)
    theme = apply_theme(theme_choice)
    
    st.markdown("---")
    st.markdown("### Visualization Settings")
    default_chart_height = st.slider("Default Chart Height", 300, 800, 450)
    animation_enabled = st.checkbox("Enable Chart Animations", True)
    
    st.markdown("---")
    
    st.markdown("### About")
    st.markdown("AutoDash AI creates professional dashboards automatically from your data.")
    st.markdown("Version 2.1")

# Apply theme CSS
st.markdown(f"""
    <style>
    .main {{
        background-color: {theme['bg']};
        color: {theme['text']};
    }}
    .stButton>button {{
        background-color: {theme['primary']};
        color: white;
    }}
    .stTextInput>div>div>input {{
        background-color: {theme['secondary']};
        color: {theme['text']};
    }}
    .css-1aumxhk {{
        background-color: {theme['secondary']};
    }}
    .reportview-container .main .block-container {{
        padding-top: 2rem;
    }}
    .main-title {{
        font-size: 2.5em;
        font-weight: bold;
        color: {theme['primary']};
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .subtitle {{
        font-size: 1.2em;
        color: {theme['text']};
        text-align: center;
        margin-bottom: 2rem;
        opacity: 0.8;
    }}
    .section-header {{
        font-size: 1.5em;
        font-weight: bold;
        color: {theme['primary']};
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid {theme['primary']};
        padding-bottom: 0.3rem;
    }}
    .chat-message {{
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        max-width: 80%;
    }}
    .user-message {{
        background-color: {theme['primary']};
        color: white;
        margin-left: auto;
        border-bottom-right-radius: 0;
    }}
    .bot-message {{
        background-color: {theme['secondary']};
        color: {theme['text']};
        margin-right: auto;
        border-bottom-left-radius: 0;
    }}
    .analysis-card {{
        padding: 15px;
        border-radius: 10px;
        background-color: {theme['secondary']};
        margin-bottom: 15px;
        border-left: 4px solid {theme['primary']};
    }}
    </style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-title">AutoDash AI </div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Professional Dashboard Generator with AI Insights</div>', unsafe_allow_html=True)

# === FILE UPLOAD ===
uploaded_file = st.file_uploader("📂 Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read file based on extension
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, encoding='latin1')
    else:
        df = pd.read_excel(uploaded_file, engine='openpyxl')
    
    # Basic data cleaning
    df = df.dropna(how='all')  # Drop completely empty rows
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]  # Remove unnamed columns
    
    # Convert date columns if detected
    for col in df.columns:
        if is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass
    
    st.success(f"✅ Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Classify dataset
    with st.spinner("Classifying dataset type..."):
        dataset_type, confidence = classify_dataset(df)
        if dataset_type != "unknown":
            st.success(f"🔍 Dataset classified as: **{dataset_type}** (confidence: {confidence:.2f})")
        else:
            st.warning("⚠️ Could not confidently classify dataset type")
    
    # === DATA EXPLORER ===
    with st.expander("🔍 Data Explorer", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Preview", "Structure", "Missing Data"])
        
        with tab1:
            st.dataframe(df.to_dict('records'), use_container_width=True)
        
        with tab2:
            st.markdown("**Data Types:**")
            dtype_df = pd.DataFrame(df.dtypes, columns=['Data Type'])
            st.dataframe(dtype_df, use_container_width=True)
            
            st.markdown("**Unique Values:**")
            unique_df = pd.DataFrame(df.nunique(), columns=['Unique Values'])
            st.dataframe(unique_df, use_container_width=True)
        
        with tab3:
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing Values']
            missing_data['% Missing'] = (missing_data['Missing Values'] / len(df)) * 100
            missing_data['% Missing'] = missing_data['% Missing'].round(2)
            
            fig = px.bar(missing_data, 
                         x='Column', 
                         y='% Missing',
                         title='Percentage of Missing Values by Column',
                         color='% Missing',
                         color_continuous_scale='RdYlGn_r')
            fig.update_layout(plot_bgcolor=theme['chart_bg'],
                             paper_bgcolor=theme['bg'],
                             font_color=theme['text'])
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(missing_data, use_container_width=True)
    
    # === DASHBOARD SECTION ===
    # === DASHBOARD SECTION ===
    st.markdown('<div class="section-header">📊 Automated Dashboard</div>', unsafe_allow_html=True)

    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Relationships", "Time Analysis", "AI Insights", "Data Chatbot"])

    with tab1:  # Overview tab
        # Initialize Groq Analyzer
        groq_analyzer = GroqAnalyzer()
        
        # Show AI analysis recommendations if available
        if 'analysis_result' not in st.session_state:
            st.session_state.analysis_result = None
            
        if 'recommended_analyses' not in st.session_state:
            st.session_state.recommended_analyses = []
        
        if st.button("Analyze with AI"):
            with st.spinner("Analyzing dataset with AI..."):
                # Get available analyses for the dataset type
                available_analyses = []
                if dataset_type != "unknown":
                    try:
                        # Dynamically import the module for the dataset type
                        module = __import__(f"datasets.{dataset_type}", fromlist=[dataset_type])
                        if hasattr(module, 'analysis_options'):
                            available_analyses = module.analysis_options
                        st.session_state.recommended_analyses = available_analyses
                    except ImportError:
                        st.warning(f"Could not load analysis functions for dataset type: {dataset_type}")
                        available_analyses = []
                
                if available_analyses:
                    analysis_result = groq_analyzer.analyze_dataset_and_suggest_functions(df, available_analyses)
                    st.session_state.analysis_result = analysis_result
                else:
                    st.warning(f"No specific analysis functions found for dataset type '{dataset_type}'.")
                    st.session_state.analysis_result = None
            
            st.success("AI Analysis Complete!")
        
        # Show analysis recommendations if available
        if st.session_state.analysis_result:
            analysis_result = st.session_state.analysis_result
            
            st.subheader("AI-Powered Analysis Recommendations")
            st.write(f"Dataset Type: {dataset_type.title()}")
            
            cols = st.columns(3)
            for i, analysis in enumerate(analysis_result.get('analysis_recommendations', [])):
                with cols[i % 3]:
                    st.markdown(f"""
                    <div class="analysis-card">
                        <h4>{analysis['name'].replace('_', ' ').title()}</h4>
                        <p>{analysis['description']}</p>
                        <small>Columns: {", ".join(analysis['columns'])}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Run {analysis['name'].replace('_', ' ').title()}", key=f"run_{i}"):
                        try:
                            # Dynamically import and call the analysis function
                            module = __import__(f"datasets.{dataset_type}", fromlist=[dataset_type])
                            analysis_function = getattr(module, analysis['name'])
                            analysis_function(df)
                        except Exception as e:
                            st.error(f"Could not run analysis: {str(e)}")
                            show_general_insights(df, title=f"General Insights for {dataset_type.title()}")
        
    
    
        
        # Show key metrics and distributions
        st.markdown("### Key Metrics")
        
        # Create metric cards for numeric columns
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            cols = st.columns(min(4, len(num_cols)))
            for i, col in enumerate(num_cols[:4]):
                with cols[i]:
                    if is_year_or_time_column(df[col]):
                        st.metric(
                            label=col,
                            value=f"Range",
                            delta=f"{df[col].min():.0f} - {df[col].max():.0f}"
                        )
                    else:
                        st.metric(
                            label=col,
                            value=f"{df[col].mean():,.1f}",
                            delta=f"Range: {df[col].min():,.1f} - {df[col].max():,.1f}"
                        )
            
            # Distribution charts
            st.markdown("### Distributions")
            
            # Numeric distributions
            if num_cols:
                st.markdown("#### Numeric Features")
                cols = st.columns(2)
                for i, col in enumerate(num_cols[:4]):
                    with cols[i % 2]:
                        fig = px.histogram(
                            df, 
                            x=col,
                            nbins=50,
                            title=f'Distribution of {col}',
                            color_discrete_sequence=[theme['primary']]
                        )
                        fig.update_layout(
                            plot_bgcolor=theme['chart_bg'],
                            paper_bgcolor=theme['bg'],
                            font_color=theme['text'],
                            height=default_chart_height
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Categorical distributions
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if cat_cols:
                st.markdown("#### Categorical Features")
                cols = st.columns(2)
                for i, col in enumerate(cat_cols[:4]):
                    with cols[i % 2]:
                        value_counts = df[col].value_counts().nlargest(10)
                        fig = px.pie(
                            names=value_counts.index,
                            values=value_counts.values,
                            title=f'Top 10 {col} Distribution',
                            color_discrete_sequence=px.colors.qualitative.Pastel
                        )
                        fig.update_layout(
                            plot_bgcolor=theme['chart_bg'],
                            paper_bgcolor=theme['bg'],
                            font_color=theme['text'],
                            height=default_chart_height,
                            showlegend=True
                        )
                        st.plotly_chart(fig, use_container_width=True)

    with tab2:  # Relationships tab
        st.markdown("### Feature Relationships")
        
        # Correlation matrix
        if len(num_cols) > 1:
            st.markdown("#### Correlation Matrix")
            corr_matrix = df[num_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}"
            ))
            
            fig.update_layout(
                title='Feature Correlation Matrix',
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text'],
                height=600,
                xaxis_showgrid=False,
                yaxis_showgrid=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot matrix
        if len(num_cols) > 1:
            st.markdown("#### Scatter Plot Matrix")
            fig = px.scatter_matrix(
                df,
                dimensions=num_cols[:4],  # Limit to 4 columns for performance
                color=cat_cols[0] if cat_cols else None,
                title="Scatter Plot Matrix",
                height=800
            )
            fig.update_layout(
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text']
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:  # Time Analysis tab
        # Find datetime columns
        date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
        
        if date_cols:
            st.markdown("### Time Series Analysis")
            
            # Time series line chart
            date_col = st.selectbox("Select date column", date_cols)
            num_col = st.selectbox("Select numeric column to plot", num_cols)
            
            if date_col and num_col:
                time_df = df.set_index(date_col).sort_index()
                
                fig = px.line(
                    time_df,
                    y=num_col,
                    title=f'{num_col} Over Time',
                    markers=True
                )
                
                fig.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text'],
                    height=default_chart_height,
                    xaxis_title=date_col,
                    yaxis_title=num_col,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Time decomposition
            st.markdown("#### Time Series Decomposition")
            if len(num_cols) > 0:
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    period = st.slider("Seasonal Period", 7, 365, 30)
                    
                    decomposition = seasonal_decompose(time_df[num_cols[0]], period=period, model='additive')
                    
                    fig = make_subplots(
                        rows=4, cols=1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_df.index, y=decomposition.observed, name="Observed"),
                        row=1, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_df.index, y=decomposition.trend, name="Trend"),
                        row=2, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_df.index, y=decomposition.seasonal, name="Seasonal"),
                        row=3, col=1
                    )
                    
                    fig.add_trace(
                        go.Scatter(x=time_df.index, y=decomposition.resid, name="Residual"),
                        row=4, col=1
                    )
                    
                    fig.update_layout(
                        height=800,
                        plot_bgcolor=theme['chart_bg'],
                        paper_bgcolor=theme['bg'],
                        font_color=theme['text'],
                        showlegend=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Could not perform time series decomposition. Ensure you have sufficient data points.")
        else:
            st.info("No datetime columns found for time series analysis.")

    with tab4:  # AI Insights tab
        st.markdown("### AI-Powered Insights ")
        
        if st.button("Generate Insights"):
            with st.spinner("Analyzing data with AI..."):
                insights = get_ai_insights(df, dataset_type)
                st.markdown(insights)
                
                # Save insights to session state
                st.session_state.insights = insights
        
        if 'insights' in st.session_state:
            st.download_button(
                label="Download Insights Report",
                data=st.session_state.insights,
                file_name="data_insights_report.md",
                mime="text/markdown"
            )

    with tab5:  # Data Chatbot tab
        st.markdown("### Data Chatbot")
        st.markdown("Ask questions about your dataset and get instant answers.")
        
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Accept user input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_with_data(df, prompt)
                st.markdown(response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Add some suggested questions
        st.markdown("**Try asking:**")
        st.markdown("- What are the most important columns in this dataset?")
        st.markdown("- Are there any correlations I should be aware of?")
        st.markdown("- What visualizations would best represent this data?")
        st.markdown("- Are there any data quality issues I should address?")

    # === EXPORT SECTION ===
    st.markdown('<div class="section-header">📤 Export Dashboard</div>', unsafe_allow_html=True)

    # Export options (simplified to HTML and Excel only)
    export_format = st.selectbox(
        "Select export format:",
        ["HTML", "Excel"],
        index=0
    )

    if st.button(f"Generate {export_format} Report"):
        if export_format == "HTML":
            try:
                html_content = f"""
                <html>
                <head>
                    <title>AutoDash Report</title>
                    <style>
                        body {{ font-family: Arial; margin: 20px; }}
                        h1 {{ color: #4A90E2; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                    </style>
                </head>
                <body>
                    <h1>AutoDash Report</h1>
                    <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <h2>Dataset Type: {dataset_type}</h2>
                    <h2>Data Sample</h2>
                    {df.head().to_html()}
                    <h2>Statistics</h2>
                    {df.describe().to_html()}
                    <h2>Insights</h2>
                    <pre>{st.session_state.get('insights', '')}</pre>
                </body>
                </html>
                """
                
                st.download_button(
                    label="🌐 Download HTML Report",
                    data=html_content,
                    file_name="autodash_report.html",
                    mime="text/html"
                )
                
            except Exception as e:
                st.error(f"Error generating HTML: {str(e)}")
        
        elif export_format == "Excel":
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                    df.describe().to_excel(writer, sheet_name='Statistics')
                    if 'insights' in st.session_state:
                        pd.DataFrame({'Insights': [st.session_state.insights]}).to_excel(
                            writer, sheet_name='Insights', index=False)
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=output.getvalue(),
                    file_name="autodash_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")
        
        elif export_format == "Excel":
            try:
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='Data', index=False)
                    df.describe().to_excel(writer, sheet_name='Statistics')
                    if 'insights' in st.session_state:
                        pd.DataFrame({'Insights': [st.session_state.insights]}).to_excel(
                            writer, sheet_name='Insights', index=False)
                
                st.download_button(
                    label="📊 Download Excel Report",
                    data=output.getvalue(),
                    file_name="autodash_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            except Exception as e:
                st.error(f"Error generating Excel: {str(e)}")

