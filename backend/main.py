import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import json
import joblib
from rapidfuzz import fuzz
from groq import Groq
from dotenv import load_dotenv
import warnings
from datetime import datetime
from io import BytesIO, StringIO
import base64
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype, is_datetime64_any_dtype
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
import uuid
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import traceback
import aiofiles
from fastapi.encoders import jsonable_encoder
import importlib
from sklearn.exceptions import InconsistentVersionWarning
import warnings
from pydantic import BaseModel

# Filter out specific warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", category=FutureWarning, message="errors='ignore' is deprecated")

# Initialize FastAPI app
app = FastAPI(title="AutoDash AI Backend", version="2.1")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3004", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load environment variables
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None
    print("Warning: GROQ_API_KEY not found in environment variables")

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
except Exception as e:
    print(f"Warning: Could not mount static files or templates: {str(e)}")
    templates = None

# Global variables for classification model
model = None
vectorizer = None
label_encoder = None

# Load classification model artifacts
def load_classification_model():
    try:
        model = joblib.load("hybrid_dataset_type_classifier_groq.pkl")
        vectorizer = joblib.load("hybrid_column_vectorizer_groq.pkl")
        label_encoder = joblib.load("dataset_type_label_encoder_groq.pkl")
        return model, vectorizer, label_encoder
    except Exception as e:
        print(f"Failed to load classification model: {str(e)}")
        return None, None, None

model, vectorizer, label_encoder = load_classification_model()

# === GROQ ANALYZER CLASS ===
class GroqAnalyzer:
    def __init__(self):
        if GROQ_API_KEY:
            self.client = Groq(api_key=GROQ_API_KEY)
        else:
            self.client = None
    
    def analyze_dataset_and_suggest_functions(self, df: pd.DataFrame, available_analyses: List[str]) -> Dict[str, Any]:
        """
        Analyze the dataset using Groq LLM to determine the most relevant predefined analyses.
        """
        if not self.client:
            return {
                "analysis_recommendations": [],
                "confidence_score": 0
            }
            
        # Sample the first few rows for analysis
        sample_data = df.head(5).to_string()
        
        # Create a string of available analysis functions for the prompt
        available_analyses_str = ", ".join(available_analyses)
        
        prompt = f"""
        You are a senior data analysis expert. Analyze the following dataset sample and recommend 3-5 of the most relevant analyses to perform. 
        You must choose analysis names ONLY from the following list of available functions: {available_analyses_str}
        
        For each recommended analysis, provide a brief description and suggest which columns to use based on the dataset sample.
        
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
            print(f"Error analyzing dataset with Groq: {str(e)}")
            return {
                "analysis_recommendations": [],
                "confidence_score": 0
            }

# Initialize Groq Analyzer
groq_analyzer = GroqAnalyzer()

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
        print(f"Classifier failed, using fallback: {str(e)}")
    
    # Fallback to fuzzy matching
    try:
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
    except Exception as e:
        print(f"Fuzzy matching fallback failed: {str(e)}")
    
    # Final fallback to Groq LLM
    possible_labels = ['sales', 'marketing', 'student', 'employee', 'finance', 
                      'healthcare', 'ecommerce', 'banking', 'manufacturing', 
                      'real_estate', 'transport', 'agriculture', 'education', 
                      'retail', 'logistics']
    
    prompt = f"""
You are a highly intelligent AI model trained to classify datasets into specific types based on their column names alone.

Your task is to choose the single most appropriate dataset type from the list below:

Dataset Types:
{', '.join(possible_labels)}

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
        if not client:
            return "unknown", 0.0
            
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        llm_output = response.choices[0].message.content.strip()
        if llm_output in possible_labels:
            return llm_output, 0.9
        else:
            return "unknown", 0.0
    except Exception as e:
        print(f"Groq fallback failed: {e}")
        return "unknown", 0.0

def get_ai_insights(df, dataset_type):
    """Get AI insights about the dataset including model recommendations"""
    if not client:
        return "Error: Groq client not initialized. Please check your API key."
        
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
    if not client:
        return "Error: Groq client not initialized. Please check your API key."

    # Safeguard: empty dataset
    if df.empty or df.shape[1] == 0:
        return "⚠️ The dataset has no valid columns after cleaning. Please upload a file with proper headers."

    # Handle numeric summary safely
    if not df.select_dtypes(include=np.number).empty:
        numeric_summary = df.describe().to_string()
    else:
        numeric_summary = "No numeric columns available"

    # Handle categorical summary safely
    if not df.select_dtypes(include=['object', 'category']).empty:
        categorical_summary = df.select_dtypes(include=['object', 'category']).describe().to_string()
    else:
        categorical_summary = "No categorical columns available"

    # Final dataset summary
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
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    return {
        "total_records": total_records,
        "total_features": len(df.columns),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols)
    }

def analyze_relationships(df, theme):
    """Analyze feature relationships (correlation and scatter plots)"""
    results = {}
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Correlation matrix
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            
            # Create a more visually appealing correlation matrix
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmin=-1,
                zmax=1,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                hoverinfo="text"
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
            
            results['correlation_matrix'] = fig.to_json()
        except Exception as e:
            results['correlation_error'] = f"Error creating correlation matrix: {str(e)}"
    
    # Scatter plot matrix
    if len(numeric_cols) > 1:
        try:
            # Get categorical columns for coloring
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            color_col = categorical_cols[0] if categorical_cols else None
            
            fig = px.scatter_matrix(
                df,
                dimensions=numeric_cols[:4],  # Limit to 4 columns for performance
                color=color_col,
                title="Scatter Plot Matrix",
                height=800
            )
            
            fig.update_layout(
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text']
            )
            
            results['scatter_matrix'] = fig.to_json()
        except Exception as e:
            results['scatter_error'] = f"Error creating scatter matrix: {str(e)}"
    
    return results

def analyze_time_series(df, theme, date_col=None, num_col=None, period=30):
    """Analyze time series data with specific columns"""
    results = {}
    
    # If no columns provided, return available options
    if not date_col or not num_col:
        # Find datetime columns
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if is_datetime64_any_dtype(df[col]):
                        date_cols.append(col)
                except:
                    continue
            elif is_datetime64_any_dtype(df[col]):
                date_cols.append(col)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return {
            "date_cols": date_cols,
            "numeric_cols": numeric_cols,
            "message": "Please select date and numeric columns"
        }
    
    try:
        # Convert date column to datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col, num_col])
        time_df = df.set_index(date_col).sort_index()
        
        # Line chart
        fig_line = px.line(
            time_df,
            y=num_col,
            title=f'{num_col} Over Time',
            markers=True
        )
        
        fig_line.update_layout(
            plot_bgcolor=theme['chart_bg'],
            paper_bgcolor=theme['bg'],
            font_color=theme['text'],
            xaxis_title=date_col,
            yaxis_title=num_col,
            hovermode="x unified"
        )
        
        results['line_chart'] = fig_line.to_json()
        
        # Time decomposition
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            clean_series = time_df[num_col].dropna()
            
            if len(clean_series) > period * 2:
                decomposition = seasonal_decompose(clean_series, period=period, model='additive')
                
                fig_dec = make_subplots(
                    rows=4, cols=1,
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual")
                )
                
                fig_dec.add_trace(
                    go.Scatter(x=clean_series.index, y=decomposition.observed, name="Observed"),
                    row=1, col=1
                )
                
                fig_dec.add_trace(
                    go.Scatter(x=clean_series.index, y=decomposition.trend, name="Trend"),
                    row=2, col=1
                )
                
                fig_dec.add_trace(
                    go.Scatter(x=clean_series.index, y=decomposition.seasonal, name="Seasonal"),
                    row=3, col=1
                )
                
                fig_dec.add_trace(
                    go.Scatter(x=clean_series.index, y=decomposition.resid, name="Residual"),
                    row=4, col=1
                )
                
                fig_dec.update_layout(
                    height=800,
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text'],
                    showlegend=False
                )
                
                results['decomposition'] = fig_dec.to_json()
            else:
                results['decomposition_warning'] = "Not enough data for time series decomposition"
        except Exception as e:
            results['decomposition_error'] = f"Time series decomposition failed: {str(e)}"
    
    except Exception as e:
        results['timeseries_error'] = f"Time series analysis failed: {str(e)}"
    
    return results

def generate_visualizations(df, theme):
    """Generate dashboard visualizations for the report"""
    visualizations = {}
    
    # Get basic visualizations
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if numeric_cols:
        # Histograms and box plots for first 2 numeric columns
        for i, col in enumerate(numeric_cols[:2]):
            try:
                # Histogram
                fig_hist = px.histogram(df, x=col, nbins=50, title=f'Distribution of {col}')
                fig_hist.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'histogram_{col}'] = fig_hist.to_json()
                
                # Box plot
                fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
                fig_box.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'boxplot_{col}'] = fig_box.to_json()
            except Exception as e:
                print(f"Error creating visualization for {col}: {str(e)}")
    
    # Add relationships analysis
    relationships = analyze_relationships(df, theme)
    visualizations.update(relationships)
    
    # Add time series analysis (without specific columns)
    timeseries = analyze_time_series(df, theme)
    visualizations.update(timeseries)
    
    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        try:
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", 
                               title="Correlation Between Numeric Features")
            fig_corr.update_layout(
                plot_bgcolor=theme['chart_bg'],
                paper_bgcolor=theme['bg'],
                font_color=theme['text']
            )
            visualizations['correlation_matrix'] = fig_corr.to_json()
        except Exception as e:
            print(f"Error creating correlation matrix: {str(e)}")
    
    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if categorical_cols:
        for i, col in enumerate(categorical_cols[:2]):
            try:
                value_counts = df[col].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                fig_bar = px.bar(value_counts.head(10), x='Value', y='Count', 
                               title=f"Distribution of {col}")
                fig_bar.update_layout(
                    plot_bgcolor=theme['chart_bg'],
                    paper_bgcolor=theme['bg'],
                    font_color=theme['text']
                )
                visualizations[f'barchart_{col}'] = fig_bar.to_json()
            except Exception as e:
                print(f"Error creating bar chart for {col}: {str(e)}")
    
    return visualizations

def generate_pdf_report(df, dataset_type, insights, analysis_results=None, theme=None):
    """Generate a PDF report with insights, analysis, and dashboard visualizations"""
    if theme is None:
        theme = {
            "bg": "#FFFFFF",
            "text": "#31333F",
            "primary": "#4A90E2",
            "secondary": "#F0F2F6",
            "chart_bg": "#FFFFFF",
            "grid": "#E5E5E5"
        }
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#4A90E2')
    )
    story.append(Paragraph("AutoDash AI Analysis Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Dataset Info
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    key_metrics = show_key_metrics(df)
    dataset_info = [
        ["Dataset Type", dataset_type],
        ["Number of Rows", str(df.shape[0])],
        ["Number of Columns", str(df.shape[1])],
        ["Numeric Features", str(key_metrics['numeric_features'])],
        ["Categorical Features", str(key_metrics['categorical_features'])],
        ["Columns", ", ".join(df.columns.tolist())]
    ]
    table = Table(dataset_info, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F0F2F6')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Key Metrics
    story.append(Paragraph("Key Metrics", styles['Heading2']))
    metrics_data = [
        ["Metric", "Value"],
        ["Total Records", str(key_metrics['total_records'])],
        ["Total Features", str(key_metrics['total_features'])],
        ["Numeric Features", str(key_metrics['numeric_features'])],
        ["Categorical Features", str(key_metrics['categorical_features'])]
    ]
    metrics_table = Table(metrics_data, colWidths=[2*inch, 2*inch])
    metrics_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # AI Insights
    story.append(Paragraph("AI Insights & Recommendations", styles['Heading2']))
    insights_paragraphs = insights.split('\n\n')
    for para in insights_paragraphs:
        if para.strip():
            story.append(Paragraph(para.strip(), styles['Normal']))
            story.append(Spacer(1, 12))
    
    # Add Relationships Analysis section
    story.append(Paragraph("Relationships Analysis", styles['Heading2']))
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 1:
        story.append(Paragraph("✓ Correlation analysis performed", styles['Normal']))
        story.append(Paragraph("✓ Scatter plot matrix generated", styles['Normal']))
    else:
        story.append(Paragraph("Not enough numeric columns for relationships analysis", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add Time Series Analysis section
    story.append(Paragraph("Time Series Analysis", styles['Heading2']))
    date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
    if date_cols and numeric_cols:
        story.append(Paragraph("✓ Time series charts generated", styles['Normal']))
        story.append(Paragraph("✓ Seasonal decomposition performed", styles['Normal']))
    else:
        if not date_cols:
            story.append(Paragraph("No datetime columns found for time series analysis", styles['Normal']))
        else:
            story.append(Paragraph("No numeric columns found for time series analysis", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Data Preview
    story.append(Paragraph("Data Preview (First 5 rows)", styles['Heading2']))
    preview_data = [df.columns.tolist()] + df.head().values.tolist()
    preview_table = Table(preview_data, repeatRows=1)
    preview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(preview_table)
    story.append(Spacer(1, 20))
    
    # Basic Statistics
    story.append(Paragraph("Basic Statistics", styles['Heading2']))
    if df.select_dtypes(include=[np.number]).shape[1] > 0:
        stats = df.describe().reset_index()
        stats_data = [stats.columns.tolist()] + stats.values.tolist()
        stats_table = Table(stats_data, repeatRows=1)
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4A90E2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 7),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(stats_table)
    else:
        story.append(Paragraph("No numeric columns for statistical analysis", styles['Normal']))
    
    # Dashboard Visualizations section
    story.append(Paragraph("Dashboard Visualizations", styles['Heading2']))
    story.append(Paragraph("The following visualizations are available in the interactive dashboard:", styles['Normal']))
    
    visualization_list = []
    if len(numeric_cols) > 0:
        visualization_list.append("• Distribution charts (histograms, box plots)")
    if len(numeric_cols) > 1:
        visualization_list.append("• Correlation matrix")
        visualization_list.append("• Scatter plot matrix")
    if date_cols and numeric_cols:
        visualization_list.append("• Time series charts")
        visualization_list.append("• Seasonal decomposition")
    
    for item in visualization_list:
        story.append(Paragraph(item, styles['Normal']))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("Note: Please use the web interface for interactive charts and detailed analysis.", styles['Italic']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# === PYDANTIC MODELS ===
class ChatRequest(BaseModel):
    message: str
    preview: List[Dict]
    history: Optional[List[Dict]] = []

class ExportRequest(BaseModel):
    preview: List[Dict]
    format: str
    insights: str = ""
    dataset_type: str = "unknown"
    theme: Dict = None

# === API ENDPOINTS ===
@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and initial processing"""
    try:
        # Read file based on extension with better error handling
        file_extension = file.filename.split('.')[-1].lower()
        
        if file_extension == 'csv':
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    file.file.seek(0)  # Reset file pointer
                    df = pd.read_csv(file.file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    continue
            
            if df is None:
                # Fallback: read as bytes and decode with errors ignored
                file.file.seek(0)
                content = await file.read()
                try:
                    # Try to decode as string
                    content_str = content.decode('utf-8', errors='ignore')
                    df = pd.read_csv(StringIO(content_str))
                except:
                    # Final fallback: use pandas with error handling
                    file.file.seek(0)
                    df = pd.read_csv(file.file, encoding='utf-8', errors='ignore')
                    
        elif file_extension in ['xlsx', 'xls']:
            file.file.seek(0)
            df = pd.read_excel(file.file, engine='openpyxl')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload CSV or Excel files.")
        
        # Basic data cleaning
        df = df.dropna(how='all')
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        
        # ... continuing from the previous backend code

        # Convert date columns if detected
        for col in df.columns:
            if is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
            elif df[col].dtype == 'object':
                try:
                    # Try to convert to datetime
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                except:
                    pass
        
        # Classify dataset
        dataset_type, confidence = classify_dataset(df)
        
        # Get sample data for preview (handle large datasets)
        preview_data = df.head(10).to_dict(orient="records")
        
        response = {
            "status": "success",
            "filename": file.filename,
            "rows": df.shape[0],
            "columns": df.shape[1],
            "dataset_type": dataset_type,
            "confidence": confidence,
            "preview": preview_data,
            "columns_info": [{"name": col, "dtype": str(df[col].dtype)} for col in df.columns],
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "date_columns": [col for col in df.columns if is_datetime64_any_dtype(df[col])]
        }
        
        return response
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.post("/api/get_ai_insights")
async def get_insights(request: Request):
    """Get AI insights about the dataset"""
    try:
        data = await request.json()
        df_data = data["preview"]
        df = pd.DataFrame(df_data)
        dataset_type = data.get("dataset_type", "unknown")
        
        insights = get_ai_insights(df, dataset_type)
        return {"status": "success", "insights": insights}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat")
async def chat_endpoint(request: Request):
    """Chat with the data using AI"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        question = data.get("message", "")
        chat_history = data.get("history", [])
        
        if not df_data or not question:
            raise HTTPException(status_code=400, detail="Missing preview data or question")
        
        df = pd.DataFrame(df_data)
        
        response = chat_with_data(df, question)
        return {"status": "success", "response": response}
    except Exception as e:
        print(f"Chat error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/analyze_with_ai")
async def analyze_with_ai(request: Request):
    """Analyze dataset with AI and get recommendations - matches Streamlit functionality"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        dataset_type = data.get("dataset_type", "unknown")
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        
        # Get available analyses for the dataset type
        available_analyses = []
        if dataset_type != "unknown":
            try:
                module = __import__(f"datasets.{dataset_type}", fromlist=[dataset_type])
                if hasattr(module, 'analysis_options'):
                    available_analyses = module.analysis_options
            except ImportError:
                print(f"No analysis module found for {dataset_type}")
        
        if available_analyses:
            analysis_result = groq_analyzer.analyze_dataset_and_suggest_functions(df, available_analyses)
            return {
                "status": "success",
                "analysis_result": analysis_result,
                "available_analyses": available_analyses
            }
        else:
            return {
                "status": "success",
                "analysis_result": {
                    "analysis_recommendations": [],
                    "confidence_score": 0
                },
                "message": f"No specific analysis functions found for dataset type '{dataset_type}'"
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/run_analysis_function")
async def run_analysis_function(request: Request):
    """Run a specific analysis function - matches Streamlit functionality"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        analysis_name = data.get("analysis_name")
        dataset_type = data.get("dataset_type", "unknown")
        
        if not df_data or not analysis_name:
            raise HTTPException(status_code=400, detail="Missing data or analysis name")
        
        df = pd.DataFrame(df_data)
        
        if dataset_type == "unknown":
            # Fallback to general insights
            results = show_general_insights_fallback(df)
            return {
                "status": "success",
                "analysis_name": analysis_name,
                "result": results
            }
            
        try:
            module = __import__(f"datasets.{dataset_type}", fromlist=[dataset_type])
            analysis_function = getattr(module, analysis_name, None)
            
            if analysis_function and callable(analysis_function):
                result = analysis_function(df)
                return {
                    "status": "success",
                    "analysis_name": analysis_name,
                    "result": result
                }
            else:
                # Fallback to general analysis
                results = show_general_insights_fallback(df)
                return {
                    "status": "success",
                    "analysis_name": analysis_name,
                    "result": results
                }
                
        except ImportError:
            # Fallback to general analysis
            results = show_general_insights_fallback(df)
            return {
                "status": "success",
                "analysis_name": analysis_name,
                "result": results
            }
            
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

def show_general_insights_fallback(df):
    """Fallback analysis when specific analysis functions aren't available"""
    return {
        "summary": "General data analysis completed",
        "insights": [
            f"Dataset contains {len(df)} rows and {len(df.columns)} columns",
            f"Numeric columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}",
            f"Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}"
        ],
        "recommendations": [
            "Consider uploading a more specific dataset type for specialized analysis",
            "Explore the relationships between numeric features",
            "Check for missing values in your dataset"
        ]
    }

@app.post("/api/get_time_series_data")
async def get_time_series_data(request: Request):
    """Get time series analysis data - matches Streamlit functionality"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        date_col = data.get("date_col")
        num_col = data.get("num_col")
        period = data.get("period", 30)
        theme = data.get("theme", {})
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        
        # If no specific columns provided, return available options
        if not date_col or not num_col:
            date_cols = [col for col in df.columns if is_datetime64_any_dtype(df[col])]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            return {
                "status": "success",
                "date_cols": date_cols,
                "numeric_cols": numeric_cols,
                "message": "Please select date and numeric columns"
            }
        
        # Perform time series analysis
        results = analyze_time_series(df, theme, date_col, num_col, period)
        return {"status": "success", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/get_relationships_data")
async def get_relationships_data(request: Request):
    """Get relationships analysis data - matches Streamlit functionality"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        theme = data.get("theme", {})
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        results = analyze_relationships(df, theme)
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/generate_visualizations")
async def generate_dashboard_visualizations(request: Request):
    """Generate dashboard visualizations for the dataset"""
    try:
        data = await request.json()
        df_data = data.get("preview", [])
        theme = data.get("theme", {
            "bg": "#0A071E",
            "text": "#ffffff",
            "primary": "#8a2be2",
            "secondary": "#1a183c",
            "chart_bg": "#1a183c",
            "grid": "#3a3863"
        })
        
        if not df_data:
            raise HTTPException(status_code=400, detail="Missing preview data")
        
        df = pd.DataFrame(df_data)
        visualizations = generate_visualizations(df, theme)
        key_metrics = show_key_metrics(df)
        
        return {
            "status": "success",
            "visualizations": visualizations,
            "key_metrics": key_metrics
        }
    except Exception as e:
        print(f"Visualization error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/chat_with_data")
async def chat_with_data_endpoint(request: ChatRequest):
    """Enhanced chat endpoint matching Streamlit functionality"""
    try:
        df = pd.DataFrame(request.preview)
        response = chat_with_data(df, request.message)
        
        return {
            "status": "success", 
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/export_dashboard")
async def export_dashboard(request: ExportRequest):
    """Export dashboard matching Streamlit functionality"""
    try:
        df = pd.DataFrame(request.preview)
        
        if request.format == "pdf":
            pdf_buffer = generate_pdf_report(
                df, 
                request.dataset_type, 
                request.insights, 
                theme=request.theme
            )
            return StreamingResponse(
                pdf_buffer,
                media_type="application/pdf",
                headers={"Content-Disposition": f"attachment; filename=autodash_report.pdf"}
            )
        else:
            return {"status": "error", "message": "Only PDF export is supported"}
            
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/get_available_analyses")
async def get_available_analyses(request: Request):
    """Get available analysis functions for dataset type"""
    try:
        data = await request.json()
        dataset_type = data.get("dataset_type", "unknown")
        
        available_analyses = []
        if dataset_type != "unknown":
            try:
                # Try to import the module dynamically
                module = __import__(f"datasets.{dataset_type}", fromlist=[dataset_type])
                if hasattr(module, 'analysis_options'):
                    available_analyses = module.analysis_options
            except ImportError:
                print(f"No analysis module found for {dataset_type}")
        
        return {
            "status": "success",
            "available_analyses": available_analyses,
            "dataset_type": dataset_type
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# === FRONTEND ROUTES ===
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serve the frontend HTML"""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        return HTMLResponse(content="<h1>Error: Templates not configured</h1>")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)