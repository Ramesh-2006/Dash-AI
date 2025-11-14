import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


analysis_options = [
    "get_general_insights"
]

def clean_metrics(d):
    """
    Recursively convert numpy types in a dictionary or list to Python native types
    for JSON serialization.
    """
    if isinstance(d, dict):
        return {k: clean_metrics(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [clean_metrics(i) for i in d]
    elif isinstance(d, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(d)
    elif isinstance(d, (np.float_, np.float16, np.float32, np.float64)):
        # Handle potential NaN/Inf
        if np.isnan(d) or np.isinf(d):
            return None  # Or str(d) if you prefer 'NaN'
        return float(d)
    elif isinstance(d, np.bool_):
        return bool(d)
    elif isinstance(d, np.datetime64):
        return pd.to_datetime(str(d)).isoformat()
    elif isinstance(d, pd.Timestamp):
        return d.isoformat()
    elif pd.isna(d):
        return None
    return d


def get_key_metrics(df):
    """Return key metrics about the dataset as a dictionary."""
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    return {
        "total_records": total_records,
        "total_features": len(df.columns),
        "numeric_features_count": len(numeric_cols),
        "categorical_features_count": len(categorical_cols),
        "numeric_features_list": numeric_cols,
        "categorical_features_list": categorical_cols,
    }


def get_general_insights(df, analysis_type="general_analysis", fallback_message=""):
    """
    Generate general data visualizations and return as a structured dictionary.
    This serves as a fallback when a specific analysis can't be run.
    """
    metrics = get_key_metrics(df)
    visualizations = {}
    insights = [fallback_message] if fallback_message else ["Showing general data insights."]

    try:
        # Numeric columns analysis
        numeric_cols = metrics["numeric_features_list"]
        if len(numeric_cols) > 0:
            selected_num_col = numeric_cols[0]
            insights.append(f"Analyzed numeric feature: {selected_num_col}")
            fig1 = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
            visualizations["numeric_distribution_histogram"] = fig1.to_json()
            fig2 = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
            visualizations["numeric_distribution_boxplot"] = fig2.to_json()
        else:
            insights.append("No numeric columns found for analysis.")

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            insights.append("Analyzing feature correlations.")
            # Ensure only numeric columns are used for correlation
            corr_df = df[numeric_cols].corr()
            fig3 = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
            visualizations["correlation_heatmap"] = fig3.to_json()

        # Categorical columns analysis
        categorical_cols = metrics["categorical_features_list"]
        if len(categorical_cols) > 0:
            selected_cat_col = categorical_cols[0]
            insights.append(f"Analyzing categorical feature: {selected_cat_col}")
            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']
            fig4 = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Top 10 Values for {selected_cat_col}")
            visualizations["categorical_distribution_barchart"] = fig4.to_json()
        else:
            insights.append("No categorical columns found for analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "fallback",
            "matched_columns": {},
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"Error during general insights generation: {str(e)}",
            "metrics": {},
            "visualizations": {},
            "insights": [str(e)]
        }
