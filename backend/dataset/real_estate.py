import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, linregress
from fuzzywuzzy import process # Now using fuzzy matching
import json
import warnings
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ========== UTILITY FUNCTIONS (Adapted from your example) ==========

class NumpyJSONEncoder(json.JSONEncoder):
    """ Custom encoder for numpy and pandas data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif pd.isna(obj):
            return None
        return super(NumpyJSONEncoder, self).default(obj)

def convert_to_native_types(data):
    """
    Converts a structure (dict, list) containing numpy/pandas types
    to one with only native Python types using a custom JSON encoder.
    """
    try:
        # Round floats to a reasonable precision for clean JSON output
        def round_floats(o):
            if isinstance(o, float):
                return round(o, 4)
            if isinstance(o, dict):
                return {k: round_floats(v) for k, v in o.items()}
            if isinstance(o, list):
                return [round_floats(v) for v in o]
            return o
        
        rounded_data = round_floats(data)
        return json.loads(json.dumps(rounded_data, cls=NumpyJSONEncoder))
    except Exception:
        # Fallback in case rounding or conversion fails
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder, default=str))


def fuzzy_match_column(df, target_columns):
    """Improved fuzzy matching with better handling"""
    matched = {}
    available = df.columns.tolist()
    
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
            
        # Standardize for better matching
        target_std = target.lower().replace("_", "").replace(" ", "").replace("-", "")
        available_std = {col.lower().replace("_", "").replace(" ", "").replace("-", ""): col for col in available}
        
        # Try exact match after standardization
        if target_std in available_std:
            matched[target] = available_std[target_std]
            continue

        # If no direct match, use fuzzy matching
        try:
            # process.extractOne returns (match, score)
            match_result = process.extractOne(target, available)
            if match_result and match_result[1] >= 80: # Using a threshold of 80
                matched[target] = match_result[0]
            else:
                matched[target] = None
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    rename_map = {v: k for k, v in matched.items() if v is not None and v in df.columns}
    return df.rename(columns=rename_map)

def load_data(file_path, encoding='utf-8'):
    """Load data from CSV or Excel file with robust encoding support"""
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            print("[ERROR] Failed to decode file with common encodings.")
            return None
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            print("[ERROR] Unsupported file format. Please provide CSV or Excel file.")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

# ========== FALLBACK & GENERAL INSIGHTS FUNCTIONS ==========

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, formatted for API return."""
    analysis_type = "General Insights"
    visualizations = {}
    metrics = {}
    insights = []
    
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        
        # Compile metrics
        metrics = {
            "dataset_overview": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "duplicate_rows": int(duplicate_rows),
            },
            "data_types_breakdown": {
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "datetime_columns_count": len(datetime_cols),
            },
            "data_quality": {
                "total_missing_values": int(missing_values.sum()),
                "columns_with_missing_count": len(columns_with_missing),
            }
        }
        
        # Generate insights
        insights.append(f"Dataset contains {total_rows:,} rows and {total_columns} columns.")
        insights.append(f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns.")
        
        if duplicate_rows > 0:
            insights.append(f"⚠️ Found {duplicate_rows:,} duplicate rows.")
        else:
            insights.append("✅ No duplicate rows found.")
            
        if len(columns_with_missing) > 0:
            insights.append(f"⚠️ {len(columns_with_missing)} columns have missing values.")
        else:
            insights.append("✅ No missing values found.")

        # Add missing columns warning if provided (for fallbacks)
        if missing_cols and len(missing_cols) > 0:
            insights.append("---")
            insights.append(f"⚠️ The requested analysis '{analysis_name.replace('General Insights (Fallback for ', '').replace(')', '')}' could not be run.")
            insights.append("The following critical columns were not found:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"  - {col}{match_info}")
            insights.append("Showing General Data Overview instead.")

        # Visualizations
        
        # 1. Missing values
        if len(columns_with_missing) > 0:
            missing_df = pd.DataFrame({
                'column': columns_with_missing.index,
                'missing_percentage': (columns_with_missing.values / total_rows) * 100
            }).sort_values('missing_percentage', ascending=False)
            
            fig_missing = px.bar(
                missing_df.head(10), 
                x='column', y='missing_percentage',
                title='Top 10 Columns with Missing Values (%)'
            )
            visualizations["missing_values_percentage"] = fig_missing.to_json()
        
        # 2. Numeric distributions (first 2 numeric cols)
        for col in numeric_cols[:2]:
            try:
                fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                visualizations[f"{col}_distribution"] = fig_hist.to_json()
            except Exception as e:
                print(f"Could not plot histogram for {col}: {e}")

        # 3. Categorical distributions (first 2 categorical cols)
        for col in categorical_cols[:2]:
            try:
                top_10_counts = df[col].value_counts().head(10).reset_index()
                top_10_counts.columns = ['Value', 'Count']
                fig_bar = px.bar(top_10_counts, x='Value', y='Count', title=f'Top 10 Values for {col}')
                visualizations[f"{col}_distribution"] = fig_bar.to_json()
            except Exception as e:
                print(f"Could not plot bar chart for {col}: {e}")

        # 4. Correlation heatmap (if at least 2 numeric cols)
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Numeric Feature Correlation Heatmap")
                visualizations["correlation_heatmap"] = fig_corr.to_json()
                insights.append("Generated correlation heatmap for numeric features.")
            except Exception as e:
                print(f"Could not plot correlation heatmap: {e}")

        return {
            "analysis_type": analysis_type,
            "status": "success", # This function itself succeeded
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights,
            "missing_columns": missing_cols or []
        }
        
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": f"Error in show_general_insights: {str(e)}",
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides general insights as a fallback.
    """
    print(f"--- ⚠️ Required Columns Not Found for {analysis_name} ---")
    print(f"Missing: {missing_cols}")
    print("Falling back to General Insights.")
    
    try:
        # Generate the fallback general insights
        general_insights_data = show_general_insights(
            df, 
            f"General Insights (Fallback for {analysis_name})",
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
    except Exception as e:
        # Failsafe if general insights also fails
        general_insights_data = {
            "visualizations": {},
            "metrics": {},
            "insights": [f"Fallback to general insights failed: {str(e)}"]
        }

    return {
        "analysis_type": analysis_name,
        "status": "fallback",
        "message": f"Required columns were missing for '{analysis_name}'. Falling back to general insights.",
        "missing_columns": missing_cols,
        "matched_columns": matched_cols,
        "visualizations": general_insights_data.get("visualizations", {}),
        "metrics": general_insights_data.get("metrics", {}),
        "insights": general_insights_data.get("insights", [])
    }

# ========== REAL ESTATE ANALYSIS FUNCTIONS (Refactored) ==========

def property_analysis(df):
    analysis_name = "Property Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'address', 'property_type', 'bedrooms', 
                    'bathrooms', 'sqft', 'year_built', 'price', 'location']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['price', 'sqft'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)

        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
        df = df.dropna(subset=['price', 'sqft'])
        
        total_properties = len(df)
        avg_price = df['price'].mean()
        avg_sqft = df['sqft'].mean()
        avg_price_per_sqft = avg_price / avg_sqft if avg_sqft > 0 else 0

        metrics = {
            "total_properties": total_properties,
            "avg_price": avg_price,
            "avg_sqft": avg_sqft,
            "avg_price_per_sqft": avg_price_per_sqft
        }
        
        insights.append(f"Analyzed {total_properties} properties.")
        insights.append(f"Average Price: ${avg_price:,.0f}")
        insights.append(f"Average Sqft: {avg_sqft:,.0f}")
        insights.append(f"Average Price/Sqft: ${avg_price_per_sqft:,.2f}")

        if 'property_type' in df.columns:
            fig1 = px.pie(df, names='property_type', title="Property Type Distribution")
            visualizations["property_type_distribution"] = fig1.to_json()
        
        if 'price' in df.columns:
            fig2 = px.histogram(df, x='price', title="Price Distribution")
            visualizations["price_distribution"] = fig2.to_json()
        
        if 'price' in df.columns and 'sqft' in df.columns:
            fig3 = px.scatter(df, x='sqft', y='price', 
                              color='property_type' if 'property_type' in df.columns else None,
                              title="Price vs Square Footage", trendline="ols")
            visualizations["price_vs_sqft"] = fig3.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def location_analysis(df):
    analysis_name = "Location Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'address', 'latitude', 'longitude', 
                    'neighborhood', 'zipcode', 'price']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['latitude', 'longitude', 'price'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)

        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['latitude', 'longitude', 'price'])
        
        unique_neighborhoods = df['neighborhood'].nunique() if 'neighborhood' in df.columns else 0
        unique_zipcodes = df['zipcode'].nunique() if 'zipcode' in df.columns else 0
        
        metrics = {
            "total_properties": len(df),
            "unique_neighborhoods": unique_neighborhoods,
            "unique_zipcodes": unique_zipcodes
        }
        
        insights.append(f"Analyzed {len(df)} properties with valid location data.")
        if unique_neighborhoods > 0:
            insights.append(f"Data covers {unique_neighborhoods} unique neighborhoods.")
        
        fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                                 color='price', size='price',
                                 hover_name='address' if 'address' in df.columns else None,
                                 zoom=10, height=600,
                                 title="Property Price Map")
        fig1.update_layout(mapbox_style="open-street-map")
        visualizations["property_price_map"] = fig1.to_json()
        
        if 'neighborhood' in df.columns:
            neighborhood_prices = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False).reset_index()
            metrics["top_neighborhood_by_price"] = neighborhood_prices.iloc[0].to_dict() if not neighborhood_prices.empty else {}
            
            fig2 = px.bar(neighborhood_prices.head(20), x='neighborhood', y='price',
                          title="Average Price by Neighborhood (Top 20)")
            visualizations["price_by_neighborhood"] = fig2.to_json()
            insights.append(f"Top neighborhood by average price: {metrics['top_neighborhood_by_price'].get('neighborhood')}")

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def price_trends_analysis(df):
    analysis_name = "Price Trends Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'date', 'price', 'property_type', 'sqft']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['date', 'price'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)
        
        if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['date', 'price'])
        
        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No valid data for analysis.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": insights
            }

        df['month'] = df['date'].dt.to_period('M').astype(str)
        monthly_trends = df.groupby('month')['price'].agg(['mean', 'count']).reset_index()
        
        avg_price_change = (monthly_trends['mean'].pct_change() * 100).mean()
        total_sales = monthly_trends['count'].sum()
        
        metrics = {
            "avg_monthly_price_change_percent": avg_price_change,
            "total_sales_in_period": total_sales,
            "monthly_trend_data": monthly_trends.to_dict('records')
        }
        
        insights.append(f"Average Monthly Price Change: {avg_price_change:.1f}%")
        insights.append(f"Total Sales in Period: {total_sales}")
        
        fig1 = px.line(monthly_trends, x='month', y='mean', title="Average Price Over Time")
        visualizations["avg_price_over_time"] = fig1.to_json()
        
        if 'property_type' in df.columns:
            type_trends = df.groupby(['month', 'property_type'])['price'].mean().reset_index()
            fig2 = px.line(type_trends, x='month', y='price', color='property_type',
                           title="Price Trends by Property Type")
            visualizations["price_trends_by_type"] = fig2.to_json()
            insights.append("Generated price trend plot by property type.")

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def rental_analysis(df):
    analysis_name = "Rental Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'address', 'property_type', 'bedrooms', 
                    'bathrooms', 'sqft', 'monthly_rent', 'location', 'occupancy_status']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['monthly_rent'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)

        df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')
        df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce') if 'sqft' in df.columns else None
        df = df.dropna(subset=['monthly_rent'])
        
        total_rentals = len(df)
        avg_rent = df['monthly_rent'].mean()
        occupancy_rate = 0
        if 'occupancy_status' in df.columns:
            # Handle variations in "Occupied"
            occupied_count = df['occupancy_status'].str.lower().isin(['occupied', 'rented']).sum()
            occupancy_rate = (occupied_count / total_rentals) * 100 if total_rentals > 0 else 0
            
        avg_rent_per_sqft = 0
        if 'sqft' in df.columns and df['sqft'].mean() > 0:
            avg_rent_per_sqft = avg_rent / df['sqft'].mean()
        
        metrics = {
            "total_rentals": total_rentals,
            "avg_monthly_rent": avg_rent,
            "occupancy_rate_percent": occupancy_rate,
            "avg_rent_per_sqft": avg_rent_per_sqft
        }
        
        insights.append(f"Total Rentals: {total_rentals}")
        insights.append(f"Average Monthly Rent: ${avg_rent:,.0f}")
        if 'occupancy_status' in df.columns:
            insights.append(f"Occupancy Rate: {occupancy_rate:.1f}%")
        if 'sqft' in df.columns:
            insights.append(f"Average Rent/Sqft: ${avg_rent_per_sqft:,.2f}")
        
        fig1 = px.histogram(df, x='monthly_rent', title="Monthly Rent Distribution")
        visualizations["monthly_rent_distribution"] = fig1.to_json()
        
        if 'bedrooms' in df.columns:
            fig2 = px.box(df, x='bedrooms', y='monthly_rent', title="Rent by Number of Bedrooms")
            visualizations["rent_by_bedrooms"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def investment_analysis(df):
    analysis_name = "Investment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'purchase_price', 'purchase_date', 
                    'current_value', 'rental_income', 'expenses', 'location']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['purchase_price', 'current_value'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)

        df['purchase_price'] = pd.to_numeric(df['purchase_price'], errors='coerce')
        df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce')
        df['rental_income'] = pd.to_numeric(df['rental_income'], errors='coerce') if 'rental_income' in df.columns else None
        df['expenses'] = pd.to_numeric(df['expenses'], errors='coerce') if 'expenses' in df.columns else None
        if 'purchase_date' in df.columns:
             df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

        df = df.dropna(subset=['purchase_price', 'current_value'])
        
        total_investment = df['purchase_price'].sum()
        current_value_total = df['current_value'].sum()
        total_roi = ((current_value_total - total_investment) / total_investment) * 100 if total_investment > 0 else 0
        annual_rental_income = 0
        if 'rental_income' in df.columns:
            annual_rental_income = df['rental_income'].sum() * 12
        
        metrics = {
            "total_investment": total_investment,
            "current_value_total": current_value_total,
            "total_roi_percent": total_roi,
            "estimated_annual_rental_income": annual_rental_income
        }
        
        insights.append(f"Total Investment: ${total_investment:,.0f}")
        insights.append(f"Current Value: ${current_value_total:,.0f}")
        insights.append(f"Total ROI: {total_roi:.1f}%")
        if 'rental_income' in df.columns:
            insights.append(f"Annual Rental Income: ${annual_rental_income:,.0f}")
        
        if 'purchase_price' in df.columns and 'current_value' in df.columns:
            df['roi'] = ((df['current_value'] - df['purchase_price']) / df['purchase_price']) * 100
            fig1 = px.histogram(df, x='roi', title="Return on Investment (ROI) Distribution")
            visualizations["roi_distribution"] = fig1.to_json()
        
        if 'purchase_date' in df.columns and not df['purchase_date'].isnull().all():
            df['year'] = df['purchase_date'].dt.year
            yearly_performance = df.groupby('year').agg(
                purchase_price=('purchase_price', 'sum'),
                current_value=('current_value', 'sum')
            ).reset_index()
            yearly_performance['roi'] = ((yearly_performance['current_value'] - yearly_performance['purchase_price']) / 
                                         yearly_performance['purchase_price']) * 100
            
            fig2 = px.line(yearly_performance, x='year', y='roi', title="Investment Performance by Purchase Year")
            visualizations["performance_by_purchase_year"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def market_comparison_analysis(df):
    analysis_name = "Market Comparison Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'price', 'sqft', 'bedrooms', 
                    'bathrooms', 'property_type', 'location']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['price', 'sqft'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)

        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
        df = df.dropna(subset=['price', 'sqft'])
        
        df['price_per_sqft'] = df['price'] / df['sqft']
        
        # Determine comparison column
        compare_by = None
        if 'property_type' in df.columns and df['property_type'].nunique() > 1:
            compare_by = 'property_type'
        elif 'bedrooms' in df.columns and df['bedrooms'].nunique() > 1:
            compare_by = 'bedrooms'
        elif 'location' in df.columns and df['location'].nunique() > 1:
            compare_by = 'location'
        
        if not compare_by:
            insights.append("Could not find a suitable column (like 'property_type', 'bedrooms', or 'location') for comparison.")
            # Still return success, but with limited plots
        else:
            insights.append(f"Comparing market by '{compare_by}'.")
            
            fig1 = px.box(df, x=compare_by, y='price',
                          title=f"Price Distribution by {compare_by}")
            visualizations[f"price_by_{compare_by}"] = fig1.to_json()
            
            fig2 = px.box(df, x=compare_by, y='price_per_sqft',
                          title=f"Price per Sqft by {compare_by}")
            visualizations[f"price_per_sqft_by_{compare_by}"] = fig2.to_json()

        metrics = {
            "avg_price": df['price'].mean(),
            "avg_price_per_sqft": df['price_per_sqft'].mean()
        }
        
        insights.insert(0, f"Overall average price: ${metrics['avg_price']:,.0f}")
        insights.insert(1, f"Overall average price per sqft: ${metrics['avg_price_per_sqft']:,.2f}")

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def affordability_analysis(df):
    analysis_name = "Affordability Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['property_id', 'price', 'bedrooms', 'bathrooms', 
                    'sqft', 'property_type', 'location']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['price'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = safe_rename(df, matched)
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df.dropna(subset=['price'])

        # Default values for affordability calculation
        annual_income = 100000
        down_payment_pct = 20
        interest_rate = 5.0
        loan_term = 30
        dti_ratio = 36
        
        down_payment = down_payment_pct / 100
        monthly_income = annual_income / 12
        max_monthly_payment = monthly_income * (dti_ratio / 100)
        
        monthly_interest_rate = (interest_rate / 100) / 12
        loan_term_months = loan_term * 12
        
        def calculate_max_price(monthly_payment):
            if monthly_interest_rate == 0:
                loan_amount = monthly_payment * loan_term_months
            else:
                loan_amount = (monthly_payment * (1 - (1 + monthly_interest_rate) ** -loan_term_months)) / monthly_interest_rate
            return loan_amount / (1 - down_payment)
            
        max_affordable_price = calculate_max_price(max_monthly_payment)
        
        affordable_properties = df[df['price'] <= max_affordable_price]
        num_affordable = len(affordable_properties)
        pct_affordable = (num_affordable / len(df)) * 100 if len(df) > 0 else 0
        
        metrics = {
            "assumed_annual_income": annual_income,
            "assumed_down_payment_percent": down_payment_pct,
            "assumed_interest_rate_percent": interest_rate,
            "max_affordable_home_price": max_affordable_price,
            "affordable_properties_count": num_affordable,
            "affordable_properties_percent": pct_affordable
        }
        
        insights.append(f"With an assumed annual income of ${annual_income:,.0f} and a {down_payment_pct}% down payment:")
        insights.append(f"Maximum affordable home price: ${max_affordable_price:,.0f}")
        insights.append(f"Affordable properties in market: {num_affordable} ({pct_affordable:.1f}% of total)")

        # Visualization: Pie chart of affordable vs unaffordable
        affordability_data = pd.DataFrame({
            'Category': ['Affordable', 'Unaffordable'],
            'Count': [num_affordable, len(df) - num_affordable]
        })
        fig1 = px.pie(affordability_data, names='Category', values='Count', title="Market Affordability")
        visualizations["affordability_pie_chart"] = fig1.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def boston_housing_price_prediction_analysis(df):
    analysis_name = "Boston Housing Price Prediction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['rm', 'lstat', 'medv'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['rm', 'lstat', 'medv'], inplace=True)

        median_value = df['medv'].median() * 1000
        avg_rooms = df['rm'].mean()
        lstat_corr = df['lstat'].corr(df['medv'])

        metrics = {
            "median_home_value_usd": median_value,
            "avg_rooms_per_dwelling": avg_rooms,
            "lower_status_population_correlation_with_value": lstat_corr
        }

        insights.append(f"Median Home Value: ${median_value:,.0f}")
        insights.append(f"Average Rooms per Dwelling: {avg_rooms:.2f}")
        insights.append(f"Correlation between % Lower Status Population and Value: {lstat_corr:.2f} (Strong negative correlation)")

        fig1 = px.scatter(df, x='rm', y='medv', 
                          color='crim' if 'crim' in df.columns else None,
                          title="Median Value vs. Average Number of Rooms",
                          labels={'rm': 'Average Rooms', 'medv': 'Median Value ($1000s)', 'crim': 'Crime Rate'})
        visualizations["value_vs_rooms"] = fig1.to_json()

        fig2 = px.scatter(df, x='lstat', y='medv', trendline='ols',
                          title="Median Value vs. % Lower Status of Population",
                          labels={'lstat': '% Lower Status Population', 'medv': 'Median Value ($1000s)'})
        visualizations["value_vs_lstat"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_listing_description_and_time_on_market_analysis(df):
    analysis_name = "Listing Description and Time-on-Market Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['full_description', 'deal_days']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['full_description', 'deal_days'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['deal_days'] = pd.to_numeric(df['deal_days'], errors='coerce')
        df.dropna(subset=['full_description', 'deal_days'], inplace=True)
        df['desc_length'] = df['full_description'].str.len()

        avg_days_on_market = df['deal_days'].mean()
        avg_desc_length = df['desc_length'].mean()
        corr = df['desc_length'].corr(df['deal_days'])

        metrics = {
            "avg_days_on_market": avg_days_on_market,
            "avg_description_length_chars": avg_desc_length,
            "description_length_vs_days_on_market_correlation": corr
        }

        insights.append(f"Average Days on Market: {avg_days_on_market:.1f}")
        insights.append(f"Average Description Length: {avg_desc_length:.0f} characters")
        insights.append(f"Correlation between description length and days on market: {corr:.2f}")

        fig1 = px.histogram(df, x='deal_days', nbins=50, title="Distribution of Days on Market")
        visualizations["days_on_market_distribution"] = fig1.to_json()

        fig2 = px.scatter(df, x='desc_length', y='deal_days', trendline='ols',
                          title="Days on Market vs. Description Length",
                          labels={'desc_length': 'Description Length (characters)', 'deal_days': 'Days on Market'})
        visualizations["days_vs_description_length"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_valuation_zestimate_and_feature_analysis(df):
    analysis_name = "Property Valuation (Zestimate) and Feature Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['zestimate', 'bedroom_number', 'bathroom_number', 'price_per_unit', 'living_space', 'property_type']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['zestimate', 'living_space', 'bedroom_number'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['zestimate', 'bedroom_number', 'bathroom_number', 'price_per_unit', 'living_space']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['zestimate', 'living_space', 'bedroom_number'], inplace=True)

        avg_zestimate = df['zestimate'].mean()
        avg_living_space = df['living_space'].mean()
        
        metrics = {
            "avg_zestimate": avg_zestimate,
            "avg_living_space_sqft": avg_living_space
        }

        if 'price_per_unit' in df.columns:
            metrics['avg_price_per_unit'] = df['price_per_unit'].mean()
            insights.append(f"Average Price per Unit: ${metrics['avg_price_per_unit']:,.0f}")

        insights.insert(0, f"Average Zestimate: ${avg_zestimate:,.0f}")
        insights.insert(1, f"Average Living Space: {avg_living_space:,.0f} sqft")

        fig1 = px.scatter(df, x='living_space', y='zestimate', 
                          color='property_type' if 'property_type' in df.columns else None,
                          title="Zestimate vs. Living Space by Property Type")
        visualizations["zestimate_vs_living_space"] = fig1.to_json()

        fig2 = px.box(df, x='bedroom_number', y='zestimate', title="Zestimate Distribution by Number of Bedrooms")
        visualizations["zestimate_by_bedrooms"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_transaction_analysis_by_area_and_furnishing(df):
    analysis_name = "Transaction Analysis by Area and Furnishing"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['area', 'furnishing', 'transaction']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['area', 'furnishing', 'transaction'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
        df.dropna(inplace=True)
        
        summary = df.groupby(['transaction', 'furnishing']).agg(
            count=('area', 'count'),
            avg_area=('area', 'mean')
        ).reset_index()
        
        metrics = {
            "transaction_summary": summary.to_dict('records')
        }
        
        insights.append("Generated summary of transactions by furnishing status and type.")
        
        fig1 = px.sunburst(df, path=['transaction', 'furnishing'], title="Hierarchical View of Transactions by Furnishing")
        visualizations["transaction_sunburst"] = fig1.to_json()
        
        fig2 = px.box(df, x='furnishing', y='area', color='transaction', title="Area Distribution by Furnishing and Transaction Type")
        visualizations["area_by_furnishing_transaction"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_price_prediction_based_on_location_and_features(df):
    analysis_name = "Price Prediction (Location & Features)"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['house_age', 'distance_to_the_nearest_mrt_station', 'number_of_convenience_stores', 
                    'latitude', 'longitude', 'house_price_of_unit_area']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['distance_to_the_nearest_mrt_station', 'latitude', 'longitude', 'house_price_of_unit_area'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=critical_missing, inplace=True) # Drop based on critical columns

        avg_price = df['house_price_of_unit_area'].mean()
        corr_mrt = df['distance_to_the_nearest_mrt_station'].corr(df['house_price_of_unit_area'])
        
        metrics = {
            "avg_house_price_of_unit_area": avg_price,
            "correlation_price_vs_mrt_distance": corr_mrt
        }
        
        insights.append(f"Average House Price of Unit Area: {avg_price:.2f}")
        insights.append(f"Correlation between Price and Distance to MRT: {corr_mrt:.2f} (Negative correlation expected)")

        fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='house_price_of_unit_area', size='house_price_of_unit_area',
                                 hover_name='house_price_of_unit_area', zoom=10, height=600, title="Geospatial Price Distribution")
        fig1.update_layout(mapbox_style="open-street-map")
        visualizations["geospatial_price_map"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='distance_to_the_nearest_mrt_station', y='house_price_of_unit_area',
                          color='number_of_convenience_stores' if 'number_of_convenience_stores' in df.columns else None, 
                          title="Price vs. Distance to MRT")
        visualizations["price_vs_mrt_distance"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_sales_and_assessment_ratio_analysis(df):
    analysis_name = "Property Sales and Assessment Ratio Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['town', 'assessed_value', 'sale_amount', 'sales_ratio', 'property_type', 'residential_type']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['assessed_value', 'sale_amount'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['assessed_value', 'sale_amount', 'sales_ratio']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['assessed_value', 'sale_amount'], inplace=True)

        # Calculate sales_ratio if not present
        if 'sales_ratio' not in df.columns or df['sales_ratio'].isnull().all():
            if df['assessed_value'].mean() > 0: # Avoid division by zero
                df['sales_ratio'] = df['sale_amount'] / df['assessed_value']
                insights.append("Calculated 'sales_ratio' (Sale Amount / Assessed Value).")
            else:
                insights.append("Could not calculate 'sales_ratio' (Assessed Value is zero or missing).")
        
        avg_ratio = df['sales_ratio'].mean() if 'sales_ratio' in df.columns else 0
        
        metrics = {
            "avg_sales_ratio": avg_ratio
        }
        
        insights.insert(0, f"Average Sales Ratio (Sale Amount / Assessed Value): {avg_ratio:.3f}")

        if 'town' in df.columns and 'sales_ratio' in df.columns:
            ratio_by_town = df.groupby('town')['sales_ratio'].mean().nlargest(20).reset_index()
            fig1 = px.bar(ratio_by_town, x='town', y='sales_ratio', title="Top 20 Towns by Average Sales Ratio")
            visualizations["avg_sales_ratio_by_town"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='assessed_value', y='sale_amount', 
                          color='property_type' if 'property_type' in df.columns else None,
                          title="Sale Amount vs. Assessed Value", log_x=True, log_y=True)
        visualizations["sale_vs_assessed_value"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def neighborhood_property_characteristics_analysis(df):
    analysis_name = "Neighborhood Property Characteristics Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['beds', 'size', 'baths', 'neighborhood']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['beds', 'size', 'neighborhood'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['beds', 'size', 'baths']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['beds', 'size', 'neighborhood'], inplace=True)

        summary = df.groupby('neighborhood').agg(
            avg_beds=('beds', 'mean'),
            avg_baths=('baths', 'mean') if 'baths' in df.columns else pd.NamedAgg(column='beds', aggfunc='count'), # Placeholder if no baths
            avg_size=('size', 'mean'),
            property_count=('size', 'count')
        ).reset_index()
        
        metrics = {
            "neighborhood_summary": summary.to_dict('records')
        }
        
        insights.append("Generated summary of property characteristics by neighborhood.")

        fig = px.scatter(summary, x='avg_size', y='property_count', color='neighborhood',
                         size='avg_beds', title="Property Count vs. Average Size by Neighborhood")
        visualizations["neighborhood_summary_scatter"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def house_price_prediction_based_on_property_features(df):
    analysis_name = "House Price Prediction based on Property Features"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'parking', 'furnishingstatus']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['price', 'area', 'bedrooms'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['price', 'area', 'bedrooms'], inplace=True)

        median_price = df['price'].median()
        avg_area = df['area'].mean()
        price_area_corr = df['area'].corr(df['price'])
        
        metrics = {
            "median_price": median_price,
            "avg_area_sqft": avg_area,
            "price_area_correlation": price_area_corr
        }
        
        insights.append(f"Median Price: ${median_price:,.0f}")
        insights.append(f"Average Area (sqft): {avg_area:,.0f}")
        insights.append(f"Price/Area Correlation: {price_area_corr:.2f} (Strong positive correlation expected)")
        
        fig1 = px.scatter(df, x='area', y='price', 
                          color='furnishingstatus' if 'furnishingstatus' in df.columns else None,
                          title="Price vs. Area by Furnishing Status")
        visualizations["price_vs_area"] = fig1.to_json()
        
        fig2 = px.box(df, x='bedrooms', y='price', 
                      color='stories' if 'stories' in df.columns else None, 
                      title="Price by Bedrooms and Stories")
        visualizations["price_by_bedrooms_stories"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_sales_and_appraisal_data_analysis(df):
    analysis_name = "Property Sales and Appraisal Data Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['saledate', 'totalappraisedvalue', 'totalfinishedarea', 'livingunits', 'xrprimaryneighborhoodid']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['saledate', 'totalappraisedvalue', 'totalfinishedarea'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        for col in ['totalappraisedvalue', 'totalfinishedarea', 'livingunits']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['saledate', 'totalappraisedvalue', 'totalfinishedarea'], inplace=True)

        avg_appraisal = df['totalappraisedvalue'].mean()
        
        metrics = {
            "avg_appraised_value": avg_appraisal
        }
        
        insights.append(f"Average Appraised Value: ${avg_appraisal:,.0f}")

        appraisal_over_time = df.groupby(df['saledate'].dt.to_period('Y').astype(str))['totalappraisedvalue'].mean().reset_index()
        fig1 = px.bar(appraisal_over_time, x='saledate', y='totalappraisedvalue', title="Average Appraised Value Over Years")
        visualizations["avg_appraisal_over_time"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='totalfinishedarea', y='totalappraisedvalue', 
                          color='xrprimaryneighborhoodid' if 'xrprimaryneighborhoodid' in df.columns else None,
                          title="Appraised Value vs. Finished Area by Neighborhood")
        visualizations["appraisal_vs_finished_area"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_pricing_and_feature_analysis(df):
    analysis_name = "Real Estate Pricing and Feature Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['rate', 'carpet_area', 'floor', 'bedroom', 'bathroom', 'parking', 'ownership']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['rate', 'carpet_area', 'bedroom'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['rate', 'carpet_area', 'bedroom', 'bathroom']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['rate', 'carpet_area', 'bedroom'], inplace=True)

        avg_rate = df['rate'].mean()
        
        metrics = {
            "avg_rate": avg_rate
        }
        
        insights.append(f"Average Rate: ${avg_rate:,.0f}")
        
        fig1 = px.scatter(df, x='carpet_area', y='rate', 
                          color='ownership' if 'ownership' in df.columns else None,
                          title="Rate vs. Carpet Area by Ownership Type")
        visualizations["rate_vs_carpet_area"] = fig1.to_json()
        
        rate_by_bedroom = df.groupby('bedroom')['rate'].mean().reset_index()
        fig2 = px.bar(rate_by_bedroom, x='bedroom', y='rate', title="Average Rate by Number of Bedrooms")
        visualizations["avg_rate_by_bedrooms"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_listing_time_on_market_analysis(df):
    analysis_name = "Property Listing Time-on-Market Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['bedrooms', 'bathrooms', 'square_feet', 'days_on_market', 'property_type']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['square_feet', 'days_on_market'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['bedrooms', 'bathrooms', 'square_feet', 'days_on_market']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['square_feet', 'days_on_market'], inplace=True)

        avg_dom = df['days_on_market'].mean()
        
        metrics = {
            "avg_days_on_market": avg_dom
        }
        
        insights.append(f"Average Days on Market: {avg_dom:.1f}")
        
        if 'property_type' in df.columns:
            dom_by_type = df.groupby('property_type')['days_on_market'].mean().reset_index()
            fig1 = px.bar(dom_by_type, x='property_type', y='days_on_market', title="Average Days on Market by Property Type")
            visualizations["dom_by_property_type"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='square_feet', y='days_on_market', title="Days on Market vs. Square Feet")
        visualizations["dom_vs_sqft"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_sales_price_analysis(df):
    analysis_name = "Real Estate Sales Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['list_price', 'sale_price', 'bedrooms', 'bathrooms', 'square_footage', 'year_built']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['list_price', 'sale_price', 'square_footage'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['list_price', 'sale_price', 'square_footage'], inplace=True)
        
        df['sale_to_list_ratio'] = df['sale_price'] / df['list_price']
        avg_ratio = df['sale_to_list_ratio'].mean()
        
        metrics = {
            "avg_sale_to_list_price_ratio": avg_ratio
        }
        
        insights.append(f"Average Sale-to-List Price Ratio: {avg_ratio:.3f} (e.g., 1.0 = sold at list price)")
        
        fig1 = px.scatter(df, x='list_price', y='sale_price', title="Sale Price vs. List Price")
        fig1.add_shape(type='line', x0=df['list_price'].min(), y0=df['list_price'].min(), 
                       x1=df['list_price'].max(), y1=df['list_price'].max(),
                       line=dict(color='red', dash='dash'))
        visualizations["sale_vs_list_price"] = fig1.to_json()

        fig2 = px.scatter(df, x='square_footage', y='sale_price', color='sale_to_list_ratio',
                          title="Sale Price vs. Square Footage (Colored by Sale-to-List Ratio)")
        visualizations["sale_price_vs_sqft_by_ratio"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def neighborhood_property_sales_trend_analysis(df):
    analysis_name = "Neighborhood Property Sales Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['neighborhood', 'area_sqft', 'bedrooms', 'sale_date', 'sale_price']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['neighborhood', 'sale_date', 'sale_price'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
        df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
        df.dropna(subset=['neighborhood', 'sale_date', 'sale_price'], inplace=True)
        
        df['sale_year'] = df['sale_date'].dt.year
        price_over_time = df.groupby(['sale_year', 'neighborhood'])['sale_price'].mean().reset_index()
        
        metrics = {
            "trend_data": price_over_time.to_dict('records')
        }
        
        insights.append("Analyzed average sale price trends over time by neighborhood.")
        
        fig = px.line(price_over_time, x='sale_year', y='sale_price', color='neighborhood',
                      title="Average Sale Price Over Time by Neighborhood")
        visualizations["avg_sale_price_trend_by_neighborhood"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_listing_price_vs_final_sale_price_analysis(df):
    analysis_name = "Listing Price vs. Final Sale Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['list_price', 'final_price', 'beds', 'baths', 'living_area']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['list_price', 'final_price'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['list_price', 'final_price', 'beds', 'baths', 'living_area']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['list_price', 'final_price'], inplace=True)
        
        # Filter out outliers where list_price is 0
        df = df[df['list_price'] > 0]

        df['negotiation_diff'] = df['list_price'] - df['final_price']
        df['negotiation_perc'] = (df['negotiation_diff'] / df['list_price']) * 100
        
        avg_negotiation_perc = df['negotiation_perc'].mean()
        
        metrics = {
            "avg_negotiation_discount_percent": avg_negotiation_perc
        }
        
        insights.append(f"Average Negotiation Discount (from list price): {avg_negotiation_perc:.2f}%")
        
        fig1 = px.histogram(df, x='negotiation_perc', title="Distribution of Negotiation Percentage")
        visualizations["negotiation_percent_distribution"] = fig1.to_json()
        
        if 'living_area' in df.columns:
            fig2 = px.scatter(df, x='living_area', y='negotiation_perc',
                              title="Negotiation % vs. Living Area")
            visualizations["negotiation_vs_living_area"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def housing_market_analysis_by_postal_code(df):
    analysis_name = "Housing Market Analysis by Postal Code"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['postal_code', 'asking_price', 'closed_price', 'square_feet', 'date_listed']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['postal_code', 'closed_price', 'square_feet'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['asking_price', 'closed_price', 'square_feet']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['closed_price', 'square_feet'], inplace=True)
        
        # Filter out outliers
        df = df[df['square_feet'] > 0]
        
        df['price_per_sqft'] = df['closed_price'] / df['square_feet']
        df['postal_code'] = df['postal_code'].astype(str)
        
        summary = df.groupby('postal_code').agg(
            avg_price=('closed_price', 'mean'),
            avg_price_sqft=('price_per_sqft', 'mean'),
            num_listings=('postal_code', 'count')
        ).reset_index()
        
        metrics = {
            "market_summary_by_postal_code": summary.to_dict('records')
        }
        
        insights.append("Generated market summary by postal code.")
        
        # Show top 10 postal codes by number of listings
        top_10_summary = summary.nlargest(10, 'num_listings')

        fig = px.scatter(top_10_summary, x='avg_price', y='avg_price_sqft', size='num_listings',
                         color='postal_code', title="Avg. Price vs. Avg. Price/SqFt by Postal Code (Top 10 by Volume)")
        visualizations["postal_code_summary_scatter"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def residential_property_feature_and_price_analysis(df):
    analysis_name = "Residential Property Feature and Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['saleprice', 'bedcount', 'bathcount', 'floorarea', 'landarea', 'yearbuilt']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['saleprice', 'bedcount', 'floorarea'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['saleprice', 'bedcount', 'floorarea'], inplace=True)
        
        metrics = {
            "avg_saleprice": df['saleprice'].mean(),
            "avg_floorarea": df['floorarea'].mean(),
            "avg_bedcount": df['bedcount'].mean()
        }
        
        insights.append(f"Average Sale Price: ${metrics['avg_saleprice']:,.0f}")
        insights.append(f"Average Floor Area: {metrics['avg_floorarea']:,.0f} sqft")

        fig1 = px.scatter(df, x='floorarea', y='saleprice', 
                          color='bedcount',
                          title="Sale Price vs. Floor Area (Colored by Bedroom Count)")
        visualizations["price_vs_floorarea_by_beds"] = fig1.to_json()
        
        if 'landarea' in df.columns and 'yearbuilt' in df.columns:
            # Sample for 3D plot to avoid performance issues
            sample_df = df.sample(n=min(1000, len(df)))
            fig2 = px.scatter_3d(sample_df, x='floorarea', y='landarea', z='saleprice', color='yearbuilt',
                                  title="3D View: Price by Floor Area, Land Area, and Year Built (Sampled)")
            visualizations["price_3d_scatter"] = fig2.to_json()
            insights.append("Generated 3D scatter plot (on a sample of 1000 properties).")

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def county_level_real_estate_market_analysis(df):
    analysis_name = "County-Level Real Estate Market Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['county', 'price_usd', 'beds', 'baths', 'sqft', 'saledate']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['county', 'price_usd', 'sqft'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'saledate' in df.columns:
            df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        for col in ['price_usd', 'beds', 'baths', 'sqft']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['county', 'price_usd', 'sqft'], inplace=True)
        
        summary = df.groupby('county').agg(
            median_price=('price_usd', 'median'),
            avg_sqft=('sqft', 'mean'),
            num_sales=('price_usd', 'count')
        ).reset_index()
        
        metrics = {
            "market_summary_by_county": summary.to_dict('records')
        }
        
        insights.append("Generated market summary by county.")
        
        fig = px.bar(summary, x='county', y='median_price', color='avg_sqft',
                     title="Median Price by County (Colored by Average SqFt)")
        visualizations["median_price_by_county"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_sales_data_analysis_by_realtor(df):
    analysis_name = "Real Estate Sales Data Analysis by Realtor"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['realtor', 'saledate', 'saleprice', 'listing_status']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['realtor', 'saleprice'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
        df.dropna(subset=['realtor', 'saleprice'], inplace=True)
        
        summary = df.groupby('realtor').agg(
            total_sales_value=('saleprice', 'sum'),
            num_sales=('saleprice', 'count'),
            avg_sale_price=('saleprice', 'mean')
        ).nlargest(15, 'total_sales_value').reset_index()
        
        metrics = {
            "top_15_realtors_by_volume": summary.to_dict('records')
        }
        
        insights.append("Analyzed and ranked top 15 realtors by total sales volume.")
        if not summary.empty:
            top_realtor = summary.iloc[0]
            insights.append(f"Top realtor: {top_realtor['realtor']} with ${top_realtor['total_sales_value']:,.0f} in sales.")
        
        fig = px.bar(summary, x='realtor', y='total_sales_value', color='avg_sale_price',
                     title="Top Realtors by Total Sales Value (Colored by Average Sale Price)")
        visualizations["top_realtors_by_sales"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def web_scraped_real_estate_listing_analysis(df):
    analysis_name = "Web-Scraped Real Estate Listing Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['beds', 'baths', 'area', 'lotsize', 'yearbuilt', 'agentname']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['agentname', 'yearbuilt'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['beds', 'baths', 'area', 'lotsize', 'yearbuilt']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['agentname', 'yearbuilt'], inplace=True)
        
        top_agents = df['agentname'].value_counts().nlargest(15).reset_index()
        top_agents.columns = ['AgentName', 'ListingsCount']
        
        metrics = {
            "top_15_agents_by_listings": top_agents.to_dict('records'),
            "avg_year_built": df['yearbuilt'].mean()
        }
        
        insights.append(f"Average property construction year: {metrics['avg_year_built']:.0f}")
        if not top_agents.empty:
            top_agent = top_agents.iloc[0]
            insights.append(f"Top agent by listing count: {top_agent['AgentName']} ({top_agent['ListingsCount']} listings)")

        fig1 = px.bar(top_agents, x='AgentName', y='ListingsCount', title="Top 15 Agents by Number of Listings")
        visualizations["top_agents_by_listings"] = fig1.to_json()
        
        fig2 = px.histogram(df, x='yearbuilt', title="Distribution of Property Construction Year")
        visualizations["year_built_distribution"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_market_analysis_by_agency_and_zip_code(df):
    analysis_name = "Market Analysis by Agency and ZIP Code"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['zip_code', 'listing_price', 'sale_amount', 'agentcompany']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['zip_code', 'sale_amount', 'agentcompany'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['listing_price', 'sale_amount']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['zip_code', 'sale_amount', 'agentcompany'], inplace=True)
        
        summary = df.groupby(['agentcompany', 'zip_code'])['sale_amount'].agg(['sum', 'count']).reset_index()
        
        metrics = {
            "sales_summary_by_agency_zip": summary.to_dict('records')
        }
        
        insights.append("Generated treemap of sales volume by agency and ZIP code.")
        
        fig = px.treemap(summary, path=[px.Constant("All Agencies"), 'agentcompany', 'zip_code'],
                         values='sum', color='count',
                         title="Sales Volume by Agency and ZIP Code (Color by Number of Sales)")
        visualizations["sales_treemap_agency_zip"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_listing_and_sales_data_correlation(df):
    analysis_name = "Property Listing and Sales Data Correlation"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['listprice', 'soldprice', 'bedrooms', 'bathrooms', 'livingsqft', 'landsqft', 'yearbuilt']
        matched = fuzzy_match_column(df, expected)
        # Find all numeric columns that were successfully matched
        numeric_matched = [col for col in expected if matched.get(col) is not None and pd.api.types.is_numeric_dtype(df[matched.get(col)])]
        
        # Rename only the matched numeric columns for correlation
        rename_map = {matched[col]: col for col in numeric_matched}
        corr_df = df.rename(columns=rename_map)
        
        for col in numeric_matched:
            corr_df[col] = pd.to_numeric(corr_df[col], errors='coerce')
        corr_df = corr_df.dropna(subset=numeric_matched)

        if len(numeric_matched) < 2:
             return create_fallback_response(analysis_name, ['listprice', 'soldprice', 'livingsqft'], matched, df)
            
        corr_matrix = corr_df[numeric_matched].corr()
        
        metrics = {
            "correlation_matrix": corr_matrix.to_dict()
        }
        
        insights.append("Generated correlation matrix for numeric property features.")
        if 'soldprice' in numeric_matched and 'livingsqft' in numeric_matched:
            insights.append(f"Correlation(Sold Price, Living SqFt): {corr_matrix.loc['soldprice', 'livingsqft']:.2f}")
        
        fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Heatmap")
        visualizations["feature_correlation_heatmap"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def neighborhood_based_property_price_analysis(df):
    analysis_name = "Neighborhood-Based Property Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['neighborhood', 'listprice', 'saleprice', 'beds', 'baths', 'livingarea']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['neighborhood', 'saleprice'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['listprice', 'saleprice', 'beds', 'baths', 'livingarea']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['neighborhood', 'saleprice'], inplace=True)
        
        summary = df.groupby('neighborhood')['saleprice'].agg(['mean', 'median', 'count']).reset_index()
        summary = summary.sort_values('median', ascending=False)
        
        metrics = {
            "neighborhood_price_summary": summary.to_dict('records')
        }
        
        insights.append("Generated summary of sales price by neighborhood.")
        if not summary.empty:
            top_hood = summary.iloc[0]
            insights.append(f"Most expensive neighborhood (median): {top_hood['neighborhood']} (${top_hood['median']:,.0f})")

        fig = px.bar(summary.head(20), # Show top 20
                     x='neighborhood', y='median', color='count',
                     title="Median Sale Price by Neighborhood (Top 20, Colored by Number of Sales)")
        visualizations["median_price_by_neighborhood"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_market_dynamics_analysis(df):
    analysis_name = "Real Estate Market Dynamics Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['county', 'zip', 'list_price', 'sold_price', 'daysonmarket']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['county', 'list_price', 'sold_price', 'daysonmarket'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['list_price', 'sold_price', 'daysonmarket']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['county', 'list_price', 'sold_price', 'daysonmarket'], inplace=True)
        
        # Filter out 0 list prices
        df = df[df['list_price'] > 0]
        
        df['sale_to_list_ratio'] = df['sold_price'] / df['list_price']
        
        summary = df.groupby('county').agg(
            avg_dom=('daysonmarket', 'mean'),
            avg_ratio=('sale_to_list_ratio', 'mean'),
            median_price=('sold_price', 'median')
        ).reset_index()
        
        metrics = {
            "market_dynamics_by_county": summary.to_dict('records')
        }
        
        insights.append("Generated market dynamics summary by county.")
        
        fig = px.scatter(summary, x='avg_dom', y='avg_ratio', size='median_price',
                         color='county', title="Market Dynamics by County",
                         labels={'avg_dom': 'Avg. Days on Market', 'avg_ratio': 'Avg. Sale/List Ratio'})
        visualizations["market_dynamics_scatter"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_sales_price_vs_list_price_analysis(df):
    analysis_name = "Property Sales Price vs. List Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['pricesold', 'pricelist', 'beds', 'baths']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['pricesold', 'pricelist'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['pricesold', 'pricelist'], inplace=True)
        
        # Filter out 0 list prices
        df = df[df['pricelist'] > 0]
        
        df['ratio'] = df['pricesold'] / df['pricelist']
        
        metrics = {
            "avg_sale_to_list_ratio": df['ratio'].mean(),
            "median_sale_to_list_ratio": df['ratio'].median()
        }
        
        insights.append(f"Average Sale-to-List Ratio: {metrics['avg_sale_to_list_ratio']:.3f}")
        insights.append(f"Median Sale-to-List Ratio: {metrics['median_sale_to_list_ratio']:.3f}")
        
        fig = px.histogram(df, x='ratio', title="Distribution of Sale-to-List Price Ratios (1.0 = Sold at List)")
        visualizations["sale_to_list_ratio_distribution"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def zip_code_level_housing_market_trend_analysis(df):
    analysis_name = "ZIP Code-Level Housing Market Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['zip', 'beds', 'baths', 'livingspace', 'saledate', 'saleprice']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['zip', 'saledate', 'saleprice'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
        df['zip'] = df['zip'].astype(str)
        df.dropna(subset=['saledate', 'saleprice', 'zip'], inplace=True)
        
        df['sale_year_month'] = df['saledate'].dt.to_period('M').astype(str)
        summary = df.groupby(['sale_year_month', 'zip'])['saleprice'].median().reset_index()
        
        # Non-interactive selection: just show for the top N ZIPs by sales volume
        top_zips = df['zip'].value_counts().nlargest(3).index.tolist()
        filtered_df = summary[summary['zip'].isin(top_zips)]

        metrics = {
            "trend_data_top_zips": filtered_df.to_dict('records'),
            "top_zips_analyzed": top_zips
        }
        
        insights.append(f"Showing median sale price trend for top {len(top_zips)} ZIP codes by sales volume.")
        
        fig = px.line(filtered_df, x='sale_year_month', y='saleprice', color='zip',
                      title="Median Sale Price Trend by ZIP Code (Top 3 by Volume)")
        visualizations["median_price_trend_by_zip"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def county_level_housing_price_and_feature_analysis(df):
    analysis_name = "County-Level Housing Price and Feature Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['county', 'zip', 'listprice', 'closedprice', 'bedrooms', 'bathrooms', 'sqft']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['county', 'closedprice', 'sqft', 'bedrooms'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['listprice', 'closedprice', 'bedrooms', 'bathrooms', 'sqft']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['county', 'closedprice', 'sqft', 'bedrooms'], inplace=True)
        
        summary = df.groupby('county').agg(
            median_price=('closedprice', 'median'),
            avg_sqft=('sqft', 'mean'),
            avg_beds=('bedrooms', 'mean')
        ).reset_index()
        
        metrics = {
            "county_housing_summary": summary.to_dict('records')
        }
        
        insights.append("Generated county-level housing summary (price, sqft, beds).")
        
        fig = px.scatter(summary, x='avg_sqft', y='median_price', size='avg_beds',
                         color='county', title="Median Price vs. Avg. SqFt by County (Sized by Avg. Bedrooms)")
        visualizations["county_summary_scatter"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_listing_duration_and_price_analysis(df):
    analysis_name = "Listing Duration and Price Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['zipcode', 'askingprice', 'finalsaleprice', 'dayslisted']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['askingprice', 'finalsaleprice', 'dayslisted'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['askingprice', 'finalsaleprice', 'dayslisted']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['askingprice', 'finalsaleprice', 'dayslisted'], inplace=True)
        
        # Filter out 0 prices
        df = df[df['askingprice'] > 0]
        
        df['price_diff_perc'] = (df['askingprice'] - df['finalsaleprice']) / df['askingprice'] * 100
        
        metrics = {
            "avg_price_reduction_percent": df['price_diff_perc'].mean(),
            "avg_days_listed": df['dayslisted'].mean(),
            "correlation_days_vs_reduction": df['dayslisted'].corr(df['price_diff_perc'])
        }
        
        insights.append(f"Average Price Reduction Percentage: {metrics['avg_price_reduction_percent']:.2f}%")
        insights.append(f"Average Days Listed: {metrics['avg_days_listed']:.1f}")
        insights.append(f"Correlation (Days Listed vs. Price Reduction): {metrics['correlation_days_vs_reduction']:.2f}")
        
        fig = px.scatter(df, x='dayslisted', y='price_diff_perc',
                         title="Price Reduction % vs. Days Listed",
                         labels={'dayslisted': 'Days Listed', 'price_diff_perc': 'Price Reduction (%)'})
        visualizations["reduction_vs_days_listed"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def agency_performance_in_real_estate_sales(df):
    analysis_name = "Agency Performance in Real Estate Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['agency', 'pricelist', 'pricesold']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['agency', 'pricelist', 'pricesold'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['pricelist', 'pricesold']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['agency', 'pricelist', 'pricesold'], inplace=True)
        
        df = df[df['pricelist'] > 0] # Avoid division by zero
        
        df['sale_to_list_ratio'] = df['pricesold'] / df['pricelist']
        
        summary = df.groupby('agency').agg(
            total_volume=('pricesold', 'sum'),
            num_sales=('pricesold', 'count'),
            avg_ratio=('sale_to_list_ratio', 'mean')
        ).nlargest(15, 'total_volume').reset_index()
        
        metrics = {
            "top_15_agencies_by_volume": summary.to_dict('records')
        }
        
        insights.append("Analyzed and ranked top 15 agencies by total sales volume.")
        if not summary.empty:
            top_agency = summary.iloc[0]
            insights.append(f"Top agency: {top_agency['agency']} with ${top_agency['total_volume']:,.0f} in sales.")
        
        fig = px.bar(summary, x='agency', y='total_volume', color='avg_ratio',
                     title="Top 15 Agencies by Sales Volume (Colored by Average Sale-to-List Ratio)")
        visualizations["top_agencies_by_volume"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_transaction_and_status_analysis(df):
    analysis_name = "Real Estate Transaction and Status Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['addr', 'zip', 'listprice', 'saleprice', 'mls_status', 'closingdate']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['mls_status'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        status_counts = df['mls_status'].value_counts().reset_index()
        status_counts.columns = ['MLS_Status', 'Count']
        
        metrics = {
            "mls_status_distribution": status_counts.to_dict('records')
        }
        
        insights.append("Generated distribution of MLS statuses.")
        
        fig = px.pie(status_counts, names='MLS_Status', values='Count', title="Distribution of MLS Statuses")
        visualizations["mls_status_distribution_pie"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def property_sales_data_analysis_by_location(df):
    analysis_name = "Property Sales Data Analysis by Location"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['fulladdress', 'zipcode', 'listingprice', 'soldprice', 'squarefeet']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['zipcode', 'soldprice', 'squarefeet'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['listingprice', 'soldprice', 'squarefeet']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['zipcode', 'soldprice', 'squarefeet'], inplace=True)
        
        df = df[df['squarefeet'] > 0] # Avoid division by zero
        
        df['price_per_sqft'] = df['soldprice'] / df['squarefeet']
        
        summary = df.groupby('zipcode')['price_per_sqft'].median().reset_index()
        summary = summary.sort_values('price_per_sqft', ascending=False)
        
        metrics = {
            "median_price_per_sqft_by_zip": summary.to_dict('records')
        }
        
        insights.append("Generated summary of median price per square foot by ZIP code.")
        if not summary.empty:
            top_zip = summary.iloc[0]
            insights.append(f"Highest median price/sqft: ${top_zip['price_per_sqft']:.2f} in ZIP {top_zip['zipcode']}")
        
        fig = px.bar(summary.head(20), # Showing top 20
                     x='zipcode', y='price_per_sqft', title="Median Price per Square Foot by ZIP Code (Top 20)")
        visualizations["median_price_per_sqft_by_zip"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def neighborhood_specific_real_estate_market_analysis(df):
    analysis_name = "Neighborhood-Specific Real Estate Market Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['neighborhood', 'list_price', 'sale_price', 'beds', 'baths', 'livingarea']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['neighborhood', 'sale_price', 'livingarea', 'beds'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['list_price', 'sale_price', 'beds', 'baths', 'livingarea']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['neighborhood', 'sale_price', 'livingarea', 'beds'], inplace=True)
        
        # Non-interactive: just pick the most common neighborhood
        if df['neighborhood'].empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No neighborhood data found.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No neighborhood data found."]
            }
            
        target_neighborhood = df['neighborhood'].mode()[0]
        df_hood = df[df['neighborhood'] == target_neighborhood]
        
        metrics = {
            "analyzed_neighborhood": target_neighborhood,
            "median_sale_price": df_hood['sale_price'].median(),
            "avg_living_area_sqft": df_hood['livingarea'].mean(),
            "num_sales_in_neighborhood": len(df_hood)
        }
        
        insights.append(f"Analyzed market for most common neighborhood: {target_neighborhood}")
        insights.append(f"Median Sale Price: ${metrics['median_sale_price']:,.0f}")
        insights.append(f"Average Living Area: {metrics['avg_living_area_sqft']:,.0f} sqft")
        
        fig = px.scatter(df_hood, x='livingarea', y='sale_price', color='beds',
                         title=f"Sale Price vs. Living Area in {target_neighborhood} (Colored by Bedrooms)")
        visualizations[f"price_vs_area_in_{target_neighborhood}"] = fig.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_estate_market_time_series_analysis(df):
    analysis_name = "Real Estate Market Time-Series Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['county', 'zipcode', 'listprice', 'saleprice', 'saledate']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['saledate', 'saleprice'] if matched.get(col) is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
        df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
        df.dropna(subset=['saledate', 'saleprice'], inplace=True)
        
        df['sale_month'] = df['saledate'].dt.to_period('M').astype(str)
        
        summary = df.groupby('sale_month').agg(
            median_price=('saleprice', 'median'),
            num_sales=('saleprice', 'count')
        ).reset_index()
        
        metrics = {
            "market_summary_over_time": summary.to_dict('records')
        }
        
        insights.append("Generated time-series summary of median price and sales volume.")
        
        fig1 = px.line(summary, x='sale_month', y='median_price', title="Median Sale Price Over Time")
        visualizations["median_price_over_time"] = fig1.to_json()
        
        fig2 = px.bar(summary, x='sale_month', y='num_sales', title="Number of Sales Over Time")
        visualizations["sales_volume_over_time"] = fig2.to_json()

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

# ========== MAIN EXECUTION LOGIC ==========

# Map of all analysis functions
analysis_functions_map = {
    "property_analysis": property_analysis,
    "location_analysis": location_analysis,
    "price_trends_analysis": price_trends_analysis,
    "rental_analysis": rental_analysis,
    "investment_analysis": investment_analysis,
    "market_comparison": market_comparison_analysis,
    "affordability_analysis": affordability_analysis,
    "boston_housing_price_prediction_analysis": boston_housing_price_prediction_analysis,
    "real_estate_listing_description_and_time-on-market_analysis": real_estate_listing_description_and_time_on_market_analysis,
    "property_valuation_(zestimate)_and_feature_analysis": property_valuation_zestimate_and_feature_analysis,
    "real_estate_transaction_analysis_by_area_and_furnishing": real_estate_transaction_analysis_by_area_and_furnishing,
    "real_estate_price_prediction_based_on_location_and_features": real_estate_price_prediction_based_on_location_and_features,
    "property_sales_and_assessment_ratio_analysis": property_sales_and_assessment_ratio_analysis,
    "neighborhood_property_characteristics_analysis": neighborhood_property_characteristics_analysis,
    "house_price_prediction_based_on_property_features": house_price_prediction_based_on_property_features,
    "property_sales_and_appraisal_data_analysis": property_sales_and_appraisal_data_analysis,
    "real_estate_pricing_and_feature_analysis": real_estate_pricing_and_feature_analysis,
    "property_listing_time-on-market_analysis": property_listing_time_on_market_analysis,
    "real_estate_sales_price_analysis": real_estate_sales_price_analysis,
    "neighborhood_property_sales_trend_analysis": neighborhood_property_sales_trend_analysis,
    "property_listing_price_vs._final_sale_price_analysis": property_listing_price_vs_final_sale_price_analysis,
    "housing_market_analysis_by_postal_code": housing_market_analysis_by_postal_code,
    "residential_property_feature_and_price_analysis": residential_property_feature_and_price_analysis,
    "county-level_real_estate_market_analysis": county_level_real_estate_market_analysis,
    "real_estate_sales_data_analysis_by_realtor": real_estate_sales_data_analysis_by_realtor,
    "web-scraped_real_estate_listing_analysis": web_scraped_real_estate_listing_analysis,
    "real_estate_market_analysis_by_agency_and_zip_code": real_estate_market_analysis_by_agency_and_zip_code,
    "property_listing_and_sales_data_correlation": property_listing_and_sales_data_correlation,
    "neighborhood-based_property_price_analysis": neighborhood_based_property_price_analysis,
    "real_estate_market_dynamics_analysis": real_estate_market_dynamics_analysis,
    "property_sales_price_vs._list_price_analysis": property_sales_price_vs_list_price_analysis,
    "zip_code-level_housing_market_trend_analysis": zip_code_level_housing_market_trend_analysis,
    "county-level_housing_price_and_feature_analysis": county_level_housing_price_and_feature_analysis,
    "real_estate_listing_duration_and_price_analysis": real_estate_listing_duration_and_price_analysis,
    "agency_performance_in_real_estate_sales": agency_performance_in_real_estate_sales,
    "real_estate_transaction_and_status_analysis": real_estate_transaction_and_status_analysis,
    "property_sales_data_analysis_by_location": property_sales_data_analysis_by_location,
    "neighborhood-specific_real_estate_market_analysis": neighborhood_specific_real_estate_market_analysis,
    "real_estate_market_time-series_analysis": real_estate_market_time_series_analysis,
    "general_insights": show_general_insights # Add general insights to the map
}
analysis_options_list = sorted(list(analysis_functions_map.keys()))


def main_backend(file_path, analysis_name, encoding='utf-8'):
    """
    Main function to be called by an API.
    
    Parameters:
    - file_path: path to the data file (CSV or Excel)
    - analysis_name: The string key of the analysis to run (e.g., "property_analysis")
    - encoding: file encoding (default: 'utf-8')
    
    Returns:
    - Dictionary with analysis results
    """
    
    # Load data
    df = load_data(file_path, encoding)
    if df is None:
        return {
            "analysis_type": "Data Loading",
            "status": "error",    
            "error_message": f"Failed to load data file from {file_path}"
        }
    
    # Find and run the selected analysis
    if analysis_name in analysis_functions_map:
        selected_function = analysis_functions_map[analysis_name]
        try:
            result = selected_function(df.copy())
            return result
        except Exception as e:
            return {
                "analysis_type": analysis_name,
                "status": "error",
                "error_message": f"An unexpected error occurred during analysis: {str(e)}",
            }
    else:
        return {
            "analysis_type": "Selection Error",
            "status": "error",
            "error_message": f"Analysis '{analysis_name}' not found."
        }


# Main execution logic for the console application (for testing)
def main():
    print("🏠 Real Estate Analytics Console Application (Testing Mode)")
    
    file_path = input("Enter path to your real estate data file (e.g., data.csv or data.xlsx): ")
    encoding = input("Enter file encoding (e.g., utf-8, latin1), or press Enter for 'utf-8': ")
    if not encoding:
        encoding = 'utf-8'
    
    # Test data loading immediately
    df_test = load_data(file_path, encoding=encoding)
    if df_test is None:
        print("Could not load data. Please check the file path and encoding. Exiting.")
        return
    print(f"Data loaded successfully! Found {len(df_test)} rows and {len(df_test.columns)} columns.")
    del df_test # Clear memory

    while True:
        print("\n--- Available Analyses ---")
        for i, option_key in enumerate(analysis_options_list):
            print(f"{i + 1}. {option_key.replace('_', ' ').title()}")
        print("0. Exit")

        choice_str = input("Enter the number corresponding to your desired analysis: ")
        
        if choice_str == '0':
            print("Exiting application. Goodbye!")
            break

        try:
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(analysis_options_list):
                selected_analysis_name = analysis_options_list[choice_idx]
                
                print(f"\n--- Running: {selected_analysis_name.replace('_', ' ').title()} ---")
                
                # Call the main_backend function
                result = main_backend(file_path, selected_analysis_name, encoding)
                
                # Pretty-print the JSON-like result
                print("\n--- API Result ---")
                print(f"Status: {result.get('status')}")
                print(f"Analysis Type: {result.get('analysis_type')}")
                
                if result.get('status') == 'error':
                    print(f"Error: {result.get('error_message')}")
                
                if result.get('status') == 'fallback':
                    print(f"Message: {result.get('message')}")
                    print(f"Missing Columns: {result.get('missing_columns')}")

                print("\nInsights:")
                for insight in result.get('insights', []):
                    print(f"- {insight}")
                    
                print("\nMetrics:")
                # Use json.dumps for pretty printing the metrics dict
                metrics_json = json.dumps(result.get('metrics', {}), indent=2)
                print(metrics_json)
                
                print(f"\nVisualizations Created: {len(result.get('visualizations', {}))}")
                for viz_name in result.get('visualizations', {}).keys():
                    print(f"- {viz_name}")
                print("--- End of Result ---")

            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()