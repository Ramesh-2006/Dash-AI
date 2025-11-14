import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "sales_analysis",
    "customer_analysis",
    "inventory_analysis",
    "product_analysis",
    "store_analysis",
    "promotion_analysis",
    "basket_analysis",
    "seasonal_analysis",
    "customer_purchase_behavior_and_rfm_analysis",
    "retail_transaction_analysis_by_product_and_country",
    "retail_order_status_and_item_analysis",
    "regional_sales_and_customer_analysis",
    "sales_channel_performance_analysis",
    "international_sales_and_transaction_analysis",
    "invoice_type_and_customer_purchase_pattern_analysis",
    "order_delivery_and_customer_location_analysis",
    "time_of_day_sales_pattern_analysis",
    "customer_order_and_status_tracking_analysis",
    "payment_method_preference_analysis",
    "product_return_rate_analysis",
    "promotional_code_effectiveness_analysis",
    "discount_impact_on_sales_analysis",
    "product_cost_and_sales_price_margin_analysis",
    "store_level_sales_performance_analysis",
    "product_category_sales_analysis",
    "sales_performance_by_channel_type",
    "customer_order_value_analysis",
    "weekly_sales_trend_analysis",
    "yearly_sales_performance_analysis",
    "monthly_sales_trend_analysis",
    "week_over_week_sales_growth_analysis",
    "holiday_sales_impact_analysis",
    "customer_type_segmentation_and_sales_analysis",
    "online_vs_offline_sales_analysis",
    "international_retail_transaction_analysis",
    "regional_customer_purchase_analysis",
    "delivery_method_preference_analysis",
    "point_of_sale_transaction_analysis",
    "sales_tax_and_revenue_analysis",
    "sales_organization_performance_analysis",
    "order_payment_status_analysis",
    "product_sales_and_cost_analysis",
    "customer_transaction_history_analysis",
    "customer_segment_based_purchasing_behavior",
    "per_unit_price_and_sales_analysis",
    "promotion_id_impact_on_sales",
    "store_location_sales_performance_analysis",
    "sales_representative_performance_analysis",
    "regional_sales_and_product_analysis",
    "multi_channel_sales_performance_analysis",
    "sales_classification_analysis",
    "retail_receipt_data_analysis",
    "invoice_line_item_total_analysis",
    "product_category_sales_performance",
    "transactional_promotion_effectiveness_analysis",
    "order_status_and_item_details_analysis",
    "sales_source_attribution_analysis",
    "customer_regional_sales_analysis",
]

# ========== UTILITY FUNCTIONS ==========

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
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception:
        # Fallback for complex unhandled types
        return str(data)

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
    visualizations = {}
    metrics = {}
    insights = []
    
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
        other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
        
        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / total_rows) * 100 if total_rows > 0 else 0
        columns_with_missing = missing_values[missing_values > 0]
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Compile metrics
        metrics = {
            "dataset_overview": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "memory_usage_mb": float(memory_usage_mb),
                "duplicate_rows": int(duplicate_rows),
                "duplicate_percentage": float(duplicate_percentage)
            },
            "data_types_breakdown": {
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "other_columns": len(other_cols)
            },
            "data_quality": {
                "total_missing_values": int(missing_values.sum()),
                "columns_with_missing": len(columns_with_missing),
                "complete_columns": len(df.columns) - len(columns_with_missing)
            }
        }
        
        # Generate insights
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns",
        ]

        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append(f"The following columns are needed for the requested '{analysis_name}' but weren't found:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"  - {col}{match_info}")
            insights.append("")
            insights.append("Showing General Analysis instead of the requested specific analysis.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data)")
        else:
            insights.append("No duplicate rows found")
        
        if len(columns_with_missing) > 0:
            insights.append(f"{len(columns_with_missing)} columns have missing values")
        else:
            insights.append("No missing values found in the dataset")

        # Create visualizations
        # 1. Data types distribution
        try:
            dtype_counts = {
                'Numeric': len(numeric_cols),
                'Categorical': len(categorical_cols),
                'Datetime': len(datetime_cols),
                'Other': len(other_cols)
            }
            fig_dtypes = px.pie(
                values=list(dtype_counts.values()), 
                names=list(dtype_counts.keys()),
                title='Data Types Distribution'
            )
            visualizations["data_types_distribution"] = fig_dtypes.to_json()
        except Exception:
            pass
            
        # 2. Missing values visualization
        try:
            if len(columns_with_missing) > 0:
                missing_df = pd.DataFrame({
                    'column': columns_with_missing.index,
                    'missing_count': columns_with_missing.values,
                    'missing_percentage': missing_percentage[columns_with_missing.index]
                }).sort_values('missing_percentage', ascending=False)
                
                fig_missing = px.bar(
                    missing_df.head(10), 
                    x='column', 
                    y='missing_percentage',
                    title='Top 10 Columns with Missing Values (%)'
                )
                visualizations["missing_values"] = fig_missing.to_json()
            else:
                insights.append("No missing values to visualize.")
        except Exception:
            pass

        # 3. Numeric columns distributions (first 2)
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:2]):
                try:
                    fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception:
                    pass
        
        # 4. Categorical columns distributions (first 2)
        if categorical_cols:
            for i, col in enumerate(categorical_cols[:2]):
                try:
                    top_10_cats = df[col].value_counts().nlargest(10).reset_index()
                    top_10_cats.columns = [col, 'count']
                    fig_bar = px.bar(top_10_cats, x=col, y='count', title=f'Top 10 Categories for {col}')
                    visualizations[f"{col}_distribution"] = fig_bar.to_json()
                except Exception:
                    pass

        insights.append(f"Generated {len(visualizations)} visualizations for data exploration")

        return {
            "analysis_type": analysis_type,
            "status": "success",
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
            "error_message": str(e),
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights,
            "missing_columns": missing_cols or []
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides general insights as a fallback.
    """
    print(f"--- ⚠️ Required Columns Not Found for {analysis_name} ---")
    print(f"Missing: {missing_cols}")
    print("Falling back to General Insights.")
    
    try:
        general_insights_data = show_general_insights(
            df, 
            analysis_name, # Pass the original name for context
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
    except Exception as fallback_error:
        print(f"General insights fallback also failed: {fallback_error}")
        general_insights_data = {
            "visualizations": {},
            "metrics": {},
            "insights": ["Fallback to general insights failed.", str(fallback_error)]
        }

    missing_info = {}
    for col in missing_cols:
        match_info = f" (fuzzy matched to: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (No match found)"
        missing_info[col] = match_info

    return {
        "analysis_type": analysis_name,
        "status": "fallback",
        "message": f"Required columns were missing for '{analysis_name}'. Falling back to general insights.",
        "missing_columns_info": missing_info,
        "matched_columns": matched_cols,
        "visualizations": general_insights_data.get("visualizations", {}),
        "metrics": general_insights_data.get("metrics", {}),
        "insights": general_insights_data.get("insights", []),
        "missing_columns": missing_cols
    }

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
            match, score = process.extractOne(target, available)
            matched[target] = match if score >= 60 else None
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    rename_map = {v: k for k, v in matched.items() if v is not None and v != k}
    return df.rename(columns=rename_map)

# ========== DATA LOADING ==========
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
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

# ========== SALES ANALYSIS FUNCTIONS ==========

def sales_analysis(df):
    analysis_name = "Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'date', 'product_id', 'quantity',
                    'unit_price', 'total_amount', 'store_id', 'customer_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        for col in ['quantity', 'unit_price', 'total_amount']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Recalculate total_amount if not present but quantity and unit_price are
        if 'total_amount' not in df.columns and 'quantity' in df.columns and 'unit_price' in df.columns:
            df['total_amount'] = df['quantity'] * df['unit_price']
            insights.append("Calculated 'total_amount' from 'quantity' * 'unit_price'.")
            
        df.dropna(subset=['total_amount'], inplace=True)

        total_sales = df['total_amount'].sum()
        avg_transaction = df['total_amount'].mean()
        unique_customers = df['customer_id'].nunique() if 'customer_id' in df else "N/A"
        
        metrics = {
            "total_sales": total_sales,
            "avg_transaction": avg_transaction,
            "unique_customers": unique_customers,
            "total_transactions": len(df)
        }
        
        insights.append(f"Total sales amounted to ${total_sales:,.2f} across {len(df)} transactions.")
        insights.append(f"The average transaction value was ${avg_transaction:,.2f}.")
        if unique_customers != "N/A":
            insights.append(f"There were {unique_customers} unique customers.")

        if 'date' in df and 'total_amount' in df and not df['date'].isnull().all():
            try:
                sales_over_time = df.set_index('date').resample('M')['total_amount'].sum().reset_index()
                fig_daily_sales = px.line(sales_over_time, x='date', y='total_amount', title="Monthly Sales Trend")
                visualizations['monthly_sales'] = fig_daily_sales.to_json()
            except Exception as e:
                insights.append(f"Could not generate time series plot: {e}")
                
        if 'product_id' in df and 'total_amount' in df:
            top_products = df.groupby('product_id')['total_amount'].sum().nlargest(10).reset_index()
            fig_top_products = px.bar(top_products, x='product_id', y='total_amount', title="Top 10 Products by Revenue")
            visualizations['top_products'] = fig_top_products.to_json()
            
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

def customer_analysis(df):
    analysis_name = "Customer Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'first_purchase_date', 'total_purchases',
                    'total_spend', 'segment', 'region']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['customer_id', 'total_spend'] if matched[col] is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'first_purchase_date' in df and not pd.api.types.is_datetime64_any_dtype(df['first_purchase_date']):
            df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'], errors='coerce')
            
        df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce')
        df.dropna(subset=['total_spend'], inplace=True)

        total_customers = len(df)
        avg_spend = df['total_spend'].mean()
        top_segment = df['segment'].mode()[0] if 'segment' in df and not df['segment'].empty else "N/A"
        
        metrics = {
            "total_customers": total_customers,
            "avg_spend_per_customer": avg_spend,
            "top_segment": top_segment
        }
        
        insights.append(f"Analyzed {total_customers} unique customers.")
        insights.append(f"Average total spend per customer: ${avg_spend:,.2f}")
        if top_segment != "N/A":
            insights.append(f"The most common customer segment is '{top_segment}'.")

        if 'segment' in df:
            segment_dist = df['segment'].value_counts().reset_index()
            fig_segment_pie = px.pie(segment_dist, names='segment', values='count',
                                     title="Customer Segment Distribution")
            visualizations['segment_pie'] = fig_segment_pie.to_json()
            
        if all(col in df for col in ['recency', 'frequency', 'monetary_value']):
            insights.append("RFM columns found, generating 3D plot.")
            fig_rfm = px.scatter_3d(df, x='recency', y='frequency', z='monetary_value',
                                  color='segment' if 'segment' in df else None, title="RFM Segmentation")
            visualizations['rfm'] = fig_rfm.to_json()
            
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

def inventory_analysis(df):
    analysis_name = "Inventory Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'product_name', 'category', 'current_stock', 'reorder_level', 'lead_time', 'supplier']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'current_stock'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['current_stock', 'reorder_level', 'lead_time']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['current_stock'], inplace=True)
        
        low_stock = 0
        if 'current_stock' in df and 'reorder_level' in df and not df['reorder_level'].isnull().all():
            df['stock_status'] = np.where(df['current_stock'] < df['reorder_level'], 'Reorder Needed', 'Adequate Stock')
            low_stock = (df['stock_status'] == 'Reorder Needed').sum()
            insights.append(f"{low_stock} products need to be reordered.")
        else:
            insights.append("Cannot determine stock status: 'reorder_level' column not found or is empty.")
            
        total_products = len(df)
        avg_lead_time = df['lead_time'].mean() if 'lead_time' in df else "N/A"
        
        metrics = {
            "total_products": total_products, 
            "products_needing_reorder": low_stock, 
            "avg_lead_time_days": avg_lead_time
        }
        
        insights.insert(0, f"Analyzed {total_products} product inventory records.")

        if 'stock_status' in df:
            fig_inv_status = px.pie(df, names='stock_status', title="Inventory Status Distribution")
            visualizations['inventory_status'] = fig_inv_status.to_json()
            
        if 'category' in df and 'current_stock' in df:
            fig_cat_inv = px.treemap(df, path=['category'], values='current_stock', title="Inventory Distribution by Category")
            visualizations['category_inventory'] = fig_cat_inv.to_json()
            
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

def product_analysis(df):
    analysis_name = "Product Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'product_name', 'category', 'subcategory', 'price', 'cost', 'margin', 'rating']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'price'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['price', 'cost', 'margin', 'rating']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'margin' not in df and 'price' in df and 'cost' in df:
            df['margin'] = df['price'] - df['cost']
            insights.append("Calculated 'margin' from 'price' - 'cost'.")
            
        df.dropna(subset=['price'], inplace=True)

        total_products = len(df)
        avg_margin = df['margin'].mean() if 'margin' in df else "N/A"
        avg_rating = df['rating'].mean() if 'rating' in df else "N/A"
        
        metrics = {
            "total_products": total_products, 
            "avg_margin": avg_margin, 
            "avg_rating": avg_rating,
            "avg_price": df['price'].mean()
        }
        
        insights.append(f"Analyzed {total_products} products with an average price of ${metrics['avg_price']:,.2f}.")
        if avg_margin != "N/A":
            insights.append(f"Average margin: ${avg_margin:,.2f}.")
        if avg_rating != "N/A":
            insights.append(f"Average rating: {avg_rating:.2f}.")

        if 'price' in df:
            fig_price_dist = px.histogram(df, x='price', title="Price Distribution")
            visualizations['price_distribution'] = fig_price_dist.to_json()
            
        if 'category' in df and 'margin' in df:
            fig_margin_cat = px.box(df, x='category', y='margin', title="Margin Distribution by Category")
            visualizations['margin_by_category'] = fig_margin_cat.to_json()
            
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

def store_analysis(df):
    analysis_name = "Store Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['store_id', 'location', 'size', 'manager', 'opening_date', 'monthly_sales', 'monthly_traffic']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['store_id', 'location', 'monthly_sales'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'opening_date' in df and not pd.api.types.is_datetime64_any_dtype(df['opening_date']):
            df['opening_date'] = pd.to_datetime(df['opening_date'], errors='coerce')
        
        for col in ['size', 'monthly_sales', 'monthly_traffic']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['monthly_sales'], inplace=True)

        total_stores = len(df)
        avg_sales = df['monthly_sales'].mean()
        sales_per_sqft = (df['monthly_sales'] / df['size']).mean() if 'size' in df and not df['size'].isnull().all() else "N/A"
        
        metrics = {
            "total_stores": total_stores, 
            "avg_monthly_sales": avg_sales, 
            "sales_per_sqft": sales_per_sqft
        }
        
        insights.append(f"Analyzed {total_stores} stores with average monthly sales of ${avg_sales:,.2f}.")
        if sales_per_sqft != "N/A":
            insights.append(f"Average sales per square foot: ${sales_per_sqft:,.2f}.")

        if 'location' in df and 'monthly_sales' in df:
            fig_sales_loc = px.bar(df.sort_values('monthly_sales', ascending=False), x='location', y='monthly_sales', title="Store Sales by Location")
            visualizations['sales_by_location'] = fig_sales_loc.to_json()
            
        if 'monthly_sales' in df and 'monthly_traffic' in df:
            fig_sales_traffic = px.scatter(df, x='monthly_traffic', y='monthly_sales', trendline="ols", title="Sales vs Customer Traffic")
            visualizations['sales_vs_traffic'] = fig_sales_traffic.to_json()
            
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

def promotion_analysis(df):
    analysis_name = "Promotion Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['promotion_id', 'start_date', 'end_date', 'discount_pct', 'products_included', 'sales_increase', 'roi', 'promotion_type']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['promotion_id', 'discount_pct', 'roi'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        date_cols = ['start_date', 'end_date']
        for col in date_cols:
            if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        for col in ['discount_pct', 'sales_increase', 'roi']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['discount_pct', 'roi'], inplace=True)
                
        total_promotions = len(df)
        avg_discount = df['discount_pct'].mean()
        avg_roi = df['roi'].mean()
        
        metrics = {
            "total_promotions": total_promotions, 
            "avg_discount_percent": avg_discount, 
            "avg_roi_percent": avg_roi * 100 if avg_roi < 10 else avg_roi # Assume ROI is decimal or percent
        }
        
        insights.append(f"Analyzed {total_promotions} promotions.")
        insights.append(f"Average discount: {avg_discount:.2f}%.")
        insights.append(f"Average ROI: {metrics['avg_roi_percent']:.2f}%.")

        if 'discount_pct' in df and 'sales_increase' in df:
            fig_disc_sales = px.scatter(df, x='discount_pct', y='sales_increase', trendline="ols", title="Discount % vs Sales Increase")
            visualizations['discount_vs_sales_increase'] = fig_disc_sales.to_json()
            
        if 'promotion_type' in df and 'roi' in df:
            fig_roi_type = px.box(df, x='promotion_type', y='roi', title="ROI by Promotion Type")
            visualizations['roi_by_type'] = fig_roi_type.to_json()
            
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

def basket_analysis(df):
    analysis_name = "Basket Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'product_id', 'product_name', 'category', 'quantity', 'unit_price']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['transaction_id', 'product_id', 'quantity', 'unit_price'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['quantity', 'unit_price']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['line_total'] = df['quantity'] * df['unit_price']
        df.dropna(subset=['transaction_id', 'product_id', 'line_total'], inplace=True)

        total_transactions = df['transaction_id'].nunique()
        avg_items_per_transaction = df.groupby('transaction_id')['product_id'].count().mean()
        avg_basket_value = df.groupby('transaction_id')['line_total'].sum().mean()
        
        metrics = {
            "total_transactions": total_transactions, 
            "avg_items_per_basket": avg_items_per_transaction, 
            "avg_basket_value": avg_basket_value
        }
        
        insights.append(f"Analyzed {total_transactions} unique transactions.")
        insights.append(f"Average basket value: ${avg_basket_value:,.2f}.")
        insights.append(f"Average items per basket: {avg_items_per_transaction:.2f}.")

        if 'category' in df:
            category_sales = df.groupby('category')['line_total'].sum().reset_index()
            fig_cat_sales = px.pie(category_sales, names='category', values='line_total', title="Sales Distribution by Category")
            visualizations['category_sales_pie'] = fig_cat_sales.to_json()
            
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

def seasonal_analysis(df):
    analysis_name = "Seasonal Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'sales', 'transactions', 'customers', 'product_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['date', 'sales'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['date', 'sales'], inplace=True)

        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day_of_week'] = df['date'].dt.day_name()
        
        total_sales = df['sales'].sum()
        avg_daily_sales = df['sales'].mean()
        busiest_day = df.groupby('day_of_week')['sales'].mean().idxmax()
        
        metrics = {
            "total_sales": total_sales, 
            "avg_daily_sales": avg_daily_sales, 
            "busiest_day": busiest_day
        }
        
        insights.append(f"Total sales analyzed: ${total_sales:,.2f}.")
        insights.append(f"The busiest day of the week on average is {busiest_day}.")

        if 'month' in df and 'sales' in df:
            monthly_sales = df.groupby('month')['sales'].mean().reset_index()
            fig_monthly = px.line(monthly_sales, x='month', y='sales', title="Average Sales by Month")
            visualizations['monthly_seasonality'] = fig_monthly.to_json()
            
        if 'day_of_week' in df and 'sales' in df:
            dow_sales = df.groupby('day_of_week')['sales'].mean().reset_index()
            # Order days
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            dow_sales['day_of_week'] = pd.Categorical(dow_sales['day_of_week'], categories=day_order, ordered=True)
            dow_sales = dow_sales.sort_values('day_of_week')
            
            fig_dow = px.bar(dow_sales, x='day_of_week', y='sales', title="Average Sales by Day of Week")
            visualizations['day_of_week_pattern'] = fig_dow.to_json()
            
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

def customer_purchase_behavior_and_rfm_analysis(df):
    analysis_name = "Customer Purchase Behavior and RFM Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'invoicedate', 'unitprice', 'quantity', 'customerid']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        
        df.dropna(subset=['customerid', 'invoicedate', 'quantity', 'unitprice'], inplace=True)
        df = df[df['quantity'] > 0]
        df = df[df['unitprice'] > 0]
        
        df['total_price'] = df['quantity'] * df['unitprice']
        
        snapshot_date = df['invoicedate'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('customerid').agg({
            'invoicedate': lambda date: (snapshot_date - date.max()).days,
            'invoiceno': 'nunique',
            'total_price': 'sum'
        }).reset_index()
        rfm.columns = ['customerid', 'recency', 'frequency', 'monetary']
        
        avg_recency = rfm['recency'].mean()
        avg_frequency = rfm['frequency'].mean()
        avg_monetary = rfm['monetary'].mean()
        
        metrics = {
            "average_recency_days": avg_recency, 
            "average_frequency": avg_frequency, 
            "average_monetary_value": avg_monetary,
            "total_customers": len(rfm)
        }
        
        insights.append(f"RFM analysis completed for {len(rfm)} customers.")
        insights.append(f"Average Recency: {avg_recency:.1f} days.")
        insights.append(f"Average Frequency: {avg_frequency:.1f} purchases.")
        insights.append(f"Average Monetary Value: ${avg_monetary:,.2f}.")

        fig_rfm = px.scatter(rfm, x='recency', y='frequency', size='monetary', color='monetary', 
                             hover_name='customerid', title="RFM Customer Segmentation")
        visualizations['rfm_distribution'] = fig_rfm.to_json()
        
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

def retail_transaction_analysis_by_product_and_country(df):
    analysis_name = "Retail Transaction Analysis by Product and Country"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['description', 'qty', 'price', 'country']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['qty'] > 0]
        df['revenue'] = df['qty'] * df['price']
        df.dropna(subset=['revenue', 'country', 'description'], inplace=True)

        total_revenue = df['revenue'].sum()
        top_country = df.groupby('country')['revenue'].sum().idxmax()
        top_product = df.groupby('description')['revenue'].sum().idxmax()
        
        metrics = {
            "total_revenue": total_revenue, 
            "top_country_by_revenue": top_country, 
            "top_product_by_revenue": top_product
        }
        
        insights.append(f"Total revenue analyzed: ${total_revenue:,.2f}.")
        insights.append(f"Top country by revenue: {top_country}.")
        insights.append(f"Top product by revenue: {top_product}.")

        revenue_by_country = df.groupby('country')['revenue'].sum().nlargest(15).reset_index()
        fig_country = px.bar(revenue_by_country, x='country', y='revenue', title="Top 15 Countries by Sales Revenue")
        visualizations['revenue_by_country'] = fig_country.to_json()

        revenue_by_product = df.groupby('description')['revenue'].sum().nlargest(15).reset_index()
        fig_product = px.bar(revenue_by_product, x='description', y='revenue', title="Top 15 Products by Sales Revenue")
        visualizations['revenue_by_product'] = fig_product.to_json()
        
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

def retail_order_status_and_item_analysis(df):
    analysis_name = "Retail Order Status and Item Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['orderno', 'itemdesc', 'qty', 'invoicestatus']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        df.dropna(subset=['orderno', 'invoicestatus'], inplace=True)

        total_orders = df['orderno'].nunique()
        top_status = df['invoicestatus'].mode()[0] if not df['invoicestatus'].empty else "N/A"
        
        metrics = {
            "total_unique_orders": total_orders, 
            "most_common_status": top_status
        }
        
        insights.append(f"Analyzed {total_orders} unique orders.")
        insights.append(f"The most common invoice status is '{top_status}'.")

        status_counts = df['invoicestatus'].value_counts().reset_index()
        fig_status_pie = px.pie(status_counts, names='invoicestatus', values='count', title="Distribution of Invoice Statuses")
        visualizations['status_distribution'] = fig_status_pie.to_json()

        items_by_status = df.groupby('invoicestatus')['itemdesc'].count().reset_index()
        fig_items_status = px.bar(items_by_status, x='invoicestatus', y='itemdesc', title="Number of Items by Invoice Status")
        visualizations['items_by_status'] = fig_items_status.to_json()
        
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

def regional_sales_and_customer_analysis(df):
    analysis_name = "Regional Sales and Customer Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'unitprice', 'quantity', 'customerid', 'region']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df = df[df['quantity'] > 0]
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'region', 'customerid'], inplace=True)
        
        summary = df.groupby('region').agg(
            total_revenue=('revenue', 'sum'),
            unique_customers=('customerid', 'nunique'),
            avg_revenue_per_customer=('revenue', 'mean')
        ).reset_index()
        
        top_region = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_region_name": top_region['region'],
            "top_region_revenue": top_region['total_revenue']
        }
        
        insights.append(f"Analysis complete for {len(summary)} regions.")
        insights.append(f"Top region by revenue: {top_region['region']} with ${top_region['total_revenue']:,.2f}.")

        fig_rev_region = px.bar(summary, x='region', y='total_revenue', color='unique_customers', title="Total Revenue by Region (Colored by Customer Count)")
        visualizations['revenue_by_region'] = fig_rev_region.to_json()

        fig_rev_share = px.pie(summary, names='region', values='total_revenue', title="Share of Revenue by Region")
        visualizations['revenue_share_by_region'] = fig_rev_share.to_json()
        
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

def sales_channel_performance_analysis(df):
    analysis_name = "Sales Channel Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'unitprice', 'quantity', 'saleschannel']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df = df[df['quantity'] > 0]
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'saleschannel', 'invoiceno'], inplace=True)
        
        summary = df.groupby('saleschannel').agg(
            total_revenue=('revenue', 'sum'),
            total_orders=('invoiceno', 'nunique'),
        ).reset_index()
        summary['avg_order_value'] = summary['total_revenue'] / summary['total_orders']
        
        top_channel = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_channel_name": top_channel['saleschannel'],
            "top_channel_revenue": top_channel['total_revenue']
        }
        
        insights.append(f"Top sales channel: {top_channel['saleschannel']} with ${top_channel['total_revenue']:,.2f} in revenue.")
        insights.append(f"This channel also had an AOV of ${top_channel['avg_order_value']:,.2f}.")

        fig_rev_share = px.pie(summary, names='saleschannel', values='total_revenue', title="Share of Revenue by Sales Channel")
        visualizations['revenue_share'] = fig_rev_share.to_json()

        fig_aov = px.bar(summary, x='saleschannel', y='avg_order_value', title="Average Order Value by Sales Channel")
        visualizations['avg_order_value'] = fig_aov.to_json()
        
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

def international_sales_and_transaction_analysis(df):
    analysis_name = "International Sales and Transaction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['orderid', 'date', 'product', 'quantity', 'price', 'country', 'currency']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['quantity', 'price', 'country'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['revenue'] = df['quantity'] * df['price']
        df.dropna(subset=['revenue', 'country'], inplace=True)
        
        num_countries = df['country'].nunique()
        top_country = df.groupby('country')['revenue'].sum().idxmax()
        
        metrics = {
            "num_countries": num_countries, 
            "top_country_by_revenue": top_country,
            "total_revenue": df['revenue'].sum()
        }
        
        insights.append(f"Analyzed sales across {num_countries} countries.")
        insights.append(f"Top country by revenue: {top_country}.")

        revenue_by_country = df.groupby('country')['revenue'].sum().nlargest(20).reset_index()
        fig_rev_country = px.bar(revenue_by_country, x='country', y='revenue', title="Top 20 Countries by Revenue")
        visualizations['revenue_by_country'] = fig_rev_country.to_json()

        if 'currency' in df:
            currency_counts = df['currency'].value_counts().reset_index()
            fig_currency = px.pie(currency_counts, names='currency', values='count', title="Transaction Count by Currency")
            visualizations['transaction_count_by_currency'] = fig_currency.to_json()
            
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

def invoice_type_and_customer_purchase_pattern_analysis(df):
    analysis_name = "Invoice Type and Customer Purchase Pattern Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'invoicedate', 'unitprice', 'quantity', 'customerid', 'invoicetype']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['invoiceno', 'unitprice', 'quantity', 'invoicetype'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'invoicetype', 'invoiceno'], inplace=True)
        
        summary = df.groupby('invoicetype').agg(
            total_revenue=('revenue', 'sum'),
            num_invoices=('invoiceno', 'nunique'),
            num_customers=('customerid', 'nunique') if 'customerid' in df else ('invoiceno', 'nunique')
        ).reset_index()
        
        metrics = {"summary_table": summary.to_dict('records')}
        insights.append(f"Analyzed {df['invoiceno'].nunique()} invoices across {df['invoicetype'].nunique()} types.")

        fig_metrics_type = px.bar(summary, x='invoicetype', y=['total_revenue', 'num_invoices'], barmode='group', title="Key Metrics by Invoice Type")
        visualizations['metrics_by_invoice_type'] = fig_metrics_type.to_json()
        
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

def order_delivery_and_customer_location_analysis(df):
    analysis_name = "Order Delivery and Customer Location Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['orderid', 'customerid', 'deliveryzip']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        df.dropna(subset=['orderid', 'customerid', 'deliveryzip'], inplace=True)

        orders_by_zip = df['deliveryzip'].value_counts().nlargest(20).reset_index()
        orders_by_zip.columns = ['zipcode', 'order_count']
        
        customers_by_zip = df.groupby('deliveryzip')['customerid'].nunique().nlargest(20).reset_index()
        customers_by_zip.columns = ['zipcode', 'customer_count']

        metrics = {
            "top_zip_by_orders": orders_by_zip.iloc[0].to_dict(),
            "top_zip_by_customers": customers_by_zip.iloc[0].to_dict()
        }
        
        insights.append(f"Top ZIP code for orders: {orders_by_zip.iloc[0]['zipcode']} ({orders_by_zip.iloc[0]['order_count']} orders).")
        insights.append(f"Top ZIP code for customers: {customers_by_zip.iloc[0]['zipcode']} ({customers_by_zip.iloc[0]['customer_count']} customers).")

        fig_top_orders_zip = px.bar(orders_by_zip, x='zipcode', y='order_count', title="Top 20 ZIP Codes by Number of Orders")
        visualizations['top_orders_by_zip'] = fig_top_orders_zip.to_json()

        fig_top_cust_zip = px.bar(customers_by_zip, x='zipcode', y='customer_count', title="Top 20 ZIP Codes by Number of Unique Customers")
        visualizations['top_customers_by_zip'] = fig_top_cust_zip.to_json()
        
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

def promotional_code_effectiveness_analysis(df):
    analysis_name = "Promotional Code Effectiveness Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'promotioncode', 'unitprice', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['invoiceno', 'unitprice', 'quantity'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        
        # Handle 'promotioncode' (assuming missing means no promo)
        if 'promotioncode' in df:
            df['used_promo'] = df['promotioncode'].notna() & (df['promotioncode'] != 'None') & (df['promotioncode'] != '')
        else:
            df['used_promo'] = False # Assume no promo if col is missing
            insights.append("Warning: 'promotioncode' column not found. Assuming no promotions were used.")
            
        df.dropna(subset=['revenue', 'invoiceno'], inplace=True)
        
        # Aggregate revenue by invoice first
        invoice_data = df.groupby('invoiceno').agg(
            revenue=('revenue', 'sum'),
            used_promo=('used_promo', 'max') # If any item in invoice used promo, mark invoice
        )

        summary = invoice_data.groupby('used_promo').agg(
            total_revenue=('revenue', 'sum'),
            num_orders=('revenue', 'count')
        ).reset_index()
        
        summary['aov'] = summary['total_revenue'] / summary['num_orders']
        
        metrics = {"summary_table": summary.to_dict('records')}
        
        try:
            aov_with_promo = summary[summary['used_promo'] == True]['aov'].values[0]
            aov_without_promo = summary[summary['used_promo'] == False]['aov'].values[0]
            aov_lift = aov_with_promo - aov_without_promo
            insights.append(f"AOV with promo: ${aov_with_promo:,.2f}.")
            insights.append(f"AOV without promo: ${aov_without_promo:,.2f}.")
            insights.append(f"AOV lift from promotions: ${aov_lift:,.2f}.")
            metrics['aov_lift'] = aov_lift
        except IndexError:
            insights.append("Could not compare AOV (only one group found - promo/no-promo).")


        fig_aov_bar = px.bar(summary, x='used_promo', y='aov', title="Average Order Value (AOV) With vs. Without Promo Code")
        visualizations['aov_bar'] = fig_aov_bar.to_json()
        
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

def discount_impact_on_sales_analysis(df):
    analysis_name = "Discount Impact on Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'unitprice', 'quantity', 'discount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['unitprice', 'quantity', 'discount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['unitprice', 'quantity', 'discount'], inplace=True)
        
        # Assume discount is a decimal (e.g., 0.1 for 10%). If it's > 1, divide by 100.
        if df['discount'].max() > 1:
            df['discount'] = df['discount'] / 100
            insights.append("Converted 'discount' column from percentage to decimal.")
            
        df['revenue'] = df['quantity'] * df['unitprice'] * (1 - df['discount'])
        
        # Bin discounts
        max_discount = df['discount'].max()
        bins = [0, 0.05, 0.1, 0.2, max(0.51, max_discount + 0.01)]
        labels = ['0-5%', '5-10%', '10-20%', '20%+']
        df['discount_level'] = pd.cut(df['discount'], bins=bins, labels=labels, right=False)
        
        metrics = {
            "total_revenue": df['revenue'].sum(),
            "total_quantity_sold": df['quantity'].sum(),
            "avg_discount_percent": df['discount'].mean() * 100
        }
        
        insights.append(f"Analyzed {len(df)} items, total revenue ${metrics['total_revenue']:,.2f}.")
        insights.append(f"Average discount applied: {metrics['avg_discount_percent']:.2f}%.")

        fig_qty_disc = px.scatter(df, x='discount', y='quantity', title="Quantity Sold vs. Discount %", trendline="ols")
        visualizations['quantity_vs_discount'] = fig_qty_disc.to_json()

        revenue_by_discount = df.groupby('discount_level', observed=True)['revenue'].sum().reset_index()
        fig_rev_disc = px.bar(revenue_by_discount, x='discount_level', y='revenue', title="Total Revenue by Discount Level")
        visualizations['revenue_by_discount'] = fig_rev_disc.to_json()
        
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

def product_cost_and_sales_price_margin_analysis(df):
    analysis_name = "Product Cost and Sales Price Margin Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product', 'unitcost', 'salesprice', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product', 'unitcost', 'salesprice'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['unitcost', 'salesprice', 'quantity']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['unitcost', 'salesprice'], inplace=True)
        
        df['margin_per_unit'] = df['salesprice'] - df['unitcost']
        df['margin_perc'] = (df['margin_per_unit'] / df['salesprice']) * 100
        df.dropna(subset=['margin_perc'], inplace=True)

        avg_margin_perc = df['margin_perc'].mean()
        most_profitable_product = df.loc[df['margin_per_unit'].idxmax()]['product']
        
        metrics = {
            "average_profit_margin_percent": avg_margin_perc, 
            "most_profitable_product_by_unit": most_profitable_product,
            "avg_unit_cost": df['unitcost'].mean(),
            "avg_sales_price": df['salesprice'].mean()
        }
        
        insights.append(f"Average product profit margin: {avg_margin_perc:.2f}%.")
        insights.append(f"Most profitable product per unit: {most_profitable_product}.")

        margin_by_product = df.groupby('product')['margin_perc'].mean().nlargest(15).reset_index()
        fig_margin_prod = px.bar(margin_by_product, x='product', y='margin_perc', title="Top 15 Products by Average Profit Margin (%)")
        visualizations['margin_by_product'] = fig_margin_prod.to_json()

        fig_cost_price = px.scatter(df, x='unitcost', y='salesprice', 
                                    size='quantity' if 'quantity' in df else None, 
                                    hover_name='product', title="Sales Price vs. Unit Cost")
        visualizations['cost_vs_price'] = fig_cost_price.to_json()
        
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

def store_level_sales_performance_analysis(df):
    analysis_name = "Store Level Sales Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'itemname', 'quantity', 'unitprice', 'storeid']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['invoiceno', 'quantity', 'unitprice', 'storeid'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'storeid', 'invoiceno'], inplace=True)
        
        summary = df.groupby('storeid').agg(
            total_revenue=('revenue', 'sum'),
            num_transactions=('invoiceno', 'nunique'),
            units_sold=('quantity', 'sum')
        ).reset_index()
        
        top_store = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_store_id": top_store['storeid'],
            "top_store_revenue": top_store['total_revenue']
        }
        
        insights.append(f"Analyzed {len(summary)} stores.")
        insights.append(f"Top performing store: {top_store['storeid']} with ${top_store['total_revenue']:,.2f} in revenue.")

        fig_rev_store = px.bar(summary.sort_values('total_revenue', ascending=False), x='storeid', y='total_revenue', title="Total Revenue by Store")
        visualizations['revenue_by_store'] = fig_rev_store.to_json()

        fig_rev_trans = px.scatter(summary, x='num_transactions', y='total_revenue', 
                                   size='units_sold', color='storeid', 
                                   title="Revenue vs. Transactions (Sized by Units Sold)")
        visualizations['revenue_vs_transactions'] = fig_rev_trans.to_json()
        
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

def product_category_sales_analysis(df):
    analysis_name = "Product Category Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sku', 'description', 'price', 'category', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['category', 'price', 'quantity'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['revenue'] = df['price'] * df['quantity']
        df.dropna(subset=['revenue', 'category'], inplace=True)
        
        summary = df.groupby('category').agg(
            total_revenue=('revenue', 'sum'),
            units_sold=('quantity', 'sum'),
            num_skus=('sku', 'nunique') if 'sku' in df else ('description', 'nunique')
        ).reset_index()
        
        top_category = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_category_name": top_category['category'],
            "top_category_revenue": top_category['total_revenue']
        }
        
        insights.append(f"Analyzed {len(summary)} categories.")
        insights.append(f"Top category: {top_category['category']} with ${top_category['total_revenue']:,.2f} in revenue.")

        fig_rev_share = px.pie(summary, names='category', values='total_revenue', title="Share of Revenue by Product Category")
        visualizations['revenue_share'] = fig_rev_share.to_json()

        fig_treemap = px.treemap(df, path=[px.Constant("All Categories"), 'category', 'description'], 
                                 values='revenue', title="Hierarchical View of Revenue (Category > Product)")
        visualizations['revenue_treemap'] = fig_treemap.to_json()
        
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

def weekly_sales_trend_analysis(df):
    analysis_name = "Weekly Sales Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoicedate', 'unitprice', 'quantity', 'dayofweek']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['unitprice', 'quantity', 'dayofweek'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'dayofweek'], inplace=True)
        
        sales_by_day = df.groupby('dayofweek')['revenue'].sum().reset_index()
        
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        try:
            sales_by_day['dayofweek'] = pd.Categorical(sales_by_day['dayofweek'], categories=day_order, ordered=True)
            sales_by_day = sales_by_day.sort_values('dayofweek')
        except:
            insights.append("Could not sort days of the week automatically.")
            pass
            
        busiest_day = sales_by_day.sort_values('revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": sales_by_day.to_dict('records'),
            "busiest_day_name": busiest_day['dayofweek'],
            "busiest_day_revenue": busiest_day['revenue']
        }
        
        insights.append(f"Busiest day of the week is {busiest_day['dayofweek']} with ${busiest_day['revenue']:,.2f} in sales.")

        fig_rev_day = px.bar(sales_by_day, x='dayofweek', y='revenue', title="Total Sales Revenue by Day of the Week")
        visualizations['revenue_by_day'] = fig_rev_day.to_json()
        
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

def holiday_sales_impact_analysis(df):
    analysis_name = "Holiday Sales Impact Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoicedate', 'unitprice', 'quantity', 'holidayflag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['unitprice', 'quantity', 'holidayflag'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        
        # Standardize holidayflag (True/False, 1/0, 'Yes'/'No')
        if df['holidayflag'].dtype == 'object':
            df['is_holiday'] = df['holidayflag'].str.lower().map({'yes': True, 'true': True, '1': True, 'no': False, 'false': False, '0': False})
        else:
             df['is_holiday'] = df['holidayflag'].astype(bool)
             
        df.dropna(subset=['revenue', 'is_holiday'], inplace=True)
        
        # Assuming 'invoicedate' is present for daily aggregation
        if 'invoicedate' in df and not pd.api.types.is_datetime64_any_dtype(df['invoicedate']):
             df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
        
        # Aggregate by day
        if 'invoicedate' in df:
             daily_sales = df.groupby(['invoicedate', 'is_holiday'])['revenue'].sum().reset_index()
             summary = daily_sales.groupby('is_holiday')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
             summary.columns = ['Is Holiday', 'Avg Daily Revenue', 'Total Revenue', 'Day Count']
        else:
             # Fallback: aggregate by transaction
             summary = df.groupby('is_holiday')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
             summary.columns = ['Is Holiday', 'Avg Transaction Revenue', 'Total Revenue', 'Transaction Count']
             insights.append("Warning: 'invoicedate' not found. Metrics are per transaction, not per day.")

        metrics = {"summary_table": summary.to_dict('records')}
        
        try:
            avg_holiday = summary[summary['Is Holiday'] == True].iloc[0,1]
            avg_non_holiday = summary[summary['Is Holiday'] == False].iloc[0,1]
            lift = (avg_holiday - avg_non_holiday) / avg_non_holiday * 100
            insights.append(f"Average revenue on holidays: ${avg_holiday:,.2f}.")
            insights.append(f"Average revenue on non-holidays: ${avg_non_holiday:,.2f}.")
            insights.append(f"This represents a {lift:.2f}% lift on holidays.")
            metrics['holiday_lift_percent'] = lift
        except Exception:
            insights.append("Could not calculate holiday lift (missing holiday or non-holiday data).")

        fig_avg_daily_rev = px.bar(summary, x='Is Holiday', y=summary.columns[1], title=f"Average Revenue on Holidays vs. Non-Holidays")
        visualizations['avg_daily_rev'] = fig_avg_daily_rev.to_json()
        
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

def customer_type_segmentation_and_sales_analysis(df):
    analysis_name = "Customer Type Segmentation and Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'unitprice', 'quantity', 'customertype']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'customertype', 'invoiceno'], inplace=True)
        
        # Aggregate by invoice first
        invoice_data = df.groupby(['invoiceno', 'customertype'])['revenue'].sum().reset_index()

        summary = invoice_data.groupby('customertype').agg(
            total_revenue=('revenue', 'sum'),
            num_orders=('invoiceno', 'nunique')
        ).reset_index()
        summary['aov'] = summary['total_revenue'] / summary['num_orders']
        
        top_type = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_customer_type": top_type['customertype'],
            "top_type_revenue": top_type['total_revenue']
        }
        
        insights.append(f"Analyzed {len(summary)} customer types.")
        insights.append(f"Top customer type by revenue: {top_type['customertype']} (${top_type['total_revenue']:,.2f}).")
        insights.append(f"AOV for {top_type['customertype']}: ${top_type['aov']:,.2f}.")

        fig_rev_share = px.pie(summary, names='customertype', values='total_revenue', title="Share of Revenue by Customer Type")
        visualizations['revenue_share'] = fig_rev_share.to_json()

        fig_aov_type = px.bar(summary, x='customertype', y='aov', title="Average Order Value by Customer Type")
        visualizations['aov_by_type'] = fig_aov_type.to_json()
        
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

def online_vs_offline_sales_analysis(df):
    analysis_name = "Online vs. Offline Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['orderid', 'unitprice', 'qty', 'onlineflag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['qty'] * df['unitprice']
        
        # Standardize onlineflag (True/False, 1/0, 'Yes'/'No')
        if df['onlineflag'].dtype == 'object':
            df['is_online'] = df['onlineflag'].str.lower().map({'yes': True, 'true': True, '1': True, 'online': True})
            df['is_online'] = df['is_online'].fillna(False) # Assume others are offline
        else:
             df['is_online'] = df['onlineflag'].astype(bool)

        df['channel'] = df['is_online'].apply(lambda x: 'Online' if x else 'Offline')
        df.dropna(subset=['revenue', 'channel'], inplace=True)
        
        summary = df.groupby('channel')['revenue'].agg(['sum', 'count']).reset_index()
        summary.columns = ['Channel', 'Total Revenue', 'Transaction Count']
        
        metrics = {"summary_table": summary.to_dict('records')}
        
        try:
            online_rev = summary[summary['Channel'] == 'Online']['Total Revenue'].values[0]
            offline_rev = summary[summary['Channel'] == 'Offline']['Total Revenue'].values[0]
            insights.append(f"Online Revenue: ${online_rev:,.2f}.")
            insights.append(f"Offline Revenue: ${offline_rev:,.2f}.")
        except IndexError:
            insights.append("Could not find data for both Online and Offline channels.")

        fig_pie = px.pie(summary, names='Channel', values='Total Revenue', title="Share of Revenue: Online vs. Offline")
        visualizations['online_vs_offline'] = fig_pie.to_json()
        
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

def sales_tax_and_revenue_analysis(df):
    analysis_name = "Sales Tax and Revenue Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'priceeach', 'quantity', 'taxrate']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['priceeach', 'quantity', 'taxrate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['priceeach', 'quantity', 'taxrate'], inplace=True)
        
        # Assume taxrate is decimal (e.g., 0.08). If > 1, divide by 100.
        if df['taxrate'].max() > 1:
            df['taxrate'] = df['taxrate'] / 100
            insights.append("Converted 'taxrate' column from percentage to decimal.")
            
        df['revenue_pre_tax'] = df['priceeach'] * df['quantity']
        df['tax_amount'] = df['revenue_pre_tax'] * df['taxrate']
        df['total_revenue'] = df['revenue_pre_tax'] + df['tax_amount']
        
        total_revenue = df['total_revenue'].sum()
        total_tax = df['tax_amount'].sum()
        effective_tax_rate = (total_tax / df['revenue_pre_tax'].sum()) * 100 if df['revenue_pre_tax'].sum() > 0 else 0
        
        metrics = {
            "total_revenue_incl_tax": total_revenue, 
            "total_tax_collected": total_tax, 
            "effective_tax_rate_percent": effective_tax_rate
        }
        
        insights.append(f"Total revenue (incl. tax): ${total_revenue:,.2f}.")
        insights.append(f"Total tax collected: ${total_tax:,.2f}.")
        insights.append(f"Effective tax rate: {effective_tax_rate:.2f}%.")

        tax_by_rate = df.groupby('taxrate')['tax_amount'].sum().reset_index()
        fig_tax_rate = px.bar(tax_by_rate, x='taxrate', y='tax_amount', title="Total Tax Collected by Tax Rate")
        visualizations['tax_by_rate'] = fig_tax_rate.to_json()
        
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

def sales_organization_performance_analysis(df):
    analysis_name = "Sales Organization Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoiceno', 'unitprice', 'quantity', 'salesorg']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
        df['revenue'] = df['quantity'] * df['unitprice']
        df.dropna(subset=['revenue', 'salesorg', 'invoiceno'], inplace=True)
        
        summary = df.groupby('salesorg').agg(
            total_revenue=('revenue', 'sum'),
            num_orders=('invoiceno', 'nunique')
        ).reset_index()
        
        top_org = summary.sort_values('total_revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_sales_org": top_org['salesorg'],
            "top_org_revenue": top_org['total_revenue']
        }
        
        insights.append(f"Analyzed {len(summary)} sales organizations.")
        insights.append(f"Top sales org: {top_org['salesorg']} with ${top_org['total_revenue']:,.2f} in revenue from {top_org['num_orders']} orders.")

        fig_rev_org = px.bar(summary.sort_values('total_revenue', ascending=False),
                             x='salesorg', y='total_revenue', color='num_orders', title="Total Revenue by Sales Organization")
        visualizations['revenue_by_salesorg'] = fig_rev_org.to_json()
        
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

# --- Functions from the prompt that were not in the first list ---
# These are added to be complete, following the same refactoring pattern.

def customer_lifetime_value_clv_and_churn_risk_analysis(df):
    analysis_name = "Customer Lifetime Value (CLV) and Churn Risk Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_name', 'industry', 'lifetime_value', 'churn_risk_score', 'account_manager']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['lifetime_value', 'churn_risk_score'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['lifetime_value', 'churn_risk_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['lifetime_value', 'churn_risk_score'], inplace=True)
        
        avg_clv = df['lifetime_value'].mean()
        avg_churn_risk = df['churn_risk_score'].mean()
        
        metrics = {
            "average_lifetime_value": avg_clv, 
            "average_churn_risk_score": avg_churn_risk
        }
        
        insights.append(f"Average CLV: ${avg_clv:,.2f}.")
        insights.append(f"Average Churn Risk Score: {avg_churn_risk:.2f}.")

        fig_clv_churn = px.scatter(df, x='churn_risk_score', y='lifetime_value', 
                                   color='industry' if 'industry' in df else None, 
                                   title="CLV vs. Churn Risk Score")
        visualizations['clv_vs_churn_risk'] = fig_clv_churn.to_json()

        if 'industry' in df:
            clv_by_industry = df.groupby('industry')['lifetime_value'].mean().reset_index()
            fig_clv_ind = px.bar(clv_by_industry, x='industry', y='lifetime_value', title="Average CLV by Industry")
            visualizations['clv_by_industry'] = fig_clv_ind.to_json()
            
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

def subscription_sales_and_renewal_analysis(df):
    analysis_name = "Subscription Sales and Renewal Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['plan_type', 'monthly_fee', 'auto_renew_flag', 'cancellation_date', 'cancellation_reason']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['plan_type', 'cancellation_date'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['churned'] = df['cancellation_date'].notna()
        churn_rate = df['churned'].mean() * 100
        
        metrics = {"overall_churn_rate_percent": churn_rate}
        insights.append(f"Overall churn rate: {churn_rate:.2f}%.")

        churn_by_plan = df.groupby('plan_type')['churned'].mean().mul(100).reset_index()
        fig_churn_plan = px.bar(churn_by_plan, x='plan_type', y='churned', title="Churn Rate by Subscription Plan")
        visualizations['churn_by_plan'] = fig_churn_plan.to_json()

        if matched.get('cancellation_reason'):
            churn_reasons = df[df['churned']]['cancellation_reason'].value_counts().reset_index()
            fig_churn_reasons = px.pie(churn_reasons, names='cancellation_reason', values='count', title="Distribution of Churn Reasons")
            visualizations['churn_reasons'] = fig_churn_reasons.to_json()
            
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

def sales_channel_performance_and_conversion_analysis(df):
    analysis_name = "Sales Channel Performance and Conversion Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['channel_type', 'units_sold', 'revenue', 'profit', 'conversion_rate']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['units_sold', 'revenue', 'profit', 'conversion_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['channel_type', 'revenue', 'profit', 'conversion_rate'], inplace=True)
        
        summary = df.groupby('channel_type').agg({
            'revenue': 'sum',
            'profit': 'sum',
            'conversion_rate': 'mean'
        }).reset_index()
        
        metrics = {"summary_table": summary.to_dict('records')}
        insights.append(f"Analyzed {len(summary)} sales channels.")
        
        top_channel = summary.sort_values('revenue', ascending=False).iloc[0]
        insights.append(f"Top channel by revenue: {top_channel['channel_type']} (${top_channel['revenue']:,.2f}).")

        fig_rev_profit = px.bar(summary, x='channel_type', y=['revenue', 'profit'], barmode='group', title="Total Revenue and Profit by Sales Channel")
        visualizations['revenue_and_profit'] = fig_rev_profit.to_json()

        fig_conv = px.bar(summary, x='channel_type', y='conversion_rate', title="Average Conversion Rate by Sales Channel")
        visualizations['conversion_rate'] = fig_conv.to_json()
        
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

def cross_sell_and_upsell_opportunity_analysis(df):
    analysis_name = "Cross-Sell and Upsell Opportunity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['primary_order_id', 'upsell_product_id', 'upsell_quantity', 'upsell_price', 'profit']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['upsell_product_id', 'profit'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['upsell_quantity', 'upsell_price', 'profit']:
            if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['upsell_product_id', 'profit'], inplace=True)
        
        total_upsell_profit = df['profit'].sum()
        
        metrics = {"total_profit_from_upsells_cross_sells": total_upsell_profit}
        insights.append(f"Total profit from upsells/cross-sells: ${total_upsell_profit:,.2f}.")

        top_products = df.groupby('upsell_product_id')['profit'].sum().nlargest(15).reset_index()
        fig_top_prod = px.bar(top_products, x='upsell_product_id', y='profit', title="Top 15 Most Profitable Upsell/Cross-sell Products")
        visualizations['top_products'] = fig_top_prod.to_json()
        
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

def sales_territory_performance_and_quota_achievement_analysis(df):
    analysis_name = "Sales Territory Performance and Quota Achievement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['territory_name', 'sales_rep_id', 'quota', 'ytd_sales', 'achievement_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['territory_name', 'quota', 'ytd_sales'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['quota', 'ytd_sales', 'achievement_perc']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'achievement_perc' not in df and 'quota' in df and 'ytd_sales' in df:
            df['achievement_perc'] = (df['ytd_sales'] / df['quota']) * 100
            insights.append("Calculated 'achievement_perc' from 'ytd_sales' / 'quota'.")
            
        df.dropna(subset=['territory_name', 'quota', 'ytd_sales', 'achievement_perc'], inplace=True)
        
        overall_achievement = (df['ytd_sales'].sum() / df['quota'].sum()) * 100
        
        metrics = {"overall_quota_achievement_percent": overall_achievement}
        insights.append(f"Overall quota achievement: {overall_achievement:.2f}%.")
        
        summary = df.groupby('territory_name').agg(
            total_quota=('quota', 'sum'),
            total_sales=('ytd_sales', 'sum')
        ).reset_index()
        summary['achievement'] = (summary['total_sales'] / summary['total_quota']) * 100
        
        metrics['summary_table'] = summary.to_dict('records')

        fig_quota_sales = px.bar(summary, x='territory_name', y=['total_quota', 'total_sales'], barmode='group', title="Quota vs. YTD Sales by Territory")
        visualizations['quota_vs_sales'] = fig_quota_sales.to_json()
        
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

def product_sales_performance_and_profitability_analysis(df):
    analysis_name = "Product Sales Performance and Profitability Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'units_sold', 'revenue', 'cogs', 'gross_profit', 'profit_margin_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'revenue', 'gross_profit'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['units_sold', 'revenue', 'cogs', 'gross_profit', 'profit_margin_perc']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'gross_profit' not in df and 'revenue' in df and 'cogs' in df:
             df['gross_profit'] = df['revenue'] - df['cogs']
             insights.append("Calculated 'gross_profit' from 'revenue' - 'cogs'.")
             
        df.dropna(subset=['product_id', 'revenue', 'gross_profit'], inplace=True)
        
        summary = df.groupby('product_id').agg(
            total_revenue=('revenue', 'sum'),
            total_profit=('gross_profit', 'sum')
        ).nlargest(20, 'total_revenue').reset_index()
        
        metrics = {"summary_table": summary.to_dict('records')}
        insights.append("Analyzed top 20 products by revenue and their associated profit.")

        fig_top_prod = px.bar(summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 20 Products by Revenue and Profit")
        visualizations['top_products'] = fig_top_prod.to_json()
        
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

def product_pricing_strategy_and_tier_analysis(df):
    analysis_name = "Product Pricing Strategy and Tier Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'list_price', 'tier_1_price', 'tier_2_price', 'tier_3_price', 'channel']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'list_price', 'tier_1_price'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        price_cols = [col for col in ['list_price', 'tier_1_price', 'tier_2_price', 'tier_3_price'] if col in df.columns]
        for col in price_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['product_id'] + price_cols, inplace=True)
        
        df_long = df.melt(id_vars=[col for col in ['product_id', 'channel'] if col in df.columns], 
                          value_vars=price_cols,
                          var_name='price_tier', value_name='price')
        
        metrics = {"total_pricing_points": len(df_long)}
        insights.append(f"Analyzed {len(df_long)} pricing points across {df['product_id'].nunique()} products.")

        fig_price_tiers = px.box(df_long, x='price_tier', y='price', 
                                 color='channel' if 'channel' in df_long else None, 
                                 title="Pricing Distribution by Tier and Channel")
        visualizations['pricing_tier_distribution'] = fig_price_tiers.to_json()

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

def sales_forecasting_accuracy_analysis(df):
    analysis_name = "Sales Forecasting Accuracy Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['forecast_date', 'territory', 'pipeline_value', 'forecast_value', 'historical_sales']
        # 'actual_sales' is a common alternative to 'historical_sales'
        if not matched.get('historical_sales'):
             matched.update(fuzzy_match_column(df, {'historical_sales': 'actual_sales'}))
             
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['forecast_date', 'forecast_value', 'historical_sales'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
        
        for col in ['pipeline_value', 'forecast_value', 'historical_sales']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('forecast_date').dropna(subset=['forecast_date', 'forecast_value', 'historical_sales'])
        
        df['error'] = df['historical_sales'] - df['forecast_value']
        df['mape'] = (df['error'].abs() / df['historical_sales'].abs()) * 100
        
        mape = df['mape'].mean()
        
        metrics = {"mean_absolute_percentage_error": mape}
        insights.append(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%.")

        df_long = df.melt(id_vars='forecast_date', value_vars=['forecast_value', 'historical_sales'],
                          var_name='type', value_name='sales')
        
        fig_fc_actual = px.line(df_long, x='forecast_date', y='sales', color='type', title="Forecast vs. Actual Sales")
        visualizations['forecast_vs_actual'] = fig_fc_actual.to_json()
        
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

def channel_promotion_performance_and_roi_analysis(df):
    analysis_name = "Channel Promotion Performance and ROI Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['channel', 'promotion_start', 'discount_perc', 'projected_lift_perc', 'actual_lift_perc', 'roi_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['discount_perc', 'projected_lift_perc', 'actual_lift_perc', 'roi_perc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        metrics = {
            "avg_roi_percent": df['roi_perc'].mean(),
            "avg_actual_lift_percent": df['actual_lift_perc'].mean()
        }
        insights.append(f"Average promotion ROI: {metrics['avg_roi_percent']:.2f}%.")
        insights.append(f"Average actual sales lift: {metrics['avg_actual_lift_percent']:.2f}%.")

        fig_actual_proj = px.scatter(df, x='projected_lift_perc', y='actual_lift_perc', color='channel',
                                     title="Actual vs. Projected Sales Lift by Channel")
        visualizations['actual_vs_projected'] = fig_actual_proj.to_json()

        summary = df.groupby('channel')[['roi_perc', 'actual_lift_perc']].mean().reset_index()
        fig_roi_lift = px.bar(summary, x='channel', y=['roi_perc', 'actual_lift_perc'], barmode='group',
                              title="Average ROI and Actual Lift by Channel")
        visualizations['roi_and_lift'] = fig_roi_lift.to_json()
        
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

def customer_service_impact_on_sales_analysis(df):
    analysis_name = "Customer Service Impact on Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['case_open_date', 'resolution_time_min', 'satisfaction_score', 'case_status', 'escalation_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['resolution_time_min', 'satisfaction_score'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['resolution_time_min', 'satisfaction_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['resolution_time_min', 'satisfaction_score'], inplace=True)
        
        metrics = {
            "avg_resolution_time_min": df['resolution_time_min'].mean(),
            "avg_satisfaction_score": df['satisfaction_score'].mean(),
            "csat_resolution_correlation": df['resolution_time_min'].corr(df['satisfaction_score'])
        }
        
        insights.append(f"Average CSAT score: {metrics['avg_satisfaction_score']:.2f}.")
        insights.append(f"Average resolution time: {metrics['avg_resolution_time_min']:.1f} minutes.")
        insights.append(f"Correlation between resolution time and CSAT: {metrics['csat_resolution_correlation']:.2f}.")

        fig_csat_dist = px.histogram(df, x='satisfaction_score', title="Distribution of CSAT Scores")
        visualizations['csat_distribution'] = fig_csat_dist.to_json()

        fig_csat_res = px.scatter(df, x='resolution_time_min', y='satisfaction_score', title="Satisfaction Score vs. Resolution Time", trendline="ols")
        visualizations['csat_vs_resolution_time'] = fig_csat_res.to_json()
        
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

def sales_call_outcome_and_effectiveness_analysis(df):
    analysis_name = "Sales Call Outcome and Effectiveness Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['call_date', 'sales_rep_id', 'call_duration_sec', 'outcome', 'deal_size']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['call_duration_sec', 'outcome'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['call_duration_sec', 'deal_size']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['call_duration_sec', 'outcome'], inplace=True)
        
        outcome_counts = df['outcome'].value_counts()
        
        metrics = {
            "total_calls": len(df),
            "avg_call_duration_sec": df['call_duration_sec'].mean(),
            "outcome_distribution": outcome_counts.to_dict()
        }
        
        insights.append(f"Analyzed {len(df)} sales calls.")
        insights.append(f"Average call duration: {metrics['avg_call_duration_sec']:.1f} seconds.")
        insights.append(f"Most common outcome: {outcome_counts.idxmax()}.")

        fig_outcome_dist = px.pie(outcome_counts.reset_index(), names='outcome', values='count', title="Distribution of Call Outcomes")
        visualizations['call_outcome_distribution'] = fig_outcome_dist.to_json()

        fig_duration_outcome = px.box(df, x='outcome', y='call_duration_sec', title="Call Duration by Outcome")
        visualizations['call_duration_by_outcome'] = fig_duration_outcome.to_json()
        
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

def market_segment_revenue_and_profitability_analysis(df):
    analysis_name = "Market Segment Revenue and Profitability Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['segment_name', 'segment_revenue', 'segment_profit', 'segment_growth_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['segment_revenue', 'segment_profit', 'segment_growth_perc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        metrics = {
            "total_revenue": df['segment_revenue'].sum(),
            "total_profit": df['segment_profit'].sum(),
            "avg_growth_rate": df['segment_growth_perc'].mean()
        }
        insights.append(f"Total revenue across segments: ${metrics['total_revenue']:,.2f}.")
        insights.append(f"Total profit across segments: ${metrics['total_profit']:,.2f}.")

        fig_rev_profit = px.bar(
            df, x='segment_name', y=['segment_revenue', 'segment_profit'],
            barmode='group', title="Revenue and Profit by Market Segment"
        )
        visualizations['revenue_and_profit'] = fig_rev_profit.to_json()

        fig_profit_growth = px.scatter(
            df, x='segment_growth_perc', y='segment_profit', size='segment_revenue',
            color='segment_name', title="Profit vs. Growth % by Segment"
        )
        visualizations['profit_vs_growth'] = fig_profit_growth.to_json()
        
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

def competitor_pricing_and_feature_analysis(df):
    analysis_name = "Competitor Pricing and Feature Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['our_product_id', 'competitor_product', 'competitor_price', 'our_price', 'market_share_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['our_product_id', 'competitor_price', 'our_price'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['competitor_price', 'our_price', 'market_share_perc']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['competitor_price', 'our_price'], inplace=True)
        
        df['price_difference'] = df['our_price'] - df['competitor_price']
        
        metrics = {
            "avg_price_difference": df['price_difference'].mean(),
            "products_more_expensive": (df['price_difference'] > 0).sum(),
            "products_less_expensive": (df['price_difference'] < 0).sum()
        }
        
        insights.append(f"On average, our products are ${metrics['avg_price_difference']:,.2f} different from competitors.")
        insights.append(f"We are more expensive for {metrics['products_more_expensive']} products.")
        insights.append(f"We are less expensive for {metrics['products_less_expensive']} products.")

        # Melt for easier plotting
        df_long = df.melt(id_vars='our_product_id', value_vars=['our_price', 'competitor_price'],
                          var_name='price_type', value_name='price')
        
        fig_price_comp = px.bar(
            df_long, x='our_product_id', y='price', color='price_type',
            barmode='group', title="Our Price vs. Competitor Price by Product"
        )
        visualizations['price_comparison'] = fig_price_comp.to_json()
        
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

def product_bundle_sales_performance_analysis(df):
    analysis_name = "Product Bundle Sales Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['bundle_name', 'bundle_price', 'revenue', 'profit']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['bundle_name', 'revenue', 'profit'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['bundle_price', 'revenue', 'profit']:
             if col in df:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['bundle_name', 'revenue', 'profit'], inplace=True)
        
        summary = df.groupby('bundle_name').agg({'revenue':'sum', 'profit':'sum'}).reset_index()
        
        top_bundle = summary.sort_values('revenue', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_bundle_by_revenue": top_bundle['bundle_name'],
            "top_bundle_revenue": top_bundle['revenue']
        }
        
        insights.append(f"Analyzed {len(summary)} product bundles.")
        insights.append(f"Top bundle by revenue: {top_bundle['bundle_name']} (${top_bundle['revenue']:,.2f}).")

        fig_rev_profit = px.bar(
            summary, x='bundle_name', y=['revenue', 'profit'],
            title="Total Revenue and Profit by Product Bundle"
        )
        visualizations['revenue_and_profit_by_bundle'] = fig_rev_profit.to_json()
        
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

def international_sales_and_currency_exchange_analysis(df):
    analysis_name = "International Sales and Currency Exchange Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['currency_pair', 'exchange_rate', 'converted_amount', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['exchange_rate', 'converted_amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(inplace=True)
        
        summary = df.groupby('currency_pair').agg(
            total_value=('converted_amount', 'sum'),
            num_transactions=('transaction_id', 'count')
        ).reset_index()
        
        top_pair = summary.sort_values('total_value', ascending=False).iloc[0]

        metrics = {
            "summary_table": summary.to_dict('records'),
            "top_currency_pair": top_pair['currency_pair'],
            "top_pair_total_value": top_pair['total_value']
        }
        
        insights.append(f"Analyzed {len(summary)} currency pairs.")
        insights.append(f"Top currency pair by value: {top_pair['currency_pair']} (${top_pair['total_value']:,.2f}).")

        fig_trans_val = px.bar(
            summary, x='currency_pair', y='total_value', color='num_transactions',
            title="Total Transaction Value by Currency Pair"
        )
        visualizations['transaction_value_by_currency'] = fig_trans_val.to_json()
        
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

def sales_contract_and_renewal_analysis(df):
    analysis_name = "Sales Contract and Renewal Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'contract_value', 'renewal_option', 'renewal_probability_perc', 'contract_status', 'signed_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['contract_value', 'renewal_probability_perc', 'contract_status'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['contract_value', 'renewal_probability_perc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['contract_value', 'renewal_probability_perc', 'contract_status'], inplace=True)
        
        status_counts = df['contract_status'].value_counts()
        
        metrics = {
            "total_contracts": len(df),
            "avg_contract_value": df['contract_value'].mean(),
            "avg_renewal_probability_percent": df['renewal_probability_perc'].mean(),
            "status_distribution": status_counts.to_dict()
        }
        
        insights.append(f"Analyzed {len(df)} contracts.")
        insights.append(f"Average contract value: ${metrics['avg_contract_value']:,.2f}.")
        insights.append(f"Average renewal probability: {metrics['avg_renewal_probability_percent']:.2f}%.")

        fig_status_dist = px.pie(
            status_counts.reset_index(), names='contract_status', values='count',
            title="Distribution of Contract Statuses"
        )
        visualizations['contract_status_distribution'] = fig_status_dist.to_json()

        fig_renewal_value = px.scatter(
            df, x='contract_value', y='renewal_probability_perc', 
            color='renewal_option' if 'renewal_option' in df else None,
            title="Renewal Probability vs. Contract Value"
        )
        visualizations['renewal_vs_value'] = fig_renewal_value.to_json()
        
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

def e_commerce_sales_funnel_and_conversion_analysis(df):
    analysis_name = "E-commerce Sales Funnel and Conversion Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['session_id', 'add_to_cart_date', 'purchase_date', 'cart_abandon_flag', 'revenue']
        matched = fuzzy_match_column(df, expected)
        
        # We need a way to track funnel stages.
        # Option 1: 'cart_abandon_flag'
        # Option 2: 'add_to_cart_date' and 'purchase_date'
        
        has_flag = matched.get('cart_abandon_flag') is not None
        has_dates = matched.get('add_to_cart_date') is not None and matched.get('purchase_date') is not None
        
        if not has_flag and not has_dates:
             return create_fallback_response(analysis_name, ['cart_abandon_flag', 'add_to_cart_date', 'purchase_date'], matched, df)
            
        df = safe_rename(df, matched)
        
        if 'session_id' not in df:
             df['session_id'] = df.index # Create a proxy session
             insights.append("Warning: 'session_id' not found. Used row index as session ID.")

        total_sessions = df['session_id'].nunique()
        
        if has_dates:
            df['added_to_cart'] = df['add_to_cart_date'].notna()
            df['purchased'] = df['purchase_date'].notna()
            added_to_cart_sessions = df[df['added_to_cart'] == True]['session_id'].nunique()
            purchased_sessions = df[df['purchased'] == True]['session_id'].nunique()
            
            funnel_stages = ['Total Sessions', 'Added to Cart', 'Purchased']
            funnel_values = [total_sessions, added_to_cart_sessions, purchased_sessions]
            
            metrics = {
                "total_sessions": total_sessions,
                "added_to_cart_sessions": added_to_cart_sessions,
                "purchased_sessions": purchased_sessions,
                "session_to_cart_rate": (added_to_cart_sessions / total_sessions) * 100,
                "cart_to_purchase_rate": (purchased_sessions / added_to_cart_sessions) * 100,
                "session_to_purchase_rate": (purchased_sessions / total_sessions) * 100
            }
            insights.append(f"Overall session-to-purchase conversion rate: {metrics['session_to_purchase_rate']:.2f}%.")
            
        elif has_flag:
            # Simpler funnel based on abandon flag
            if df['cart_abandon_flag'].dtype == 'object':
                 df['purchased_flag'] = ~df['cart_abandon_flag'].str.lower().map({'yes': True, 'true': True, '1': True, 'no': False, 'false': False, '0': False})
            else:
                 df['purchased_flag'] = ~df['cart_abandon_flag'].astype(bool)

            purchased_sessions = df[df['purchased_flag'] == True]['session_id'].nunique()
            conversion_rate = (purchased_sessions / total_sessions) * 100
            
            metrics = {
                "total_sessions": total_sessions,
                "purchased_sessions": purchased_sessions,
                "session_to_purchase_conversion_rate": conversion_rate
            }
            insights.append(f"Overall session-to-purchase conversion rate: {conversion_rate:.2f}%.")
            
            funnel_stages = ['Total Sessions', 'Purchased']
            funnel_values = [total_sessions, purchased_sessions]

        fig = go.Figure(go.Funnel(
            y=funnel_stages,
            x=funnel_values,
            textinfo="value+percent previous"
        ))
        fig.update_layout(title_text="E-commerce Sales Funnel")
        visualizations["sales_funnel"] = fig.to_json()
        
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

def sales_order_fulfillment_and_status_analysis(df):
    analysis_name = "Sales Order Fulfillment and Status Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'customer_id', 'order_date', 'order_status', 'order_value', 'fulfillment_days']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['order_id', 'order_status', 'order_value'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce')
        if 'fulfillment_days' in df:
            df['fulfillment_days'] = pd.to_numeric(df['fulfillment_days'], errors='coerce')
        
        df.dropna(subset=['order_id', 'order_status', 'order_value'], inplace=True)
        
        status_counts = df['order_status'].value_counts()
        metrics = {
            "total_orders": len(df),
            "total_order_value": df['order_value'].sum(),
            "status_distribution": status_counts.to_dict()
        }
        insights.append(f"Analyzed {len(df)} orders with a total value of ${metrics['total_order_value']:,.2f}.")
        insights.append(f"Most common order status: {status_counts.idxmax()} ({status_counts.max()} orders).")

        fig_status_dist = px.pie(status_counts.reset_index(), names='order_status', values='count', title="Distribution of Order Statuses")
        visualizations['order_status_distribution'] = fig_status_dist.to_json()

        if 'fulfillment_days' in df and not df['fulfillment_days'].isnull().all():
            avg_fulfillment = df.groupby('order_status')['fulfillment_days'].mean().reset_index()
            fig_avg_fulfill = px.bar(avg_fulfillment, x='order_status', y='fulfillment_days', title="Average Fulfillment Days by Order Status")
            visualizations['avg_fulfillment'] = fig_avg_fulfill.to_json()
        
        if 'order_date' in df:
            if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
                df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
            
            if not df['order_date'].isnull().all():
                revenue_trend = df.set_index('order_date').resample('M')['order_value'].sum().reset_index()
                fig_monthly_rev = px.line(revenue_trend, x='order_date', y='order_value', markers=True, title="Monthly Revenue Trend")
                visualizations['monthly_revenue'] = fig_monthly_rev.to_json()
                
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

def sales_invoice_and_payment_reconciliation_analysis(df):
    analysis_name = "Sales Invoice and Payment Reconciliation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoice_id', 'invoice_amount', 'payment_amount', 'payment_date', 'invoicestatus']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['invoice_id', 'invoice_amount', 'payment_amount', 'invoicestatus'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['invoice_amount', 'payment_amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing payments
        df['payment_amount'] = df['payment_amount'].fillna(0)
        df.dropna(subset=['invoice_amount', 'invoicestatus'], inplace=True)
        
        df['balance_due'] = df['invoice_amount'] - df['payment_amount']
        
        summary = df.groupby('invoicestatus').agg(
            num_invoices=('invoice_id', 'nunique'),
            total_balance=('balance_due', 'sum'),
            total_invoice_amount=('invoice_amount', 'sum')
        ).reset_index()

        metrics = {
            "summary_table": summary.to_dict('records'),
            "total_outstanding_balance": df['balance_due'].sum(),
            "total_invoiced_amount": df['invoice_amount'].sum()
        }
        
        insights.append(f"Total invoiced amount: ${metrics['total_invoiced_amount']:,.2f}.")
        insights.append(f"Total outstanding balance: ${metrics['total_outstanding_balance']:,.2f}.")
        
        try:
            paid_balance = summary[summary['invoicestatus'].str.lower() == 'paid']['total_balance'].values[0]
            insights.append(f"Note: 'Paid' invoices still have a balance of ${paid_balance:,.2f}.")
        except Exception:
            pass

        fig_status_dist = px.pie(summary, names='invoicestatus', values='num_invoices', title="Distribution of Invoice Statuses (by Count)")
        visualizations['invoice_status_distribution'] = fig_status_dist.to_json()

        fig_balance_status = px.bar(summary, x='invoicestatus', y='total_balance', title="Total Balance Due by Invoice Status")
        visualizations['balance_due_by_status'] = fig_balance_status.to_json()
        
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

def sales_transaction_and_profit_margin_analysis(df):
    analysis_name = "Sales Transaction and Profit Margin Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'product_id', 'sales_price', 'cost_of_goods_sold']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'sales_price', 'cost_of_goods_sold'] if matched.get(col) is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        for col in ['sales_price', 'cost_of_goods_sold']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['product_id', 'sales_price', 'cost_of_goods_sold'], inplace=True)

        df['profit_margin'] = df['sales_price'] - df['cost_of_goods_sold']
        df['profit_margin_percent'] = (df['profit_margin'] / df['sales_price']) * 100
        
        product_summary = df.groupby('product_id').agg(
            total_revenue=('sales_price', 'sum'),
            total_profit=('profit_margin', 'sum'),
            avg_profit_margin_percent=('profit_margin_percent', 'mean')
        ).nlargest(15, 'total_revenue').reset_index()

        metrics = {
            "summary_table": product_summary.to_dict('records'),
            "overall_avg_margin_percent": df['profit_margin_percent'].mean()
        }
        
        insights.append(f"Overall average profit margin: {metrics['overall_avg_margin_percent']:.2f}%.")
        insights.append("Analyzed top 15 products by revenue.")

        fig_top_prod = px.bar(product_summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 15 Products by Revenue and Profit")
        visualizations['top_products_revenue_profit'] = fig_top_prod.to_json()

        fig_margin_dist = px.histogram(df, x='profit_margin_percent', title="Distribution of Profit Margins (%)")
        visualizations['profit_margin_distribution'] = fig_margin_dist.to_json()
        
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


def main_backend(df, category=None, general_analysis=None, specific_analysis_name=None):
    
    # Mapping of general analysis names to their corresponding functions
    general_analysis_functions = {
        "Sales Analysis": sales_analysis,
        "Customer Analysis": customer_analysis,
        "Inventory Analysis": inventory_analysis,
        "Product Analysis": product_analysis,
        "Store Analysis": store_analysis,
        "Promotion Analysis": promotion_analysis,
        "Basket Analysis": basket_analysis,
        "Seasonal Analysis": seasonal_analysis,
    }

    # Mapping of specific analysis names to their corresponding functions
    specific_retail_function_mapping = {
        # Functions from the first part of the script
        "Customer Purchase Behavior and RFM Analysis": customer_purchase_behavior_and_rfm_analysis,
        "Retail Transaction Analysis by Product and Country": retail_transaction_analysis_by_product_and_country,
        "Retail Order Status and Item Analysis": retail_order_status_and_item_analysis,
        "Regional Sales and Customer Analysis": regional_sales_and_customer_analysis,
        "Sales Channel Performance Analysis": sales_channel_performance_analysis,
        "International Sales and Transaction Analysis": international_sales_and_transaction_analysis,
        "Invoice Type and Customer Purchase Pattern Analysis": invoice_type_and_customer_purchase_pattern_analysis,
        "Order Delivery and Customer Location Analysis": order_delivery_and_customer_location_analysis,
        "Promotional Code Effectiveness Analysis": promotional_code_effectiveness_analysis,
        "Discount Impact on Sales Analysis": discount_impact_on_sales_analysis,
        "Product Cost and Sales Price Margin Analysis": product_cost_and_sales_price_margin_analysis,
        "Store Level Sales Performance Analysis": store_level_sales_performance_analysis,
        "Product Category Sales Analysis": product_category_sales_analysis,
        "Weekly Sales Trend Analysis": weekly_sales_trend_analysis,
        "Holiday Sales Impact Analysis": holiday_sales_impact_analysis,
        "Customer Type Segmentation and Sales Analysis": customer_type_segmentation_and_sales_analysis,
        "Online vs Offline Sales Analysis": online_vs_offline_sales_analysis,
        "Sales Tax and Revenue Analysis": sales_tax_and_revenue_analysis,
        "Sales Organization Performance Analysis": sales_organization_performance_analysis,
        
        # Functions from the second part of the script
        "Customer Lifetime Value (CLV) and Churn Risk Analysis": customer_lifetime_value_clv_and_churn_risk_analysis,
        "Subscription Sales and Renewal Analysis": subscription_sales_and_renewal_analysis,
        "Sales Channel Performance and Conversion Analysis": sales_channel_performance_and_conversion_analysis,
        "Cross-Sell and Upsell Opportunity Analysis": cross_sell_and_upsell_opportunity_analysis,
        "Sales Territory Performance and Quota Achievement Analysis": sales_territory_performance_and_quota_achievement_analysis,
        "Product Sales Performance and Profitability Analysis": product_sales_performance_and_profitability_analysis,
        "Product Pricing Strategy and Tier Analysis": product_pricing_strategy_and_tier_analysis,
        "Sales Forecasting Accuracy Analysis": sales_forecasting_accuracy_analysis,
        "Channel Promotion Performance and ROI Analysis": channel_promotion_performance_and_roi_analysis,
        "Customer Service Impact on Sales Analysis": customer_service_impact_on_sales_analysis,
        "Sales Call Outcome and Effectiveness Analysis": sales_call_outcome_and_effectiveness_analysis,
        "Market Segment Revenue and Profitability Analysis": market_segment_revenue_and_profitability_analysis,
        "Competitor Pricing and Feature Analysis": competitor_pricing_and_feature_analysis,
        "Product Bundle Sales Performance Analysis": product_bundle_sales_performance_analysis,
        "International Sales and Currency Exchange Analysis": international_sales_and_currency_exchange_analysis,
        "Sales Contract and Renewal Analysis": sales_contract_and_renewal_analysis,
        "E-commerce Sales Funnel and Conversion Analysis": e_commerce_sales_funnel_and_conversion_analysis,
        "Sales Order Fulfillment and Status Analysis": sales_order_fulfillment_and_status_analysis,
        "Sales Invoice and Payment Reconciliation Analysis": sales_invoice_and_payment_reconciliation_analysis,
        "Sales Transaction and Profit Margin Analysis": sales_transaction_and_profit_margin_analysis,
    }

    result = None

    try:
        if category == "General Retail Analysis":
            if not general_analysis or general_analysis == "--Select--":
                # Use the utility function from your script
                result = show_general_insights(df, "Initial Overview")
            else:
                func = general_analysis_functions.get(general_analysis)
                if func:
                    result = func(df)
                else:
                    # Fallback if name not found
                    result = show_general_insights(df, "Initial Overview")

        elif category == "Specific Retail Analysis":
            if specific_analysis_name and specific_analysis_name != "--Select--":
                func = specific_retail_function_mapping.get(specific_analysis_name)
                if func:
                    result = func(df)
                else:
                    # Handle case where the specific name isn't in the map
                    result = {
                        "analysis_type": specific_analysis_name,
                        "status": "error",
                        "error_message": f"Analysis function for '{specific_analysis_name}' not found."
                    }
            else:
                # No specific analysis was selected, show general
                result = show_general_insights(df, "Specific Analysis Not Selected")
        else:
            # Default action if no category matches (e.g., initial load)
            result = show_general_insights(df, "Initial Overview")

    except Exception as e:
        # Broad exception handler for the dispatcher, using your script's structure
        return {
            "analysis_type": "Main Dispatcher",
            "status": "error",
            "error_message": f"An unexpected error occurred in main_backend: {str(e)}",
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {e}"]
        }

    return result

# Example of how to run this script (e.g., from a main app or for testing)
if __name__ == "__main__":
    print("Running Retail Analysis Script in test mode...")
