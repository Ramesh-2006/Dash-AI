import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')

# ========== UTILITY FUNCTIONS (Adapted from your template) ==========

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
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
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
            if match_result:
                match, score = match_result
                matched[target] = match if score >= 70 else None
            else:
                matched[target] = None
        except Exception:
            matched[target] = None
    
    return matched

def show_general_insights(df, analysis_name="General Retail Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
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
        
        # Create visualizations
        visualizations = {}
        
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
                fig_no_missing = go.Figure()
                fig_no_missing.add_annotation(
                    text="No Missing Values Found!",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="green")
                )
                fig_no_missing.update_layout(title="Missing Values Analysis")
                visualizations["missing_values"] = fig_no_missing.to_json()
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
                    top_10_cats = df[col].value_counts().head(10).index.tolist()
                    df_top_10 = df[df[col].isin(top_10_cats)]
                    fig_bar = px.bar(df_top_10, x=col, title=f'Distribution of {col} (Top 10)')
                    visualizations[f"{col}_distribution"] = fig_bar.to_json()
                except Exception:
                    pass

        # Generate insights
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns.",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns.",
        ]
        
        # Add missing columns warning if provided
        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append(f"The following columns are needed for the requested analysis '{analysis_name}' but weren't found:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"  - {col}{match_info}")
            insights.append("")
            insights.append("Showing General Analysis instead.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data).")
        else:
            insights.append("No duplicate rows found.")
        
        if len(columns_with_missing) > 0:
            insights.append(f"{len(columns_with_missing)} columns have missing values.")
        else:
            insights.append("No missing values found in the dataset.")
        
        insights.append(f"Generated {len(visualizations)} visualizations for data exploration.")
        
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
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred during general insights generation: {e}"],
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
            analysis_name,
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
    except Exception as fallback_error:
        print(f"General insights fallback also failed: {fallback_error}")
        general_insights_data = {
            "visualizations": {},
            "metrics": {},
            "insights": [f"Fallback to general insights failed: {fallback_error}"],
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

# ========== RETAIL ANALYSIS FUNCTIONS (API-Ready) ==========

def sales_performance_analysis(df):
    analysis_name = "Sales Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'revenue', 'transactions', 'product_id', 'sales_rep_id', 'sales_channel']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['date', 'revenue'], inplace=True)

        total_revenue = df['revenue'].sum()
        total_transactions = df['transactions'].sum() if 'transactions' in df.columns and pd.api.types.is_numeric_dtype(df['transactions']) else len(df)

        metrics = {
            "total_revenue": total_revenue,
            "total_transactions": total_transactions,
            "avg_transaction_value": total_revenue / total_transactions if total_transactions > 0 else 0
        }

        insights.append(f"Total revenue analyzed: ${total_revenue:,.2f}")
        insights.append(f"Total transactions: {total_transactions:,}")
        insights.append(f"Average transaction value: ${metrics['avg_transaction_value']:,.2f}")

        revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
        revenue_by_rep = df.groupby('sales_rep_id')['revenue'].sum().nlargest(10).reset_index()

        fig_channel = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title="Revenue Distribution by Sales Channel")
        fig_reps = px.bar(revenue_by_rep, x='sales_rep_id', y='revenue', title="Top 10 Sales Reps by Revenue")
        
        visualizations = {
            'revenue_by_channel': fig_channel.to_json(),
            'top_reps_by_revenue': fig_reps.to_json()
        }

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
    data = {}
    matched = {}

    try:
        expected = ['customer_id', 'transaction_date', 'revenue', 'order_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

        # RFM Analysis
        current_date = df['transaction_date'].max() + pd.Timedelta(days=1)
        rfm_df = df.groupby('customer_id').agg(
            recency=('transaction_date', lambda date: (current_date - date.max()).days),
            frequency=('order_id', 'nunique'),
            monetary=('revenue', 'sum')
        ).reset_index()

        rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
        rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

        metrics = {
            "num_customers_analyzed": len(rfm_df),
            "avg_recency": rfm_df['recency'].mean(),
            "avg_frequency": rfm_df['frequency'].mean(),
            "avg_monetary": rfm_df['monetary'].mean()
        }

        insights.append(f"Analyzed {len(rfm_df)} unique customers.")
        insights.append(f"Average Recency: {metrics['avg_recency']:.1f} days")
        insights.append(f"Average Frequency: {metrics['avg_frequency']:.1f} orders")
        insights.append(f"Average Monetary Value: ${metrics['avg_monetary']:,.2f}")

        fig_recency = px.histogram(rfm_df, x='recency', nbins=50, title='Distribution of Recency (Days)')
        fig_monetary = px.histogram(rfm_df, x='monetary', nbins=50, title='Distribution of Monetary Value')

        visualizations = {
            'recency_distribution': fig_recency.to_json(),
            'monetary_distribution': fig_monetary.to_json()
        }
        
        data['rfm_data'] = rfm_df.to_dict(orient='records')

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "data": convert_to_native_types(data),
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
            "data": data,
            "insights": insights
        }

def retail_transaction_analysis_by_product_and_country(df):
    analysis_name = "Retail Transaction Analysis by Product and Country"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'product_id', 'country', 'quantity', 'price', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['revenue'] = df['quantity'] * df['price']
        df.dropna(subset=['product_id', 'country', 'revenue', 'transaction_date'], inplace=True)

        top_products = df.groupby('product_id')['revenue'].sum().nlargest(10).reset_index()
        revenue_by_country = df.groupby('country')['revenue'].sum().reset_index()

        metrics = {
            "total_transactions": len(df),
            "total_revenue": df['revenue'].sum(),
            "num_unique_products": df['product_id'].nunique(),
            "num_unique_countries": df['country'].nunique()
        }

        insights.append(f"Total revenue: ${metrics['total_revenue']:,.2f} from {metrics['total_transactions']:,} transactions.")
        insights.append(f"Analysis covers {metrics['num_unique_products']:,} products across {metrics['num_unique_countries']} countries.")

        fig_product_revenue = px.bar(top_products, x='product_id', y='revenue', title='Top 10 Products by Revenue')
        fig_country_revenue = px.choropleth(revenue_by_country, locations='country', locationmode='country names',
                                            color='revenue', hover_name='country',
                                            color_continuous_scale=px.colors.sequential.Plasma,
                                            title='Total Revenue by Country')

        visualizations = {
            'top_products_revenue': fig_product_revenue.to_json(),
            'country_revenue_map': fig_country_revenue.to_json()
        }

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
        expected = ['order_id', 'item_id', 'order_status', 'product_name', 'quantity', 'price']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['item_revenue'] = df['quantity'] * df['price']
        df.dropna(subset=['order_id', 'order_status', 'product_name'], inplace=True)

        order_status_counts = df.groupby('order_status')['order_id'].nunique().reset_index(name='count')
        top_selling_items = df.groupby('product_name')['quantity'].sum().nlargest(10).reset_index()

        metrics = {
            "total_orders": df['order_id'].nunique(),
            "total_items_sold": df['quantity'].sum(),
            "num_unique_products": df['product_name'].nunique()
        }

        insights.append(f"Analyzed {metrics['total_orders']:,} unique orders.")
        insights.append(f"Total items sold: {metrics['total_items_sold']:,} across {metrics['num_unique_products']:,} products.")

        fig_order_status = px.pie(order_status_counts, names='order_status', values='count', title='Distribution of Order Status')
        fig_top_items = px.bar(top_selling_items, x='product_name', y='quantity', title='Top 10 Selling Items by Quantity')

        visualizations = {
            'order_status_distribution': fig_order_status.to_json(),
            'top_selling_items': fig_top_items.to_json()
        }

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
        expected = ['region', 'sales', 'customer_id', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['region', 'sales', 'customer_id'], inplace=True)

        sales_by_region = df.groupby('region')['sales'].sum().reset_index()
        customers_by_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

        metrics = {
            "total_sales_across_regions": df['sales'].sum(),
            "total_unique_customers": df['customer_id'].nunique(),
            "num_unique_regions": df['region'].nunique()
        }

        insights.append(f"Total sales: ${metrics['total_sales_across_regions']:,.2f}")
        insights.append(f"Total customers: {metrics['total_unique_customers']:,} across {metrics['num_unique_regions']} regions.")

        fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
        fig_customers_by_region = px.bar(customers_by_region, x='region', y='unique_customers', title='Number of Unique Customers by Region')

        visualizations = {
            'sales_by_region': fig_sales_by_region.to_json(),
            'customers_by_region': fig_customers_by_region.to_json()
        }

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

def sales_channel_performance(df):
    analysis_name = "Sales Channel Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_channel', 'revenue', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

        revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
        transactions_by_channel = df.groupby('sales_channel')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_channels": df['sales_channel'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_channels']} sales channels.")
        insights.append(f"Total revenue: ${metrics['total_revenue']:,.2f}")

        fig_revenue_by_channel = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title='Revenue Distribution by Sales Channel')
        fig_transactions_by_channel = px.bar(transactions_by_channel, x='sales_channel', y='num_transactions', title='Number of Transactions by Sales Channel')

        visualizations = {
            'revenue_by_channel': fig_revenue_by_channel.to_json(),
            'transactions_by_channel': fig_transactions_by_channel.to_json()
        }

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
        expected = ['country', 'sales', 'transaction_id', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['country', 'sales', 'transaction_id'], inplace=True)

        sales_by_country = df.groupby('country')['sales'].sum().reset_index()
        transactions_by_country = df.groupby('country')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_international_sales": df['sales'].sum(),
            "total_international_transactions": df['transaction_id'].nunique(),
            "num_international_countries": df['country'].nunique()
        }

        insights.append(f"Total sales: ${metrics['total_international_sales']:,.2f} from {metrics['num_international_countries']} countries.")

        fig_sales_by_country = px.choropleth(sales_by_country, locations='country', locationmode='country names',
                                            color='sales', hover_name='country',
                                            color_continuous_scale=px.colors.sequential.Plasma,
                                            title='Total Sales by Country (International)')
        fig_transactions_by_country = px.bar(transactions_by_country.nlargest(10, 'num_transactions'), x='country', y='num_transactions', title='Top 10 Countries by Number of Transactions')

        visualizations = {
            'sales_by_country_map': fig_sales_by_country.to_json(),
            'transactions_by_country_bar': fig_transactions_by_country.to_json()
        }

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

def invoice_type_and_customer_purchase_pattern(df):
    analysis_name = "Invoice Type and Customer Purchase Pattern"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoice_id', 'invoice_type', 'customer_id', 'transaction_date', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['invoice_id', 'invoice_type', 'customer_id', 'revenue'], inplace=True)

        revenue_by_invoice_type = df.groupby('invoice_type')['revenue'].sum().reset_index()
        customers_by_invoice_type = df.groupby('invoice_type')['customer_id'].nunique().reset_index(name='unique_customers')

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "num_unique_invoice_types": df['invoice_type'].nunique(),
            "total_unique_customers": df['customer_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_invoice_types']} invoice types.")

        fig_revenue_invoice_type = px.pie(revenue_by_invoice_type, names='invoice_type', values='revenue', title='Revenue Distribution by Invoice Type')
        fig_customers_invoice_type = px.bar(customers_by_invoice_type, x='invoice_type', y='unique_customers', title='Number of Unique Customers by Invoice Type')

        visualizations = {
            'revenue_by_invoice_type': fig_revenue_invoice_type.to_json(),
            'customers_by_invoice_type': fig_customers_invoice_type.to_json()
        }

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

def order_delivery_and_customer_location(df):
    analysis_name = "Order Delivery and Customer Location"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'customer_location', 'delivery_status', 'delivery_time_days', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['order_id', 'customer_location', 'delivery_status'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df.dropna(subset=['order_id', 'customer_location', 'delivery_status'], inplace=True)

        delivery_status_counts = df.groupby('delivery_status')['order_id'].nunique().reset_index(name='count')

        metrics = {
            "total_orders": df['order_id'].nunique(),
            "num_unique_customer_locations": df['customer_location'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['total_orders']:,} orders across {metrics['num_unique_customer_locations']} locations.")

        fig_delivery_status = px.pie(delivery_status_counts, names='delivery_status', values='count', title='Distribution of Order Delivery Status')
        visualizations = {
            'delivery_status_distribution': fig_delivery_status.to_json(),
        }

        if 'delivery_time_days' in df.columns:
            df['delivery_time_days'] = pd.to_numeric(df['delivery_time_days'], errors='coerce')
            if not df['delivery_time_days'].isnull().all():
                avg_delivery_time_by_location = df.groupby('customer_location')['delivery_time_days'].mean().reset_index()
                avg_delivery_time_by_location = avg_delivery_time_by_location.nlargest(10, 'delivery_time_days') # Top 10 for visualization
                
                metrics["avg_delivery_time_overall"] = df['delivery_time_days'].mean()
                insights.append(f"Overall average delivery time: {metrics['avg_delivery_time_overall']:.2f} days.")
                
                fig_avg_delivery_time = px.bar(avg_delivery_time_by_location, x='customer_location', y='delivery_time_days', title='Average Delivery Time by Customer Location (Top 10 Slowest)')
                visualizations['avg_delivery_time_by_location'] = fig_avg_delivery_time.to_json()
            else:
                insights.append("Delivery time data ('delivery_time_days') found but contains no valid numeric data.")
        else:
            insights.append("Delivery time analysis skipped: 'delivery_time_days' column not found.")


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

def time_of_day_sales_pattern(df):
    analysis_name = "Time of Day Sales Pattern"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['transaction_date', 'revenue'], inplace=True)

        df['hour_of_day'] = df['transaction_date'].dt.hour
        sales_by_hour = df.groupby('hour_of_day')['revenue'].sum().reset_index()
        transactions_by_hour = df.groupby('hour_of_day').size().reset_index(name='num_transactions')

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "total_transactions": len(df),
            "peak_sales_hour": sales_by_hour.loc[sales_by_hour['revenue'].idxmax()]['hour_of_day'],
            "peak_transaction_hour": transactions_by_hour.loc[transactions_by_hour['num_transactions'].idxmax()]['hour_of_day']
        }

        insights.append(f"Peak sales hour: {metrics['peak_sales_hour']}:00")
        insights.append(f"Peak transaction hour: {metrics['peak_transaction_hour']}:00")

        fig_sales_by_hour = px.line(sales_by_hour, x='hour_of_day', y='revenue', title='Sales by Hour of Day')
        fig_transactions_by_hour = px.bar(transactions_by_hour, x='hour_of_day', y='num_transactions', title='Number of Transactions by Hour of Day')
        
        fig_sales_by_hour.update_xaxes(type='category') # Treat hours as categories
        fig_transactions_by_hour.update_xaxes(type='category')

        visualizations = {
            'sales_by_hour': fig_sales_by_hour.to_json(),
            'transactions_by_hour': fig_transactions_by_hour.to_json()
        }

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

def customer_order_and_status_tracking(df):
    analysis_name = "Customer Order and Status Tracking"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'order_id', 'order_status', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df.dropna(subset=['customer_id', 'order_id', 'order_status'], inplace=True)

        orders_per_customer = df.groupby('customer_id')['order_id'].nunique().nlargest(10).reset_index(name='num_orders')
        order_status_distribution = df.groupby('order_status')['order_id'].nunique().reset_index(name='count')

        metrics = {
            "total_unique_customers": df['customer_id'].nunique(),
            "total_unique_orders": df['order_id'].nunique()
        }
        
        insights.append(f"{metrics['total_unique_customers']:,} customers placed {metrics['total_unique_orders']:,} orders.")

        fig_orders_per_customer = px.bar(orders_per_customer, x='customer_id', y='num_orders', title='Top 10 Customers by Number of Orders')
        fig_order_status_distribution = px.pie(order_status_distribution, names='order_status', values='count', title='Overall Order Status Distribution')

        visualizations = {
            'orders_per_customer': fig_orders_per_customer.to_json(),
            'order_status_distribution': fig_order_status_distribution.to_json()
        }

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

def payment_method_preference(df):
    analysis_name = "Payment Method Preference"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['payment_method', 'transaction_id', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['payment_method', 'revenue'], inplace=True)

        revenue_by_payment_method = df.groupby('payment_method')['revenue'].sum().reset_index()
        transactions_by_payment_method = df.groupby('payment_method')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_payment_methods": df['payment_method'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_payment_methods']} payment methods.")
        
        most_popular_method_rev = revenue_by_payment_method.loc[revenue_by_payment_method['revenue'].idxmax()]
        insights.append(f"Most popular method by revenue: {most_popular_method_rev['payment_method']} (${most_popular_method_rev['revenue']:,.2f})")

        fig_revenue_payment = px.pie(revenue_by_payment_method, names='payment_method', values='revenue', title='Revenue Distribution by Payment Method')
        fig_transactions_payment = px.bar(transactions_by_payment_method, x='payment_method', y='num_transactions', title='Number of Transactions by Payment Method')

        visualizations = {
            'revenue_by_payment_method': fig_revenue_payment.to_json(),
            'transactions_by_payment_method': fig_transactions_payment.to_json()
        }

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

def product_return_rate(df):
    analysis_name = "Product Return Rate"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'product_id', 'return_status', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['order_id', 'product_id', 'return_status', 'quantity'], inplace=True)
        
        # Standardize return status
        df['return_status_clean'] = df['return_status'].astype(str).str.lower()
        is_returned = df['return_status_clean'] == 'returned'

        total_items = df['quantity'].sum()
        returned_items = df[is_returned]['quantity'].sum()
        return_rate_overall = (returned_items / total_items) * 100 if total_items > 0 else 0

        returned_products = df[is_returned].groupby('product_id')['quantity'].sum().nlargest(10).reset_index(name='returned_quantity')
        return_status_counts = df['return_status'].value_counts().reset_index()
        return_status_counts.columns = ['return_status', 'count']

        metrics = {
            "total_items_sold": total_items,
            "total_items_returned": returned_items,
            "overall_return_rate_percent": return_rate_overall
        }
        
        insights.append(f"Overall return rate: {return_rate_overall:.2f}% ({returned_items:,} items returned out of {total_items:,}).")

        fig_return_rate_pie = px.pie(return_status_counts, names='return_status', values='count', title='Overall Return Status Distribution')
        fig_top_returned_products = px.bar(returned_products, x='product_id', y='returned_quantity', title='Top 10 Products by Returned Quantity')

        visualizations = {
            'overall_return_status': fig_return_rate_pie.to_json(),
            'top_returned_products': fig_top_returned_products.to_json()
        }

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

def promotional_code_effectiveness(df):
    analysis_name = "Promotional Code Effectiveness"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['promotion_code', 'revenue', 'transaction_id', 'is_promotional_sale']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['revenue'] if matched.get(col) is None]
        
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        if matched.get('is_promotional_sale') is None and matched.get('promotion_code') is None:
             return create_fallback_response(analysis_name, ['is_promotional_sale', 'promotion_code'], matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['revenue'], inplace=True)

        if 'is_promotional_sale' not in df.columns:
            df['is_promotional_sale'] = df['promotion_code'].notna().astype(int)
        else:
            # Standardize 'is_promotional_sale' to 0/1
            if df['is_promotional_sale'].dtype == 'object':
                 df['is_promotional_sale'] = df['is_promotional_sale'].astype(str).str.lower().map({'true': 1, '1': 1, 'yes': 1, 'false': 0, '0': 0, 'no': 0}).fillna(0)
            df['is_promotional_sale'] = pd.to_numeric(df['is_promotional_sale'], errors='coerce').fillna(0).astype(int)


        revenue_by_promotion_status = df.groupby('is_promotional_sale')['revenue'].sum().reset_index()
        revenue_by_promotion_status['is_promotional_sale'] = revenue_by_promotion_status['is_promotional_sale'].map({0: 'Non-Promotional', 1: 'Promotional'})

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "promotional_revenue": df[df['is_promotional_sale'] == 1]['revenue'].sum(),
            "non_promotional_revenue": df[df['is_promotional_sale'] == 0]['revenue'].sum()
        }
        
        insights.append(f"Total revenue: ${metrics['total_revenue']:,.2f}")
        insights.append(f"Promotional sales accounted for ${metrics['promotional_revenue']:,.2f}.")
        
        fig_revenue_promotion_status = px.pie(revenue_by_promotion_status, names='is_promotional_sale', values='revenue', title='Revenue by Promotional Status')
        visualizations = {
            'revenue_by_promotion_status': fig_revenue_promotion_status.to_json(),
        }

        if 'promotion_code' in df.columns:
            top_promotion_codes = df[df['is_promotional_sale']==1].groupby('promotion_code')['revenue'].sum().nlargest(10).reset_index()
            if not top_promotion_codes.empty:
                fig_top_promotion_codes = px.bar(top_promotion_codes, x='promotion_code', y='revenue', title='Top 10 Promotion Codes by Revenue')
                visualizations['top_promotion_codes'] = fig_top_promotion_codes.to_json()
                insights.append("Generated plot for top 10 promotion codes.")
            else:
                insights.append("No specific promotion codes found for top performance analysis.")
        else:
            insights.append("Top promotion code analysis skipped: 'promotion_code' column not found.")

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

def discount_impact_on_sales(df):
    analysis_name = "Discount Impact on Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['discount_amount', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce')
        df.dropna(subset=['sales', 'discount_amount'], inplace=True)

        df['has_discount'] = (df['discount_amount'] > 0).astype(int)
        sales_by_discount_status = df.groupby('has_discount')['sales'].sum().reset_index()
        sales_by_discount_status['has_discount'] = sales_by_discount_status['has_discount'].map({0: 'No Discount', 1: 'With Discount'})

        avg_sales_by_discount = df.groupby('has_discount')['sales'].mean().reset_index()
        avg_sales_by_discount['has_discount'] = avg_sales_by_discount['has_discount'].map({0: 'No Discount', 1: 'With Discount'})

        metrics = {
            "total_sales": df['sales'].sum(),
            "sales_with_discount": df[df['has_discount'] == 1]['sales'].sum(),
            "sales_without_discount": df[df['has_discount'] == 0]['sales'].sum(),
            "avg_sales_with_discount": avg_sales_by_discount[avg_sales_by_discount['has_discount'] == 'With Discount']['sales'].values[0],
            "avg_sales_no_discount": avg_sales_by_discount[avg_sales_by_discount['has_discount'] == 'No Discount']['sales'].values[0]
        }
        
        insights.append(f"Sales with discount: ${metrics['sales_with_discount']:,.2f}")
        insights.append(f"Sales without discount: ${metrics['sales_without_discount']:,.2f}")
        insights.append(f"Avg. sale with discount: ${metrics['avg_sales_with_discount']:,.2f} vs. Avg. sale with no discount: ${metrics['avg_sales_no_discount']:,.2f}")

        fig_sales_by_discount = px.pie(sales_by_discount_status, names='has_discount', values='sales', title='Sales Distribution by Discount Presence')
        fig_avg_sales_discount = px.bar(avg_sales_by_discount, x='has_discount', y='sales', title='Average Sales per Transaction by Discount Presence')

        visualizations = {
            'sales_by_discount': fig_sales_by_discount.to_json(),
            'avg_sales_by_discount': fig_avg_sales_discount.to_json()
        }

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

def product_cost_and_sales_margin(df):
    analysis_name = "Product Cost and Sales Margin"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'sales_price', 'cost_price', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales_price'] = pd.to_numeric(df['sales_price'], errors='coerce')
        df['cost_price'] = pd.to_numeric(df['cost_price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['sales_price', 'cost_price', 'quantity'], inplace=True)

        df['total_sales'] = df['sales_price'] * df['quantity']
        df['total_cost'] = df['cost_price'] * df['quantity']
        df['gross_profit'] = df['total_sales'] - df['total_cost']

        gross_profit_by_product = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()
        overall_gross_profit_margin = (df['gross_profit'].sum() / df['total_sales'].sum()) * 100 if df['total_sales'].sum() > 0 else 0

        metrics = {
            "total_sales": df['total_sales'].sum(),
            "total_cost": df['total_cost'].sum(),
            "total_gross_profit": df['gross_profit'].sum(),
            "overall_gross_profit_margin_percent": overall_gross_profit_margin
        }
        
        insights.append(f"Total Sales: ${metrics['total_sales']:,.2f}")
        insights.append(f"Total Cost: ${metrics['total_cost']:,.2f}")
        insights.append(f"Total Gross Profit: ${metrics['total_gross_profit']:,.2f}")
        insights.append(f"Overall Gross Profit Margin: {metrics['overall_gross_profit_margin_percent']:.2f}%")

        fig_gross_profit_product = px.bar(gross_profit_by_product, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')
        
        overall_financials = pd.DataFrame({
            'Metric': ['Total Gross Profit', 'Total Cost'],
            'Value': [metrics['total_gross_profit'], metrics['total_cost']]
        })
        fig_overall_margin_breakdown = px.pie(overall_financials, names='Metric', values='Value', title='Overall Profit vs. Cost Breakdown', hole=0.4)

        visualizations = {
            'gross_profit_by_product': fig_gross_profit_product.to_json(),
            'overall_margin_breakdown': fig_overall_margin_breakdown.to_json()
        }

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

def store_level_sales_performance(df):
    analysis_name = "Store Level Sales Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['store_id', 'sales', 'transaction_id', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['store_id', 'sales'], inplace=True)

        sales_by_store = df.groupby('store_id')['sales'].sum().reset_index()
        transactions_by_store = df.groupby('store_id')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales_across_stores": df['sales'].sum(),
            "total_transactions_across_stores": df['transaction_id'].nunique(),
            "num_unique_stores": df['store_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_stores']} stores.")
        top_store = sales_by_store.loc[sales_by_store['sales'].idxmax()]
        insights.append(f"Top store by sales: {top_store['store_id']} (${top_store['sales']:,.2f})")

        fig_sales_by_store = px.bar(sales_by_store.nlargest(10, 'sales'), x='store_id', y='sales', title='Top 10 Stores by Sales')
        fig_transactions_by_store = px.bar(transactions_by_store.nlargest(10, 'num_transactions'), x='store_id', y='num_transactions', title='Top 10 Stores by Number of Transactions')

        visualizations = {
            'sales_by_store': fig_sales_by_store.to_json(),
            'transactions_by_store': fig_transactions_by_store.to_json()
        }

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

def product_category_analysis(df): # Renamed from product_category
    analysis_name = "Product Category Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_category', 'sales', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['product_category', 'sales'], inplace=True)

        sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()
        quantity_by_category = df.groupby('product_category')['quantity'].sum().reset_index()

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_quantity_sold": df['quantity'].sum(),
            "num_unique_product_categories": df['product_category'].nunique()
        }

        insights.append(f"Analyzed {metrics['num_unique_product_categories']} product categories.")
        top_cat = sales_by_category.loc[sales_by_category['sales'].idxmax()]
        insights.append(f"Top category by sales: {top_cat['product_category']} (${top_cat['sales']:,.2f})")

        fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
        fig_quantity_by_category = px.bar(quantity_by_category.nlargest(10, 'quantity'), x='product_category', y='quantity', title='Top 10 Product Categories by Quantity Sold')

        visualizations = {
            'sales_by_category': fig_sales_by_category.to_json(),
            'quantity_by_category': fig_quantity_by_category.to_json()
        }

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

def weekly_sales_trend(df):
    analysis_name = "Weekly Sales Trend"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales'], inplace=True)

        df['week_start'] = df['transaction_date'].dt.to_period('W').dt.start_time
        weekly_sales = df.groupby('week_start')['sales'].sum().reset_index()
        weekly_sales = weekly_sales.sort_values('week_start')

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_weeks_in_data": weekly_sales.shape[0],
            "avg_weekly_sales": weekly_sales['sales'].mean()
        }
        
        insights.append(f"Analyzed {metrics['num_weeks_in_data']} weeks of data.")
        insights.append(f"Average weekly sales: ${metrics['avg_weekly_sales']:,.2f}")

        fig_weekly_sales_line = px.line(weekly_sales, x='week_start', y='sales', title='Weekly Sales Trend')
        fig_weekly_sales_bar = px.bar(weekly_sales, x='week_start', y='sales', title='Weekly Sales Trend (Bar Chart)')

        visualizations = {
            'weekly_sales_line': fig_weekly_sales_line.to_json(),
            'weekly_sales_bar': fig_weekly_sales_bar.to_json()
        }

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

def yearly_sales_performance(df):
    analysis_name = "Yearly Sales Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales'], inplace=True)

        df['year'] = df['transaction_date'].dt.year
        yearly_sales = df.groupby('year')['sales'].sum().reset_index()
        yearly_sales = yearly_sales.sort_values('year')

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_years_in_data": yearly_sales.shape[0],
            "avg_yearly_sales": yearly_sales['sales'].mean()
        }
        
        insights.append(f"Analyzed {metrics['num_years_in_data']} years of data.")
        insights.append(f"Average yearly sales: ${metrics['avg_yearly_sales']:,.2f}")

        fig_yearly_sales_bar = px.bar(yearly_sales, x='year', y='sales', title='Yearly Sales Performance')
        fig_yearly_sales_line = px.line(yearly_sales, x='year', y='sales', title='Yearly Sales Trend')
        
        fig_yearly_sales_bar.update_xaxes(type='category')
        fig_yearly_sales_line.update_xaxes(type='category')

        visualizations = {
            'yearly_sales_bar': fig_yearly_sales_bar.to_json(),
            'yearly_sales_line': fig_yearly_sales_line.to_json()
        }

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

def monthly_sales_trend(df):
    analysis_name = "Monthly Sales Trend"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales'], inplace=True)

        df['month_year'] = df['transaction_date'].dt.to_period('M').dt.start_time
        monthly_sales = df.groupby('month_year')['sales'].sum().reset_index()
        monthly_sales = monthly_sales.sort_values('month_year')

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_months_in_data": monthly_sales.shape[0],
            "avg_monthly_sales": monthly_sales['sales'].mean()
        }

        insights.append(f"Analyzed {metrics['num_months_in_data']} months of data.")
        insights.append(f"Average monthly sales: ${metrics['avg_monthly_sales']:,.2f}")

        fig_monthly_sales_line = px.line(monthly_sales, x='month_year', y='sales', title='Monthly Sales Trend')
        fig_monthly_sales_bar = px.bar(monthly_sales, x='month_year', y='sales', title='Monthly Sales Trend (Bar Chart)')

        visualizations = {
            'monthly_sales_line': fig_monthly_sales_line.to_json(),
            'monthly_sales_bar': fig_monthly_sales_bar.to_json()
        }

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

def week_over_week_sales_growth(df):
    analysis_name = "Week-over-Week Sales Growth"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales'], inplace=True)

        df['week_start'] = df['transaction_date'].dt.to_period('W').dt.start_time
        weekly_sales = df.groupby('week_start')['sales'].sum().reset_index()
        weekly_sales = weekly_sales.sort_values('week_start')

        weekly_sales['previous_week_sales'] = weekly_sales['sales'].shift(1)
        weekly_sales['wow_growth'] = ((weekly_sales['sales'] - weekly_sales['previous_week_sales']) / weekly_sales['previous_week_sales']) * 100
        weekly_sales.dropna(subset=['wow_growth'], inplace=True)

        metrics = {
            "avg_wow_growth_percent": weekly_sales['wow_growth'].mean() if not weekly_sales.empty else 0,
            "max_wow_growth_percent": weekly_sales['wow_growth'].max() if not weekly_sales.empty else 0,
            "min_wow_growth_percent": weekly_sales['wow_growth'].min() if not weekly_sales.empty else 0
        }
        
        insights.append(f"Average Week-over-Week Growth: {metrics['avg_wow_growth_percent']:.2f}%")

        fig_wow_growth_line = px.line(weekly_sales, x='week_start', y='wow_growth', title='Week-over-Week Sales Growth (%)')
        fig_wow_growth_bar = px.bar(weekly_sales, x='week_start', y='wow_growth', title='Week-over-Week Sales Growth (%) (Bar Chart)')

        visualizations = {
            'wow_growth_line': fig_wow_growth_line.to_json(),
            'wow_growth_bar': fig_wow_growth_bar.to_json()
        }

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

def holiday_sales_impact(df):
    analysis_name = "Holiday Sales Impact"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'sales', 'is_holiday']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales', 'is_holiday'], inplace=True)

        # Ensure 'is_holiday' is boolean or 0/1
        if df['is_holiday'].dtype == 'object':
            df['is_holiday'] = df['is_holiday'].astype(str).str.lower().map({'true': 1, '1': 1, 'yes': 1, 'false': 0, '0': 0, 'no': 0}).fillna(0)
        df['is_holiday'] = pd.to_numeric(df['is_holiday'], errors='coerce').fillna(0).astype(int)


        sales_by_holiday = df.groupby('is_holiday')['sales'].sum().reset_index()
        sales_by_holiday['is_holiday'] = sales_by_holiday['is_holiday'].map({0: 'Non-Holiday', 1: 'Holiday'})

        df['date_only'] = df['transaction_date'].dt.date
        daily_sales = df.groupby(['date_only', 'is_holiday'])['sales'].sum().reset_index()
        avg_daily_sales_by_holiday = daily_sales.groupby('is_holiday')['sales'].mean().reset_index()
        avg_daily_sales_by_holiday['is_holiday'] = avg_daily_sales_by_holiday['is_holiday'].map({0: 'Non-Holiday', 1: 'Holiday'})
        
        avg_daily_holiday = avg_daily_sales_by_holiday[avg_daily_sales_by_holiday['is_holiday'] == 'Holiday']['sales'].values[0]
        avg_daily_non_holiday = avg_daily_sales_by_holiday[avg_daily_sales_by_holiday['is_holiday'] == 'Non-Holiday']['sales'].values[0]

        metrics = {
            "total_holiday_sales": df[df['is_holiday'] == 1]['sales'].sum(),
            "total_non_holiday_sales": df[df['is_holiday'] == 0]['sales'].sum(),
            "num_holiday_days": df[df['is_holiday'] == 1]['date_only'].nunique(),
            "num_non_holiday_days": df[df['is_holiday'] == 0]['date_only'].nunique(),
            "avg_daily_sales_holiday": avg_daily_holiday,
            "avg_daily_sales_non_holiday": avg_daily_non_holiday
        }
        
        insights.append(f"Average daily sales on holidays: ${metrics['avg_daily_sales_holiday']:,.2f}")
        insights.append(f"Average daily sales on non-holidays: ${metrics['avg_daily_sales_non_holiday']:,.2f}")

        fig_sales_by_holiday = px.bar(sales_by_holiday, x='is_holiday', y='sales', title='Total Sales: Holiday vs. Non-Holiday')
        fig_avg_daily_sales_holiday = px.bar(avg_daily_sales_by_holiday, x='is_holiday', y='sales', title='Average Daily Sales: Holiday vs. Non-Holiday')

        visualizations = {
            'sales_by_holiday': fig_sales_by_holiday.to_json(),
            'avg_daily_sales_holiday': fig_avg_daily_sales_holiday.to_json()
        }

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

def customer_type_analysis(df):
    analysis_name = "Customer Type Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_type', 'sales', 'customer_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['customer_type', 'sales'], inplace=True)

        sales_by_customer_type = df.groupby('customer_type')['sales'].sum().reset_index()
        unique_customers_by_type = df.groupby('customer_type')['customer_id'].nunique().reset_index(name='unique_customers')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_unique_customers": df['customer_id'].nunique(),
            "num_customer_types": df['customer_type'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_customer_types']} customer types.")
        top_type = sales_by_customer_type.loc[sales_by_customer_type['sales'].idxmax()]
        insights.append(f"Top customer type by sales: {top_type['customer_type']} (${top_type['sales']:,.2f})")

        fig_sales_by_customer_type = px.pie(sales_by_customer_type, names='customer_type', values='sales', title='Sales Distribution by Customer Type')
        fig_unique_customers_by_type = px.bar(unique_customers_by_type, x='customer_type', y='unique_customers', title='Number of Unique Customers by Type')

        visualizations = {
            'sales_by_customer_type': fig_sales_by_customer_type.to_json(),
            'unique_customers_by_type': fig_unique_customers_by_type.to_json()
        }

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

def online_vs_offline_sales(df):
    analysis_name = "Online vs. Offline Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_channel', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_channel', 'sales'], inplace=True)

        df['channel_type'] = df['sales_channel'].apply(lambda x: 'Online' if 'online' in str(x).lower() else ('Offline' if 'store' in str(x).lower() or 'physical' in str(x).lower() or 'offline' in str(x).lower() else 'Other'))

        sales_by_channel_type = df.groupby('channel_type')['sales'].sum().reset_index()
        transactions_by_channel_type = df.groupby('channel_type')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "online_sales": sales_by_channel_type[sales_by_channel_type['channel_type'] == 'Online']['sales'].sum(),
            "offline_sales": sales_by_channel_type[sales_by_channel_type['channel_type'] == 'Offline']['sales'].sum(),
            "other_sales": sales_by_channel_type[sales_by_channel_type['channel_type'] == 'Other']['sales'].sum()
        }
        
        insights.append(f"Online sales: ${metrics['online_sales']:,.2f}")
        insights.append(f"Offline sales: ${metrics['offline_sales']:,.2f}")

        fig_sales_channel_type = px.pie(sales_by_channel_type, names='channel_type', values='sales', title='Sales Distribution: Online vs. Offline')
        fig_transactions_channel_type = px.bar(transactions_by_channel_type, x='channel_type', y='num_transactions', title='Number of Transactions: Online vs. Offline')

        visualizations = {
            'sales_channel_type': fig_sales_channel_type.to_json(),
            'transactions_channel_type': fig_transactions_channel_type.to_json()
        }

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

def regional_customer_purchase(df):
    analysis_name = "Regional Customer Purchase"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'region', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['customer_id', 'region', 'sales'], inplace=True)

        sales_by_region = df.groupby('region')['sales'].sum().reset_index()
        customers_by_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

        metrics = {
            "total_sales_across_regions": df['sales'].sum(),
            "total_unique_customers": df['customer_id'].nunique(),
            "num_unique_regions": df['region'].nunique()
        }
        
        insights.append(f"Total sales: ${metrics['total_sales_across_regions']:,.2f} from {metrics['total_unique_customers']:,} customers in {metrics['num_unique_regions']} regions.")

        fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
        fig_customers_by_region = px.bar(customers_by_region, x='region', y='unique_customers', title='Number of Unique Customers by Region')

        visualizations = {
            'sales_by_region': fig_sales_by_region.to_json(),
            'customers_by_region': fig_customers_by_region.to_json()
        }

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

def delivery_method_preference(df):
    analysis_name = "Delivery Method Preference"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['delivery_method', 'order_id', 'customer_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df.dropna(subset=['delivery_method', 'order_id'], inplace=True)

        orders_by_delivery_method = df.groupby('delivery_method')['order_id'].nunique().reset_index(name='num_orders')
        
        metrics = {
            "total_orders": df['order_id'].nunique(),
            "num_delivery_methods": df['delivery_method'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['total_orders']:,} orders across {metrics['num_delivery_methods']} delivery methods.")
        top_method = orders_by_delivery_method.loc[orders_by_delivery_method['num_orders'].idxmax()]
        insights.append(f"Top method: {top_method['delivery_method']} ({top_method['num_orders']:,} orders)")

        fig_orders_delivery_method = px.pie(orders_by_delivery_method, names='delivery_method', values='num_orders', title='Orders Distribution by Delivery Method')
        visualizations = {
            'orders_by_delivery_method': fig_orders_delivery_method.to_json(),
        }

        if 'customer_id' in df.columns:
            customers_by_delivery_method = df.groupby('delivery_method')['customer_id'].nunique().reset_index(name='unique_customers')
            fig_customers_delivery_method = px.bar(customers_by_delivery_method, x='delivery_method', y='unique_customers', title='Unique Customers by Delivery Method')
            visualizations['customers_by_delivery_method'] = fig_customers_delivery_method.to_json()
        else:
            insights.append("Customer analysis by delivery method skipped: 'customer_id' not found.")

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

def point_of_sale_transaction(df):
    analysis_name = "Point of Sale Transaction"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'transaction_date', 'store_id', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['transaction_id', 'transaction_date', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_id', 'sales', 'transaction_date'], inplace=True)

        daily_sales_pos = df.groupby(df['transaction_date'].dt.date)['sales'].sum().reset_index()
        daily_sales_pos.columns = ['date', 'sales']

        metrics = {
            "total_pos_sales": df['sales'].sum(),
            "total_pos_transactions": df['transaction_id'].nunique(),
            "avg_sales_per_transaction": df['sales'].sum() / df['transaction_id'].nunique() if df['transaction_id'].nunique() > 0 else 0
        }
        
        insights.append(f"Total POS sales: ${metrics['total_pos_sales']:,.2f}")
        insights.append(f"Total POS transactions: {metrics['total_pos_transactions']:,}")

        fig_daily_sales_pos = px.line(daily_sales_pos, x='date', y='sales', title='Daily Sales Trend (Point of Sale)')
        visualizations = {
            'daily_sales_pos': fig_daily_sales_pos.to_json(),
        }

        if 'store_id' in df.columns:
            transactions_by_store = df.groupby('store_id')['transaction_id'].nunique().reset_index(name='num_transactions')
            transactions_by_store = transactions_by_store.nlargest(10, 'num_transactions')
            if not transactions_by_store.empty:
                fig_transactions_by_store = px.bar(transactions_by_store, x='store_id', y='num_transactions', title='Top 10 Stores by Number of POS Transactions')
                visualizations['transactions_by_store'] = fig_transactions_by_store.to_json()
                insights.append("Generated plot for transactions by store.")
            else:
                 insights.append("Store ID data found, but no transactions to plot.")
        else:
            insights.append("Store-level analysis skipped: 'store_id' column not found.")


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

def sales_tax_analysis(df):
    analysis_name = "Sales Tax Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales', 'tax_amount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df['tax_amount'] = pd.to_numeric(df['tax_amount'], errors='coerce')
        df.dropna(subset=['sales', 'tax_amount'], inplace=True)

        total_sales = df['sales'].sum()
        total_tax = df['tax_amount'].sum()
        overall_tax_rate = (total_tax / total_sales) * 100 if total_sales > 0 else 0

        metrics = {
            "total_sales_subject_to_tax": total_sales,
            "total_tax_collected": total_tax,
            "overall_tax_rate_percent": overall_tax_rate
        }
        
        insights.append(f"Total sales: ${total_sales:,.2f} | Total tax: ${total_tax:,.2f}")
        insights.append(f"Implied overall tax rate: {overall_tax_rate:.2f}%")

        fig_sales_vs_tax = px.scatter(df, x='sales', y='tax_amount', title='Sales vs. Tax Amount',
                                    trendline='ols', trendline_color_discrete='red')
        fig_tax_distribution = px.histogram(df, x='tax_amount', nbins=50, title='Distribution of Tax Amounts')

        visualizations = {
            'sales_vs_tax_scatter': fig_sales_vs_tax.to_json(),
            'tax_distribution_histogram': fig_tax_distribution.to_json()
        }

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

def sales_organization_analysis(df): # Renamed from sales_organization
    analysis_name = "Sales Organization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_organization', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_organization', 'sales'], inplace=True)

        sales_by_org = df.groupby('sales_organization')['sales'].sum().reset_index()
        transactions_by_org = df.groupby('sales_organization')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_organizations": df['sales_organization'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_organizations']} sales organizations.")
        top_org = sales_by_org.loc[sales_by_org['sales'].idxmax()]
        insights.append(f"Top organization: {top_org['sales_organization']} (${top_org['sales']:,.2f} in sales)")

        fig_sales_by_org = px.bar(sales_by_org.nlargest(10, 'sales'), x='sales_organization', y='sales', title='Top 10 Sales Organizations by Sales')
        fig_transactions_by_org = px.bar(transactions_by_org.nlargest(10, 'num_transactions'), x='sales_organization', y='num_transactions', title='Top 10 Sales Organizations by Transactions')

        visualizations = {
            'sales_by_organization': fig_sales_by_org.to_json(),
            'transactions_by_organization': fig_transactions_by_org.to_json()
        }

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

def order_payment_status(df):
    analysis_name = "Order Payment Status"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'payment_status', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['order_id', 'payment_status', 'revenue'], inplace=True)

        payment_status_counts = df.groupby('payment_status')['order_id'].nunique().reset_index(name='count')
        revenue_by_payment_status = df.groupby('payment_status')['revenue'].sum().reset_index()

        metrics = {
            "total_orders": df['order_id'].nunique(),
            "total_revenue": df['revenue'].sum(),
            "num_payment_statuses": df['payment_status'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['total_orders']:,} orders.")
        
        revenue_paid = revenue_by_payment_status[revenue_by_payment_status['payment_status'].astype(str).str.lower() == 'paid']['revenue'].sum()
        revenue_pending = revenue_by_payment_status[revenue_by_payment_status['payment_status'].astype(str).str.lower() == 'pending']['revenue'].sum()
        
        insights.append(f"Revenue (Paid): ${revenue_paid:,.2f}")
        insights.append(f"Revenue (Pending): ${revenue_pending:,.2f}")

        fig_payment_status_pie = px.pie(payment_status_counts, names='payment_status', values='count', title='Distribution of Order Payment Status')
        fig_revenue_payment_status = px.bar(revenue_by_payment_status, x='payment_status', y='revenue', title='Revenue by Payment Status')

        visualizations = {
            'payment_status_distribution': fig_payment_status_pie.to_json(),
            'revenue_by_payment_status': fig_revenue_payment_status.to_json()
        }

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

def product_sales_and_cost(df):
    analysis_name = "Product Sales and Cost"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'sales', 'cost']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df.dropna(subset=['product_id', 'sales', 'cost'], inplace=True)

        product_summary = df.groupby('product_id').agg(
            total_sales=('sales', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        product_summary['gross_profit'] = product_summary['total_sales'] - product_summary['total_cost']

        top_sales_products = product_summary.nlargest(10, 'total_sales')
        top_profit_products = product_summary.nlargest(10, 'gross_profit')

        metrics = {
            "overall_total_sales": product_summary['total_sales'].sum(),
            "overall_total_cost": product_summary['total_cost'].sum(),
            "overall_gross_profit": product_summary['gross_profit'].sum(),
            "overall_profit_margin": (product_summary['gross_profit'].sum() / product_summary['total_sales'].sum()) * 100 if product_summary['total_sales'].sum() > 0 else 0
        }
        
        insights.append(f"Overall Profit Margin: {metrics['overall_profit_margin']:.2f}%")
        top_profit_prod_id = top_profit_products.iloc[0]['product_id']
        insights.append(f"Most profitable product: {top_profit_prod_id}")

        fig_top_sales_products = px.bar(top_sales_products, x='product_id', y='total_sales', title='Top 10 Products by Sales')
        fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

        visualizations = {
            'top_sales_products': fig_top_sales_products.to_json(),
            'top_profit_products': fig_top_profit_products.to_json()
        }

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

def customer_transaction_history(df):
    analysis_name = "Customer Transaction History"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'transaction_date', 'transaction_id', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

        transactions_per_customer = df.groupby('customer_id')['transaction_id'].nunique().nlargest(10).reset_index(name='num_transactions')
        revenue_per_customer = df.groupby('customer_id')['revenue'].sum().nlargest(10).reset_index()

        metrics = {
            "total_unique_customers": df['customer_id'].nunique(),
            "total_transactions": df['transaction_id'].nunique(),
            "total_revenue_overall": df['revenue'].sum()
        }
        
        insights.append(f"Analyzed {metrics['total_unique_customers']:,} customers.")
        top_customer = revenue_per_customer.iloc[0]
        insights.append(f"Top customer by revenue: {top_customer['customer_id']} (${top_customer['revenue']:,.2f})")

        fig_transactions_per_customer = px.bar(transactions_per_customer, x='customer_id', y='num_transactions', title='Top 10 Customers by Number of Transactions')
        fig_revenue_per_customer = px.bar(revenue_per_customer, x='customer_id', y='revenue', title='Top 10 Customers by Total Revenue')

        visualizations = {
            'transactions_per_customer': fig_transactions_per_customer.to_json(),
            'revenue_per_customer': fig_revenue_per_customer.to_json()
        }

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

def customer_segment_purchasing(df):
    analysis_name = "Customer Segment Purchasing"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_segment', 'sales', 'customer_id', 'product_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['customer_segment', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['customer_segment', 'sales'], inplace=True)

        sales_by_segment = df.groupby('customer_segment')['sales'].sum().reset_index()
        
        metrics = {
            "total_sales": df['sales'].sum(),
            "num_customer_segments": df['customer_segment'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_customer_segments']} customer segments.")
        top_segment = sales_by_segment.loc[sales_by_segment['sales'].idxmax()]
        insights.append(f"Top segment by sales: {top_segment['customer_segment']} (${top_segment['sales']:,.2f})")

        fig_sales_by_segment = px.pie(sales_by_segment, names='customer_segment', values='sales', title='Sales Distribution by Customer Segment')
        visualizations = {
            'sales_by_segment': fig_sales_by_segment.to_json(),
        }

        if 'product_category' in df.columns:
            segment_product_sales = df.groupby(['customer_segment', 'product_category'])['sales'].sum().reset_index()
            # Get top 3 categories per segment
            top_categories_per_segment = segment_product_sales.sort_values('sales', ascending=False).groupby('customer_segment').head(3)

            if not top_categories_per_segment.empty:
                fig_top_categories_per_segment = px.bar(top_categories_per_segment, x='product_category', y='sales', color='customer_segment',
                                                        title='Top Product Categories Purchased by Customer Segment (Top 3 per Segment)', barmode='group')
                visualizations['top_categories_per_segment'] = fig_top_categories_per_segment.to_json()
                insights.append("Generated plot for top categories by segment.")
            else:
                insights.append("Product category data found, but no sales data to plot by segment.")
        else:
            insights.append("Product category analysis skipped: 'product_category' column not found.")

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

def per_unit_price_and_sales(df):
    analysis_name = "Per Unit Price and Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'price', 'quantity', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'price', 'quantity'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        
        if 'sales' not in df.columns:
             df['sales'] = df['price'] * df['quantity']
             insights.append("Calculated 'sales' from 'price' * 'quantity'.")
        else:
             df['sales'] = pd.to_numeric(df['sales'], errors='coerce')

        df.dropna(subset=['product_id', 'price', 'quantity', 'sales'], inplace=True)

        avg_unit_price_by_product = df.groupby('product_id')['price'].mean().nlargest(10).reset_index()
        total_sales_by_product = df.groupby('product_id')['sales'].sum().nlargest(10).reset_index()

        metrics = {
            "overall_average_unit_price": df['price'].mean(),
            "overall_total_sales": df['sales'].sum(),
            "median_unit_price": df['price'].median()
        }

        insights.append(f"Overall average unit price: ${metrics['overall_average_unit_price']:,.2f}")
        insights.append(f"Overall total sales: ${metrics['overall_total_sales']:,.2f}")
        
        fig_avg_unit_price = px.bar(avg_unit_price_by_product, x='product_id', y='price', title='Top 10 Products by Average Unit Price')
        fig_total_sales_product = px.bar(total_sales_by_product, x='product_id', y='sales', title='Top 10 Products by Total Sales')

        visualizations = {
            'avg_unit_price_product': fig_avg_unit_price.to_json(),
            'total_sales_product': fig_total_sales_product.to_json()
        }

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

def promotion_id_impact(df):
    analysis_name = "Promotion ID Impact"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['promotion_id', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales'], inplace=True)

        df['promotion_id'] = df['promotion_id'].fillna('No Promotion')
        sales_by_promotion = df.groupby('promotion_id')['sales'].sum().reset_index()
        sales_by_promotion = sales_by_promotion.sort_values('sales', ascending=False)

        transactions_by_promotion = df.groupby('promotion_id')['transaction_id'].nunique().reset_index(name='num_transactions')
        transactions_by_promotion = transactions_by_promotion.sort_values('num_transactions', ascending=False)

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_unique_promotions": df['promotion_id'].nunique() - (1 if 'No Promotion' in df['promotion_id'].unique() else 0),
            "sales_with_promotion": df[df['promotion_id'] != 'No Promotion']['sales'].sum(),
            "sales_no_promotion": df[df['promotion_id'] == 'No Promotion']['sales'].sum()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_promotions']} unique promotions.")
        insights.append(f"Sales with promotions: ${metrics['sales_with_promotion']:,.2f}")
        insights.append(f"Sales without promotions: ${metrics['sales_no_promotion']:,.2f}")

        fig_sales_by_promotion = px.bar(sales_by_promotion.nlargest(11, 'sales'), x='promotion_id', y='sales', title='Top 10 Promotions (and No Promotion) by Sales Revenue')
        fig_transactions_by_promotion = px.bar(transactions_by_promotion.nlargest(11, 'num_transactions'), x='promotion_id', y='num_transactions', title='Top 10 Promotions (and No Promotion) by Number of Transactions')

        visualizations = {
            'sales_by_promotion': fig_sales_by_promotion.to_json(),
            'transactions_by_promotion': fig_transactions_by_promotion.to_json()
        }

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

def store_location_sales(df):
    analysis_name = "Store Location Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['store_location', 'sales', 'store_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['store_location', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['store_location', 'sales'], inplace=True)

        sales_by_location = df.groupby('store_location')['sales'].sum().reset_index()

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_unique_locations": df['store_location'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_locations']} store locations.")
        top_loc = sales_by_location.loc[sales_by_location['sales'].idxmax()]
        insights.append(f"Top location by sales: {top_loc['store_location']} (${top_loc['sales']:,.2f})")

        fig_sales_by_location = px.bar(sales_by_location.nlargest(10, 'sales'), x='store_location', y='sales', title='Top 10 Store Locations by Sales')
        visualizations = {
            'sales_by_location': fig_sales_by_location.to_json(),
        }

        if 'store_id' in df.columns:
            stores_per_location = df.groupby('store_location')['store_id'].nunique().reset_index(name='num_stores')
            if not stores_per_location.empty:
                fig_stores_per_location = px.bar(stores_per_location, x='store_location', y='num_stores', title='Number of Unique Stores per Location')
                visualizations['stores_per_location'] = fig_stores_per_location.to_json()
                insights.append("Generated plot for stores per location.")
            else:
                insights.append("Store ID data found, but no stores to plot.")
        else:
            insights.append("Stores per location analysis skipped: 'store_id' column not found.")

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

def sales_representative_performance(df):
    analysis_name = "Sales Representative Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_representative_id', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_representative_id', 'sales'], inplace=True)

        sales_by_rep = df.groupby('sales_representative_id')['sales'].sum().nlargest(10).reset_index()
        transactions_by_rep = df.groupby('sales_representative_id')['transaction_id'].nunique().nlargest(10).reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_representatives": df['sales_representative_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_representatives']} sales reps.")
        top_rep = sales_by_rep.iloc[0]
        insights.append(f"Top rep by sales: {top_rep['sales_representative_id']} (${top_rep['sales']:,.2f})")

        fig_sales_by_rep = px.bar(sales_by_rep, x='sales_representative_id', y='sales', title='Top 10 Sales Representatives by Sales')
        fig_transactions_by_rep = px.bar(transactions_by_rep, x='sales_representative_id', y='num_transactions', title='Top 10 Sales Representatives by Transactions')

        visualizations = {
            'sales_by_representative': fig_sales_by_rep.to_json(),
            'transactions_by_representative': fig_transactions_by_rep.to_json()
        }

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

def regional_sales_and_product(df):
    analysis_name = "Regional Sales and Product"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['region', 'product_id', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['region', 'product_id', 'sales'], inplace=True)

        sales_by_region = df.groupby('region')['sales'].sum().reset_index()
        regional_product_sales = df.groupby(['region', 'product_id'])['sales'].sum().reset_index()
        top_product_per_region = regional_product_sales.loc[regional_product_sales.groupby('region')['sales'].idxmax()].reset_index(drop=True)

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_unique_regions": df['region'].nunique(),
            "num_unique_products": df['product_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_regions']} regions and {metrics['num_unique_products']} products.")

        fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
        fig_top_product_per_region = px.bar(top_product_per_region, x='region', y='sales', color='product_id', title='Top Selling Product per Region')

        visualizations = {
            'sales_by_region': fig_sales_by_region.to_json(),
            'top_product_per_region': fig_top_product_per_region.to_json()
        }

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

def multi_channel_sales(df):
    analysis_name = "Multi-Channel Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_channel', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_channel', 'sales'], inplace=True)

        sales_by_channel = df.groupby('sales_channel')['sales'].sum().reset_index()
        transactions_by_channel = df.groupby('sales_channel')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_channels": df['sales_channel'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_channels']} sales channels, totaling ${metrics['total_sales']:,.2f} in sales.")

        fig_sales_by_channel = px.pie(sales_by_channel, names='sales_channel', values='sales', title='Sales Distribution by Channel')
        fig_transactions_by_channel = px.bar(transactions_by_channel, x='sales_channel', y='num_transactions', title='Number of Transactions by Channel')

        visualizations = {
            'sales_by_channel': fig_sales_by_channel.to_json(),
            'transactions_by_channel': fig_transactions_by_channel.to_json()
        }

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

def sales_classification(df):
    analysis_name = "Sales Classification"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_category', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_category', 'sales'], inplace=True)

        sales_by_classification = df.groupby('sales_category')['sales'].sum().reset_index()
        transactions_by_classification = df.groupby('sales_category')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_classifications": df['sales_category'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_classifications']} sales classifications.")

        fig_sales_classification_pie = px.pie(sales_by_classification, names='sales_category', values='sales', title='Sales Distribution by Classification')
        fig_transactions_classification = px.bar(transactions_by_classification, x='sales_category', y='num_transactions', title='Number of Transactions by Sales Classification')

        visualizations = {
            'sales_classification_pie': fig_sales_classification_pie.to_json(),
            'transactions_classification_bar': fig_transactions_classification.to_json()
        }

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

def retail_receipt_data(df):
    analysis_name = "Retail Receipt Data Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['receipt_id', 'transaction_date', 'total_amount', 'item_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['receipt_id', 'transaction_date', 'total_amount'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['total_amount'] = pd.to_numeric(df['total_amount'], errors='coerce')
        df.dropna(subset=['receipt_id', 'total_amount', 'transaction_date'], inplace=True)

        daily_receipt_amount = df.groupby(df['transaction_date'].dt.date)['total_amount'].sum().reset_index()
        daily_receipt_amount.columns = ['date', 'total_amount']
        
        metrics = {
            "total_receipts": df['receipt_id'].nunique(),
            "total_revenue_from_receipts": df['total_amount'].sum()
        }
        
        visualizations = {}
        fig_daily_receipt_amount = px.line(daily_receipt_amount, x='date', y='total_amount', title='Daily Total Amount from Receipts')
        visualizations['daily_receipt_amount'] = fig_daily_receipt_amount.to_json()

        if 'item_count' in df.columns:
            df['item_count'] = pd.to_numeric(df['item_count'], errors='coerce')
            if not df['item_count'].isnull().all():
                metrics['avg_items_per_receipt'] = df['item_count'].mean()
                insights.append(f"Average items per receipt: {metrics['avg_items_per_receipt']:.2f}")
                fig_item_count_distribution = px.histogram(df, x='item_count', nbins=50, title='Distribution of Item Counts per Receipt')
                visualizations['item_count_distribution'] = fig_item_count_distribution.to_json()
            else:
                 insights.append("'item_count' column found but contains no valid data.")
        else:
            metrics['avg_items_per_receipt'] = 'N/A'
            insights.append("Item count analysis skipped: 'item_count' column not found.")

        insights.insert(0, f"Analyzed {metrics['total_receipts']:,} receipts, totaling ${metrics['total_revenue_from_receipts']:,.2f}.")

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

def invoice_line_item(df):
    analysis_name = "Invoice Line Item Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['invoice_id', 'product_id', 'quantity', 'unit_price', 'total_price']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['invoice_id', 'product_id', 'quantity', 'unit_price'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['unit_price'] = pd.to_numeric(df['unit_price'], errors='coerce')
        df.dropna(subset=['invoice_id', 'product_id', 'quantity', 'unit_price'], inplace=True)
        
        if 'total_price' not in df.columns:
             df['total_price'] = df['quantity'] * df['unit_price']
             insights.append("Calculated 'total_price' from 'quantity' * 'unit_price'.")
        else:
             df['total_price'] = pd.to_numeric(df['total_price'], errors='coerce')

        top_selling_products_line_item = df.groupby('product_id')['total_price'].sum().nlargest(10).reset_index()

        metrics = {
            "total_line_items": len(df),
            "total_revenue_from_line_items": df['total_price'].sum(),
            "num_unique_products_in_line_items": df['product_id'].nunique(),
            "avg_quantity_per_line": df['quantity'].mean()
        }
        
        insights.append(f"Analyzed {metrics['total_line_items']:,} line items.")
        insights.append(f"Average quantity per line item: {metrics['avg_quantity_per_line']:.2f}")

        fig_quantity_distribution = px.histogram(df, x='quantity', nbins=50, title='Distribution of Quantities per Line Item')
        fig_top_selling_products_line_item = px.bar(top_selling_products_line_item, x='product_id', y='total_price', title='Top 10 Products by Line Item Revenue')

        visualizations = {
            'quantity_distribution': fig_quantity_distribution.to_json(),
            'top_selling_products_line_item': fig_top_selling_products_line_item.to_json()
        }

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

def product_category_performance(df):
    analysis_name = "Product Category Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_category', 'sales', 'quantity', 'product_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_category', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['product_category', 'sales'], inplace=True)

        sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()
        
        metrics = {
            "total_sales": df['sales'].sum(),
            "num_unique_product_categories": df['product_category'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_product_categories']} categories totaling ${metrics['total_sales']:,.2f} in sales.")

        fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
        visualizations = {
            'sales_by_category': fig_sales_by_category.to_json(),
        }

        if 'product_id' in df.columns:
            products_per_category = df.groupby('product_category')['product_id'].nunique().reset_index(name='unique_products')
            if not products_per_category.empty:
                fig_products_per_category = px.bar(products_per_category, x='product_category', y='unique_products', title='Number of Unique Products per Category')
                visualizations['products_per_category'] = fig_products_per_category.to_json()
                insights.append("Generated plot for unique products per category.")
            else:
                insights.append("'product_id' column found, but no data to plot for products per category.")
        else:
            insights.append("Products per category analysis skipped: 'product_id' column not found.")

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

def transactional_promotion_effectiveness(df):
    analysis_name = "Transactional Promotion Effectiveness"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_id', 'promotion_id', 'sales', 'discount_amount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['transaction_id', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        if matched.get('promotion_id') is None and matched.get('discount_amount') is None:
            return create_fallback_response(analysis_name, ['promotion_id', 'discount_amount'], matched, df)


        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_id', 'sales'], inplace=True)

        if 'had_promotion' not in df.columns:
            if 'promotion_id' in df.columns:
                df['had_promotion'] = df['promotion_id'].notna().astype(int)
            elif 'discount_amount' in df.columns:
                df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
                df['had_promotion'] = (df['discount_amount'] > 0).astype(int)
            else:
                df['had_promotion'] = 0 # Should not be reachable due to check above
        
        avg_sales_by_promotion = df.groupby('had_promotion')['sales'].mean().reset_index()
        avg_sales_by_promotion['had_promotion'] = avg_sales_by_promotion['had_promotion'].map({0: 'No Promotion', 1: 'Had Promotion'})

        metrics = {
            "total_transactions_promoted": df[df['had_promotion'] == 1]['transaction_id'].nunique(),
            "total_transactions_non_promoted": df[df['had_promotion'] == 0]['transaction_id'].nunique(),
            "total_sales_promoted": df[df['had_promotion'] == 1]['sales'].sum(),
            "total_sales_non_promoted": df[df['had_promotion'] == 0]['sales'].sum(),
            "avg_sales_promoted": df[df['had_promotion'] == 1]['sales'].mean(),
            "avg_sales_non_promoted": df[df['had_promotion'] == 0]['sales'].mean()
        }
        
        insights.append(f"Avg. Sales (Promoted): ${metrics['avg_sales_promoted']:,.2f}")
        insights.append(f"Avg. Sales (Non-Promoted): ${metrics['avg_sales_non_promoted']:,.2f}")

        fig_avg_sales_by_promotion = px.bar(avg_sales_by_promotion, x='had_promotion', y='sales', title='Average Sales per Transaction: Promoted vs. Non-Promoted')
        visualizations = {
            'avg_sales_by_promotion': fig_avg_sales_by_promotion.to_json(),
        }

        if 'discount_amount' in df.columns and 'promotion_id' in df.columns:
            df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
            total_discount_by_promotion = df[df['promotion_id'].notna()].groupby('promotion_id')['discount_amount'].sum().nlargest(10).reset_index()
            if not total_discount_by_promotion.empty:
                fig_total_discount_by_promotion = px.bar(total_discount_by_promotion, x='promotion_id', y='discount_amount', title='Top 10 Promotions by Total Discount Amount Applied')
                visualizations['total_discount_by_promotion'] = fig_total_discount_by_promotion.to_json()
                insights.append("Generated plot for discount by promotion ID.")
            else:
                insights.append("Discount/Promotion data found, but no data to plot for top discounts.")
        else:
            insights.append("Top discount analysis skipped: 'discount_amount' or 'promotion_id' column not found.")

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

def order_status_and_item_details(df):
    analysis_name = "Order Status and Item Details"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'order_status', 'product_name', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['order_id', 'order_status', 'product_name', 'quantity'], inplace=True)

        items_by_order_status = df.groupby('order_status')['quantity'].sum().reset_index()
        
        # Find common "completed" statuses
        completed_shipped_orders = df[df['order_status'].astype(str).str.lower().isin(['completed', 'shipped', 'delivered', 'complete'])]
        if completed_shipped_orders.empty:
            insights.append("Could not find common 'completed' or 'shipped' statuses. Using all data for top products.")
            top_products_completed = df.groupby('product_name')['quantity'].sum().nlargest(10).reset_index()
        else:
             insights.append("Filtered top products on 'completed', 'shipped', 'delivered' statuses.")
             top_products_completed = completed_shipped_orders.groupby('product_name')['quantity'].sum().nlargest(10).reset_index()


        metrics = {
            "total_orders": df['order_id'].nunique(),
            "total_items_in_orders": df['quantity'].sum(),
            "num_unique_order_statuses": df['order_status'].nunique()
        }
        
        insights.insert(0, f"Analyzed {metrics['total_items_in_orders']:,} items across {metrics['total_orders']:,} orders.")

        fig_items_by_order_status = px.bar(items_by_order_status, x='order_status', y='quantity', title='Total Items by Order Status')
        fig_top_products_completed = px.bar(top_products_completed, x='product_name', y='quantity', title='Top 10 Products in Completed/Shipped Orders')

        visualizations = {
            'items_by_order_status': fig_items_by_order_status.to_json(),
            'top_products_completed': fig_top_products_completed.to_json()
        }

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

def sales_source_attribution(df):
    analysis_name = "Sales Source Attribution"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sales_source', 'sales', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales_source', 'sales'], inplace=True)

        sales_by_source = df.groupby('sales_source')['sales'].sum().reset_index()
        transactions_by_source = df.groupby('sales_source')['transaction_id'].nunique().reset_index(name='num_transactions')

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_transactions": df['transaction_id'].nunique(),
            "num_sales_sources": df['sales_source'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_sales_sources']} sales sources.")
        top_source = sales_by_source.loc[sales_by_source['sales'].idxmax()]
        insights.append(f"Top source by sales: {top_source['sales_source']} (${top_source['sales']:,.2f})")

        fig_sales_by_source = px.pie(sales_by_source, names='sales_source', values='sales', title='Sales Distribution by Source')
        fig_transactions_by_source = px.bar(transactions_by_source, x='sales_source', y='num_transactions', title='Number of Transactions by Sales Source')

        visualizations = {
            'sales_by_source': fig_sales_by_source.to_json(),
            'transactions_by_source': fig_transactions_by_source.to_json()
        }

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

def customer_regional_sales(df):
    analysis_name = "Customer Regional Sales"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['customer_id', 'region', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['customer_id', 'region', 'sales'], inplace=True)

        customer_region_sales = df.groupby(['customer_id', 'region'])['sales'].sum().nlargest(10).reset_index()
        customer_region_sales['customer_region'] = customer_region_sales['customer_id'].astype(str) + ' - ' + customer_region_sales['region']

        customers_per_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

        metrics = {
            "total_sales": df['sales'].sum(),
            "num_unique_customers": df['customer_id'].nunique(),
            "num_unique_regions": df['region'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_customers']} customers in {metrics['num_unique_regions']} regions.")

        fig_customer_region_sales = px.bar(customer_region_sales, x='customer_region', y='sales', color='region', title='Top 10 Customer-Region Sales Combinations')
        fig_customers_per_region = px.bar(customers_per_region, x='region', y='unique_customers', title='Number of Unique Customers per Region')

        visualizations = {
            'customer_region_sales': fig_customer_region_sales.to_json(),
            'customers_per_region': fig_customers_per_region.to_json()
        }

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

# ========== GENERAL ANALYSIS FUNCTIONS (API-Ready) ==========

def sales_analysis(df):
    analysis_name = "Sales Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transaction_date', 'revenue', 'quantity', 'product_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['transaction_date', 'revenue', 'product_id'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['transaction_date', 'revenue'], inplace=True)

        total_revenue = df['revenue'].sum()
        avg_transaction_value = df['revenue'].mean()
        
        metrics = {
            "total_revenue": total_revenue,
            "avg_transaction_value": avg_transaction_value,
        }
        
        if 'quantity' in df.columns:
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            metrics['total_quantity_sold'] = df['quantity'].sum()
            insights.append(f"Total quantity sold: {metrics['total_quantity_sold']:,.0f}")
        else:
            insights.append("Quantity analysis skipped: 'quantity' column not found.")
            
        insights.insert(0, f"Total revenue: ${total_revenue:,.2f}")
        insights.insert(1, f"Average revenue per transaction: ${avg_transaction_value:,.2f}")

        daily_sales = df.groupby(df['transaction_date'].dt.date)['revenue'].sum().reset_index()
        daily_sales.columns = ['date', 'revenue']
        top_products = df.groupby('product_id')['revenue'].sum().nlargest(10).reset_index()

        fig_daily_sales = px.line(daily_sales, x='date', y='revenue', title='Daily Sales Trend')
        fig_top_products = px.bar(top_products, x='product_id', y='revenue', title='Top 10 Products by Revenue')

        visualizations = {
            'daily_sales_trend': fig_daily_sales.to_json(),
            'top_products_by_revenue': fig_top_products.to_json()
        }

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
        expected = ['customer_id', 'transaction_date', 'revenue', 'order_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['customer_id', 'transaction_date', 'revenue', 'order_id'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

        num_unique_customers = df['customer_id'].nunique()
        avg_revenue_per_customer = df.groupby('customer_id')['revenue'].sum().mean()
        top_customers_by_revenue = df.groupby('customer_id')['revenue'].sum().nlargest(10).reset_index()
        customer_frequency = df.groupby('customer_id')['order_id'].nunique().reset_index(name='num_orders')

        metrics = {
            "num_unique_customers": num_unique_customers,
            "avg_revenue_per_customer": avg_revenue_per_customer,
            "avg_transactions_per_customer": customer_frequency['num_orders'].mean()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_customers']:,} unique customers.")
        insights.append(f"Average revenue per customer: ${metrics['avg_revenue_per_customer']:,.2f}")
        insights.append(f"Average orders per customer: {metrics['avg_transactions_per_customer']:.2f}")

        fig_top_customers = px.bar(top_customers_by_revenue, x='customer_id', y='revenue', title='Top 10 Customers by Total Revenue')
        fig_customer_frequency_distribution = px.histogram(customer_frequency, x='num_orders', nbins=50, title='Distribution of Customer Transaction Frequency')

        visualizations = {
            'top_customers_by_revenue': fig_top_customers.to_json(),
            'customer_frequency_distribution': fig_customer_frequency_distribution.to_json()
        }

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
        expected = ['product_id', 'stock_quantity', 'price', 'product_name', 'last_restock_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'stock_quantity'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['stock_quantity'] = pd.to_numeric(df['stock_quantity'], errors='coerce')
        df.dropna(subset=['product_id', 'stock_quantity'], inplace=True)

        num_unique_products_in_stock = df['product_id'].nunique()
        
        metrics = {
            "total_stock_quantity": df['stock_quantity'].sum(),
            "num_unique_products_in_stock": num_unique_products_in_stock
        }
        
        if 'price' in df.columns:
            df['price'] = pd.to_numeric(df['price'], errors='coerce')
            if not df['price'].isnull().all():
                metrics['total_stock_value'] = (df['stock_quantity'] * df['price']).sum()
                insights.append(f"Total stock value (based on 'price'): ${metrics['total_stock_value']:,.2f}")
            else:
                 insights.append("'price' column found but has no valid data; cannot calculate stock value.")
        else:
            insights.append("Stock value calculation skipped: 'price' column not found.")

        insights.insert(0, f"Total items in stock: {metrics['total_stock_quantity']:,} across {metrics['num_unique_products_in_stock']:,} products.")

        top_stock_products = df.groupby('product_id')['stock_quantity'].sum().nlargest(10).reset_index()
        fig_stock_distribution = px.histogram(df, x='stock_quantity', nbins=50, title='Distribution of Stock Quantities')
        fig_top_stock = px.bar(top_stock_products, x='product_id', y='stock_quantity', title='Top 10 Products by Stock Quantity')

        visualizations = {
            'top_stock_products': fig_top_stock.to_json(),
            'stock_distribution': fig_stock_distribution.to_json()
        }

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
        expected = ['product_id', 'product_name', 'sales', 'quantity', 'product_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['product_id', 'sales', 'quantity'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['product_id', 'sales', 'quantity'], inplace=True)

        top_selling_products = df.groupby('product_id')['sales'].sum().nlargest(10).reset_index()
        
        metrics = {
            "total_sales": df['sales'].sum(),
            "total_quantity_sold": df['quantity'].sum(),
            "num_unique_products": df['product_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_products']:,} products.")
        insights.append(f"Total sales: ${metrics['total_sales']:,.2f} | Total quantity: {metrics['total_quantity_sold']:,}")
        
        fig_top_selling_products = px.bar(top_selling_products, x='product_id', y='sales', title='Top 10 Selling Products by Sales')
        visualizations = {
            'top_selling_products': fig_top_selling_products.to_json(),
        }

        if 'product_category' in df.columns:
            sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()
            if not sales_by_category.empty:
                fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
                visualizations['sales_by_category'] = fig_sales_by_category.to_json()
                insights.append("Generated plot for sales by category.")
            else:
                insights.append("'product_category' found, but no data to plot.")
        else:
            insights.append("Sales by category analysis skipped: 'product_category' column not found.")

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
        expected = ['store_id', 'sales', 'transaction_id', 'region', 'store_location']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['store_id', 'sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['store_id', 'sales'], inplace=True)

        top_performing_stores = df.groupby('store_id')['sales'].sum().nlargest(10).reset_index()
        
        metrics = {
            "total_sales_across_stores": df['sales'].sum(),
            "num_unique_stores": df['store_id'].nunique()
        }
        
        insights.append(f"Analyzed {metrics['num_unique_stores']} stores totaling ${metrics['total_sales_across_stores']:,.2f} in sales.")

        fig_top_performing_stores = px.bar(top_performing_stores, x='store_id', y='sales', title='Top 10 Performing Stores by Sales')
        visualizations = {
            'top_performing_stores': fig_top_performing_stores.to_json(),
        }

        if 'region' in df.columns:
            sales_by_region = df.groupby('region')['sales'].sum().reset_index()
            if not sales_by_region.empty:
                fig_sales_by_region = px.pie(sales_by_region, names='region', values='sales', title='Sales Distribution by Region')
                visualizations['sales_by_region'] = fig_sales_by_region.to_json()
                insights.append("Generated plot for sales by region.")
            else:
                insights.append("'region' column found, but no data to plot.")
        else:
            insights.append("Sales by region analysis skipped: 'region' column not found.")

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
        expected = ['promotion_id', 'sales', 'discount_amount', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in ['sales'] if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        if matched.get('promotion_id') is None and matched.get('discount_amount') is None:
            return create_fallback_response(analysis_name, ['promotion_id', 'discount_amount'], matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['sales'], inplace=True)
        
        if 'had_promotion' not in df.columns:
            if 'promotion_id' in df.columns:
                df['had_promotion'] = df['promotion_id'].notna().astype(int)
            elif 'discount_amount' in df.columns:
                df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
                df['had_promotion'] = (df['discount_amount'] > 0).astype(int)
            else:
                df['had_promotion'] = 0

        sales_by_promotion_status = df.groupby('had_promotion')['sales'].sum().reset_index()
        sales_by_promotion_status['had_promotion'] = sales_by_promotion_status['had_promotion'].map({0: 'No Promotion', 1: 'Had Promotion'})

        metrics = {
            "total_sales": df['sales'].sum(),
            "total_promoted_sales": df[df['had_promotion'] == 1]['sales'].sum(),
        }
        
        if 'promotion_id' in df.columns:
             metrics["num_unique_promotions"] = df['promotion_id'].nunique() - (1 if df['promotion_id'].isnull().any() else 0)
             insights.append(f"Analyzed {metrics['num_unique_promotions']} unique promotions.")

        insights.append(f"Total sales with promotions: ${metrics['total_promoted_sales']:,.2f}")
        
        fig_sales_by_promo_status = px.bar(sales_by_promotion_status, x='had_promotion', y='sales', title='Sales with vs. Without Promotion')
        visualizations = {
            'sales_by_promo_status': fig_sales_by_promo_status.to_json(),
        }

        if 'discount_amount' in df.columns and 'promotion_id' in df.columns and not df['promotion_id'].isnull().all():
            df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0)
            avg_discount_per_promotion = df[df['promotion_id'].notna()].groupby('promotion_id')['discount_amount'].mean().nlargest(10).reset_index()
            if not avg_discount_per_promotion.empty:
                fig_avg_discount_per_promo = px.bar(avg_discount_per_promotion, x='promotion_id', y='discount_amount', title='Top 10 Promotions by Average Discount Given')
                visualizations['avg_discount_per_promo'] = fig_avg_discount_per_promo.to_json()
                insights.append("Generated plot for average discount by promotion.")
            else:
                insights.append("Discount/Promotion data found, but no data to plot.")
        else:
            insights.append("Average discount analysis skipped: 'discount_amount' or 'promotion_id' column not found or empty.")

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
        expected = ['transaction_id', 'product_id', 'quantity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df.dropna(subset=['transaction_id', 'product_id', 'quantity'], inplace=True)

        items_per_transaction = df.groupby('transaction_id')['quantity'].sum().reset_index(name='total_items')
        avg_items_per_transaction = items_per_transaction['total_items'].mean()
        
        product_frequency = df.groupby('product_id')['quantity'].sum().nlargest(10).reset_index()

        metrics = {
            "total_transactions": df['transaction_id'].nunique(),
            "avg_items_per_transaction": avg_items_per_transaction,
            "total_items_sold": df['quantity'].sum()
        }
        
        insights.append(f"Analyzed {metrics['total_transactions']:,} transactions.")
        insights.append(f"Average items per transaction: {metrics['avg_items_per_transaction']:.2f}")

        fig_items_per_transaction_hist = px.histogram(items_per_transaction, x='total_items', nbins=50, title='Distribution of Items per Transaction')
        fig_top_product_frequency = px.bar(product_frequency, x='product_id', y='quantity', title='Top 10 Most Frequently Purchased Products (by Quantity)')

        visualizations = {
            'items_per_transaction_distribution': fig_items_per_transaction_hist.to_json(),
            'top_product_frequency': fig_top_product_frequency.to_json()
        }

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
        expected = ['transaction_date', 'sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['sales'] = pd.to_numeric(df['sales'], errors='coerce')
        df.dropna(subset=['transaction_date', 'sales'], inplace=True)

        df['month'] = df['transaction_date'].dt.month_name()
        df['quarter'] = 'Q' + df['transaction_date'].dt.quarter.astype(str)
        df['day_of_week'] = df['transaction_date'].dt.day_name()

        sales_by_month = df.groupby('month')['sales'].sum().reset_index()
        month_order = ["January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        sales_by_month['month'] = pd.Categorical(sales_by_month['month'], categories=month_order, ordered=True)
        sales_by_month = sales_by_month.sort_values('month')

        sales_by_day_of_week = df.groupby('day_of_week')['sales'].sum().reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        sales_by_day_of_week['day_of_week'] = pd.Categorical(sales_by_day_of_week['day_of_week'], categories=day_order, ordered=True)
        sales_by_day_of_week = sales_by_day_of_week.sort_values('day_of_week')
        
        metrics = {
            "total_sales": df['sales'].sum(),
            "peak_sales_month": sales_by_month.loc[sales_by_month['sales'].idxmax()]['month'],
            "peak_sales_day_of_week": sales_by_day_of_week.loc[sales_by_day_of_week['sales'].idxmax()]['day_of_week']
        }
        
        insights.append(f"Peak sales month: {metrics['peak_sales_month']}")
        insights.append(f"Peak sales day of week: {metrics['peak_sales_day_of_week']}")

        fig_sales_by_month = px.line(sales_by_month, x='month', y='sales', title='Sales Trend by Month')
        fig_sales_by_day_of_week = px.bar(sales_by_day_of_week, x='day_of_week', y='sales', title='Sales by Day of Week')

        visualizations = {
            'sales_by_month': fig_sales_by_month.to_json(),
            'sales_by_day_of_week': fig_sales_by_day_of_week.to_json()
        }

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

# ========== MAIN DISPATCHER FUNCTION ==========

# Lists of available analyses for UI/API discovery
general_analysis_options = [
    "Sales Analysis",
    "Customer Analysis",
    "Inventory Analysis",
    "Product Analysis",
    "Store Analysis",
    "Promotion Analysis",
    "Basket Analysis",
    "Seasonal Analysis",
]

specific_analysis_options = [
    "Customer Purchase Behavior and RFM Analysis",
    "Retail Transaction Analysis by Product and Country",
    "Retail Order Status and Item Analysis",
    "Regional Sales and Customer Analysis",
    "Sales Channel Performance",
    "International Sales and Transaction Analysis",
    "Invoice Type and Customer Purchase Pattern",
    "Order Delivery and Customer Location",
    "Time-of-Day Sales Pattern",
    "Customer Order and Status Tracking",
    "Payment Method Preference",
    "Product Return Rate",
    "Promotional Code Effectiveness",
    "Discount Impact on Sales",
    "Product Cost and Sales Margin",
    "Store-Level Sales Performance",
    "Product Category Analysis",
    "Weekly Sales Trend",
    "Yearly Sales Performance",
    "Monthly Sales Trend",
    "Week-over-Week Sales Growth",
    "Holiday Sales Impact",
    "Customer Type Analysis",
    "Online vs Offline Sales",
    "Regional Customer Purchase",
    "Delivery Method Preference",
    "Point of Sale Transaction",
    "Sales Tax Analysis",
    "Sales Organization Analysis",
    "Order Payment Status",
    "Product Sales and Cost",
    "Customer Transaction History",
    "Customer Segment Purchasing",
    "Per Unit Price and Sales",
    "Promotion ID Impact",
    "Store Location Sales",
    "Sales Representative Performance",
    "Regional Sales and Product",
    "Multi-channel Sales",
    "Sales Classification",
    "Retail Receipt Data",
    "Invoice Line Item",
    "Product Category Performance",
    "Transactional Promotion Effectiveness",
    "Order Status and Item Details",
    "Sales Source Attribution",
    "Customer Regional Sales",
]

def main_backend(df, category=None, general_analysis=None, specific_analysis_name=None):
    
    # Mapping of analysis names to their corresponding functions
    specific_retail_function_mapping = {
        "Customer Purchase Behavior and RFM Analysis": customer_purchase_behavior_and_rfm_analysis,
        "Retail Transaction Analysis by Product and Country": retail_transaction_analysis_by_product_and_country,
        "Retail Order Status and Item Analysis": retail_order_status_and_item_analysis,
        "Regional Sales and Customer Analysis": regional_sales_and_customer_analysis,
        "Sales Channel Performance": sales_channel_performance,
        "International Sales and Transaction Analysis": international_sales_and_transaction_analysis,
        "Invoice Type and Customer Purchase Pattern": invoice_type_and_customer_purchase_pattern,
        "Order Delivery and Customer Location": order_delivery_and_customer_location,
        "Time-of-Day Sales Pattern": time_of_day_sales_pattern,
        "Customer Order and Status Tracking": customer_order_and_status_tracking,
        "Payment Method Preference": payment_method_preference,
        "Product Return Rate": product_return_rate,
        "Promotional Code Effectiveness": promotional_code_effectiveness,
        "Discount Impact on Sales": discount_impact_on_sales,
        "Product Cost and Sales Margin": product_cost_and_sales_margin,
        "Store-Level Sales Performance": store_level_sales_performance,
        "Product Category Analysis": product_category_analysis, # Mapped renamed function
        "Weekly Sales Trend": weekly_sales_trend,
        "Yearly Sales Performance": yearly_sales_performance,
        "Monthly Sales Trend": monthly_sales_trend,
        "Week-over-Week Sales Growth": week_over_week_sales_growth,
        "Holiday Sales Impact": holiday_sales_impact,
        "Customer Type Analysis": customer_type_analysis,
        "Online vs Offline Sales": online_vs_offline_sales,
        "Regional Customer Purchase": regional_customer_purchase,
        "Delivery Method Preference": delivery_method_preference,
        "Point of Sale Transaction": point_of_sale_transaction,
        "Sales Tax Analysis": sales_tax_analysis,
        "Sales Organization Analysis": sales_organization_analysis, # Mapped renamed function
        "Order Payment Status": order_payment_status,
        "Product Sales and Cost": product_sales_and_cost,
        "Customer Transaction History": customer_transaction_history,
        "Customer Segment Purchasing": customer_segment_purchasing,
        "Per Unit Price and Sales": per_unit_price_and_sales,
        "Promotion ID Impact": promotion_id_impact,
        "Store Location Sales": store_location_sales,
        "Sales Representative Performance": sales_representative_performance,
        "Regional Sales and Product": regional_sales_and_product,
        "Multi-channel Sales": multi_channel_sales,
        "Sales Classification": sales_classification,
        "Retail Receipt Data": retail_receipt_data,
        "Invoice Line Item": invoice_line_item,
        "Product Category Performance": product_category_performance,
        "Transactional Promotion Effectiveness": transactional_promotion_effectiveness,
        "Order Status and Item Details": order_status_and_item_details,
        "Sales Source Attribution": sales_source_attribution,
        "Customer Regional Sales": customer_regional_sales,
    }

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

    result = None

    try:
        if category == "General Retail Analysis":
            if not general_analysis or general_analysis == "--Select--":
                result = show_general_insights(df, "Initial Overview")
            else:
                func = general_analysis_functions.get(general_analysis)
                if func:
                    result = func(df)
                else:
                    result = show_general_insights(df, "Initial Overview")

        elif category == "Specific Retail Analysis":
            if specific_analysis_name and specific_analysis_name != "--Select--":
                func = specific_retail_function_mapping.get(specific_analysis_name)
                if func:
                    result = func(df)
                else:
                    result = {
                        "analysis_type": specific_analysis_name,
                        "status": "error",
                        "error_message": f"Analysis '{specific_analysis_name}' not found."
                    }
            else:
                 result = show_general_insights(df, "Specific Analysis Not Selected")
        else:
            # Default action if no category matches
            result = show_general_insights(df, "Initial Overview")

    except Exception as e:
        # Broad exception handler for the dispatcher
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
    

    