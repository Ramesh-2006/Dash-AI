import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import json

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "sales_summary",
    "top_products",
    "customer_analysis",
    "revenue_trends",
    "marketing_analysis",
    "regional_channel_analysis",
    "conversion_analysis",
    "sales_order_fulfillment_and_status_analysis",
    "sales_invoice_and_payment_reconciliation_analysis",
    "sales_transaction_and_profit_margin_analysis",
    "sales_representative_performance_and_revenue_analysis",
    "sales_channel_and_customer_segment_performance_analysis",
    "sales_opportunity_and_pipeline_analysis",
    "sales_quote_conversion_and_pricing_analysis",
    "sales_return_and_refund_analysis",
    "sales_lead_and_opportunity_conversion_analysis",
    "customer_payment_and_reconciliation_analysis",
    "lead_management_and_conversion_funnel_analysis",
    "customer_lifetime_value_and_churn_risk_analysis",
    "subscription_sales_and_renewal_analysis",
    "sales_channel_performance_and_conversion_analysis",
    "cross_sell_and_upsell_opportunity_analysis",
    "sales_territory_performance_and_quota_achievement_analysis",
    "product_sales_performance_and_profitability_analysis",
    "product_pricing_strategy_and_tier_analysis",
    "sales_forecasting_accuracy_analysis",
    "channel_promotion_performance_and_roi_analysis",
    "customer_service_impact_on_sales_analysis",
    "sales_call_outcome_and_effectiveness_analysis",
    "market_segment_revenue_and_profitability_analysis",
    "competitor_pricing_and_feature_analysis",
    "product_bundle_sales_performance_analysis",
    "international_sales_and_currency_exchange_analysis",
    "sales_contract_and_renewal_analysis",
    "e_commerce_sales_funnel_and_conversion_analysis",
    "field_sales_visit_effectiveness_analysis",
    "sales_key_performance_indicator_kpi_trend_analysis",
    "sales_refund_and_reason_code_analysis",
    "lead_nurturing_campaign_effectiveness_analysis",
]

# --- JSON Serialization Helpers ---

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
    This ensures the data is JSON serializable.
    """
    try:
        # Dump to string and reload to force type conversion
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception:
        # Fallback for complex un-serializable objects
        return str(data)

# --- Helper Functions ---

def show_missing_columns_warning(missing_cols, matched_cols=None):
    """Creates a warning message about missing columns."""
    warning_msg = "Missing required columns: " + ", ".join(missing_cols)
    if matched_cols:
        matched_info = []
        for col in missing_cols:
            if matched_cols.get(col):
                matched_info.append(f"{col} (matched to: {matched_cols[col]})")
            else:
                matched_info.append(col)
        warning_msg = "Missing required columns (with matches): " + ", ".join(matched_info)
    return {"warning": warning_msg}


def fuzzy_match_column(df, target_columns):
    """Finds the best fuzzy match for target columns in the dataframe."""
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available) if available else (None, 0)
        matched[target] = match if score >= 70 else None
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    return df.rename(columns={v: k for k, v in matched.items() if v is not None})

def get_key_metrics(df):
    """Calculates general key metrics from the dataframe."""
    return {
        "total_orders": df['order_id'].nunique() if 'order_id' in df.columns else len(df),
        "total_revenue": df['revenue'].sum() if 'revenue' in df.columns else 0,
        "total_customers": df['customer_id'].nunique() if 'customer_id' in df.columns else 0,
        "total_products": df['product_id'].nunique() if 'product_id' in df.columns else 0
    }

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
        
        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / total_rows) * 100
        columns_with_missing = missing_values[missing_values > 0]
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_stats = {}
        if categorical_cols:
            for col in categorical_cols[:5]:  # Limit to first 5 for brevity
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(5).to_dict()
                categorical_stats[col] = {
                    "unique_count": int(unique_count),
                    "top_values": convert_to_native_types(top_values)
                }
        
        # Create visualizations
        visualizations = {}
        
        # 1. Data types distribution
        dtype_counts = {
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'Datetime': len(datetime_cols),
            'Other': len(other_cols)
        }
        fig_dtypes = px.pie(
            values=list(dtype_counts.values()), 
            names=list(dtype_counts.keys()),
            title='Data Types Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        visualizations["data_types_distribution"] = fig_dtypes.to_json()
        
        # 2. Missing values visualization
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
                title='Top 10 Columns with Missing Values (%)',
                labels={'missing_percentage': 'Missing %', 'column': 'Column'},
                color='missing_percentage',
                color_continuous_scale='reds'
            )
            visualizations["missing_values"] = fig_missing.to_json()
        else:
            # Create a simple message visualization when no missing values
            fig_no_missing = go.Figure()
            fig_no_missing.add_annotation(
                text="No Missing Values Found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20, color="green")
            )
            fig_no_missing.update_layout(
                title="Missing Values Analysis",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            visualizations["missing_values"] = fig_no_missing.to_json()
        
        # 3. Numeric columns distributions (first 3 columns)
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:3]):
                try:
                    fig_hist = px.histogram(
                        df, 
                        x=col, 
                        title=f'Distribution of {col}',
                        marginal='box',
                        color_discrete_sequence=['#6366f1']
                    )
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception as e:
                    print(f"Could not create histogram for {col}: {e}")
            
            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr().round(2)
                    fig_corr = px.imshow(
                        corr_matrix, 
                        title='Correlation Matrix of Numeric Columns',
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        text_auto=True
                    )
                    visualizations["correlation_matrix"] = fig_corr.to_json()
                except Exception as e:
                    print(f"Could not create correlation matrix: {e}")
        
        # 4. Categorical columns (first 2 columns)
        if categorical_cols:
            for i, col in enumerate(categorical_cols[:2]):
                try:
                    value_counts = df[col].value_counts().head(10)  # Top 10 values
                    if len(value_counts) > 0:
                        fig_bar = px.bar(
                            x=value_counts.index.astype(str), 
                            y=value_counts.values,
                            title=f'Top 10 Values in {col}',
                            labels={'x': col, 'y': 'Count'},
                            color=value_counts.values,
                            color_continuous_scale='blues'
                        )
                        visualizations[f"{col}_top_values"] = fig_bar.to_json()
                except Exception as e:
                    print(f"Could not create bar chart for {col}: {e}")
        
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
            },
            "sample_columns": {
                "numeric_columns_sample": numeric_cols[:5],
                "categorical_columns_sample": categorical_cols[:5]
            }
        }
        
        # Add numeric statistics if available
        if numeric_stats:
            metrics["numeric_statistics_sample"] = {k: numeric_stats[k] for k in list(numeric_stats.keys())[:3]}
        
        # Add categorical statistics if available
        if categorical_stats:
            metrics["categorical_statistics_sample"] = categorical_stats
        
        # Generate insights - NOW INCLUDING MISSING COLUMNS WARNINGS
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns ({memory_usage_mb:.1f} MB)",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns",
        ]
        
        # Add missing columns warning if provided
        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append("The following columns are needed for the requested analysis but weren't found in your data:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"   - {col}{match_info}")
            insights.append("")
            insights.append("Showing General Analysis instead of the requested specific analysis.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data)")
        else:
            insights.append("No duplicate rows found")
        
        if len(columns_with_missing) > 0:
            top_missing_col = columns_with_missing.idxmax()
            top_missing_pct = missing_percentage[top_missing_col]
            insights.append(f"{len(columns_with_missing)} columns have missing values (max: {top_missing_col} with {top_missing_pct:.1f}% missing)")
        else:
            insights.append("No missing values found in the dataset")
        
        if numeric_cols:
            sample_numeric = ', '.join(numeric_cols[:3])
            insights.append(f"Numeric columns include: {sample_numeric}{'...' if len(numeric_cols) > 3 else ''}")
        
        if categorical_cols:
            sample_categorical = ', '.join(categorical_cols[:3])
            insights.append(f"Categorical columns include: {sample_categorical}{'...' if len(categorical_cols) > 3 else ''}")
        
        insights.append(f"Generated {len(visualizations)} visualizations for comprehensive data exploration")
        
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
        # Ultra-safe fallback with missing columns info
        basic_insights = [
            f"Basic dataset info: {len(df)} rows, {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}",
            "Limited analysis due to data compatibility"
        ]
        
        # Add missing columns warning even in error case
        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append("The following columns are needed for the requested analysis but weren't found in your data:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"   - {col}{match_info}")
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
        
        insights.append(f"Generated {len(visualizations)} visualizations for data exploration")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",  # Still return success for basic info
            "matched_columns": matched_cols or {},
            "visualizations": {},
            "metrics": {
                "dataset_basic_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "column_names": df.columns.tolist()[:10]
                }
            },
            "insights": basic_insights,
            "missing_columns": missing_cols or []
        }

def get_fallback_analysis(df, analysis_type, missing_columns):
    """Helper function for consistent fallback analysis"""
    general_result = show_general_insights(df, analysis_type)
    return {
        "analysis_type": analysis_type,
        "status": "fallback",
        "matched_columns": {},
        "missing_columns": missing_columns,
        "visualizations": general_result["visualizations"],
        "metrics": general_result["metrics"],
            "insights": [
                f"⚠️ REQUIRED COLUMNS NOT FOUND for '{analysis_type}'",
                f"Missing columns: {', '.join(missing_columns)}",
                f"Dataset has {len(df)} rows and {len(df.columns)} columns",
                "Available columns: " + ", ".join(df.columns.tolist()[:8]) + ("..." if len(df.columns) > 8 else ""),
                "Showing General Analysis due to missing required columns."
            ]
    }
# --- Core Analysis Functions (Refactored) ---

def sales_summary(df):
    analysis_type = "Sales Summary"
    try:
        expected = ['order_id', 'order_date', 'customer_id', 'product_id', 'quantity', 'price', 'revenue', 'channel', 'country']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
            df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        
        if 'quantity' in df.columns and 'price' in df.columns:
             df['revenue'] = df['revenue'].fillna(df['quantity'] * df['price'])
        
        df['month'] = df['order_date'].dt.to_period('M').astype(str)
        monthly = df.groupby('month')['revenue'].sum().reset_index()
        fig = px.line(monthly, x='month', y='revenue', title='Monthly Revenue')
        
        visualizations = {
            "monthly_revenue": fig.to_json()
        }
        
        metrics = get_key_metrics(df)
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": [f"Successfully generated {analysis_type}."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def top_products(df):
    analysis_type = "Top Products"
    try:
        expected = ['product_id', 'product_name', 'quantity', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            general_metrics = get_key_metrics(df)
            return {
                "analysis_type": analysis_type,
                "status": "fallback",
                "matched_columns": matched,
                "missing_columns": missing,
                "visualizations": {},
                "metrics": convert_to_native_types(general_metrics),
                "insights": [show_general_insights(df, analysis_type), f"Missing columns for Top Products: {missing}"]
            }

        df = safe_rename(df, matched)
        prod_rev = df.groupby(['product_id', 'product_name']).agg({'quantity':'sum','revenue':'sum'}).reset_index()
        top_by_rev = prod_rev.sort_values('revenue', ascending=False).head(20)
        fig_rev = px.bar(top_by_rev, x='product_name', y='revenue', title='Top Products by Revenue')
        
        visualizations = {
            "top_revenue_products": fig_rev.to_json()
        }
        metrics = get_key_metrics(df)

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": [f"Successfully generated {analysis_type}."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

# --- Specific Sales Analysis Functions (Refactored) ---

def sales_order_fulfillment_and_status_analysis(df):
    analysis_type = "Sales Order Fulfillment and Status Analysis"
    try:
        expected = ['order_id', 'order_date', 'delivery_date', 'order_status', 'customer_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
        df.dropna(subset=['order_id', 'order_status'], inplace=True)

        order_status_counts = df['order_status'].value_counts(normalize=True).reset_index()
        order_status_counts.columns = ['order_status', 'proportion']
        fig_order_status_pie = px.pie(order_status_counts, names='order_status', values='proportion', title='Order Status Distribution')

        avg_fulfillment_time = 'N/A'
        fig_fulfillment_time_dist = go.Figure().add_annotation(text="No completed orders or date data.",
                                                                xref="paper", yref="paper", showarrow=False)

        df_completed = df[df['order_status'].astype(str).str.lower() == 'completed'].copy()
        if not df_completed.empty and 'delivery_date' in df_completed.columns and 'order_date' in df_completed.columns:
            df_completed['fulfillment_time_days'] = (df_completed['delivery_date'] - df_completed['order_date']).dt.days
            if not df_completed['fulfillment_time_days'].isnull().all():
                avg_fulfillment_time = df_completed['fulfillment_time_days'].mean()
                fig_fulfillment_time_dist = px.histogram(df_completed.dropna(subset=['fulfillment_time_days']), 
                                                         x='fulfillment_time_days', nbins=50, 
                                                         title='Distribution of Order Fulfillment Times (Days)')
        
        visualizations = {
            'order_status_distribution': fig_order_status_pie.to_json(),
            'fulfillment_time_distribution': fig_fulfillment_time_dist.to_json()
        }

        metrics = {
            "total_orders": df['order_id'].nunique(),
            "avg_fulfillment_time_days": avg_fulfillment_time
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed order status and fulfillment times."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_invoice_and_payment_reconciliation_analysis(df):
    analysis_type = "Sales Invoice and Payment Reconciliation Analysis"
    try:
        expected = ['invoice_id', 'order_id', 'invoice_amount', 'payment_received_amount', 'payment_status', 'invoice_date', 'payment_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['invoice_id', 'invoice_amount', 'payment_status'], inplace=True)

        df['outstanding_amount'] = df['invoice_amount'] - df.get('payment_received_amount', 0)
        df['reconciliation_status'] = df.apply(lambda x: 'Fully Paid' if x['outstanding_amount'] <= 0.01 else 'Outstanding' if x['outstanding_amount'] > 0 else 'Overpaid', axis=1)

        payment_status_counts = df['payment_status'].value_counts(normalize=True).reset_index()
        payment_status_counts.columns = ['payment_status', 'proportion']
        outstanding_by_status = df.groupby('reconciliation_status')['outstanding_amount'].sum().reset_index()

        fig_payment_status = px.pie(payment_status_counts, names='payment_status', values='proportion', title='Payment Status Distribution')
        fig_outstanding_amount = px.bar(outstanding_by_status, x='reconciliation_status', y='outstanding_amount', title='Total Outstanding Amount by Reconciliation Status')

        visualizations = {
            'payment_status_distribution': fig_payment_status.to_json(),
            'outstanding_amount_by_reconciliation_status': fig_outstanding_amount.to_json()
        }

        metrics = {
            "total_invoices": df['invoice_id'].nunique(),
            "total_billed_amount": df['invoice_amount'].sum(),
            "total_outstanding_amount": df['outstanding_amount'].sum()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed payment status and outstanding amounts."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_transaction_and_profit_margin_analysis(df):
    analysis_type = "Sales Transaction and Profit Margin Analysis"
    try:
        expected = ['transaction_id', 'sale_amount', 'cost_of_goods_sold', 'product_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['transaction_id', 'sale_amount', 'cost_of_goods_sold'], inplace=True)

        df['gross_profit'] = df['sale_amount'] - df['cost_of_goods_sold']
        df['profit_margin_percent'] = (df['gross_profit'] / df['sale_amount']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True) 

        fig_profit_margin_dist = px.histogram(df['profit_margin_percent'].dropna(), nbins=50, title='Distribution of Profit Margins (%)')
        
        fig_top_profit_products = go.Figure().add_annotation(text="Product ID not available.", xref="paper", yref="paper", showarrow=False)
        if 'product_id' in df.columns:
            top_profit_products = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()
            fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

        visualizations = {
            'profit_margin_distribution': fig_profit_margin_dist.to_json(),
            'top_products_by_gross_profit': fig_top_profit_products.to_json()
        }

        metrics = {
            "total_sales_amount": df['sale_amount'].sum(),
            "total_gross_profit": df['gross_profit'].sum(),
            "overall_profit_margin_percent": (df['gross_profit'].sum() / df['sale_amount'].sum()) * 100 if df['sale_amount'].sum() > 0 else 0
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed profit margins and top profitable products."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_representative_performance_and_revenue_analysis(df):
    analysis_type = "Sales Representative Performance and Revenue Analysis"
    try:
        expected = ['sales_representative_id', 'revenue', 'number_of_deals', 'customer_id', 'region']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['sales_representative_id', 'revenue'], inplace=True)

        revenue_by_rep = df.groupby('sales_representative_id')['revenue'].sum().nlargest(10).reset_index()
        fig_revenue_by_rep = px.bar(revenue_by_rep, x='sales_representative_id', y='revenue', title='Top 10 Sales Representatives by Total Revenue')

        fig_avg_revenue_per_deal = go.Figure().add_annotation(text="Number of deals data not available.", xref="paper", yref="paper", showarrow=False)
        if 'number_of_deals' in df.columns:
            df['avg_revenue_per_deal'] = df['revenue'] / df['number_of_deals']
            avg_revenue_per_deal_by_rep = df.groupby('sales_representative_id')['avg_revenue_per_deal'].mean().nlargest(10).reset_index()
            fig_avg_revenue_per_deal = px.bar(avg_revenue_per_deal_by_rep, x='sales_representative_id', y='avg_revenue_per_deal', title='Top 10 Sales Reps by Average Revenue Per Deal')

        visualizations = {
            'revenue_by_sales_representative': fig_revenue_by_rep.to_json(),
            'average_revenue_per_deal_by_representative': fig_avg_revenue_per_deal.to_json()
        }

        metrics = {
            "total_revenue_overall": df['revenue'].sum(),
            "num_unique_sales_reps": df['sales_representative_id'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales representative performance by revenue."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_channel_and_customer_segment_performance_analysis(df):
    analysis_type = "Sales Channel and Customer Segment Performance Analysis"
    try:
        expected = ['sales_channel', 'customer_segment', 'revenue', 'transaction_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['sales_channel', 'customer_segment', 'revenue'], inplace=True)

        revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
        fig_revenue_by_channel_pie = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title='Revenue Distribution by Sales Channel')

        revenue_channel_segment = df.groupby(['sales_channel', 'customer_segment'])['revenue'].sum().unstack(fill_value=0)
        fig_revenue_channel_segment = px.bar(revenue_channel_segment, x=revenue_channel_segment.index, y=revenue_channel_segment.columns,
                                             title='Revenue by Customer Segment per Sales Channel', barmode='group')

        visualizations = {
            'revenue_distribution_by_sales_channel': fig_revenue_by_channel_pie.to_json(),
            'revenue_by_customer_segment_per_channel': fig_revenue_channel_segment.to_json()
        }

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "num_unique_sales_channels": df['sales_channel'].nunique(),
            "num_unique_customer_segments": df['customer_segment'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed revenue by sales channel and customer segment."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_opportunity_and_pipeline_analysis(df):
    analysis_type = "Sales Opportunity and Pipeline Analysis"
    try:
        expected = ['opportunity_id', 'stage', 'expected_revenue', 'close_date', 'sales_representative_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
        df.dropna(subset=['opportunity_id', 'stage', 'expected_revenue'], inplace=True)

        opportunity_by_stage = df['stage'].value_counts().reset_index()
        opportunity_by_stage.columns = ['stage', 'count']
        fig_opportunity_by_stage = px.pie(opportunity_by_stage, names='stage', values='count', title='Opportunity Count by Stage')

        revenue_by_stage = df.groupby('stage')['expected_revenue'].sum().reset_index()
        fig_revenue_by_stage = px.bar(revenue_by_stage, x='stage', y='expected_revenue', title='Total Expected Revenue by Opportunity Stage')

        visualizations = {
            'opportunity_count_by_stage': fig_opportunity_by_stage.to_json(),
            'expected_revenue_by_stage': fig_revenue_by_stage.to_json()
        }

        metrics = {
            "total_opportunities": df['opportunity_id'].nunique(),
            "total_expected_pipeline_revenue": df['expected_revenue'].sum()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales opportunity pipeline by stage and expected revenue."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_quote_conversion_and_pricing_analysis(df):
    analysis_type = "Sales Quote Conversion and Pricing Analysis"
    try:
        expected = ['quote_id', 'customer_id', 'quoted_price', 'conversion_status', 'conversion_date', 'product_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['quote_id', 'quoted_price', 'conversion_status'], inplace=True)

        converted_quotes = df[df['conversion_status'].astype(str).str.lower() == 'converted']
        conversion_rate = (len(converted_quotes) / len(df)) * 100 if len(df) > 0 else 0

        fig_quoted_price_dist = px.histogram(df, x='quoted_price', color='conversion_status', barmode='overlay',
                                             title='Distribution of Quoted Prices by Conversion Status')

        avg_quoted_price_by_status = df.groupby('conversion_status')['quoted_price'].mean().reset_index()
        fig_avg_quoted_price = px.bar(avg_quoted_price_by_status, x='conversion_status', y='quoted_price', title='Average Quoted Price by Conversion Status')

        visualizations = {
            'quoted_price_distribution_by_conversion_status': fig_quoted_price_dist.to_json(),
            'average_quoted_price_by_conversion_status': fig_avg_quoted_price.to_json()
        }

        metrics = {
            "total_quotes": len(df),
            "total_converted_quotes": len(converted_quotes),
            "conversion_rate_percent": conversion_rate
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales quote conversion rates and pricing distribution."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_return_and_refund_analysis(df):
    analysis_type = "Sales Return and Refund Analysis"
    try:
        expected = ['return_id', 'order_id', 'refund_amount', 'return_reason', 'return_date', 'sale_amount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
        df.dropna(subset=['return_id', 'refund_amount'], inplace=True)

        total_refund_amount = df['refund_amount'].sum()
        
        fig_return_reasons = go.Figure().add_annotation(text="Return reason data not available.", xref="paper", yref="paper", showarrow=False)
        if 'return_reason' in df.columns:
            return_reasons_counts = df['return_reason'].value_counts().nlargest(10).reset_index()
            return_reasons_counts.columns = ['reason', 'count']
            fig_return_reasons = px.bar(return_reasons_counts, x='reason', y='count', title='Top 10 Sales Return Reasons')

        monthly_refunds = df.groupby(df['return_date'].dt.to_period('M').dt.start_time)['refund_amount'].sum().reset_index()
        monthly_refunds.columns = ['month_year', 'refund_amount']
        monthly_refunds = monthly_refunds.sort_values('month_year')
        fig_monthly_refunds = px.line(monthly_refunds, x='month_year', y='refund_amount', title='Monthly Sales Refund Trend')

        visualizations = {
            'monthly_refund_trend': fig_monthly_refunds.to_json(),
            'top_return_reasons': fig_return_reasons.to_json()
        }

        metrics = {
            "total_returns": len(df),
            "total_refund_amount": total_refund_amount
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales returns, refund amounts, and top return reasons."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_lead_and_opportunity_conversion_analysis(df):
    analysis_type = "Sales Lead and Opportunity Conversion Analysis"
    try:
        expected = ['lead_id', 'opportunity_id', 'conversion_status', 'lead_source', 'deal_value']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['lead_id', 'conversion_status'], inplace=True)

        total_leads = len(df)
        converted_leads = df[df['conversion_status'].astype(str).str.lower() == 'converted']
        conversion_rate = (len(converted_leads) / total_leads) * 100 if total_leads > 0 else 0

        fig_conversion_by_source = go.Figure().add_annotation(text="Lead source data not available.", xref="paper", yref="paper", showarrow=False)
        if 'lead_source' in df.columns:
            conversion_by_source = df.groupby('lead_source')['conversion_status'].apply(
                lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
            ).reset_index(name='conversion_rate_percent')
            fig_conversion_by_source = px.bar(conversion_by_source, x='lead_source', y='conversion_rate_percent', title='Lead Conversion Rate by Lead Source (%)')

        conversion_status_counts = df['conversion_status'].value_counts(normalize=True).reset_index()
        conversion_status_counts.columns = ['status', 'proportion']
        fig_conversion_status_pie = px.pie(conversion_status_counts, names='status', values='proportion', title='Lead Conversion Status Distribution')

        visualizations = {
            'lead_conversion_status_distribution': fig_conversion_status_pie.to_json(),
            'lead_conversion_rate_by_source': fig_conversion_by_source.to_json()
        }

        metrics = {
            "total_leads": total_leads,
            "total_converted_leads": len(converted_leads),
            "overall_conversion_rate_percent": conversion_rate
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed lead conversion rates and status distribution."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def customer_payment_and_reconciliation_analysis(df):
    analysis_type = "Customer Payment and Reconciliation Analysis"
    try:
        expected = ['customer_id', 'payment_id', 'payment_amount', 'invoice_id', 'payment_status', 'payment_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
        df.dropna(subset=['payment_id', 'payment_amount', 'payment_status'], inplace=True)

        total_payment_received = df['payment_amount'].sum()

        payment_status_dist = df['payment_status'].value_counts(normalize=True).reset_index()
        payment_status_dist.columns = ['status', 'proportion']
        fig_payment_status_pie = px.pie(payment_status_dist, names='status', values='proportion', title='Customer Payment Status Distribution')
        
        fig_monthly_payments = go.Figure().add_annotation(text="Payment date not available for trend.", xref="paper", yref="paper", showarrow=False)
        if not df['payment_date'].isnull().all():
            monthly_payments = df.groupby(df['payment_date'].dt.to_period('M').dt.start_time)['payment_amount'].sum().reset_index()
            monthly_payments.columns = ['month_year', 'payment_amount']
            monthly_payments = monthly_payments.sort_values('month_year')
            fig_monthly_payments = px.line(monthly_payments, x='month_year', y='payment_amount', title='Monthly Customer Payments Received Trend')

        visualizations = {
            'payment_status_distribution': fig_payment_status_pie.to_json(),
            'monthly_payments_trend': fig_monthly_payments.to_json()
        }

        metrics = {
            "total_payments_received": total_payment_received,
            "num_unique_customers_paying": df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed customer payment statuses and trends."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def lead_management_and_conversion_funnel_analysis(df):
    analysis_type = "Lead Management and Conversion Funnel Analysis"
    try:
        expected = ['lead_id', 'lead_status', 'lead_source', 'conversion_date', 'deal_value']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['lead_id', 'lead_status'], inplace=True)

        funnel_stages_counts = df['lead_status'].value_counts().reset_index()
        funnel_stages_counts.columns = ['stage', 'count']
        fig_funnel_stages = px.bar(funnel_stages_counts, x='stage', y='count', title='Lead Conversion Funnel Stages')
        
        fig_conversion_by_source = go.Figure().add_annotation(text="Lead source data not available.", xref="paper", yref="paper", showarrow=False)
        if 'lead_source' in df.columns:
            conversion_by_source = df.groupby('lead_source')['lead_status'].apply(
                lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
            ).reset_index(name='conversion_rate_percent')
            fig_conversion_by_source = px.bar(conversion_by_source, x='lead_source', y='conversion_rate_percent', title='Lead Conversion Rate by Lead Source (%)')

        visualizations = {
            'lead_funnel_stages': fig_funnel_stages.to_json(),
            'lead_conversion_rate_by_source': fig_conversion_by_source.to_json()
        }
        
        total_converted = df[df['lead_status'].astype(str).str.lower() == 'converted'].shape[0]
        total_leads = len(df)
        overall_conversion_rate = (total_converted / total_leads) * 100 if total_leads > 0 else 0

        metrics = {
            "total_leads": total_leads,
            "total_converted_leads": total_converted,
            "overall_conversion_rate_percent": overall_conversion_rate
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed lead management funnel stages and conversion by source."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def customer_lifetime_value_and_churn_risk_analysis(df):
    analysis_type = "Customer Lifetime Value and Churn Risk Analysis"
    try:
        expected = ['customer_id', 'total_revenue', 'number_of_purchases', 'first_purchase_date', 'last_purchase_date', 'churn_status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['customer_id', 'total_revenue', 'number_of_purchases'], inplace=True)
        
        avg_clv_by_customer = df['total_revenue'].mean()
        fig_total_revenue_dist = px.histogram(df, x='total_revenue', nbins=50, title='Distribution of Customer Total Revenue (Proxy for CLV)')

        fig_churn_status = go.Figure().add_annotation(text="Churn status data not available.", xref="paper", yref="paper", showarrow=False)
        churn_rate = 0
        if 'churn_status' in df.columns:
            churn_status_counts = df['churn_status'].value_counts(normalize=True).reset_index()
            churn_status_counts.columns = ['status', 'proportion']
            fig_churn_status = px.pie(churn_status_counts, names='status', values='proportion', title='Customer Churn Status Distribution')
            churned_prop = churn_status_counts[churn_status_counts['status'].astype(str).str.lower() == 'churned']['proportion']
            if not churned_prop.empty:
                churn_rate = churned_prop.sum() * 100

        visualizations = {
            'customer_total_revenue_distribution': fig_total_revenue_dist.to_json(),
            'customer_churn_status_distribution': fig_churn_status.to_json()
        }

        metrics = {
            "total_customers": df['customer_id'].nunique(),
            "avg_clv_estimate": avg_clv_by_customer,
            "churn_rate_percent": churn_rate
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed customer total revenue (as CLV proxy) and churn status."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def subscription_sales_and_renewal_analysis(df):
    analysis_type = "Subscription Sales and Renewal Analysis"
    try:
        expected = ['subscription_id', 'customer_id', 'subscription_type', 'start_date', 'end_date', 'renewal_status', 'revenue_per_subscription']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['subscription_id', 'renewal_status', 'revenue_per_subscription'], inplace=True)

        total_subscriptions = len(df)
        renewed_subscriptions = df[df['renewal_status'].astype(str).str.lower() == 'renewed']
        renewal_rate_percent = (len(renewed_subscriptions) / total_subscriptions) * 100 if total_subscriptions > 0 else 0

        revenue_by_subscription_type = df.groupby('subscription_type')['revenue_per_subscription'].sum().reset_index()
        fig_revenue_by_sub_type = px.bar(revenue_by_subscription_type, x='subscription_type', y='revenue_per_subscription', title='Revenue by Subscription Type')

        renewal_status_counts = df['renewal_status'].value_counts(normalize=True).reset_index()
        renewal_status_counts.columns = ['status', 'proportion']
        fig_renewal_status = px.pie(renewal_status_counts, names='status', values='proportion', title='Subscription Renewal Status Distribution')

        visualizations = {
            'renewal_status_distribution': fig_renewal_status.to_json(),
            'revenue_by_subscription_type': fig_revenue_by_sub_type.to_json()
        }

        metrics = {
            "total_subscriptions": total_subscriptions,
            "total_revenue_from_subscriptions": df['revenue_per_subscription'].sum(),
            "renewal_rate_percent": renewal_rate_percent
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed subscription sales, revenue by type, and renewal rates."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_channel_performance_and_conversion_analysis(df):
    analysis_type = "Sales Channel Performance and Conversion Analysis"
    try:
        expected = ['sales_channel', 'revenue', 'transactions', 'leads_generated', 'converted_customers']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

        revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
        fig_revenue_by_channel_bar = px.bar(revenue_by_channel, x='sales_channel', y='revenue', title='Total Revenue by Sales Channel')

        fig_conversion_rate_channel = go.Figure().add_annotation(text="Leads or conversion data missing.", xref="paper", yref="paper", showarrow=False)
        if 'leads_generated' in df.columns and 'converted_customers' in df.columns:
            df['conversion_rate_percent'] = (df['converted_customers'] / df['leads_generated']) * 100
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            avg_conversion_rate_by_channel = df.groupby('sales_channel')['conversion_rate_percent'].mean().reset_index()
            fig_conversion_rate_channel = px.bar(avg_conversion_rate_by_channel.dropna(), x='sales_channel', y='conversion_rate_percent', title='Average Conversion Rate by Sales Channel (%)')

        visualizations = {
            'revenue_by_sales_channel': fig_revenue_by_channel_bar.to_json(),
            'conversion_rate_by_sales_channel': fig_conversion_rate_channel.to_json()
        }

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "num_unique_sales_channels": df['sales_channel'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales channel revenue and conversion rates."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def cross_sell_and_upsell_opportunity_analysis(df):
    analysis_type = "Cross-Sell and Up-Sell Opportunity Analysis"
    try:
        expected = ['customer_id', 'product_id', 'total_sales_value', 'cross_sell_potential', 'upsell_potential', 'customer_segment']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['customer_id'], inplace=True)
        
        metrics = {"total_customers_analyzed": df['customer_id'].nunique()}
        visualizations = {}

        fig_cross_sell_pie = go.Figure().add_annotation(text="Cross-sell potential data not available.", xref="paper", yref="paper", showarrow=False)
        if 'cross_sell_potential' in df.columns:
            cross_sell_counts = df['cross_sell_potential'].value_counts(normalize=True).reset_index()
            cross_sell_counts.columns = ['potential', 'proportion']
            fig_cross_sell_pie = px.pie(cross_sell_counts, names='potential', values='proportion', title='Customer Cross-Sell Potential Distribution')
            metrics["total_cross_sell_potential_customers"] = df[df['cross_sell_potential'].astype(str).str.lower() == 'yes']['customer_id'].nunique()
        else:
            metrics["total_cross_sell_potential_customers"] = 'N/A'

        fig_upsell_segment = go.Figure().add_annotation(text="Up-sell potential or customer segment data not available.", xref="paper", yref="paper", showarrow=False)
        if 'upsell_potential' in df.columns and 'customer_segment' in df.columns:
            upsell_potential_by_segment = df.groupby('customer_segment')['upsell_potential'].sum().reset_index()
            fig_upsell_segment = px.bar(upsell_potential_by_segment, x='customer_segment', y='upsell_potential', title='Total Up-Sell Potential by Customer Segment')
            metrics["total_upsell_potential_value"] = df['upsell_potential'].sum()
        else:
            metrics["total_upsell_potential_value"] = 'N/A'

        visualizations = {
            'customer_cross_sell_potential': fig_cross_sell_pie.to_json(),
            'upsell_potential_by_customer_segment': fig_upsell_segment.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed cross-sell and up-sell opportunities by customer segment."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_territory_performance_and_quota_achievement_analysis(df):
    analysis_type = "Sales Territory Performance and Quota Achievement Analysis"
    try:
        expected = ['territory_id', 'sales_representative_id', 'revenue', 'quota', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['territory_id', 'revenue'], inplace=True)

        revenue_by_territory = df.groupby('territory_id')['revenue'].sum().reset_index()
        fig_revenue_by_territory = px.bar(revenue_by_territory, x='territory_id', y='revenue', title='Total Revenue by Sales Territory')

        fig_quota_achievement = go.Figure().add_annotation(text="Quota data not available for achievement rate analysis.", xref="paper", yref="paper", showarrow=False)
        if 'quota' in df.columns:
            territory_performance = df.groupby('territory_id').agg(
                total_revenue=('revenue', 'sum'),
                total_quota=('quota', 'sum')
            ).reset_index()
            territory_performance['achievement_rate_percent'] = (territory_performance['total_revenue'] / territory_performance['total_quota']) * 100
            territory_performance.replace([np.inf, -np.inf], np.nan, inplace=True)
            fig_quota_achievement = px.bar(territory_performance.dropna(), x='territory_id', y='achievement_rate_percent', title='Sales Territory Quota Achievement Rate (%)')

        visualizations = {
            'revenue_by_sales_territory': fig_revenue_by_territory.to_json(),
            'quota_achievement_rate_by_territory': fig_quota_achievement.to_json()
        }

        metrics = {
            "total_revenue_across_territories": df['revenue'].sum(),
            "num_unique_territories": df['territory_id'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales territory performance and quota achievement rates."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def product_sales_performance_and_profitability_analysis(df):
    analysis_type = "Product Sales Performance and Profitability Analysis"
    try:
        expected = ['product_id', 'product_category', 'sales_amount', 'cost_of_goods_sold', 'quantity_sold']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['product_id', 'sales_amount', 'cost_of_goods_sold'], inplace=True)

        df['gross_profit'] = df['sales_amount'] - df['cost_of_goods_sold']
        df['profit_margin_percent'] = (df['gross_profit'] / df['sales_amount']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        top_profit_products = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()
        fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

        fig_category_profitability = go.Figure().add_annotation(text="Product category data not available for category profitability.", xref="paper", yref="paper", showarrow=False)
        if 'product_category' in df.columns:
            category_profitability = df.groupby('product_category').agg(
                total_sales=('sales_amount', 'sum'),
                total_gross_profit=('gross_profit', 'sum')
            ).reset_index()
            fig_category_profitability = px.bar(category_profitability, x='product_category', y=['total_sales', 'total_gross_profit'],
                                                title='Total Sales and Gross Profit by Product Category', barmode='group')

        visualizations = {
            'top_products_by_gross_profit': fig_top_profit_products.to_json(),
            'sales_and_profit_by_product_category': fig_category_profitability.to_json()
        }

        metrics = {
            "total_sales_amount": df['sales_amount'].sum(),
            "total_gross_profit": df['gross_profit'].sum(),
            "overall_product_profit_margin_percent": (df['gross_profit'].sum() / df['sales_amount'].sum()) * 100 if df['sales_amount'].sum() > 0 else 0
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed product sales performance and profitability by category."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

# Continue with the remaining analysis functions following the same pattern...

def product_pricing_strategy_and_tier_analysis(df):
    analysis_type = "Product Pricing Strategy and Tier Analysis"
    try:
        expected = ['product_id', 'price_tier', 'unit_price', 'sales_volume', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['product_id', 'unit_price', 'sales_volume'], inplace=True)

        fig_revenue_by_tier = go.Figure().add_annotation(text="Price tier or revenue data not available.", xref="paper", yref="paper", showarrow=False)
        if 'price_tier' in df.columns and 'revenue' in df.columns:
            revenue_by_tier = df.groupby('price_tier')['revenue'].sum().reset_index()
            fig_revenue_by_tier = px.pie(revenue_by_tier, names='price_tier', values='revenue', title='Revenue Distribution by Price Tier')

        fig_price_vs_volume = go.Figure().add_annotation(text="Sales volume or unit price data not available for scatter plot.", xref="paper", yref="paper", showarrow=False)
        if 'sales_volume' in df.columns and 'unit_price' in df.columns:
            product_agg = df.groupby('product_id').agg(
                avg_unit_price=('unit_price', 'mean'),
                total_sales_volume=('sales_volume', 'sum')
            ).reset_index()
            fig_price_vs_volume = px.scatter(product_agg, x='avg_unit_price', y='total_sales_volume',
                                             title='Average Unit Price vs. Total Sales Volume',
                                             hover_name='product_id')

        visualizations = {
            'revenue_distribution_by_price_tier': fig_revenue_by_tier.to_json(),
            'average_unit_price_vs_total_sales_volume': fig_price_vs_volume.to_json()
        }

        metrics = {
            "total_revenue_overall": df['revenue'].sum() if 'revenue' in df.columns else 'N/A',
            "num_unique_price_tiers": df['price_tier'].nunique() if 'price_tier' in df.columns else 'N/A'
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed pricing strategy and revenue distribution by price tier."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }


def sales_forecasting_accuracy_analysis(df):
    analysis_type = "Sales Forecasting Accuracy Analysis"
    try:
        expected = ['period', 'actual_sales', 'forecasted_sales']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['period', 'actual_sales', 'forecasted_sales'], inplace=True)

        df['forecast_error'] = df['actual_sales'] - df['forecasted_sales']
        df['absolute_forecast_error'] = df['forecast_error'].abs()
        df['percentage_error'] = (df['forecast_error'] / df['actual_sales']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        mape = df['percentage_error'].abs().mean() if df['actual_sales'].sum() > 0 else 0

        fig_actual_vs_forecast = px.line(df, x='period', y=['actual_sales', 'forecasted_sales'],
                                         title='Actual vs. Forecasted Sales Over Time')

        fig_error_distribution = px.histogram(df['forecast_error'].dropna(), nbins=50, title='Distribution of Forecast Errors')

        visualizations = {
            'actual_vs_forecasted_sales_trend': fig_actual_vs_forecast.to_json(),
            'forecast_error_distribution': fig_error_distribution.to_json()
        }

        metrics = {
            "total_actual_sales": df['actual_sales'].sum(),
            "total_forecasted_sales": df['forecasted_sales'].sum(),
            "mean_absolute_percentage_error": mape
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales forecasting accuracy and error distribution."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def channel_promotion_performance_and_roi_analysis(df):
    analysis_type = "Channel Promotion Performance and ROI Analysis"
    try:
        expected = ['promotion_id', 'sales_channel', 'revenue', 'promotion_cost', 'start_date', 'end_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['promotion_id', 'revenue', 'promotion_cost'], inplace=True)

        df['promotion_roi'] = ((df['revenue'] - df['promotion_cost']) / df['promotion_cost']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        top_roi_promotions = df.groupby('promotion_id')['promotion_roi'].mean().nlargest(10).reset_index()

        fig_revenue_by_channel_promo = go.Figure().add_annotation(text="Sales channel data not available.", xref="paper", yref="paper", showarrow=False)
        if 'sales_channel' in df.columns:
            revenue_by_channel_promo = df.groupby('sales_channel')['revenue'].sum().reset_index()
            fig_revenue_by_channel_promo = px.bar(revenue_by_channel_promo, x='sales_channel', y='revenue', title='Total Promotional Revenue by Sales Channel')

        fig_top_roi_promotions = px.bar(top_roi_promotions.dropna(), x='promotion_id', y='promotion_roi', title='Top 10 Promotions by ROI (%)')

        visualizations = {
            'top_promotions_by_roi': fig_top_roi_promotions.to_json(),
            'promotional_revenue_by_sales_channel': fig_revenue_by_channel_promo.to_json()
        }

        metrics = {
            "total_promotional_revenue": df['revenue'].sum(),
            "total_promotion_cost": df['promotion_cost'].sum(),
            "avg_promotion_roi_percent": df['promotion_roi'].mean()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed promotion performance and ROI across sales channels."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def customer_service_impact_on_sales_analysis(df):
    analysis_type = "Customer Service Impact on Sales Analysis"
    try:
        expected = ['customer_id', 'sales_amount', 'customer_service_interaction_count', 'satisfaction_score', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['customer_id', 'sales_amount'], inplace=True)

        fig_sales_vs_interactions = go.Figure().add_annotation(text="Customer service interaction count data not available.", xref="paper", yref="paper", showarrow=False)
        if 'customer_service_interaction_count' in df.columns:
            avg_sales_by_interactions = df.groupby('customer_service_interaction_count')['sales_amount'].mean().reset_index()
            fig_sales_vs_interactions = px.bar(avg_sales_by_interactions, x='customer_service_interaction_count', y='sales_amount',
                                                title='Average Sales Amount by Customer Service Interaction Count')

        fig_sales_by_satisfaction = go.Figure().add_annotation(text="Customer satisfaction score data not available.", xref="paper", yref="paper", showarrow=False)
        if 'satisfaction_score' in df.columns:
            sales_by_satisfaction = df.groupby('satisfaction_score')['sales_amount'].sum().reset_index()
            fig_sales_by_satisfaction = px.bar(sales_by_satisfaction, x='satisfaction_score', y='sales_amount', title='Total Sales by Customer Satisfaction Score')

        visualizations = {
            'average_sales_by_interactions': fig_sales_vs_interactions.to_json(),
            'total_sales_by_customer_satisfaction_score': fig_sales_by_satisfaction.to_json()
        }

        metrics = {
            "total_sales": df['sales_amount'].sum(),
            "num_customers_with_interactions": df['customer_id'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed impact of customer service interactions and satisfaction on sales."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_call_outcome_and_effectiveness_analysis(df):
    analysis_type = "Sales Call Outcome and Effectiveness Analysis"
    try:
        expected = ['call_id', 'sales_representative_id', 'call_outcome', 'call_duration_minutes', 'deal_closed']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['call_id', 'call_outcome'], inplace=True)

        call_outcome_counts = df['call_outcome'].value_counts(normalize=True).reset_index()
        call_outcome_counts.columns = ['outcome', 'proportion']

        call_conversion_rate = 'N/A'
        fig_avg_call_duration = go.Figure().add_annotation(text="Deal closed data not available for conversion analysis.", xref="paper", yref="paper", showarrow=False)
        if 'deal_closed' in df.columns:
            total_calls = len(df)
            converted_calls = df[df['deal_closed'].astype(bool) == True]
            call_conversion_rate = (len(converted_calls) / total_calls) * 100 if total_calls > 0 else 0

            if 'call_duration_minutes' in df.columns:
                avg_duration_by_conversion = df.groupby('deal_closed')['call_duration_minutes'].mean().reset_index()
                avg_duration_by_conversion['deal_closed'] = avg_duration_by_conversion['deal_closed'].map({True: 'Deal Closed', False: 'No Deal'})
                fig_avg_call_duration = px.bar(avg_duration_by_conversion, x='deal_closed', y='call_duration_minutes', title='Average Call Duration for Deal Closed vs. No Deal (Minutes)')

        fig_call_outcome_pie = px.pie(call_outcome_counts, names='outcome', values='proportion', title='Sales Call Outcome Distribution')

        visualizations = {
            'call_outcome_distribution': fig_call_outcome_pie.to_json(),
            'average_call_duration_by_conversion': fig_avg_call_duration.to_json()
        }

        metrics = {
            "total_sales_calls": len(df),
            "call_conversion_rate_percent": call_conversion_rate
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed sales call outcomes, conversion rates, and call duration effectiveness."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def market_segment_revenue_and_profitability_analysis(df):
    analysis_type = "Market Segment Revenue and Profitability Analysis"
    try:
        expected = ['market_segment', 'revenue', 'cost_of_goods_sold', 'customer_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['market_segment', 'revenue', 'cost_of_goods_sold'], inplace=True)

        df['gross_profit'] = df['revenue'] - df['cost_of_goods_sold']
        df['profit_margin_percent'] = (df['gross_profit'] / df['revenue']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        segment_performance = df.groupby('market_segment').agg(
            total_revenue=('revenue', 'sum'),
            total_gross_profit=('gross_profit', 'sum'),
            avg_profit_margin=('profit_margin_percent', 'mean')
        ).reset_index()

        fig_segment_revenue = px.bar(segment_performance, x='market_segment', y='total_revenue', title='Total Revenue by Market Segment')
        fig_segment_profit_margin = px.bar(segment_performance.dropna(), x='market_segment', y='avg_profit_margin', title='Average Profit Margin by Market Segment (%)')

        visualizations = {
            'revenue_by_market_segment': fig_segment_revenue.to_json(),
            'profit_margin_by_market_segment': fig_segment_profit_margin.to_json()
        }

        metrics = {
            "total_revenue_overall": df['revenue'].sum(),
            "total_gross_profit_overall": df['gross_profit'].sum(),
            "num_unique_market_segments": df['market_segment'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed revenue and profitability across different market segments."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def competitor_pricing_and_feature_analysis(df):
    analysis_type = "Competitor Pricing and Feature Analysis"
    try:
        expected = ['product_name', 'our_price', 'competitor_price_1', 'competitor_price_2', 'feature_advantage_score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['product_name', 'our_price'], inplace=True)

        fig_price_comparison = go.Figure().add_annotation(text="Competitor pricing data not available for comparison.", xref="paper", yref="paper", showarrow=False)
        if 'competitor_price_1' in df.columns and 'competitor_price_2' in df.columns:
            sample_products = df.head(5).copy()
            price_comparison_data = []
            for index, row in sample_products.iterrows():
                price_comparison_data.append({'Product': row['product_name'], 'Source': 'Our Price', 'Price': row['our_price']})
                price_comparison_data.append({'Product': row['product_name'], 'Source': 'Competitor 1', 'Price': row['competitor_price_1']})
                price_comparison_data.append({'Product': row['product_name'], 'Source': 'Competitor 2', 'Price': row['competitor_price_2']})
            price_comparison_df = pd.DataFrame(price_comparison_data)
            fig_price_comparison = px.bar(price_comparison_df, x='Product', y='Price', color='Source',
                                          barmode='group', title='Price Comparison for Sample Products')

        fig_feature_advantage_dist = go.Figure().add_annotation(text="Feature advantage score data not available.", xref="paper", yref="paper", showarrow=False)
        if 'feature_advantage_score' in df.columns:
            fig_feature_advantage_dist = px.histogram(df['feature_advantage_score'].dropna(), nbins=50, title='Distribution of Feature Advantage Scores')

        visualizations = {
            'price_comparison_with_competitors': fig_price_comparison.to_json(),
            'feature_advantage_score_distribution': fig_feature_advantage_dist.to_json()
        }

        metrics = {
            "num_products_analyzed": len(df),
            "avg_our_price": df['our_price'].mean()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed competitor pricing and feature advantage scores."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def product_bundle_sales_performance_analysis(df):
    analysis_type = "Product Bundle Sales Performance Analysis"
    try:
        expected = ['bundle_id', 'bundle_name', 'total_sales_revenue', 'quantity_sold_bundles', 'product_ids_in_bundle']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['bundle_id', 'total_sales_revenue', 'quantity_sold_bundles'], inplace=True)

        top_revenue_bundles = df.groupby('bundle_name')['total_sales_revenue'].sum().nlargest(10).reset_index()

        df['avg_revenue_per_bundle'] = df['total_sales_revenue'] / df['quantity_sold_bundles']
        avg_revenue_per_bundle_overall = df['avg_revenue_per_bundle'].mean()

        fig_top_revenue_bundles = px.bar(top_revenue_bundles, x='bundle_name', y='total_sales_revenue', title='Top 10 Product Bundles by Total Sales Revenue')

        fig_quantity_sold_bundles_dist = px.histogram(df['quantity_sold_bundles'], nbins=20, title='Distribution of Quantity Sold for Bundles')

        visualizations = {
            'top_revenue_bundles': fig_top_revenue_bundles.to_json(),
            'quantity_sold_bundles_distribution': fig_quantity_sold_bundles_dist.to_json()
        }

        metrics = {
            "total_bundle_revenue": df['total_sales_revenue'].sum(),
            "total_bundles_sold": df['quantity_sold_bundles'].sum(),
            "avg_revenue_per_bundle_sold": avg_revenue_per_bundle_overall
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed product bundle sales performance and revenue distribution."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def international_sales_and_currency_exchange_analysis(df):
    analysis_type = "International Sales and Currency Exchange Analysis"
    try:
        expected = ['transaction_id', 'country', 'sales_amount_local_currency', 'exchange_rate_to_usd', 'sales_amount_usd', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['transaction_id', 'country', 'sales_amount_local_currency'], inplace=True)

        if 'sales_amount_usd' not in df.columns and 'exchange_rate_to_usd' in df.columns:
            df['sales_amount_usd'] = df['sales_amount_local_currency'] * df['exchange_rate_to_usd']
        elif 'sales_amount_usd' not in df.columns:
            df['sales_amount_usd'] = df['sales_amount_local_currency']

        sales_usd_by_country = df.groupby('country')['sales_amount_usd'].sum().reset_index()

        fig_exchange_rate_trend = go.Figure().add_annotation(text="Transaction date or exchange rate data not available for trend.", xref="paper", yref="paper", showarrow=False)
        if 'transaction_date' in df.columns and 'exchange_rate_to_usd' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
            monthly_avg_exchange_rate = df.groupby(df['transaction_date'].dt.to_period('M').dt.start_time)['exchange_rate_to_usd'].mean().reset_index()
            monthly_avg_exchange_rate.columns = ['month_year', 'avg_exchange_rate']
            monthly_avg_exchange_rate = monthly_avg_exchange_rate.sort_values('month_year')
            fig_exchange_rate_trend = px.line(monthly_avg_exchange_rate, x='month_year', y='avg_exchange_rate', title='Monthly Average Exchange Rate to USD Trend')

        fig_sales_usd_by_country = px.choropleth(sales_usd_by_country, locations='country', locationmode='country names',
                                                 color='sales_amount_usd', hover_name='country',
                                                 color_continuous_scale=px.colors.sequential.Plasma,
                                                 title='Total Sales in USD by Country')

        visualizations = {
            'sales_in_usd_by_country_map': fig_sales_usd_by_country.to_json(),
            'average_exchange_rate_trend': fig_exchange_rate_trend.to_json()
        }

        metrics = {
            "total_international_sales_usd": df['sales_amount_usd'].sum(),
            "num_unique_countries": df['country'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed international sales performance and currency exchange trends."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_contract_and_renewal_analysis(df):
    analysis_type = "Sales Contract and Renewal Analysis"
    try:
        expected = ['contract_id', 'customer_id', 'contract_value', 'start_date', 'end_date', 'renewal_status', 'sales_representative_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['contract_id', 'contract_value', 'renewal_status'], inplace=True)

        renewal_status_counts = df['renewal_status'].value_counts(normalize=True).reset_index()
        renewal_status_counts.columns = ['status', 'proportion']

        contract_value_by_renewal_status = df.groupby('renewal_status')['contract_value'].sum().reset_index()
        fig_contract_value_renewal = px.bar(contract_value_by_renewal_status, x='renewal_status', y='contract_value', title='Total Contract Value by Renewal Status')

        fig_renewal_status_pie = px.pie(renewal_status_counts, names='status', values='proportion', title='Contract Renewal Status Distribution')

        visualizations = {
            'contract_renewal_status_distribution': fig_renewal_status_pie.to_json(),
            'total_contract_value_by_renewal_status': fig_contract_value_renewal.to_json()
        }

        metrics = {
            "total_contracts": len(df),
            "total_contract_value_overall": df['contract_value'].sum(),
            "renewal_rate_percent": renewal_status_counts[renewal_status_counts['status'].astype(str).str.lower() == 'renewed']['proportion'].sum() * 100 if 'renewed' in renewal_status_counts['status'].astype(str).str.lower().values else 0
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed contract renewal rates and value distribution by renewal status."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def e_commerce_sales_funnel_and_conversion_analysis(df):
    analysis_type = "E-commerce Sales Funnel and Conversion Analysis"
    try:
        expected = ['session_id', 'product_view_count', 'add_to_cart_count', 'checkout_initiated_count', 'purchase_completed_count', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['session_id'], inplace=True)

        funnel_data = {
            'Stage': [],
            'Count': []
        }
        if 'product_view_count' in df.columns:
            funnel_data['Stage'].append('Product Views')
            funnel_data['Count'].append(df['product_view_count'].sum())
        if 'add_to_cart_count' in df.columns:
            funnel_data['Stage'].append('Add to Carts')
            funnel_data['Count'].append(df['add_to_cart_count'].sum())
        if 'checkout_initiated_count' in df.columns:
            funnel_data['Stage'].append('Checkouts Initiated')
            funnel_data['Count'].append(df['checkout_initiated_count'].sum())
        if 'purchase_completed_count' in df.columns:
            funnel_data['Stage'].append('Purchases Completed')
            funnel_data['Count'].append(df['purchase_completed_count'].sum())

        funnel_df = pd.DataFrame(funnel_data)

        overall_conversion_rate = 'N/A'
        if 'purchase_completed_count' in df.columns and 'product_view_count' in df.columns and df['product_view_count'].sum() > 0:
            overall_conversion_rate = (df['purchase_completed_count'].sum() / df['product_view_count'].sum()) * 100
        elif 'purchase_completed_count' in df.columns and len(df) > 0:
            overall_conversion_rate = (df['purchase_completed_count'].sum() / len(df)) * 100

        fig_revenue_conversion_status = go.Figure().add_annotation(text="Revenue or purchase completion data not available.", xref="paper", yref="paper", showarrow=False)
        if 'revenue' in df.columns and 'purchase_completed_count' in df.columns:
            df['is_purchased'] = (df['purchase_completed_count'] > 0).astype(str)
            revenue_by_purchase_status = df.groupby('is_purchased')['revenue'].sum().reset_index()
            revenue_by_purchase_status['is_purchased'] = revenue_by_purchase_status['is_purchased'].map({'True': 'Purchased', 'False': 'Not Purchased'})
            fig_revenue_conversion_status = px.pie(revenue_by_purchase_status, names='is_purchased', values='revenue', title='Revenue by Purchase Completion Status')

        fig_sales_funnel = px.funnel(funnel_df, x='Count', y='Stage', title='E-commerce Sales Funnel')

        visualizations = {
            'e_commerce_sales_funnel': fig_sales_funnel.to_json(),
            'revenue_by_purchase_completion_status': fig_revenue_conversion_status.to_json()
        }

        metrics = {
            "total_sessions": len(df),
            "overall_purchase_conversion_rate_percent": overall_conversion_rate,
            "total_e_commerce_revenue": df['revenue'].sum() if 'revenue' in df.columns else 'N/A'
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed e-commerce sales funnel conversion rates and revenue distribution."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def field_sales_visit_effectiveness_analysis(df):
    analysis_type = "Field Sales Visit Effectiveness Analysis"
    try:
        expected = ['visit_id', 'sales_representative_id', 'visit_date', 'outcome_status', 'deal_value_closed', 'customer_id', 'visit_duration_minutes']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df.dropna(subset=['visit_id', 'outcome_status'], inplace=True)

        visit_outcome_counts = df['outcome_status'].value_counts(normalize=True).reset_index()
        visit_outcome_counts.columns = ['outcome', 'proportion']

        avg_deal_value_per_visit = 'N/A'
        fig_deal_value_dist = go.Figure().add_annotation(text="Deal value data not available for effectiveness analysis.", xref="paper", yref="paper", showarrow=False)
        if 'deal_value_closed' in df.columns:
            successful_visits = df[df['outcome_status'].astype(str).str.lower() == 'deal closed'].copy()
            if not successful_visits.empty:
                avg_deal_value_per_visit = successful_visits['deal_value_closed'].mean()
                fig_deal_value_dist = px.histogram(successful_visits, x='deal_value_closed', nbins=50, title='Distribution of Deal Values Closed from Visits')

        fig_visit_outcome_pie = px.pie(visit_outcome_counts, names='outcome', values='proportion', title='Field Sales Visit Outcome Distribution')

        visualizations = {
            'visit_outcome_distribution': fig_visit_outcome_pie.to_json(),
            'deal_value_closed_distribution_from_visits': fig_deal_value_dist.to_json()
        }

        metrics = {
            "total_field_visits": len(df),
            "success_rate_percent": visit_outcome_counts[visit_outcome_counts['outcome'].astype(str).str.lower() == 'deal closed']['proportion'].sum() * 100 if 'deal closed' in visit_outcome_counts['outcome'].astype(str).str.lower().values else 0,
            "avg_deal_value_per_successful_visit": avg_deal_value_per_visit
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed field sales visit effectiveness and deal closure rates."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_key_performance_indicator_kpi_trend_analysis(df):
    analysis_type = "Sales Key Performance Indicator (KPI) Trend Analysis"
    try:
        expected = ['date', 'kpi_name', 'kpi_value', 'sales_representative_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=['date', 'kpi_name', 'kpi_value'], inplace=True)

        fig_kpi_trend = go.Figure().add_annotation(text="KPI name data not available for trend analysis.", xref="paper", yref="paper", showarrow=False)
        if not df['kpi_name'].empty:
            target_kpi = 'Revenue' if 'revenue' in df['kpi_name'].astype(str).str.lower().values else df['kpi_name'].mode()[0]
            kpi_trend = df[df['kpi_name'] == target_kpi].groupby(df['date'].dt.to_period('M').dt.start_time)['kpi_value'].sum().reset_index()
            kpi_trend.columns = ['month_year', 'kpi_value']
            kpi_trend = kpi_trend.sort_values('month_year')
            fig_kpi_trend = px.line(kpi_trend, x='month_year', y='kpi_value', title=f'Monthly Trend for {target_kpi} KPI')

        fig_kpi_distribution = px.box(df, x='kpi_name', y='kpi_value', title='KPI Value Distribution by KPI Name')

        visualizations = {
            'kpi_monthly_trend': fig_kpi_trend.to_json(),
            'kpi_value_distribution_by_kpi_name': fig_kpi_distribution.to_json()
        }

        metrics = {
            "total_kpi_records": len(df),
            "num_unique_kpis": df['kpi_name'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed KPI trends and distribution across different performance indicators."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def sales_refund_and_reason_code_analysis(df):
    analysis_type = "Sales Refund and Reason Code Analysis"
    try:
        expected = ['refund_id', 'order_id', 'refund_amount', 'reason_code', 'refund_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['refund_date'] = pd.to_datetime(df['refund_date'], errors='coerce')
        df.dropna(subset=['refund_id', 'refund_amount'], inplace=True)

        fig_refund_amount_by_reason = go.Figure().add_annotation(text="Reason code data not available.", xref="paper", yref="paper", showarrow=False)
        if 'reason_code' in df.columns:
            refund_amount_by_reason = df.groupby('reason_code')['refund_amount'].sum().nlargest(10).reset_index()
            fig_refund_amount_by_reason = px.bar(refund_amount_by_reason, x='reason_code', y='refund_amount', title='Total Refund Amount by Top 10 Reason Codes')

        monthly_refunds = df.groupby(df['refund_date'].dt.to_period('M').dt.start_time)['refund_amount'].sum().reset_index()
        monthly_refunds.columns = ['month_year', 'refund_amount']
        monthly_refunds = monthly_refunds.sort_values('month_year')

        fig_monthly_refunds_trend = px.line(monthly_refunds, x='month_year', y='refund_amount', title='Monthly Sales Refund Amount Trend')

        visualizations = {
            'total_refund_amount_by_reason_code': fig_refund_amount_by_reason.to_json(),
            'monthly_sales_refund_amount_trend': fig_monthly_refunds_trend.to_json()
        }

        metrics = {
            "total_refunds_processed": len(df),
            "total_refund_amount_overall": df['refund_amount'].sum()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed refund patterns by reason code and monthly trends."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def lead_nurturing_campaign_effectiveness_analysis(df):
    analysis_type = "Lead Nurturing Campaign Effectiveness Analysis"
    try:
        expected = ['campaign_id', 'lead_id', 'engagement_score', 'conversion_status', 'channel', 'cost_per_lead']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['campaign_id', 'lead_id', 'conversion_status'], inplace=True)

        conversion_by_campaign = df.groupby('campaign_id')['conversion_status'].apply(
            lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
        ).reset_index(name='conversion_rate_percent')

        fig_avg_engagement_channel = go.Figure().add_annotation(text="Engagement score or channel data not available.", xref="paper", yref="paper", showarrow=False)
        if 'engagement_score' in df.columns and 'channel' in df.columns:
            avg_engagement_by_channel = df.groupby('channel')['engagement_score'].mean().reset_index()
            fig_avg_engagement_channel = px.bar(avg_engagement_by_channel, x='channel', y='engagement_score', title='Average Engagement Score by Campaign Channel')

        fig_conversion_by_campaign = px.bar(conversion_by_campaign, x='campaign_id', y='conversion_rate_percent', title='Lead Conversion Rate by Nurturing Campaign (%)')

        visualizations = {
            'conversion_rate_by_campaign': fig_conversion_by_campaign.to_json(),
            'average_engagement_by_campaign_channel': fig_avg_engagement_channel.to_json()
        }

        metrics = {
            "total_leads_in_campaigns": len(df),
            "num_unique_campaigns": df['campaign_id'].nunique(),
            "overall_conversion_rate_percent": (df[df['conversion_status'].astype(str).str.lower() == 'converted'].shape[0] / len(df)) * 100 if len(df) > 0 else 0
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed lead nurturing campaign effectiveness and engagement metrics."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

# Update the analysis function mapping with all completed functions


# Add the remaining general analysis functions
def customer_analysis(df):
    analysis_type = "Customer Analysis"
    try:
        expected = ['customer_id', 'transaction_date', 'revenue', 'order_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

        num_unique_customers = df['customer_id'].nunique()
        avg_revenue_per_customer = df.groupby('customer_id')['revenue'].sum().mean()

        top_customers_by_revenue = df.groupby('customer_id')['revenue'].sum().nlargest(10).reset_index()

        customer_frequency = df.groupby('customer_id')['order_id'].nunique().reset_index(name='num_orders')

        fig_top_customers = px.bar(top_customers_by_revenue, x='customer_id', y='revenue', title='Top 10 Customers by Total Revenue')
        fig_customer_frequency_distribution = px.histogram(customer_frequency, x='num_orders', nbins=50, title='Distribution of Customer Transaction Frequency')

        visualizations = {
            'top_customers_by_revenue': fig_top_customers.to_json(),
            'customer_frequency_distribution': fig_customer_frequency_distribution.to_json()
        }

        metrics = {
            "num_unique_customers": num_unique_customers,
            "avg_revenue_per_customer": avg_revenue_per_customer,
            "avg_transactions_per_customer": customer_frequency['num_orders'].mean()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed customer behavior, revenue distribution, and transaction frequency."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def revenue_trends(df):
    analysis_type = "Revenue Trends"
    try:
        expected = ['order_date', 'revenue']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df.dropna(subset=['order_date', 'revenue'], inplace=True)

        df['week'] = df['order_date'].dt.to_period('W').astype(str)
        weekly_revenue = df.groupby('week')['revenue'].sum().reset_index()

        fig_line = px.line(weekly_revenue, x='week', y='revenue', title='Weekly Revenue Trends')
        fig_bar = px.bar(weekly_revenue, x='week', y='revenue', title='Weekly Revenue')

        visualizations = {
            'line_chart': fig_line.to_json(),
            'bar_chart': fig_bar.to_json()
        }

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "average_weekly_revenue": weekly_revenue['revenue'].mean()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed weekly revenue trends and patterns."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def marketing_analysis(df):
    analysis_type = "Marketing Analysis"
    try:
        expected = ['order_id', 'order_date', 'revenue', 'utm_source']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
        df.dropna(subset=['order_id', 'order_date', 'revenue', 'utm_source'], inplace=True)

        summary = df.groupby('utm_source').agg(
            orders=('order_id', 'nunique'),
            total_revenue=('revenue', 'sum')
        ).reset_index()
        summary['avg_order_value'] = summary['total_revenue'] / summary['orders']

        fig_bar_revenue = px.bar(summary, x='utm_source', y='total_revenue', title='Revenue by UTM Source')
        fig_bar_orders = px.bar(summary, x='utm_source', y='orders', title='Orders by UTM Source')

        visualizations = {
            'revenue_by_source': fig_bar_revenue.to_json(),
            'orders_by_source': fig_bar_orders.to_json()
        }

        metrics = {
            "total_revenue": summary['total_revenue'].sum(),
            "total_orders": summary['orders'].sum(),
            "average_order_value": summary['avg_order_value'].mean()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed marketing performance by UTM source and order values."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def regional_channel_analysis(df):
    analysis_type = "Regional Channel Analysis"
    try:
        expected = ['revenue', 'country', 'region', 'channel']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)
        df = safe_rename(df, matched)
        df.dropna(subset=['revenue', 'country', 'region', 'channel'], inplace=True)

        country_revenue = df.groupby('country')['revenue'].sum().reset_index()
        region_revenue = df.groupby('region')['revenue'].sum().reset_index()
        channel_revenue = df.groupby('channel')['revenue'].sum().reset_index()

        fig_country = px.bar(country_revenue.sort_values('revenue', ascending=False).head(10),
                             x='country', y='revenue', title='Top 10 Countries by Revenue')
        fig_region = px.bar(region_revenue.sort_values('revenue', ascending=False).head(10),
                            x='region', y='revenue', title='Top 10 Regions by Revenue')
        fig_channel = px.pie(channel_revenue, names='channel', values='revenue', title='Revenue by Channel')

        visualizations = {
            'top_countries': fig_country.to_json(),
            'top_regions': fig_region.to_json(),
            'channel_share': fig_channel.to_json()
        }

        metrics = {
            "total_revenue": df['revenue'].sum(),
            "unique_countries": df['country'].nunique(),
            "unique_regions": df['region'].nunique(),
            "unique_channels": df['channel'].nunique()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed regional and channel performance across countries and regions."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def conversion_analysis(df):
    analysis_type = "Conversion Analysis"
    try:
        expected = ['impressions', 'clicks', 'add_to_cart', 'order_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing)

        df = safe_rename(df, matched)

        funnel = {}
        if 'impressions' in df.columns:
            funnel['Impressions'] = int(df['impressions'].sum())
        if 'clicks' in df.columns:
            funnel['Clicks'] = int(df['clicks'].sum())
        if 'add_to_cart' in df.columns:
            funnel['Add to Cart'] = int(df['add_to_cart'].sum())
        if 'order_id' in df.columns:
            funnel['Orders'] = int(df['order_id'].nunique())

        funnel_df = pd.DataFrame(list(funnel.items()), columns=['Stage', 'Count'])

        conversion_rates = {}
        keys = list(funnel.keys())
        for i in range(1, len(keys)):
            prev_val = funnel[keys[i-1]]
            curr_val = funnel[keys[i]]
            if prev_val:
                conversion_rates[f"{keys[i-1]} to {keys[i]}"] = round(100 * curr_val / prev_val, 2)
            else:
                conversion_rates[f"{keys[i-1]} to {keys[i]}"] = None

        fig_bar = px.bar(funnel_df, x='Stage', y='Count', title='Conversion Funnel')
        fig_rate = px.bar(
            pd.DataFrame(list(conversion_rates.items()), columns=['Step', 'Rate']),
            x='Step', y='Rate', title='Conversion Rates Between Steps'
        )

        visualizations = {
            'funnel_counts': fig_bar.to_json(),
            'conversion_rates': fig_rate.to_json()
        }

        metrics = {**funnel, **conversion_rates}

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": ["Analyzed conversion funnel performance and step-by-step conversion rates."]
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }


# Main analysis mapping dictionary
analysis_function_mapping = {
    "sales_summary": sales_summary,
    "top_products": top_products,
    "customer_analysis": customer_analysis,
    "revenue_trends": revenue_trends,
    "marketing_analysis": marketing_analysis,
    "regional_channel_analysis": regional_channel_analysis,
    "conversion_analysis": conversion_analysis,
    "sales_order_fulfillment_and_status_analysis": sales_order_fulfillment_and_status_analysis,
    "sales_invoice_and_payment_reconciliation_analysis": sales_invoice_and_payment_reconciliation_analysis,
    "sales_transaction_and_profit_margin_analysis": sales_transaction_and_profit_margin_analysis,
    "sales_representative_performance_and_revenue_analysis": sales_representative_performance_and_revenue_analysis,
    "sales_channel_and_customer_segment_performance_analysis": sales_channel_and_customer_segment_performance_analysis,
    "sales_opportunity_and_pipeline_analysis": sales_opportunity_and_pipeline_analysis,
    "sales_quote_conversion_and_pricing_analysis": sales_quote_conversion_and_pricing_analysis,
    "sales_return_and_refund_analysis": sales_return_and_refund_analysis,
    "sales_lead_and_opportunity_conversion_analysis": sales_lead_and_opportunity_conversion_analysis,
    "customer_payment_and_reconciliation_analysis": customer_payment_and_reconciliation_analysis,
    "lead_management_and_conversion_funnel_analysis": lead_management_and_conversion_funnel_analysis,
    "customer_lifetime_value_and_churn_risk_analysis": customer_lifetime_value_and_churn_risk_analysis,
    "subscription_sales_and_renewal_analysis": subscription_sales_and_renewal_analysis,
    "sales_channel_performance_and_conversion_analysis": sales_channel_performance_and_conversion_analysis,
    "cross_sell_and_upsell_opportunity_analysis": cross_sell_and_upsell_opportunity_analysis,
    "sales_territory_performance_and_quota_achievement_analysis": sales_territory_performance_and_quota_achievement_analysis,
    "product_sales_performance_and_profitability_analysis": product_sales_performance_and_profitability_analysis,
    "product_pricing_strategy_and_tier_analysis": product_pricing_strategy_and_tier_analysis,
    "sales_forecasting_accuracy_analysis": sales_forecasting_accuracy_analysis,
    "channel_promotion_performance_and_roi_analysis": channel_promotion_performance_and_roi_analysis,
    "customer_service_impact_on_sales_analysis": customer_service_impact_on_sales_analysis,
    "sales_call_outcome_and_effectiveness_analysis": sales_call_outcome_and_effectiveness_analysis,
    "market_segment_revenue_and_profitability_analysis": market_segment_revenue_and_profitability_analysis,
    "competitor_pricing_and_feature_analysis": competitor_pricing_and_feature_analysis,
    "product_bundle_sales_performance_analysis": product_bundle_sales_performance_analysis,
    "international_sales_and_currency_exchange_analysis": international_sales_and_currency_exchange_analysis,
    "sales_contract_and_renewal_analysis": sales_contract_and_renewal_analysis,
    "e_commerce_sales_funnel_and_conversion_analysis": e_commerce_sales_funnel_and_conversion_analysis,
    "field_sales_visit_effectiveness_analysis": field_sales_visit_effectiveness_analysis,
    "sales_key_performance_indicator_kpi_trend_analysis": sales_key_performance_indicator_kpi_trend_analysis,
    "sales_refund_and_reason_code_analysis": sales_refund_and_reason_code_analysis,
    "lead_nurturing_campaign_effectiveness_analysis": lead_nurturing_campaign_effectiveness_analysis,
}

def run_analysis(df, analysis_name):
    """Main function to run any analysis by name"""
    func = analysis_function_mapping.get(analysis_name)
    if func is None:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": f"No analysis function found for '{analysis_name}'",
            "visualizations": {},
            "metrics": {},
            "insights": []
        }
    
    try:
        return func(df)
    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def load_data(file_path, encoding='utf-8'):
    """Load data from CSV or Excel file"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please provide CSV or Excel file.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main_backend(file_path, analysis_name, encoding='utf-8'):
    """Main backend function to load data and run analysis"""
    # Load data
    df = load_data(file_path, encoding)
    if df is None:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": "Failed to load data from file",
            "visualizations": {},
            "metrics": {},
            "insights": []
        }
    
    # Run the requested analysis
    return run_analysis(df, analysis_name)

# Example usage
if __name__ == "__main__":
    # Example of how to use the analysis system
    sample_data = pd.DataFrame({
        'order_id': [1, 2, 3, 4, 5],
        'order_date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'revenue': [100, 200, 150, 300, 250],
        'customer_id': [101, 102, 103, 104, 105],
        'product_id': [201, 202, 203, 204, 205]
    })
    
    # Test sales summary analysis
    result = sales_summary(sample_data)
    print("Sales Summary Result:")
    print(json.dumps(result, indent=2))