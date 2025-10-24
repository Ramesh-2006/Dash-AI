import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')


def fuzzy_match_column(df, target_columns):
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available)
        matched[target] = match if score >= 70 else None
    return matched


def show_missing_columns_warning(missing_cols, matched_cols=None):
    warning_msg = {
        "missing_columns": missing_cols,
        "matched_columns": {col: matched_cols[col] for col in missing_cols} if matched_cols else {}
    }
    return warning_msg


def show_general_insights(df, title="General Insights"):
    insights = {
        "record_count": len(df),
        "column_count": len(df.columns),
        "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
        "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist()
    }
    plots = {}
    numeric_cols = insights["numeric_columns"]
    if numeric_cols:
        selected = numeric_cols[0]
        plots['histogram'] = px.histogram(df, x=selected, title=f"Distribution of {selected}")
        plots['boxplot'] = px.box(df, y=selected, title=f"Boxplot of {selected}")
    return {"insights": insights, "plots": plots}


# Example of one analysis function - all other functions follow the same pattern
def sales_performance_analysis(df):
    expected = ['date', 'revenue', 'transactions', 'product_id', 'sales_rep_id', 'sales_channel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Performance Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'revenue'], inplace=True)

    total_revenue = df['revenue'].sum()
    total_transactions = df['transactions'].sum() if 'transactions' in df else len(df)

    metrics = {
        "total_revenue": total_revenue,
        "total_transactions": total_transactions,
        "avg_transaction_value": total_revenue / total_transactions if total_transactions > 0 else 0
    }

    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
    revenue_by_rep = df.groupby('sales_rep_id')['revenue'].sum().nlargest(10).reset_index()

    plots = {
        'revenue_by_channel': px.pie(revenue_by_channel, names='sales_channel', values='revenue', title="Revenue Distribution by Sales Channel"),
        'top_reps_by_revenue': px.bar(revenue_by_rep, x='sales_rep_id', y='revenue', title="Top 10 Sales Reps by Revenue")
    }

    return {"metrics": metrics, "plots": plots}
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming these utility functions are defined elsewhere in your environment
# For the purpose of this response, I'll include a basic placeholder for them.
def fuzzy_match_column(df, expected_columns):
    matched_columns = {}
    df_columns = [col.lower() for col in df.columns]
    for expected in expected_columns:
        if expected.lower() in df_columns:
            matched_columns[expected] = df.columns[df_columns.index(expected.lower())]
        else:
            # Simple fuzzy matching (can be improved)
            for col in df.columns:
                if expected.lower() in col.lower() or col.lower() in expected.lower():
                    matched_columns[expected] = col
                    break
    return matched_columns

def show_missing_columns_warning(missing_columns, matched_columns):
    warning_message = f"Warning: The following expected columns are missing or could not be matched: {', '.join(missing_columns)}. "
    if matched_columns:
        warning_message += f"Successfully matched: {', '.join([f'{k}:{v}' for k, v in matched_columns.items() if v])}."
    return warning_message

def show_general_insights(df, analysis_name):
    return f"General insights for {analysis_name}: Dataset has {len(df)} rows and {len(df.columns)} columns."

def customer_purchase_behavior_and_rfm_analysis(df):
    expected = ['customer_id', 'transaction_date', 'revenue', 'order_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Purchase Behavior and RFM Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

    # RFM Analysis
    current_date = df['transaction_date'].max() + pd.Timedelta(days=1)
    rfm_df = df.groupby('customer_id').agg(
        recency=('transaction_date', lambda date: (current_date - date.max()).days),
        frequency=('order_id', 'nunique'),
        monetary=('revenue', 'sum')
    ).reset_index()

    # Simple RFM segmentation (can be more complex with quantiles)
    rfm_df['R_Score'] = pd.qcut(rfm_df['recency'], 5, labels=[5, 4, 3, 2, 1], duplicates='drop')
    rfm_df['F_Score'] = pd.qcut(rfm_df['frequency'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm_df['M_Score'] = pd.qcut(rfm_df['monetary'], 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
    rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)

    # Top N customers by monetary value
    top_customers = rfm_df.nlargest(10, 'monetary')

    # Distribution of Recency, Frequency, Monetary
    fig_recency = px.histogram(rfm_df, x='recency', nbins=50, title='Distribution of Recency (Days)')
    fig_monetary = px.histogram(rfm_df, x='monetary', nbins=50, title='Distribution of Monetary Value')

    plots = {
        'recency_distribution': fig_recency,
        'monetary_distribution': fig_monetary
    }

    metrics = {
        "num_customers_analyzed": len(rfm_df),
        "avg_recency": rfm_df['recency'].mean(),
        "avg_frequency": rfm_df['frequency'].mean(),
        "avg_monetary": rfm_df['monetary'].mean()
    }

    return {"metrics": metrics, "plots": plots, "rfm_data": rfm_df.to_dict(orient='records')}

def retail_transaction_analysis_by_product_and_country(df):
    expected = ['transaction_id', 'product_id', 'country', 'quantity', 'price', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Retail Transaction Analysis by Product and Country")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df['revenue'] = df['quantity'] * df['price']
    df.dropna(subset=['product_id', 'country', 'revenue', 'transaction_date'], inplace=True)

    # Top N products by revenue
    top_products = df.groupby('product_id')['revenue'].sum().nlargest(10).reset_index()

    # Revenue by country
    revenue_by_country = df.groupby('country')['revenue'].sum().reset_index()

    fig_product_revenue = px.bar(top_products, x='product_id', y='revenue', title='Top 10 Products by Revenue')
    fig_country_revenue = px.choropleth(revenue_by_country, locations='country', locationmode='country names',
                                       color='revenue', hover_name='country',
                                       color_continuous_scale=px.colors.sequential.Plasma,
                                       title='Total Revenue by Country')

    plots = {
        'top_products_revenue': fig_product_revenue,
        'country_revenue_map': fig_country_revenue
    }

    metrics = {
        "total_transactions": len(df),
        "total_revenue": df['revenue'].sum(),
        "num_unique_products": df['product_id'].nunique(),
        "num_unique_countries": df['country'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def retail_order_status_and_item_analysis(df):
    expected = ['order_id', 'item_id', 'order_status', 'product_name', 'quantity', 'price']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Retail Order Status and Item Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['item_revenue'] = df['quantity'] * df['price']
    df.dropna(subset=['order_id', 'order_status', 'product_name'], inplace=True)

    # Order status distribution
    order_status_counts = df.groupby('order_status')['order_id'].nunique().reset_index(name='count')

    # Top selling items (by quantity)
    top_selling_items = df.groupby('product_name')['quantity'].sum().nlargest(10).reset_index()

    fig_order_status = px.pie(order_status_counts, names='order_status', values='count', title='Distribution of Order Status')
    fig_top_items = px.bar(top_selling_items, x='product_name', y='quantity', title='Top 10 Selling Items by Quantity')

    plots = {
        'order_status_distribution': fig_order_status,
        'top_selling_items': fig_top_items
    }

    metrics = {
        "total_orders": df['order_id'].nunique(),
        "total_items_sold": df['quantity'].sum(),
        "num_unique_products": df['product_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def regional_sales_and_customer_analysis(df):
    expected = ['region', 'sales', 'customer_id', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Regional Sales and Customer Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['region', 'sales', 'customer_id'], inplace=True)

    # Total sales by region
    sales_by_region = df.groupby('region')['sales'].sum().reset_index()

    # Number of unique customers by region
    customers_by_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
    fig_customers_by_region = px.bar(customers_by_region, x='region', y='unique_customers', title='Number of Unique Customers by Region')

    plots = {
        'sales_by_region': fig_sales_by_region,
        'customers_by_region': fig_customers_by_region
    }

    metrics = {
        "total_sales_across_regions": df['sales'].sum(),
        "total_unique_customers": df['customer_id'].nunique(),
        "num_unique_regions": df['region'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_channel_performance(df):
    expected = ['sales_channel', 'revenue', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Channel Performance")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

    # Revenue by sales channel
    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()

    # Number of transactions by sales channel
    transactions_by_channel = df.groupby('sales_channel')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_revenue_by_channel = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title='Revenue Distribution by Sales Channel')
    fig_transactions_by_channel = px.bar(transactions_by_channel, x='sales_channel', y='num_transactions', title='Number of Transactions by Sales Channel')

    plots = {
        'revenue_by_channel': fig_revenue_by_channel,
        'transactions_by_channel': fig_transactions_by_channel
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_channels": df['sales_channel'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def international_sales_and_transaction_analysis(df):
    expected = ['country', 'sales', 'transaction_id', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "International Sales and Transaction Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['country', 'sales', 'transaction_id'], inplace=True)

    # Sales by country
    sales_by_country = df.groupby('country')['sales'].sum().reset_index()

    # Number of transactions by country
    transactions_by_country = df.groupby('country')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_by_country = px.choropleth(sales_by_country, locations='country', locationmode='country names',
                                        color='sales', hover_name='country',
                                        color_continuous_scale=px.colors.sequential.Plasma,
                                        title='Total Sales by Country (International)')
    fig_transactions_by_country = px.bar(transactions_by_country.nlargest(10, 'num_transactions'), x='country', y='num_transactions', title='Top 10 Countries by Number of Transactions')

    plots = {
        'sales_by_country_map': fig_sales_by_country,
        'transactions_by_country_bar': fig_transactions_by_country
    }

    metrics = {
        "total_international_sales": df['sales'].sum(),
        "total_international_transactions": df['transaction_id'].nunique(),
        "num_international_countries": df['country'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def invoice_type_and_customer_purchase_pattern(df):
    expected = ['invoice_id', 'invoice_type', 'customer_id', 'transaction_date', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Invoice Type and Customer Purchase Pattern")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['invoice_id', 'invoice_type', 'customer_id', 'revenue'], inplace=True)

    # Revenue by invoice type
    revenue_by_invoice_type = df.groupby('invoice_type')['revenue'].sum().reset_index()

    # Number of unique customers by invoice type
    customers_by_invoice_type = df.groupby('invoice_type')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_revenue_invoice_type = px.pie(revenue_by_invoice_type, names='invoice_type', values='revenue', title='Revenue Distribution by Invoice Type')
    fig_customers_invoice_type = px.bar(customers_by_invoice_type, x='invoice_type', y='unique_customers', title='Number of Unique Customers by Invoice Type')

    plots = {
        'revenue_by_invoice_type': fig_revenue_invoice_type,
        'customers_by_invoice_type': fig_customers_invoice_type
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "num_unique_invoice_types": df['invoice_type'].nunique(),
        "total_unique_customers": df['customer_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def order_delivery_and_customer_location(df):
    expected = ['order_id', 'customer_location', 'delivery_status', 'delivery_time_days', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Order Delivery and Customer Location")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['order_id', 'customer_location', 'delivery_status'], inplace=True)

    # Delivery status distribution
    delivery_status_counts = df.groupby('delivery_status')['order_id'].nunique().reset_index(name='count')

    # Average delivery time by customer location (if 'delivery_time_days' exists)
    avg_delivery_time_by_location = None
    if 'delivery_time_days' in df.columns:
        avg_delivery_time_by_location = df.groupby('customer_location')['delivery_time_days'].mean().reset_index()
        avg_delivery_time_by_location = avg_delivery_time_by_location.nlargest(10, 'delivery_time_days') # Top 10 for visualization

    fig_delivery_status = px.pie(delivery_status_counts, names='delivery_status', values='count', title='Distribution of Order Delivery Status')
    
    plots = {
        'delivery_status_distribution': fig_delivery_status,
    }

    if avg_delivery_time_by_location is not None and not avg_delivery_time_by_location.empty:
        fig_avg_delivery_time = px.bar(avg_delivery_time_by_location, x='customer_location', y='delivery_time_days', title='Average Delivery Time by Customer Location (Top 10)')
        plots['avg_delivery_time_by_location'] = fig_avg_delivery_time
    else:
        plots['avg_delivery_time_by_location_warning'] = "Delivery time data not available for visualization."


    metrics = {
        "total_orders": df['order_id'].nunique(),
        "num_unique_customer_locations": df['customer_location'].nunique()
    }
    if 'delivery_time_days' in df.columns:
        metrics["avg_delivery_time_overall"] = df['delivery_time_days'].mean()

    return {"metrics": metrics, "plots": plots}

def time_of_day_sales_pattern(df):
    expected = ['transaction_date', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Time of Day Sales Pattern")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'revenue'], inplace=True)

    df['hour_of_day'] = df['transaction_date'].dt.hour

    # Sales by hour of day
    sales_by_hour = df.groupby('hour_of_day')['revenue'].sum().reset_index()

    # Number of transactions by hour of day
    transactions_by_hour = df.groupby('hour_of_day').size().reset_index(name='num_transactions')

    fig_sales_by_hour = px.line(sales_by_hour, x='hour_of_day', y='revenue', title='Sales by Hour of Day')
    fig_transactions_by_hour = px.bar(transactions_by_hour, x='hour_of_day', y='num_transactions', title='Number of Transactions by Hour of Day')

    plots = {
        'sales_by_hour': fig_sales_by_hour,
        'transactions_by_hour': fig_transactions_by_hour
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "total_transactions": len(df)
    }

    return {"metrics": metrics, "plots": plots}

def customer_order_and_status_tracking(df):
    expected = ['customer_id', 'order_id', 'order_status', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Order and Status Tracking")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['customer_id', 'order_id', 'order_status'], inplace=True)

    # Number of orders per customer (top 10)
    orders_per_customer = df.groupby('customer_id')['order_id'].nunique().nlargest(10).reset_index(name='num_orders')

    # Order status distribution
    order_status_distribution = df.groupby('order_status')['order_id'].nunique().reset_index(name='count')

    fig_orders_per_customer = px.bar(orders_per_customer, x='customer_id', y='num_orders', title='Top 10 Customers by Number of Orders')
    fig_order_status_distribution = px.pie(order_status_distribution, names='order_status', values='count', title='Overall Order Status Distribution')

    plots = {
        'orders_per_customer': fig_orders_per_customer,
        'order_status_distribution': fig_order_status_distribution
    }

    metrics = {
        "total_unique_customers": df['customer_id'].nunique(),
        "total_unique_orders": df['order_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def payment_method_preference(df):
    expected = ['payment_method', 'transaction_id', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Payment Method Preference")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['payment_method', 'revenue'], inplace=True)

    # Revenue by payment method
    revenue_by_payment_method = df.groupby('payment_method')['revenue'].sum().reset_index()

    # Number of transactions by payment method
    transactions_by_payment_method = df.groupby('payment_method')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_revenue_payment = px.pie(revenue_by_payment_method, names='payment_method', values='revenue', title='Revenue Distribution by Payment Method')
    fig_transactions_payment = px.bar(transactions_by_payment_method, x='payment_method', y='num_transactions', title='Number of Transactions by Payment Method')

    plots = {
        'revenue_by_payment_method': fig_revenue_payment,
        'transactions_by_payment_method': fig_transactions_payment
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_payment_methods": df['payment_method'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def product_return_rate(df):
    expected = ['order_id', 'product_id', 'return_status', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Return Rate")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['order_id', 'product_id', 'return_status', 'quantity'], inplace=True)

    total_items = df['quantity'].sum()
    returned_items = df[df['return_status'].astype(str).str.lower() == 'returned']['quantity'].sum()
    return_rate_overall = (returned_items / total_items) * 100 if total_items > 0 else 0

    # Return rate by product (top 10 products with highest return count)
    returned_products = df[df['return_status'].astype(str).str.lower() == 'returned'].groupby('product_id')['quantity'].sum().nlargest(10).reset_index(name='returned_quantity')
    
    # Return status distribution
    return_status_counts = df['return_status'].value_counts().reset_index()
    return_status_counts.columns = ['return_status', 'count']

    fig_return_rate_pie = px.pie(return_status_counts, names='return_status', values='count', title='Overall Return Status Distribution')
    fig_top_returned_products = px.bar(returned_products, x='product_id', y='returned_quantity', title='Top 10 Products by Returned Quantity')

    plots = {
        'overall_return_status': fig_return_rate_pie,
        'top_returned_products': fig_top_returned_products
    }

    metrics = {
        "total_items_sold": total_items,
        "total_items_returned": returned_items,
        "overall_return_rate_percent": return_rate_overall
    }

    return {"metrics": metrics, "plots": plots}


def promotional_code_effectiveness(df):
    expected = ['promotion_code', 'revenue', 'transaction_id', 'is_promotional_sale']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Promotional Code Effectiveness")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['revenue'], inplace=True)

    # Ensure 'is_promotional_sale' is numeric (0 or 1) or boolean
    if 'is_promotional_sale' not in df.columns:
        df['is_promotional_sale'] = df['promotion_code'].notna().astype(int)
    
    # Revenue with and without promotion
    revenue_by_promotion_status = df.groupby('is_promotional_sale')['revenue'].sum().reset_index()
    revenue_by_promotion_status['is_promotional_sale'] = revenue_by_promotion_status['is_promotional_sale'].map({0: 'Non-Promotional', 1: 'Promotional'})

    # Top performing promotion codes (by revenue)
    if 'promotion_code' in df.columns:
        top_promotion_codes = df.groupby('promotion_code')['revenue'].sum().nlargest(10).reset_index()
    else:
        top_promotion_codes = pd.DataFrame(columns=['promotion_code', 'revenue'])

    fig_revenue_promotion_status = px.pie(revenue_by_promotion_status, names='is_promotional_sale', values='revenue', title='Revenue by Promotional Status')
    
    plots = {
        'revenue_by_promotion_status': fig_revenue_promotion_status,
    }

    if not top_promotion_codes.empty:
        fig_top_promotion_codes = px.bar(top_promotion_codes, x='promotion_code', y='revenue', title='Top 10 Promotion Codes by Revenue')
        plots['top_promotion_codes'] = fig_top_promotion_codes
    else:
        plots['top_promotion_codes_warning'] = "No specific promotion codes found for top performance analysis."

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "promotional_revenue": df[df['is_promotional_sale'] == 1]['revenue'].sum(),
        "non_promotional_revenue": df[df['is_promotional_sale'] == 0]['revenue'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def discount_impact_on_sales(df):
    expected = ['discount_amount', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Discount Impact on Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales'], inplace=True)

    # Sales with and without discount (assuming discount_amount > 0 for discount)
    df['has_discount'] = (df['discount_amount'] > 0).astype(int)
    sales_by_discount_status = df.groupby('has_discount')['sales'].sum().reset_index()
    sales_by_discount_status['has_discount'] = sales_by_discount_status['has_discount'].map({0: 'No Discount', 1: 'With Discount'})

    # Average sales per transaction by discount presence
    avg_sales_by_discount = df.groupby('has_discount')['sales'].mean().reset_index()
    avg_sales_by_discount['has_discount'] = avg_sales_by_discount['has_discount'].map({0: 'No Discount', 1: 'With Discount'})

    fig_sales_by_discount = px.pie(sales_by_discount_status, names='has_discount', values='sales', title='Sales Distribution by Discount Presence')
    fig_avg_sales_discount = px.bar(avg_sales_by_discount, x='has_discount', y='sales', title='Average Sales per Transaction by Discount Presence')

    plots = {
        'sales_by_discount': fig_sales_by_discount,
        'avg_sales_by_discount': fig_avg_sales_discount
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "sales_with_discount": df[df['has_discount'] == 1]['sales'].sum(),
        "sales_without_discount": df[df['has_discount'] == 0]['sales'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def product_cost_and_sales_margin(df):
    expected = ['product_id', 'sales_price', 'cost_price', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Cost and Sales Margin")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_price', 'cost_price', 'quantity'], inplace=True)

    df['total_sales'] = df['sales_price'] * df['quantity']
    df['total_cost'] = df['cost_price'] * df['quantity']
    df['gross_profit'] = df['total_sales'] - df['total_cost']

    # Gross profit by product (top 10)
    gross_profit_by_product = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()

    # Overall gross profit margin
    overall_gross_profit_margin = (df['gross_profit'].sum() / df['total_sales'].sum()) * 100 if df['total_sales'].sum() > 0 else 0

    fig_gross_profit_product = px.bar(gross_profit_by_product, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

    plots = {
        'gross_profit_by_product': fig_gross_profit_product,
    }

    # Donut chart for overall margin breakdown (Sales vs. Cost)
    overall_financials = pd.DataFrame({
        'Metric': ['Total Sales', 'Total Cost'],
        'Value': [df['total_sales'].sum(), df['total_cost'].sum()]
    })
    fig_overall_margin_breakdown = px.pie(overall_financials, names='Metric', values='Value', title='Overall Sales vs. Cost Breakdown')
    plots['overall_margin_breakdown'] = fig_overall_margin_breakdown


    metrics = {
        "total_sales": df['total_sales'].sum(),
        "total_cost": df['total_cost'].sum(),
        "total_gross_profit": df['gross_profit'].sum(),
        "overall_gross_profit_margin_percent": overall_gross_profit_margin
    }

    return {"metrics": metrics, "plots": plots}


def store_level_sales_performance(df):
    expected = ['store_id', 'sales', 'transaction_id', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Store Level Sales Performance")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['store_id', 'sales'], inplace=True)

    # Total sales by store
    sales_by_store = df.groupby('store_id')['sales'].sum().reset_index()

    # Number of transactions by store
    transactions_by_store = df.groupby('store_id')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_by_store = px.bar(sales_by_store.nlargest(10, 'sales'), x='store_id', y='sales', title='Top 10 Stores by Sales')
    fig_transactions_by_store = px.bar(transactions_by_store.nlargest(10, 'num_transactions'), x='store_id', y='num_transactions', title='Top 10 Stores by Number of Transactions')

    plots = {
        'sales_by_store': fig_sales_by_store,
        'transactions_by_store': fig_transactions_by_store
    }

    metrics = {
        "total_sales_across_stores": df['sales'].sum(),
        "total_transactions_across_stores": df['transaction_id'].nunique(),
        "num_unique_stores": df['store_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def product_category(df):
    expected = ['product_category', 'sales', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Category Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_category', 'sales'], inplace=True)

    # Sales by product category
    sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()

    # Quantity sold by product category
    quantity_by_category = df.groupby('product_category')['quantity'].sum().reset_index()

    fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
    fig_quantity_by_category = px.bar(quantity_by_category.nlargest(10, 'quantity'), x='product_category', y='quantity', title='Top 10 Product Categories by Quantity Sold')

    plots = {
        'sales_by_category': fig_sales_by_category,
        'quantity_by_category': fig_quantity_by_category
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_quantity_sold": df['quantity'].sum(),
        "num_unique_product_categories": df['product_category'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def weekly_sales_trend(df):
    expected = ['transaction_date', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Weekly Sales Trend")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales'], inplace=True)

    df['week_start'] = df['transaction_date'].dt.to_period('W').dt.start_time
    weekly_sales = df.groupby('week_start')['sales'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values('week_start')

    fig_weekly_sales_line = px.line(weekly_sales, x='week_start', y='sales', title='Weekly Sales Trend')
    fig_weekly_sales_bar = px.bar(weekly_sales, x='week_start', y='sales', title='Weekly Sales Trend (Bar Chart)')

    plots = {
        'weekly_sales_line': fig_weekly_sales_line,
        'weekly_sales_bar': fig_weekly_sales_bar
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_weeks_in_data": weekly_sales.shape[0]
    }

    return {"metrics": metrics, "plots": plots}

def yearly_sales_performance(df):
    expected = ['transaction_date', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Yearly Sales Performance")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales'], inplace=True)

    df['year'] = df['transaction_date'].dt.year
    yearly_sales = df.groupby('year')['sales'].sum().reset_index()
    yearly_sales = yearly_sales.sort_values('year')

    fig_yearly_sales_bar = px.bar(yearly_sales, x='year', y='sales', title='Yearly Sales Performance')
    fig_yearly_sales_line = px.line(yearly_sales, x='year', y='sales', title='Yearly Sales Trend')

    plots = {
        'yearly_sales_bar': fig_yearly_sales_bar,
        'yearly_sales_line': fig_yearly_sales_line
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_years_in_data": yearly_sales.shape[0]
    }

    return {"metrics": metrics, "plots": plots}

def monthly_sales_trend(df):
    expected = ['transaction_date', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Monthly Sales Trend")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales'], inplace=True)

    df['month_year'] = df['transaction_date'].dt.to_period('M').dt.start_time
    monthly_sales = df.groupby('month_year')['sales'].sum().reset_index()
    monthly_sales = monthly_sales.sort_values('month_year')

    fig_monthly_sales_line = px.line(monthly_sales, x='month_year', y='sales', title='Monthly Sales Trend')
    fig_monthly_sales_bar = px.bar(monthly_sales, x='month_year', y='sales', title='Monthly Sales Trend (Bar Chart)')

    plots = {
        'monthly_sales_line': fig_monthly_sales_line,
        'monthly_sales_bar': fig_monthly_sales_bar
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_months_in_data": monthly_sales.shape[0]
    }

    return {"metrics": metrics, "plots": plots}

def week_over_week_sales_growth(df):
    expected = ['transaction_date', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Week-over-Week Sales Growth")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales'], inplace=True)

    df['week_start'] = df['transaction_date'].dt.to_period('W').dt.start_time
    weekly_sales = df.groupby('week_start')['sales'].sum().reset_index()
    weekly_sales = weekly_sales.sort_values('week_start')

    weekly_sales['previous_week_sales'] = weekly_sales['sales'].shift(1)
    weekly_sales['wow_growth'] = ((weekly_sales['sales'] - weekly_sales['previous_week_sales']) / weekly_sales['previous_week_sales']) * 100
    weekly_sales.dropna(subset=['wow_growth'], inplace=True)

    fig_wow_growth_line = px.line(weekly_sales, x='week_start', y='wow_growth', title='Week-over-Week Sales Growth (%)')
    fig_wow_growth_bar = px.bar(weekly_sales, x='week_start', y='wow_growth', title='Week-over-Week Sales Growth (%) (Bar Chart)')

    plots = {
        'wow_growth_line': fig_wow_growth_line,
        'wow_growth_bar': fig_wow_growth_bar
    }

    metrics = {
        "avg_wow_growth_percent": weekly_sales['wow_growth'].mean() if not weekly_sales.empty else 0,
        "max_wow_growth_percent": weekly_sales['wow_growth'].max() if not weekly_sales.empty else 0,
        "min_wow_growth_percent": weekly_sales['wow_growth'].min() if not weekly_sales.empty else 0
    }

    return {"metrics": metrics, "plots": plots}

def holiday_sales_impact(df):
    expected = ['transaction_date', 'sales', 'is_holiday']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Holiday Sales Impact")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales', 'is_holiday'], inplace=True)

    # Ensure 'is_holiday' is boolean or 0/1
    df['is_holiday'] = df['is_holiday'].astype(bool)

    # Sales on holidays vs. non-holidays
    sales_by_holiday = df.groupby('is_holiday')['sales'].sum().reset_index()
    sales_by_holiday['is_holiday'] = sales_by_holiday['is_holiday'].map({False: 'Non-Holiday', True: 'Holiday'})

    # Average daily sales on holidays vs. non-holidays
    df['date_only'] = df['transaction_date'].dt.date
    daily_sales = df.groupby(['date_only', 'is_holiday'])['sales'].sum().reset_index()
    avg_daily_sales_by_holiday = daily_sales.groupby('is_holiday')['sales'].mean().reset_index()
    avg_daily_sales_by_holiday['is_holiday'] = avg_daily_sales_by_holiday['is_holiday'].map({False: 'Non-Holiday', True: 'Holiday'})

    fig_sales_by_holiday = px.bar(sales_by_holiday, x='is_holiday', y='sales', title='Total Sales: Holiday vs. Non-Holiday')
    fig_avg_daily_sales_holiday = px.bar(avg_daily_sales_by_holiday, x='is_holiday', y='sales', title='Average Daily Sales: Holiday vs. Non-Holiday')

    plots = {
        'sales_by_holiday': fig_sales_by_holiday,
        'avg_daily_sales_holiday': fig_avg_daily_sales_holiday
    }

    metrics = {
        "total_holiday_sales": df[df['is_holiday'] == True]['sales'].sum(),
        "total_non_holiday_sales": df[df['is_holiday'] == False]['sales'].sum(),
        "num_holiday_days": df[df['is_holiday'] == True]['date_only'].nunique(),
        "num_non_holiday_days": df[df['is_holiday'] == False]['date_only'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def customer_type_analysis(df):
    expected = ['customer_type', 'sales', 'customer_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Type Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_type', 'sales'], inplace=True)

    # Sales by customer type
    sales_by_customer_type = df.groupby('customer_type')['sales'].sum().reset_index()

    # Number of unique customers by type
    unique_customers_by_type = df.groupby('customer_type')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_sales_by_customer_type = px.pie(sales_by_customer_type, names='customer_type', values='sales', title='Sales Distribution by Customer Type')
    fig_unique_customers_by_type = px.bar(unique_customers_by_type, x='customer_type', y='unique_customers', title='Number of Unique Customers by Type')

    plots = {
        'sales_by_customer_type': fig_sales_by_customer_type,
        'unique_customers_by_type': fig_unique_customers_by_type
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_unique_customers": df['customer_id'].nunique(),
        "num_customer_types": df['customer_type'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def online_vs_offline_sales(df):
    expected = ['sales_channel', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Online vs. Offline Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'sales'], inplace=True)

    # Assuming 'Online' and 'Offline' are values in 'sales_channel'
    # Or derive based on store_id/web_transaction_id if specific channel names aren't present
    df['channel_type'] = df['sales_channel'].apply(lambda x: 'Online' if 'online' in str(x).lower() else ('Offline' if 'store' in str(x).lower() or 'physical' in str(x).lower() else 'Other'))

    # Sales by channel type
    sales_by_channel_type = df.groupby('channel_type')['sales'].sum().reset_index()

    # Number of transactions by channel type
    transactions_by_channel_type = df.groupby('channel_type')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_channel_type = px.pie(sales_by_channel_type, names='channel_type', values='sales', title='Sales Distribution: Online vs. Offline')
    fig_transactions_channel_type = px.bar(transactions_by_channel_type, x='channel_type', y='num_transactions', title='Number of Transactions: Online vs. Offline')

    plots = {
        'sales_channel_type': fig_sales_channel_type,
        'transactions_channel_type': fig_transactions_channel_type
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "online_sales": df[df['channel_type'] == 'Online']['sales'].sum(),
        "offline_sales": df[df['channel_type'] == 'Offline']['sales'].sum()
    }

    return {"metrics": metrics, "plots": plots}


def regional_customer_purchase(df):
    expected = ['customer_id', 'region', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Regional Customer Purchase")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_id', 'region', 'sales'], inplace=True)

    # Total sales by region
    sales_by_region = df.groupby('region')['sales'].sum().reset_index()

    # Number of unique customers by region
    customers_by_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
    fig_customers_by_region = px.bar(customers_by_region, x='region', y='unique_customers', title='Number of Unique Customers by Region')

    plots = {
        'sales_by_region': fig_sales_by_region,
        'customers_by_region': fig_customers_by_region
    }

    metrics = {
        "total_sales_across_regions": df['sales'].sum(),
        "total_unique_customers": df['customer_id'].nunique(),
        "num_unique_regions": df['region'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def delivery_method_preference(df):
    expected = ['delivery_method', 'order_id', 'customer_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Delivery Method Preference")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['delivery_method', 'order_id'], inplace=True)

    # Number of orders by delivery method
    orders_by_delivery_method = df.groupby('delivery_method')['order_id'].nunique().reset_index(name='num_orders')

    # Number of unique customers using each delivery method
    customers_by_delivery_method = df.groupby('delivery_method')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_orders_delivery_method = px.pie(orders_by_delivery_method, names='delivery_method', values='num_orders', title='Orders Distribution by Delivery Method')
    fig_customers_delivery_method = px.bar(customers_by_delivery_method, x='delivery_method', y='unique_customers', title='Unique Customers by Delivery Method')

    plots = {
        'orders_by_delivery_method': fig_orders_delivery_method,
        'customers_by_delivery_method': fig_customers_delivery_method
    }

    metrics = {
        "total_orders": df['order_id'].nunique(),
        "num_delivery_methods": df['delivery_method'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def point_of_sale_transaction(df):
    expected = ['transaction_id', 'transaction_date', 'store_id', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Point of Sale Transaction")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_id', 'sales'], inplace=True)

    # Daily sales trend (POS)
    daily_sales_pos = df.groupby(df['transaction_date'].dt.date)['sales'].sum().reset_index()
    daily_sales_pos.columns = ['date', 'sales']

    # Transactions by store (if store_id is present)
    transactions_by_store = None
    if 'store_id' in df.columns:
        transactions_by_store = df.groupby('store_id')['transaction_id'].nunique().reset_index(name='num_transactions')
        transactions_by_store = transactions_by_store.nlargest(10, 'num_transactions')

    fig_daily_sales_pos = px.line(daily_sales_pos, x='date', y='sales', title='Daily Sales Trend (Point of Sale)')
    
    plots = {
        'daily_sales_pos': fig_daily_sales_pos,
    }

    if transactions_by_store is not None and not transactions_by_store.empty:
        fig_transactions_by_store = px.bar(transactions_by_store, x='store_id', y='num_transactions', title='Top 10 Stores by Number of POS Transactions')
        plots['transactions_by_store'] = fig_transactions_by_store
    else:
        plots['transactions_by_store_warning'] = "Store ID data not available for transaction count by store."


    metrics = {
        "total_pos_sales": df['sales'].sum(),
        "total_pos_transactions": df['transaction_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_tax_analysis(df):
    expected = ['sales', 'tax_amount']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Tax Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales', 'tax_amount'], inplace=True)

    total_sales = df['sales'].sum()
    total_tax = df['tax_amount'].sum()
    overall_tax_rate = (total_tax / total_sales) * 100 if total_sales > 0 else 0

    # Relationship between sales and tax amount
    fig_sales_vs_tax = px.scatter(df, x='sales', y='tax_amount', title='Sales vs. Tax Amount',
                                  trendline='ols', trendline_color_discrete=['red'])

    # Distribution of tax amounts
    fig_tax_distribution = px.histogram(df, x='tax_amount', nbins=50, title='Distribution of Tax Amounts')

    plots = {
        'sales_vs_tax_scatter': fig_sales_vs_tax,
        'tax_distribution_histogram': fig_tax_distribution
    }

    metrics = {
        "total_sales_subject_to_tax": total_sales,
        "total_tax_collected": total_tax,
        "overall_tax_rate_percent": overall_tax_rate
    }

    return {"metrics": metrics, "plots": plots}

def sales_organization(df):
    expected = ['sales_organization', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Organization Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_organization', 'sales'], inplace=True)

    # Sales by sales organization
    sales_by_org = df.groupby('sales_organization')['sales'].sum().reset_index()

    # Number of transactions by sales organization
    transactions_by_org = df.groupby('sales_organization')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_by_org = px.bar(sales_by_org.nlargest(10, 'sales'), x='sales_organization', y='sales', title='Top 10 Sales Organizations by Sales')
    fig_transactions_by_org = px.bar(transactions_by_org.nlargest(10, 'num_transactions'), x='sales_organization', y='num_transactions', title='Top 10 Sales Organizations by Transactions')

    plots = {
        'sales_by_organization': fig_sales_by_org,
        'transactions_by_organization': fig_transactions_by_org
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_organizations": df['sales_organization'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def order_payment_status(df):
    expected = ['order_id', 'payment_status', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Order Payment Status")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['order_id', 'payment_status', 'revenue'], inplace=True)

    # Distribution of payment statuses
    payment_status_counts = df.groupby('payment_status')['order_id'].nunique().reset_index(name='count')

    # Revenue by payment status
    revenue_by_payment_status = df.groupby('payment_status')['revenue'].sum().reset_index()

    fig_payment_status_pie = px.pie(payment_status_counts, names='payment_status', values='count', title='Distribution of Order Payment Status')
    fig_revenue_payment_status = px.bar(revenue_by_payment_status, x='payment_status', y='revenue', title='Revenue by Payment Status')

    plots = {
        'payment_status_distribution': fig_payment_status_pie,
        'revenue_by_payment_status': fig_revenue_payment_status
    }

    metrics = {
        "total_orders": df['order_id'].nunique(),
        "total_revenue": df['revenue'].sum(),
        "num_payment_statuses": df['payment_status'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def product_sales_and_cost(df):
    expected = ['product_id', 'sales', 'cost']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Sales and Cost")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'sales', 'cost'], inplace=True)

    # Total sales and cost by product
    product_summary = df.groupby('product_id').agg(
        total_sales=('sales', 'sum'),
        total_cost=('cost', 'sum')
    ).reset_index()
    product_summary['gross_profit'] = product_summary['total_sales'] - product_summary['total_cost']

    # Top 10 products by sales
    top_sales_products = product_summary.nlargest(10, 'total_sales')

    # Top 10 products by gross profit
    top_profit_products = product_summary.nlargest(10, 'gross_profit')

    fig_top_sales_products = px.bar(top_sales_products, x='product_id', y='total_sales', title='Top 10 Products by Sales')
    fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

    plots = {
        'top_sales_products': fig_top_sales_products,
        'top_profit_products': fig_top_profit_products
    }

    metrics = {
        "overall_total_sales": product_summary['total_sales'].sum(),
        "overall_total_cost": product_summary['total_cost'].sum(),
        "overall_gross_profit": product_summary['gross_profit'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def customer_transaction_history(df):
    expected = ['customer_id', 'transaction_date', 'transaction_id', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Transaction History")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

    # Number of transactions per customer (top 10)
    transactions_per_customer = df.groupby('customer_id')['transaction_id'].nunique().nlargest(10).reset_index(name='num_transactions')

    # Total revenue per customer (top 10)
    revenue_per_customer = df.groupby('customer_id')['revenue'].sum().nlargest(10).reset_index()

    fig_transactions_per_customer = px.bar(transactions_per_customer, x='customer_id', y='num_transactions', title='Top 10 Customers by Number of Transactions')
    fig_revenue_per_customer = px.bar(revenue_per_customer, x='customer_id', y='revenue', title='Top 10 Customers by Total Revenue')

    plots = {
        'transactions_per_customer': fig_transactions_per_customer,
        'revenue_per_customer': fig_revenue_per_customer
    }

    metrics = {
        "total_unique_customers": df['customer_id'].nunique(),
        "total_transactions": df['transaction_id'].nunique(),
        "total_revenue_overall": df['revenue'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def customer_segment_purchasing(df):
    expected = ['customer_segment', 'sales', 'customer_id', 'product_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Segment Purchasing")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_segment', 'sales'], inplace=True)

    # Sales by customer segment
    sales_by_segment = df.groupby('customer_segment')['sales'].sum().reset_index()

    # Top product categories purchased by each segment (top 3 categories per segment)
    if 'product_category' in df.columns:
        segment_product_sales = df.groupby(['customer_segment', 'product_category'])['sales'].sum().reset_index()
        top_categories_per_segment = segment_product_sales.loc[segment_product_sales.groupby('customer_segment')['sales'].nlargest(3).index].reset_index(drop=True)
    else:
        top_categories_per_segment = pd.DataFrame(columns=['customer_segment', 'product_category', 'sales'])

    fig_sales_by_segment = px.pie(sales_by_segment, names='customer_segment', values='sales', title='Sales Distribution by Customer Segment')
    
    plots = {
        'sales_by_segment': fig_sales_by_segment,
    }

    if not top_categories_per_segment.empty:
        fig_top_categories_per_segment = px.bar(top_categories_per_segment, x='product_category', y='sales', color='customer_segment',
                                                title='Top Product Categories Purchased by Customer Segment', barmode='group')
        plots['top_categories_per_segment'] = fig_top_categories_per_segment
    else:
        plots['top_categories_per_segment_warning'] = "Product category data not available for segment-wise product analysis."

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_customer_segments": df['customer_segment'].nunique()
    }

    return {"metrics": metrics, "plots": plots}


def per_unit_price_and_sales(df):
    expected = ['product_id', 'price', 'quantity', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Per Unit Price and Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'price', 'quantity', 'sales'], inplace=True)

    # Calculate actual sales from price * quantity if 'sales' column is missing or incorrect
    if 'sales' not in df.columns or not df['sales'].sum() == (df['price'] * df['quantity']).sum(): # Simple check
        df['sales'] = df['price'] * df['quantity']

    # Average unit price by product (top 10)
    avg_unit_price_by_product = df.groupby('product_id')['price'].mean().nlargest(10).reset_index()

    # Total sales by product (top 10)
    total_sales_by_product = df.groupby('product_id')['sales'].sum().nlargest(10).reset_index()

    fig_avg_unit_price = px.bar(avg_unit_price_by_product, x='product_id', y='price', title='Top 10 Products by Average Unit Price')
    fig_total_sales_product = px.bar(total_sales_by_product, x='product_id', y='sales', title='Top 10 Products by Total Sales')

    plots = {
        'avg_unit_price_product': fig_avg_unit_price,
        'total_sales_product': fig_total_sales_product
    }

    metrics = {
        "overall_average_unit_price": df['price'].mean(),
        "overall_total_sales": df['sales'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def promotion_id_impact(df):
    expected = ['promotion_id', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Promotion ID Impact")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales'], inplace=True)

    # Sales by promotion ID (include 'No Promotion' if applicable)
    df['promotion_id'] = df['promotion_id'].fillna('No Promotion')
    sales_by_promotion = df.groupby('promotion_id')['sales'].sum().reset_index()
    sales_by_promotion = sales_by_promotion.sort_values('sales', ascending=False)

    # Number of transactions by promotion ID
    transactions_by_promotion = df.groupby('promotion_id')['transaction_id'].nunique().reset_index(name='num_transactions')
    transactions_by_promotion = transactions_by_promotion.sort_values('num_transactions', ascending=False)


    fig_sales_by_promotion = px.bar(sales_by_promotion.nlargest(10, 'sales'), x='promotion_id', y='sales', title='Top 10 Promotions by Sales Revenue')
    fig_transactions_by_promotion = px.bar(transactions_by_promotion.nlargest(10, 'num_transactions'), x='promotion_id', y='num_transactions', title='Top 10 Promotions by Number of Transactions')

    plots = {
        'sales_by_promotion': fig_sales_by_promotion,
        'transactions_by_promotion': fig_transactions_by_promotion
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_unique_promotions": df['promotion_id'].nunique() - (1 if 'No Promotion' in df['promotion_id'].unique() else 0)
    }

    return {"metrics": metrics, "plots": plots}

def store_location_sales(df):
    expected = ['store_location', 'sales', 'store_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Store Location Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['store_location', 'sales'], inplace=True)

    # Total sales by store location
    sales_by_location = df.groupby('store_location')['sales'].sum().reset_index()

    # Number of stores per location (if store_id is available)
    stores_per_location = None
    if 'store_id' in df.columns:
        stores_per_location = df.groupby('store_location')['store_id'].nunique().reset_index(name='num_stores')

    fig_sales_by_location = px.bar(sales_by_location.nlargest(10, 'sales'), x='store_location', y='sales', title='Top 10 Store Locations by Sales')
    
    plots = {
        'sales_by_location': fig_sales_by_location,
    }

    if stores_per_location is not None and not stores_per_location.empty:
        fig_stores_per_location = px.bar(stores_per_location, x='store_location', y='num_stores', title='Number of Unique Stores per Location')
        plots['stores_per_location'] = fig_stores_per_location
    else:
        plots['stores_per_location_warning'] = "Store ID data not available for number of stores per location."

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_unique_locations": df['store_location'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_representative_performance(df):
    expected = ['sales_representative_id', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Representative Performance")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_representative_id', 'sales'], inplace=True)

    # Total sales by sales representative (top 10)
    sales_by_rep = df.groupby('sales_representative_id')['sales'].sum().nlargest(10).reset_index()

    # Number of transactions by sales representative (top 10)
    transactions_by_rep = df.groupby('sales_representative_id')['transaction_id'].nunique().nlargest(10).reset_index(name='num_transactions')

    fig_sales_by_rep = px.bar(sales_by_rep, x='sales_representative_id', y='sales', title='Top 10 Sales Representatives by Sales')
    fig_transactions_by_rep = px.bar(transactions_by_rep, x='sales_representative_id', y='num_transactions', title='Top 10 Sales Representatives by Transactions')

    plots = {
        'sales_by_representative': fig_sales_by_rep,
        'transactions_by_representative': fig_transactions_by_rep
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_representatives": df['sales_representative_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def regional_sales_and_product(df):
    expected = ['region', 'product_id', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Regional Sales and Product")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['region', 'product_id', 'sales'], inplace=True)

    # Sales by region
    sales_by_region = df.groupby('region')['sales'].sum().reset_index()

    # Top product by sales for each region
    regional_product_sales = df.groupby(['region', 'product_id'])['sales'].sum().reset_index()
    top_product_per_region = regional_product_sales.loc[regional_product_sales.groupby('region')['sales'].idxmax()].reset_index(drop=True)

    fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales', title='Total Sales by Region')
    fig_top_product_per_region = px.bar(top_product_per_region, x='region', y='sales', color='product_id', title='Top Selling Product per Region')

    plots = {
        'sales_by_region': fig_sales_by_region,
        'top_product_per_region': fig_top_product_per_region
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_unique_regions": df['region'].nunique(),
        "num_unique_products": df['product_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def multi_channel_sales(df):
    expected = ['sales_channel', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Multi-Channel Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'sales'], inplace=True)

    # Sales distribution by sales channel
    sales_by_channel = df.groupby('sales_channel')['sales'].sum().reset_index()

    # Number of transactions by sales channel
    transactions_by_channel = df.groupby('sales_channel')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_by_channel = px.pie(sales_by_channel, names='sales_channel', values='sales', title='Sales Distribution by Channel')
    fig_transactions_by_channel = px.bar(transactions_by_channel, x='sales_channel', y='num_transactions', title='Number of Transactions by Channel')

    plots = {
        'sales_by_channel': fig_sales_by_channel,
        'transactions_by_channel': fig_transactions_by_channel
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_channels": df['sales_channel'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_classification(df):
    expected = ['sales_category', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Classification")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_category', 'sales'], inplace=True)

    # Sales by classification (e.g., 'High Value', 'Medium Value', 'Low Value')
    sales_by_classification = df.groupby('sales_category')['sales'].sum().reset_index()

    # Number of transactions by classification
    transactions_by_classification = df.groupby('sales_category')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_classification_pie = px.pie(sales_by_classification, names='sales_category', values='sales', title='Sales Distribution by Classification')
    fig_transactions_classification = px.bar(transactions_by_classification, x='sales_category', y='num_transactions', title='Number of Transactions by Sales Classification')

    plots = {
        'sales_classification_pie': fig_sales_classification_pie,
        'transactions_classification_bar': fig_transactions_classification
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_classifications": df['sales_category'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def retail_receipt_data(df):
    expected = ['receipt_id', 'transaction_date', 'total_amount', 'item_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Retail Receipt Data Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['receipt_id', 'total_amount'], inplace=True)

    # Daily total amount from receipts
    daily_receipt_amount = df.groupby(df['transaction_date'].dt.date)['total_amount'].sum().reset_index()
    daily_receipt_amount.columns = ['date', 'total_amount']

    # Distribution of item counts per receipt
    if 'item_count' in df.columns:
        fig_item_count_distribution = px.histogram(df, x='item_count', nbins=50, title='Distribution of Item Counts per Receipt')
    else:
        fig_item_count_distribution = go.Figure().add_annotation(text="Item count data not available for distribution.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))


    fig_daily_receipt_amount = px.line(daily_receipt_amount, x='date', y='total_amount', title='Daily Total Amount from Receipts')

    plots = {
        'daily_receipt_amount': fig_daily_receipt_amount,
        'item_count_distribution': fig_item_count_distribution
    }

    metrics = {
        "total_receipts": df['receipt_id'].nunique(),
        "total_revenue_from_receipts": df['total_amount'].sum(),
        "avg_items_per_receipt": df['item_count'].mean() if 'item_count' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def invoice_line_item(df):
    expected = ['invoice_id', 'product_id', 'quantity', 'unit_price', 'total_price']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Invoice Line Item Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['invoice_id', 'product_id', 'quantity', 'unit_price'], inplace=True)

    # Calculate total_price if not present or incorrect
    if 'total_price' not in df.columns or not df['total_price'].sum() == (df['quantity'] * df['unit_price']).sum():
        df['total_price'] = df['quantity'] * df['unit_price']

    # Top selling products by revenue (based on line item total price)
    top_selling_products_line_item = df.groupby('product_id')['total_price'].sum().nlargest(10).reset_index()

    # Distribution of quantities per line item
    fig_quantity_distribution = px.histogram(df, x='quantity', nbins=50, title='Distribution of Quantities per Line Item')
    
    fig_top_selling_products_line_item = px.bar(top_selling_products_line_item, x='product_id', y='total_price', title='Top 10 Products by Line Item Revenue')

    plots = {
        'quantity_distribution': fig_quantity_distribution,
        'top_selling_products_line_item': fig_top_selling_products_line_item
    }

    metrics = {
        "total_line_items": len(df),
        "total_revenue_from_line_items": df['total_price'].sum(),
        "num_unique_products_in_line_items": df['product_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def product_category_performance(df):
    expected = ['product_category', 'sales', 'quantity', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Category Performance")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_category', 'sales'], inplace=True)

    # Sales by product category
    sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()

    # Number of unique products per category
    if 'product_id' in df.columns:
        products_per_category = df.groupby('product_category')['product_id'].nunique().reset_index(name='unique_products')
    else:
        products_per_category = pd.DataFrame(columns=['product_category', 'unique_products'])

    fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
    
    plots = {
        'sales_by_category': fig_sales_by_category,
    }

    if not products_per_category.empty:
        fig_products_per_category = px.bar(products_per_category, x='product_category', y='unique_products', title='Number of Unique Products per Category')
        plots['products_per_category'] = fig_products_per_category
    else:
        plots['products_per_category_warning'] = "Product ID data not available for unique products per category."

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_unique_product_categories": df['product_category'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def transactional_promotion_effectiveness(df):
    expected = ['transaction_id', 'promotion_id', 'sales', 'discount_amount']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Transactional Promotion Effectiveness")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['transaction_id', 'sales'], inplace=True)

    # Determine if a transaction had a promotion
    df['had_promotion'] = df['promotion_id'].notna().astype(int)

    # Compare average sales for promoted vs. non-promoted transactions
    avg_sales_by_promotion = df.groupby('had_promotion')['sales'].mean().reset_index()
    avg_sales_by_promotion['had_promotion'] = avg_sales_by_promotion['had_promotion'].map({0: 'No Promotion', 1: 'Had Promotion'})

    # If discount_amount is available, total discount amount applied per promotion (top 10)
    total_discount_by_promotion = None
    if 'discount_amount' in df.columns:
        total_discount_by_promotion = df.groupby('promotion_id')['discount_amount'].sum().nlargest(10).reset_index()

    fig_avg_sales_by_promotion = px.bar(avg_sales_by_promotion, x='had_promotion', y='sales', title='Average Sales per Transaction: Promoted vs. Non-Promoted')
    
    plots = {
        'avg_sales_by_promotion': fig_avg_sales_by_promotion,
    }

    if total_discount_by_promotion is not None and not total_discount_by_promotion.empty:
        fig_total_discount_by_promotion = px.bar(total_discount_by_promotion, x='promotion_id', y='discount_amount', title='Top 10 Promotions by Total Discount Amount Applied')
        plots['total_discount_by_promotion'] = fig_total_discount_by_promotion
    else:
        plots['total_discount_by_promotion_warning'] = "Discount amount data not available for promotion effectiveness analysis."

    metrics = {
        "total_transactions_promoted": df[df['had_promotion'] == 1]['transaction_id'].nunique(),
        "total_transactions_non_promoted": df[df['had_promotion'] == 0]['transaction_id'].nunique(),
        "total_sales_promoted": df[df['had_promotion'] == 1]['sales'].sum(),
        "total_sales_non_promoted": df[df['had_promotion'] == 0]['sales'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def order_status_and_item_details(df):
    expected = ['order_id', 'order_status', 'product_name', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Order Status and Item Details")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['order_id', 'order_status', 'product_name'], inplace=True)

    # Number of items per order status
    items_by_order_status = df.groupby('order_status')['quantity'].sum().reset_index()

    # Top 10 products associated with a 'Completed' or 'Shipped' status (adjust based on actual statuses)
    completed_shipped_orders = df[df['order_status'].astype(str).str.lower().isin(['completed', 'shipped'])]
    top_products_completed = completed_shipped_orders.groupby('product_name')['quantity'].sum().nlargest(10).reset_index()

    fig_items_by_order_status = px.bar(items_by_order_status, x='order_status', y='quantity', title='Total Items by Order Status')
    fig_top_products_completed = px.bar(top_products_completed, x='product_name', y='quantity', title='Top 10 Products in Completed/Shipped Orders')

    plots = {
        'items_by_order_status': fig_items_by_order_status,
        'top_products_completed': fig_top_products_completed
    }

    metrics = {
        "total_orders": df['order_id'].nunique(),
        "total_items_in_orders": df['quantity'].sum(),
        "num_unique_order_statuses": df['order_status'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_source_attribution(df):
    expected = ['sales_source', 'sales', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Source Attribution")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_source', 'sales'], inplace=True)

    # Sales by sales source
    sales_by_source = df.groupby('sales_source')['sales'].sum().reset_index()

    # Number of transactions by sales source
    transactions_by_source = df.groupby('sales_source')['transaction_id'].nunique().reset_index(name='num_transactions')

    fig_sales_by_source = px.pie(sales_by_source, names='sales_source', values='sales', title='Sales Distribution by Source')
    fig_transactions_by_source = px.bar(transactions_by_source, x='sales_source', y='num_transactions', title='Number of Transactions by Sales Source')

    plots = {
        'sales_by_source': fig_sales_by_source,
        'transactions_by_source': fig_transactions_by_source
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_transactions": df['transaction_id'].nunique(),
        "num_sales_sources": df['sales_source'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def customer_regional_sales(df):
    expected = ['customer_id', 'region', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Regional Sales")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_id', 'region', 'sales'], inplace=True)

    # Sales per region per customer (top 10 customer-region combinations by sales)
    customer_region_sales = df.groupby(['customer_id', 'region'])['sales'].sum().nlargest(10).reset_index()
    customer_region_sales['customer_region'] = customer_region_sales['customer_id'].astype(str) + ' - ' + customer_region_sales['region']

    # Number of unique customers per region
    customers_per_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')

    fig_customer_region_sales = px.bar(customer_region_sales, x='customer_region', y='sales', color='region', title='Top 10 Customer-Region Sales Combinations')
    fig_customers_per_region = px.bar(customers_per_region, x='region', y='unique_customers', title='Number of Unique Customers per Region')

    plots = {
        'customer_region_sales': fig_customer_region_sales,
        'customers_per_region': fig_customers_per_region
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "num_unique_customers": df['customer_id'].nunique(),
        "num_unique_regions": df['region'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

# General Analyses - I will provide a more comprehensive version for these.

def sales_analysis(df):
    expected = ['transaction_date', 'revenue', 'quantity', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'revenue'], inplace=True)

    total_revenue = df['revenue'].sum()
    avg_transaction_value = df['revenue'].mean()
    total_quantity_sold = df['quantity'].sum() if 'quantity' in df.columns else 'N/A'

    # Daily sales trend
    daily_sales = df.groupby(df['transaction_date'].dt.date)['revenue'].sum().reset_index()
    daily_sales.columns = ['date', 'revenue']

    # Top 10 products by revenue
    top_products = df.groupby('product_id')['revenue'].sum().nlargest(10).reset_index()

    fig_daily_sales = px.line(daily_sales, x='date', y='revenue', title='Daily Sales Trend')
    fig_top_products = px.bar(top_products, x='product_id', y='revenue', title='Top 10 Products by Revenue')

    plots = {
        'daily_sales_trend': fig_daily_sales,
        'top_products_by_revenue': fig_top_products
    }

    metrics = {
        "total_revenue": total_revenue,
        "avg_transaction_value": avg_transaction_value,
        "total_quantity_sold": total_quantity_sold
    }

    return {"metrics": metrics, "plots": plots}

def customer_analysis(df):
    expected = ['customer_id', 'transaction_date', 'revenue', 'order_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['customer_id', 'transaction_date', 'revenue'], inplace=True)

    num_unique_customers = df['customer_id'].nunique()
    avg_revenue_per_customer = df.groupby('customer_id')['revenue'].sum().mean()

    # Top 10 customers by total revenue
    top_customers_by_revenue = df.groupby('customer_id')['revenue'].sum().nlargest(10).reset_index()

    # Distribution of customer transaction frequency
    customer_frequency = df.groupby('customer_id')['order_id'].nunique().reset_index(name='num_orders')

    fig_top_customers = px.bar(top_customers_by_revenue, x='customer_id', y='revenue', title='Top 10 Customers by Total Revenue')
    fig_customer_frequency_distribution = px.histogram(customer_frequency, x='num_orders', nbins=50, title='Distribution of Customer Transaction Frequency')

    plots = {
        'top_customers_by_revenue': fig_top_customers,
        'customer_frequency_distribution': fig_customer_frequency_distribution
    }

    metrics = {
        "num_unique_customers": num_unique_customers,
        "avg_revenue_per_customer": avg_revenue_per_customer,
        "avg_transactions_per_customer": customer_frequency['num_orders'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def inventory_analysis(df):
    expected = ['product_id', 'stock_quantity', 'product_name', 'last_restock_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Inventory Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'stock_quantity'], inplace=True)

    total_stock_value = (df['stock_quantity'] * df['price']).sum() if 'price' in df.columns else 'N/A' # Assuming 'price' refers to cost or selling price
    num_unique_products_in_stock = df['product_id'].nunique()

    # Top 10 products by stock quantity
    top_stock_products = df.groupby('product_id')['stock_quantity'].sum().nlargest(10).reset_index()

    # Distribution of stock quantities
    fig_stock_distribution = px.histogram(df, x='stock_quantity', nbins=50, title='Distribution of Stock Quantities')

    plots = {
        'top_stock_products': top_stock_products,
        'stock_distribution': fig_stock_distribution
    }

    metrics = {
        "total_stock_quantity": df['stock_quantity'].sum(),
        "total_stock_value": total_stock_value,
        "num_unique_products_in_stock": num_unique_products_in_stock
    }

    return {"metrics": metrics, "plots": plots}

def product_analysis(df):
    expected = ['product_id', 'product_name', 'sales', 'quantity', 'product_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'sales', 'quantity'], inplace=True)

    # Top 10 products by sales
    top_selling_products = df.groupby('product_id')['sales'].sum().nlargest(10).reset_index()

    # Sales by product category
    if 'product_category' in df.columns:
        sales_by_category = df.groupby('product_category')['sales'].sum().reset_index()
    else:
        sales_by_category = pd.DataFrame(columns=['product_category', 'sales'])

    fig_top_selling_products = px.bar(top_selling_products, x='product_id', y='sales', title='Top 10 Selling Products by Sales')
    
    plots = {
        'top_selling_products': fig_top_selling_products,
    }

    if not sales_by_category.empty:
        fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales', title='Sales Distribution by Product Category')
        plots['sales_by_category'] = fig_sales_by_category
    else:
        plots['sales_by_category_warning'] = "Product category data not available for sales distribution."

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_quantity_sold": df['quantity'].sum(),
        "num_unique_products": df['product_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def store_analysis(df):
    expected = ['store_id', 'sales', 'transaction_id', 'region', 'store_location']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Store Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['store_id', 'sales'], inplace=True)

    # Top 10 stores by sales
    top_performing_stores = df.groupby('store_id')['sales'].sum().nlargest(10).reset_index()

    # Sales distribution by region (if available)
    if 'region' in df.columns:
        sales_by_region = df.groupby('region')['sales'].sum().reset_index()
    else:
        sales_by_region = pd.DataFrame(columns=['region', 'sales'])

    fig_top_performing_stores = px.bar(top_performing_stores, x='store_id', y='sales', title='Top 10 Performing Stores by Sales')
    
    plots = {
        'top_performing_stores': fig_top_performing_stores,
    }

    if not sales_by_region.empty:
        fig_sales_by_region = px.pie(sales_by_region, names='region', values='sales', title='Sales Distribution by Region')
        plots['sales_by_region'] = fig_sales_by_region
    else:
        plots['sales_by_region_warning'] = "Region data not available for sales distribution."

    metrics = {
        "total_sales_across_stores": df['sales'].sum(),
        "num_unique_stores": df['store_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def promotion_analysis(df):
    expected = ['promotion_id', 'sales', 'discount_amount', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Promotion Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales'], inplace=True)

    # Categorize transactions by promotion presence
    df['had_promotion'] = df['promotion_id'].notna().astype(int)
    sales_by_promotion_status = df.groupby('had_promotion')['sales'].sum().reset_index()
    sales_by_promotion_status['had_promotion'] = sales_by_promotion_status['had_promotion'].map({0: 'No Promotion', 1: 'Had Promotion'})

    # Average discount given per promotion (for promotions that offered discount)
    if 'discount_amount' in df.columns and 'promotion_id' in df.columns:
        avg_discount_per_promotion = df[df['promotion_id'].notna()].groupby('promotion_id')['discount_amount'].mean().nlargest(10).reset_index()
    else:
        avg_discount_per_promotion = pd.DataFrame(columns=['promotion_id', 'discount_amount'])

    fig_sales_by_promo_status = px.bar(sales_by_promotion_status, x='had_promotion', y='sales', title='Sales with vs. Without Promotion')
    
    plots = {
        'sales_by_promo_status': fig_sales_by_promo_status,
    }

    if not avg_discount_per_promotion.empty:
        fig_avg_discount_per_promo = px.bar(avg_discount_per_promotion, x='promotion_id', y='discount_amount', title='Top 10 Promotions by Average Discount Given')
        plots['avg_discount_per_promo'] = fig_avg_discount_per_promo
    else:
        plots['avg_discount_per_promo_warning'] = "Discount amount or promotion ID data not available for average discount analysis."

    metrics = {
        "total_sales": df['sales'].sum(),
        "total_promoted_sales": df[df['had_promotion'] == 1]['sales'].sum(),
        "num_unique_promotions": df['promotion_id'].nunique() - (1 if 'No Promotion' in df['promotion_id'].unique() else 0 if 'promotion_id' in df.columns else 0)
    }

    return {"metrics": metrics, "plots": plots}


def basket_analysis(df):
    expected = ['transaction_id', 'product_id', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Basket Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['transaction_id', 'product_id', 'quantity'], inplace=True)

    # Average items per transaction
    items_per_transaction = df.groupby('transaction_id')['quantity'].sum().reset_index(name='total_items')
    avg_items_per_transaction = items_per_transaction['total_items'].mean()

    # Top 10 most frequently purchased products (single items, not pairs for this simplified version)
    product_frequency = df.groupby('product_id')['quantity'].sum().nlargest(10).reset_index()

    fig_items_per_transaction_hist = px.histogram(items_per_transaction, x='total_items', nbins=50, title='Distribution of Items per Transaction')
    fig_top_product_frequency = px.bar(product_frequency, x='product_id', y='quantity', title='Top 10 Most Frequently Purchased Products')

    plots = {
        'items_per_transaction_distribution': fig_items_per_transaction_hist,
        'top_product_frequency': fig_top_product_frequency
    }

    metrics = {
        "total_transactions": df['transaction_id'].nunique(),
        "avg_items_per_transaction": avg_items_per_transaction,
        "total_items_sold": df['quantity'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def seasonal_analysis(df):
    expected = ['transaction_date', 'sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Seasonal Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(subset=['transaction_date', 'sales'], inplace=True)

    df['month'] = df['transaction_date'].dt.month_name()
    df['quarter'] = df['transaction_date'].dt.quarter
    df['day_of_week'] = df['transaction_date'].dt.day_name()

    # Sales by month
    sales_by_month = df.groupby('month')['sales'].sum().reset_index()
    # Ensure correct order for months
    month_order = ["January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]
    sales_by_month['month'] = pd.Categorical(sales_by_month['month'], categories=month_order, ordered=True)
    sales_by_month = sales_by_month.sort_values('month')

    # Sales by day of week
    sales_by_day_of_week = df.groupby('day_of_week')['sales'].sum().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sales_by_day_of_week['day_of_week'] = pd.Categorical(sales_by_day_of_week['day_of_week'], categories=day_order, ordered=True)
    sales_by_day_of_week = sales_by_day_of_week.sort_values('day_of_week')


    fig_sales_by_month = px.line(sales_by_month, x='month', y='sales', title='Sales Trend by Month')
    fig_sales_by_day_of_week = px.bar(sales_by_day_of_week, x='day_of_week', y='sales', title='Sales by Day of Week')

    plots = {
        'sales_by_month': fig_sales_by_month,
        'sales_by_day_of_week': fig_sales_by_day_of_week
    }

    metrics = {
        "total_sales": df['sales'].sum(),
        "peak_sales_month": sales_by_month.loc[sales_by_month['sales'].idxmax()]['month'],
        "peak_sales_day_of_week": sales_by_day_of_week.loc[sales_by_day_of_week['sales'].idxmax()]['day_of_week']
    }

    return {"metrics": metrics, "plots": plots}



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
        "Product Category": product_category,
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
        "Sales Organization": sales_organization,
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

    result = None

    if category == "General Retail Analysis":
        if not general_analysis or general_analysis == "--Select--":
            result = show_general_insights(df, "Initial Overview")
        else:
            # Map your general analysis strings to functions accordingly, example:
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
            func = general_analysis_functions.get(general_analysis)
            if func:
                result = func(df)
            else:
                result = show_general_insights(df, "Initial Overview")

    elif category == "Specific Retail Analysis":
        if specific_analysis_name and specific_analysis_name != "--Select--":
            func = specific_retail_function_mapping.get(specific_analysis_name)
            if func:
                try:
                    result = func(df)
                except Exception as e:
                    result = {"error": f"Error running analysis '{specific_analysis_name}': {str(e)}"}
            else:
                result = {"error": f"Analysis '{specific_analysis_name}' not found."}
        else:
            result = {"error": "No specific analysis selected."}
    else:
        result = show_general_insights(df, "Initial Overview")

    return result

