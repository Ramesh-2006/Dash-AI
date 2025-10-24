import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import warnings
warnings.filterwarnings('ignore')

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

def show_key_metrics(df):
    metrics = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "numeric_features": len(df.select_dtypes(include=['int64', 'float64']).columns),
        "categorical_features": len(df.select_dtypes(include=['object', 'category']).columns),
    }
    return metrics

def show_missing_columns_warning(missing_cols, matched_cols=None):
    warning = {
        "missing_columns": missing_cols,
        "matched_columns": {col: matched_cols[col] for col in missing_cols} if matched_cols else {}
    }
    return warning

def show_general_insights(df, title="General Insights"):
    insights = {
        "key_metrics": show_key_metrics(df),
    }
    plots = {}

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        selected_num_col = numeric_cols[0]
        plots['histogram'] = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
        plots['boxplot'] = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            plots['correlation'] = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
    else:
        insights['numeric_analysis'] = "No numeric columns found for analysis."

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        selected_cat_col = categorical_cols[0]
        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        plots['cat_bar'] = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Distribution of {selected_cat_col}")
    else:
        insights['categorical_analysis'] = "No categorical columns found for analysis."

    return {"insights": insights, "plots": plots}

def load_data(file, encoding='utf-8'):
    try:
        if file.name.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            return None
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        else:
            return None
    except Exception:
        return None

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

def sales_analysis(df):
    expected = ['transaction_id', 'date', 'product_id', 'quantity',
                'unit_price', 'total_amount', 'store_id', 'customer_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        warn = show_missing_columns_warning(missing, matched)
        insights = show_general_insights(df, "General Analysis")
        return {"warning": warn, "fallback": insights}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    total_sales = df['total_amount'].sum()
    avg_transaction = df['total_amount'].mean()
    unique_customers = df['customer_id'].nunique() if 'customer_id' in df else 0
    metrics = {
        "total_sales": total_sales,
        "avg_transaction": avg_transaction,
        "unique_customers": unique_customers
    }
    plots = {}
    if 'date' in df and 'total_amount' in df:
        sales_over_time = df.groupby('date')['total_amount'].sum().reset_index()
        plots['daily_sales'] = px.line(sales_over_time, x='date', y='total_amount', title="Daily Sales Trend")
    if 'product_id' in df and 'total_amount' in df:
        top_products = df.groupby('product_id')['total_amount'].sum().nlargest(10).reset_index()
        plots['top_products'] = px.bar(top_products, x='product_id', y='total_amount', title="Top Products by Revenue")
    return {"metrics": metrics, "plots": plots}

def customer_analysis(df):
    expected = ['customer_id', 'first_purchase_date', 'total_purchases',
                'total_spend', 'segment', 'region']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        warn = show_missing_columns_warning(missing, matched)
        insights = show_general_insights(df, "General Analysis")
        return {"warning": warn, "fallback": insights}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if 'first_purchase_date' in df and not pd.api.types.is_datetime64_any_dtype(df['first_purchase_date']):
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'])
    total_customers = len(df)
    avg_spend = df['total_spend'].mean()
    top_segment = df['segment'].mode()[0] if 'segment' in df else "N/A"
    metrics = {
        "total_customers": total_customers,
        "avg_spend": avg_spend,
        "top_segment": top_segment
    }
    plots = {}
    if 'segment' in df:
        segment_dist = df['segment'].value_counts().reset_index()
        plots['segment_pie'] = px.pie(segment_dist, names='segment', values='count',
                                      title="Customer Segment Distribution")
    if all(col in df for col in ['recency', 'frequency', 'monetary_value']):
        plots['rfm'] = px.scatter_3d(df, x='recency', y='frequency', z='monetary_value',
                                    color='segment', title="RFM Segmentation")
    return {"metrics": metrics, "plots": plots}

def inventory_analysis(df):
    expected = ['product_id', 'product_name', 'category', 'current_stock', 'reorder_level', 'lead_time', 'supplier']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if 'current_stock' in df and 'reorder_level' in df:
        df['stock_status'] = np.where(df['current_stock'] < df['reorder_level'], 'Reorder Needed', 'Adequate Stock')
    total_products = len(df)
    low_stock = (df['stock_status'] == 'Reorder Needed').sum() if 'stock_status' in df else 0
    avg_lead_time = df['lead_time'].mean() if 'lead_time' in df else 0
    metrics = {"total_products": total_products, "products_needing_reorder": low_stock, "avg_lead_time_days": avg_lead_time}
    plots = {}
    if 'stock_status' in df:
        plots['inventory_status'] = px.pie(df, names='stock_status', title="Inventory Status Distribution")
    if 'category' in df and 'current_stock' in df:
        plots['category_inventory'] = px.treemap(df, path=['category'], values='current_stock', title="Inventory Distribution by Category")
    return {"metrics": metrics, "plots": plots}

def product_analysis(df):
    expected = ['product_id', 'product_name', 'category', 'subcategory', 'price', 'cost', 'margin', 'rating']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    total_products = len(df)
    avg_margin = df['margin'].mean() if 'margin' in df else 0
    avg_rating = df['rating'].mean() if 'rating' in df else 0
    metrics = {"total_products": total_products, "avg_margin": avg_margin, "avg_rating": avg_rating}
    plots = {}
    if 'price' in df:
        plots['price_distribution'] = px.histogram(df, x='price', title="Price Distribution")
    if 'category' in df and 'margin' in df:
        plots['margin_by_category'] = px.box(df, x='category', y='margin', title="Margin Distribution by Category")
    return {"metrics": metrics, "plots": plots}

def store_analysis(df):
    expected = ['store_id', 'location', 'size', 'manager', 'opening_date', 'monthly_sales', 'monthly_traffic']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if 'opening_date' in df and not pd.api.types.is_datetime64_any_dtype(df['opening_date']):
        df['opening_date'] = pd.to_datetime(df['opening_date'])
    total_stores = len(df)
    avg_sales = df['monthly_sales'].mean() if 'monthly_sales' in df else 0
    sales_per_sqft = (df['monthly_sales'] / df['size']).mean() if all(col in df for col in ['monthly_sales', 'size']) else 0
    metrics = {"total_stores": total_stores, "avg_monthly_sales": avg_sales, "sales_per_sqft": sales_per_sqft}
    plots = {}
    if 'location' in df and 'monthly_sales' in df:
        plots['sales_by_location'] = px.bar(df.sort_values('monthly_sales', ascending=False), x='location', y='monthly_sales', title="Store Sales by Location")
    if 'monthly_sales' in df and 'monthly_traffic' in df:
        plots['sales_vs_traffic'] = px.scatter(df, x='monthly_traffic', y='monthly_sales', trendline="ols", title="Sales vs Customer Traffic")
    return {"metrics": metrics, "plots": plots}

def promotion_analysis(df):
    expected = ['promotion_id', 'start_date', 'end_date', 'discount_pct', 'products_included', 'sales_increase', 'roi']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    date_cols = ['start_date', 'end_date']
    for col in date_cols:
        if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col])
    total_promotions = len(df)
    avg_discount = df['discount_pct'].mean() if 'discount_pct' in df else 0
    avg_roi = df['roi'].mean() if 'roi' in df else 0
    metrics = {"total_promotions": total_promotions, "avg_discount": avg_discount, "avg_roi": avg_roi}
    plots = {}
    if 'discount_pct' in df and 'sales_increase' in df:
        plots['discount_vs_sales_increase'] = px.scatter(df, x='discount_pct', y='sales_increase', trendline="ols", title="Discount % vs Sales Increase")
    if 'promotion_type' in df and 'roi' in df:
        plots['roi_by_type'] = px.box(df, x='promotion_type', y='roi', title="ROI by Promotion Type")
    return {"metrics": metrics, "plots": plots}

def basket_analysis(df):
    expected = ['transaction_id', 'product_id', 'product_name', 'category', 'quantity', 'unit_price']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    total_transactions = df['transaction_id'].nunique()
    avg_items_per_transaction = df.groupby('transaction_id')['product_id'].count().mean()
    avg_basket_value = df.groupby('transaction_id')['unit_price'].sum().mean()
    metrics = {"total_transactions": total_transactions, "avg_items_per_basket": avg_items_per_transaction, "avg_basket_value": avg_basket_value}
    plots = {}
    if 'transaction_id' in df and 'product_name' in df:
        top_products = df['product_name'].value_counts().nlargest(10).index.tolist()
        filtered_df = df[df['product_name'].isin(top_products)]
        product_pairs = filtered_df.groupby('transaction_id')['product_name'].agg(list).reset_index()
        product_pairs['product_name'] = product_pairs['product_name'].apply(lambda x: list(set(x)))
    if 'transaction_id' in df and 'category' in df:
        plots['category_mix'] = px.bar(df.groupby(['transaction_id', 'category']).size().reset_index(name='count'), x='transaction_id', y='count', color='category', title="Category Mix in Baskets")
    return {"metrics": metrics, "plots": plots}

def seasonal_analysis(df):
    expected = ['date', 'sales', 'transactions', 'customers', 'product_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    total_sales = df['sales'].sum()
    avg_daily_sales = df['sales'].mean()
    busiest_day = df.groupby('day_of_week')['sales'].mean().idxmax()
    metrics = {"total_sales": total_sales, "avg_daily_sales": avg_daily_sales, "busiest_day": busiest_day}
    plots = {}
    if 'month' in df and 'sales' in df:
        monthly_sales = df.groupby('month')['sales'].mean().reset_index()
        plots['monthly_seasonality'] = px.line(monthly_sales, x='month', y='sales', title="Monthly Sales Seasonality")
    if 'day_of_week' in df and 'sales' in df:
        dow_sales = df.groupby('day_of_week')['sales'].mean().reset_index()
        plots['day_of_week_pattern'] = px.bar(dow_sales, x='day_of_week', y='sales', title="Sales by Day of Week")
    return {"metrics": metrics, "plots": plots}

def customer_purchase_behavior_and_rfm_analysis(df):
    expected = ['invoiceno', 'invoicedate', 'unitprice', 'quantity', 'customerid']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched), "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df.dropna(subset=['customerid', 'invoicedate'], inplace=True)
    df = df[df['quantity'] > 0]
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
    metrics = {"average_recency_days": avg_recency, "average_frequency": avg_frequency, "average_monetary_value": avg_monetary}
    plots = {}
    plots['rfm_distribution'] = px.scatter(rfm, x='recency', y='frequency', size='monetary', color='monetary', hover_name='customerid', title="RFM Customer Segmentation")
    return {"metrics": metrics, "plots": plots, "rfm_head": rfm.head()}
def retail_transaction_analysis_by_product_and_country(df):
    expected = ['description', 'qty', 'price', 'country']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['qty'] > 0]
    df['revenue'] = df['qty'] * df['price']
    df.dropna(inplace=True)
    total_revenue = df['revenue'].sum()
    top_country = df.groupby('country')['revenue'].sum().idxmax()
    top_product = df.groupby('description')['revenue'].sum().idxmax()
    metrics = {"total_revenue": total_revenue, "top_country": top_country, "top_product": top_product}
    plots = {}
    revenue_by_country = df.groupby('country')['revenue'].sum().nlargest(15).reset_index()
    plots['revenue_by_country'] = px.bar(revenue_by_country, x='country', y='revenue', title="Top 15 Countries by Sales Revenue")
    revenue_by_product = df.groupby('description')['revenue'].sum().nlargest(15).reset_index()
    plots['revenue_by_product'] = px.bar(revenue_by_product, x='description', y='revenue', title="Top 15 Products by Sales Revenue")
    return {"metrics": metrics, "plots": plots}

def retail_order_status_and_item_analysis(df):
    expected = ['orderno', 'itemdesc', 'qty', 'invoicestatus']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    total_orders = df['orderno'].nunique()
    top_status = df['invoicestatus'].mode()[0]
    metrics = {"total_unique_orders": total_orders, "most_common_status": top_status}
    plots = {}
    status_counts = df['invoicestatus'].value_counts().reset_index()
    plots['status_distribution'] = px.pie(status_counts, names='index', values='invoicestatus', title="Distribution of Invoice Statuses")
    items_by_status = df.groupby('invoicestatus')['itemdesc'].count().reset_index()
    plots['items_by_status'] = px.bar(items_by_status, x='invoicestatus', y='itemdesc', title="Number of Items by Invoice Status")
    return {"metrics": metrics, "plots": plots}

def regional_sales_and_customer_analysis(df):
    expected = ['invoiceno', 'unitprice', 'quantity', 'customerid', 'region']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df = df[df['quantity'] > 0]
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('region').agg(
        total_revenue=('revenue', 'sum'),
        unique_customers=('customerid', 'nunique'),
        avg_revenue_per_customer=('revenue', 'mean')
    ).reset_index()
    plots = {}
    plots['revenue_by_region'] = px.bar(summary, x='region', y='total_revenue', color='unique_customers', title="Total Revenue by Region (Colored by Customer Count)")
    plots['revenue_share_by_region'] = px.pie(summary, names='region', values='total_revenue', title="Share of Revenue by Region")
    return {"summary_table": summary, "plots": plots}

def sales_channel_performance_analysis(df):
    expected = ['invoiceno', 'unitprice', 'quantity', 'saleschannel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df = df[df['quantity'] > 0]
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('saleschannel').agg(
        total_revenue=('revenue', 'sum'),
        total_orders=('invoiceno', 'nunique'),
    ).reset_index()
    summary['avg_order_value'] = summary['total_revenue'] / summary['total_orders']
    plots = {}
    plots['revenue_share'] = px.pie(summary, names='saleschannel', values='total_revenue', title="Share of Revenue by Sales Channel")
    plots['avg_order_value'] = px.bar(summary, x='saleschannel', y='avg_order_value', title="Average Order Value by Sales Channel")
    return {"summary_table": summary, "plots": plots}

def international_sales_and_transaction_analysis(df):
    expected = ['orderid', 'date', 'product', 'quantity', 'price', 'country', 'currency']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['revenue'] = df['quantity'] * df['price']
    df.dropna(inplace=True)
    num_countries = df['country'].nunique()
    top_country = df.groupby('country')['revenue'].sum().idxmax()
    metrics = {"num_countries": num_countries, "top_country_by_revenue": top_country}
    plots = {}
    revenue_by_country = df.groupby('country')['revenue'].sum().nlargest(20).reset_index()
    plots['revenue_by_country'] = px.bar(revenue_by_country, x='country', y='revenue', title="Top 20 Countries by Revenue")
    currency_counts = df['currency'].value_counts().reset_index()
    plots['transaction_count_by_currency'] = px.pie(currency_counts, names='index', values='currency', title="Transaction Count by Currency")
    return {"metrics": metrics, "plots": plots}

def invoice_type_and_customer_purchase_pattern_analysis(df):
    expected = ['invoiceno', 'invoicedate', 'unitprice', 'quantity', 'customerid', 'invoicetype']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('invoicetype').agg(
        total_revenue=('revenue', 'sum'),
        num_invoices=('invoiceno', 'nunique'),
        num_customers=('customerid', 'nunique')
    ).reset_index()
    plots = {}
    plots['metrics_by_invoice_type'] = px.bar(summary, x='invoicetype', y=['total_revenue', 'num_invoices'], barmode='group', title="Key Metrics by Invoice Type")
    return {"summary_table": summary, "plots": plots}

def order_delivery_and_customer_location_analysis(df):
    expected = ['orderid', 'customerid', 'deliveryzip']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    orders_by_zip = df['deliveryzip'].value_counts().nlargest(20).reset_index()
    customers_by_zip = df.groupby('deliveryzip')['customerid'].nunique().nlargest(20).reset_index()
    plots = {}
    plots['top_orders_by_zip'] = px.bar(orders_by_zip, x='index', y='deliveryzip', title="Top 20 ZIP Codes by Number of Orders")
    plots['top_customers_by_zip'] = px.bar(customers_by_zip, x='deliveryzip', y='customerid', title="Top 20 ZIP Codes by Number of Unique Customers")
    return {"plots": plots}
def promotional_code_effectiveness_analysis(df):
    expected = ['invoiceno', 'promotioncode', 'unitprice', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "Promo Code Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df['used_promo'] = df['promotioncode'].notna() & (df['promotioncode'] != 'None')
    df.dropna(inplace=True)
    summary = df.groupby('used_promo').agg(
        total_revenue=('revenue', 'sum'),
        num_orders=('invoiceno', 'nunique')
    ).reset_index()
    summary['aov'] = summary['total_revenue'] / summary['num_orders']
    plots = {}
    plots['aov_bar'] = px.bar(summary, x='used_promo', y='aov', title="Average Order Value (AOV) With vs. Without Promo Code")
    return {"summary_table": summary, "plots": plots}

def discount_impact_on_sales_analysis(df):
    expected = ['invoiceno', 'unitprice', 'quantity', 'discount']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "general Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['unitprice', 'quantity', 'discount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['revenue'] = df['quantity'] * df['unitprice'] * (1 - df['discount'])
    df['discount_level'] = pd.cut(df['discount'], bins=[0, 0.05, 0.1, 0.2, 1], labels=['0-5%', '5-10%', '10-20%', '20%+'], right=False)
    plots = {}
    plots['quantity_vs_discount'] = px.scatter(df, x='discount', y='quantity', title="Quantity Sold vs. Discount %")
    revenue_by_discount = df.groupby('discount_level')['revenue'].sum().reset_index()
    plots['revenue_by_discount'] = px.bar(revenue_by_discount, x='discount_level', y='revenue', title="Total Revenue by Discount Level")
    return {"plots": plots}

def product_cost_and_sales_price_margin_analysis(df):
    expected = ['product', 'unitcost', 'salesprice', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['unitcost', 'salesprice', 'quantity']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['margin_per_unit'] = df['salesprice'] - df['unitcost']
    df['margin_perc'] = (df['margin_per_unit'] / df['salesprice']) * 100
    avg_margin_perc = df['margin_perc'].mean()
    most_profitable_product = df.loc[df['margin_per_unit'].idxmax()]['product']
    metrics = {"average_profit_margin": avg_margin_perc, "most_profitable_product": most_profitable_product}
    plots = {}
    margin_by_product = df.groupby('product')['margin_perc'].mean().nlargest(15).reset_index()
    plots['margin_by_product'] = px.bar(margin_by_product, x='product', y='margin_perc', title="Top 15 Products by Average Profit Margin")
    plots['cost_vs_price'] = px.scatter(df, x='unitcost', y='salesprice', size='quantity', hover_name='product', title="Sales Price vs. Unit Cost (Sized by Quantity)")
    return {"metrics": metrics, "plots": plots}

def store_level_sales_performance_analysis(df):
    expected = ['invoiceno', 'itemname', 'quantity', 'unitprice', 'storeid']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('storeid').agg(
        total_revenue=('revenue', 'sum'),
        num_transactions=('invoiceno', 'nunique'),
        units_sold=('quantity', 'sum')
    ).reset_index()
    plots = {}
    plots['revenue_by_store'] = px.bar(summary, x='storeid', y='total_revenue', title="Total Revenue by Store")
    plots['revenue_vs_transactions'] = px.scatter(summary, x='num_transactions', y='total_revenue', size='units_sold', color='storeid', title="Revenue vs. Transactions (Sized by Units Sold)")
    return {"summary_table": summary, "plots": plots}

def product_category_sales_analysis(df):
    expected = ['sku', 'description', 'price', 'category', 'quantity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in ['category', 'price', 'quantity'] if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['revenue'] = df['price'] * df['quantity']
    df.dropna(inplace=True)
    summary = df.groupby('category').agg(
        total_revenue=('revenue', 'sum'),
        units_sold=('quantity', 'sum'),
        num_skus=('sku', 'nunique')
    ).reset_index()
    plots = {}
    plots['revenue_share'] = px.pie(summary, names='category', values='total_revenue', title="Share of Revenue by Product Category")
    plots['revenue_treemap'] = px.treemap(df, path=[px.Constant("All Categories"), 'category', 'description'], values='revenue', title="Hierarchical View of Revenue (Category > Product)")
    return {"summary_table": summary, "plots": plots}

def weekly_sales_trend_analysis(df):
    expected = ['invoicedate', 'unitprice', 'quantity', 'dayofweek']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    sales_by_day = df.groupby('dayofweek')['revenue'].sum().reset_index()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    try:
        sales_by_day['dayofweek'] = pd.Categorical(sales_by_day['dayofweek'], categories=day_order, ordered=True)
        sales_by_day = sales_by_day.sort_values('dayofweek')
    except:
        pass
    plots = {}
    plots['revenue_by_day'] = px.bar(sales_by_day, x='dayofweek', y='revenue', title="Total Sales Revenue by Day of the Week")
    return {"plots": plots}

def holiday_sales_impact_analysis(df):
    expected = ['invoicedate', 'unitprice', 'quantity', 'holidayflag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('holidayflag')['revenue'].agg(['mean', 'sum', 'count']).reset_index()
    summary.columns = ['Is Holiday', 'Avg Daily Revenue', 'Total Revenue', 'Day Count']
    plots = {}
    plots['avg_daily_rev'] = px.bar(summary, x='Is Holiday', y='Avg Daily Revenue', title="Average Daily Revenue on Holidays vs. Non-Holidays")
    return {"summary_table": summary, "plots": plots}

def customer_type_segmentation_and_sales_analysis(df):
    expected = ['invoiceno', 'unitprice', 'quantity', 'customertype']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('customertype').agg(
        total_revenue=('revenue', 'sum'),
        num_orders=('invoiceno', 'nunique')
    ).reset_index()
    summary['aov'] = summary['total_revenue'] / summary['num_orders']
    plots = {}
    plots['revenue_share'] = px.pie(summary, names='customertype', values='total_revenue', title="Share of Revenue by Customer Type")
    plots['aov_by_type'] = px.bar(summary, x='customertype', y='aov', title="Average Order Value by Customer Type")
    return {"summary_table": summary, "plots": plots}

def online_vs_offline_sales_analysis(df):
    expected = ['orderid', 'unitprice', 'qty', 'onlineflag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['qty'] * df['unitprice']
    df['channel'] = df['onlineflag'].apply(lambda x: 'Online' if x else 'Offline')
    df.dropna(inplace=True)
    summary = df.groupby('channel')['revenue'].agg(['sum', 'count']).reset_index()
    summary.columns = ['Channel', 'Total Revenue', 'Transaction Count']
    plots = {}
    plots['online_vs_offline'] = px.pie(summary, names='Channel', values='Total Revenue', title="Share of Revenue: Online vs. Offline")
    return {"summary_table": summary, "plots": plots}

def sales_tax_and_revenue_analysis(df):
    expected = ['invoiceno', 'priceeach', 'quantity', 'taxrate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['priceeach', 'quantity', 'taxrate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['revenue_pre_tax'] = df['priceeach'] * df['quantity']
    df['tax_amount'] = df['revenue_pre_tax'] * df['taxrate']
    df['total_revenue'] = df['revenue_pre_tax'] + df['tax_amount']
    total_revenue = df['total_revenue'].sum()
    total_tax = df['tax_amount'].sum()
    effective_tax_rate = (total_tax / df['revenue_pre_tax'].sum()) * 100
    metrics = {"total_revenue_incl_tax": total_revenue, "total_tax_collected": total_tax, "effective_tax_rate": effective_tax_rate}
    tax_by_rate = df.groupby('taxrate')['tax_amount'].sum().reset_index()
    plots = {}
    plots['tax_by_rate'] = px.bar(tax_by_rate, x='taxrate', y='tax_amount', title="Total Tax Collected by Tax Rate")
    return {"metrics": metrics, "plots": plots}

def sales_organization_performance_analysis(df):
    expected = ['invoiceno', 'unitprice', 'quantity', 'salesorg']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
    df['unitprice'] = pd.to_numeric(df['unitprice'], errors='coerce')
    df['revenue'] = df['quantity'] * df['unitprice']
    df.dropna(inplace=True)
    summary = df.groupby('salesorg').agg(
        total_revenue=('revenue', 'sum'),
        num_orders=('invoiceno', 'nunique')
    ).reset_index()
    plots = {}
    plots['revenue_by_salesorg'] = px.bar(summary.sort_values('total_revenue', ascending=False),
                                          x='salesorg', y='total_revenue', color='num_orders', title="Total Revenue by Sales Organization")
    return {"summary_table": summary, "plots": plots}
def customer_lifetime_value_clv_and_churn_risk_analysis(df):
    expected = ['customer_name', 'industry', 'lifetime_value', 'churn_risk_score', 'account_manager']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in ['lifetime_value', 'churn_risk_score'] if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['lifetime_value', 'churn_risk_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    avg_clv = df['lifetime_value'].mean()
    avg_churn_risk = df['churn_risk_score'].mean()
    metrics = {"average_lifetime_value": avg_clv, "average_churn_risk_score": avg_churn_risk}
    plots = {}
    plots['clv_vs_churn_risk'] = px.scatter(df, x='churn_risk_score', y='lifetime_value', color='industry', title="CLV vs. Churn Risk Score by Industry")
    clv_by_industry = df.groupby('industry')['lifetime_value'].mean().reset_index()
    plots['clv_by_industry'] = px.bar(clv_by_industry, x='industry', y='lifetime_value', title="Average CLV by Industry")
    return {"metrics": metrics, "plots": plots}

def subscription_sales_and_renewal_analysis(df):
    expected = ['plan_type', 'monthly_fee', 'auto_renew_flag', 'cancellation_date', 'cancellation_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in ['plan_type', 'cancellation_date'] if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['churned'] = df['cancellation_date'].notna()
    churn_rate = df['churned'].mean() * 100
    metrics = {"overall_churn_rate_percent": churn_rate}
    churn_by_plan = df.groupby('plan_type')['churned'].mean().mul(100).reset_index()
    plots = {}
    plots['churn_by_plan'] = px.bar(churn_by_plan, x='plan_type', y='churned', title="Churn Rate by Subscription Plan")
    if matched.get('cancellation_reason'):
        churn_reasons = df[df['churned']]['cancellation_reason'].value_counts().reset_index()
        plots['churn_reasons'] = px.pie(churn_reasons, names='index', values='cancellation_reason', title="Distribution of Churn Reasons")
    return {"metrics": metrics, "plots": plots}

def sales_channel_performance_and_conversion_analysis(df):
    expected = ['channel_type', 'units_sold', 'revenue', 'profit', 'conversion_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['units_sold', 'revenue', 'profit', 'conversion_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    summary = df.groupby('channel_type').agg({
        'revenue': 'sum',
        'profit': 'sum',
        'conversion_rate': 'mean'
    }).reset_index()
    plots = {}
    plots['revenue_and_profit'] = px.bar(summary, x='channel_type', y=['revenue', 'profit'], barmode='group', title="Total Revenue and Profit by Sales Channel")
    plots['conversion_rate'] = px.bar(summary, x='channel_type', y='conversion_rate', title="Average Conversion Rate by Sales Channel")
    return {"summary": summary, "plots": plots}

def cross_sell_and_upsell_opportunity_analysis(df):
    expected = ['primary_order_id', 'upsell_product_id', 'upsell_quantity', 'upsell_price', 'profit']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['upsell_quantity', 'upsell_price', 'profit']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    total_upsell_profit = df['profit'].sum()
    metrics = {"total_profit_from_upsells_cross_sells": total_upsell_profit}
    top_products = df.groupby('upsell_product_id')['profit'].sum().nlargest(15).reset_index()
    plots = {}
    plots['top_products'] = px.bar(top_products, x='upsell_product_id', y='profit', title="Top 15 Most Profitable Upsell/Cross-sell Products")
    return {"metrics": metrics, "plots": plots}

def sales_territory_performance_and_quota_achievement_analysis(df):
    expected = ['territory_name', 'sales_rep_id', 'quota', 'ytd_sales', 'achievement_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['quota', 'ytd_sales', 'achievement_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    overall_achievement = (df['ytd_sales'].sum() / df['quota'].sum()) * 100
    metrics = {"overall_quota_achievement_percent": overall_achievement}
    summary = df.groupby('territory_name').agg(
        total_quota=('quota', 'sum'),
        total_sales=('ytd_sales', 'sum')
    ).reset_index()
    summary['achievement'] = (summary['total_sales'] / summary['total_quota']) * 100
    plots = {}
    plots['quota_vs_sales'] = px.bar(summary, x='territory_name', y=['total_quota', 'total_sales'], barmode='group', title="Quota vs. YTD Sales by Territory")
    return {"metrics": metrics, "summary": summary, "plots": plots}

def product_sales_performance_and_profitability_analysis(df):
    expected = ['product_id', 'units_sold', 'revenue', 'cogs', 'gross_profit', 'profit_margin_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in ['product_id', 'revenue', 'gross_profit'] if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['units_sold', 'revenue', 'cogs', 'gross_profit', 'profit_margin_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    summary = df.groupby('product_id').agg(
        total_revenue=('revenue', 'sum'),
        total_profit=('gross_profit', 'sum')
    ).nlargest(20, 'total_revenue').reset_index()
    plots = {}
    plots['top_products'] = px.bar(summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 20 Products by Revenue and Profit")
    return {"summary": summary, "plots": plots}

def product_pricing_strategy_and_tier_analysis(df):
    expected = ['product_id', 'list_price', 'tier_1_price', 'tier_2_price', 'tier_3_price', 'channel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['list_price', 'tier_1_price', 'tier_2_price', 'tier_3_price']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df_long = df.melt(id_vars=['product_id', 'channel'], 
                      value_vars=['list_price', 'tier_1_price', 'tier_2_price', 'tier_3_price'],
                      var_name='price_tier', value_name='price')
    # Example: returns pricing tiers for all products for further use
    return {"pricing_tiers": df_long}

def sales_forecasting_accuracy_analysis(df):
    expected = ['forecast_date', 'territory', 'pipeline_value', 'forecast_value', 'historical_sales']
    matched = fuzzy_match_column(df, expected)
    actual_col = matched.get('historical_sales')
    if not matched.get('forecast_value') or not actual_col:
        return {"missing_columns": show_missing_columns_warning(['forecast_value', 'actual_sales/historical_sales'], matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
    for col in ['pipeline_value', 'forecast_value', 'historical_sales']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('forecast_date').dropna()
    df['error'] = df['historical_sales'] - df['forecast_value']
    df['mape'] = (df['error'].abs() / df['historical_sales'].abs()) * 100
    mape = df['mape'].mean()
    metrics = {"mean_absolute_percentage_error": mape}
    df_long = df.melt(id_vars='forecast_date', value_vars=['forecast_value', 'historical_sales'],
                      var_name='type', value_name='sales')
    plots = {}
    plots['forecast_vs_actual'] = px.line(df_long, x='forecast_date', y='sales', color='type', title="Forecast vs. Actual Sales")
    return {"metrics": metrics, "plots": plots}

def channel_promotion_performance_and_roi_analysis(df):
    expected = ['channel', 'promotion_start', 'discount_perc', 'projected_lift_perc', 'actual_lift_perc', 'roi_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['discount_perc', 'projected_lift_perc', 'actual_lift_perc', 'roi_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    plots = {}
    plots['actual_vs_projected'] = px.scatter(df, x='projected_lift_perc', y='actual_lift_perc', color='channel',
                                              title="Actual vs. Projected Sales Lift by Channel")
    summary = df.groupby('channel')[['roi_perc', 'actual_lift_perc']].mean().reset_index()
    plots['roi_and_lift'] = px.bar(summary, x='channel', y=['roi_perc', 'actual_lift_perc'], barmode='group',
                                   title="Average ROI and Actual Lift by Channel")
    return {"plots": plots}

def customer_service_impact_on_sales_analysis(df):
    expected = ['case_open_date', 'resolution_time_min', 'satisfaction_score', 'case_status', 'escalation_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['resolution_time_min', 'satisfaction_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    plots = {}
    plots['csat_distribution'] = px.histogram(df, x='satisfaction_score', title="Distribution of CSAT Scores")
    plots['csat_vs_resolution_time'] = px.scatter(df, x='resolution_time_min', y='satisfaction_score', title="Satisfaction Score vs. Resolution Time")
    return {"plots": plots}

def sales_call_outcome_and_effectiveness_analysis(df):
    expected = ['call_date', 'sales_rep_id', 'call_duration_sec', 'outcome', 'deal_size']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['call_duration_sec', 'deal_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    outcome_counts = df['outcome'].value_counts().reset_index()
    plots = {}
    plots['call_outcome_distribution'] = px.pie(outcome_counts, names='index', values='outcome', title="Distribution of Call Outcomes")
    plots['call_duration_by_outcome'] = px.box(df, x='outcome', y='call_duration_sec', title="Call Duration by Outcome")
    return {"plots": plots}
def market_segment_revenue_and_profitability_analysis(df):
    expected = ['segment_name', 'segment_revenue', 'segment_profit', 'segment_growth_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['segment_revenue', 'segment_profit', 'segment_growth_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    plots = {}
    plots['revenue_and_profit'] = px.bar(
        df, x='segment_name', y=['segment_revenue', 'segment_profit'],
        barmode='group', title="Revenue and Profit by Market Segment"
    )
    plots['profit_vs_growth'] = px.scatter(
        df, x='segment_growth_perc', y='segment_profit', size='segment_revenue',
        color='segment_name', title="Profit vs. Growth % by Segment"
    )
    return {"plots": plots}

def competitor_pricing_and_feature_analysis(df):
    expected = ['our_product_id', 'competitor_product', 'competitor_price', 'our_price', 'market_share_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['competitor_price', 'our_price', 'market_share_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['price_difference'] = df['our_price'] - df['competitor_price']
    plots = {}
    plots['price_comparison'] = px.bar(
        df, x='our_product_id', y=['our_price', 'competitor_price'],
        barmode='group', title="Our Price vs. Competitor Price by Product"
    )
    return {"plots": plots}

def product_bundle_sales_performance_analysis(df):
    expected = ['bundle_name', 'bundle_price', 'revenue', 'profit']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['bundle_price', 'revenue', 'profit']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    summary = df.groupby('bundle_name').agg({'revenue':'sum', 'profit':'sum'}).reset_index()
    plots = {}
    plots['revenue_and_profit_by_bundle'] = px.bar(
        summary, x='bundle_name', y=['revenue', 'profit'],
        title="Total Revenue and Profit by Product Bundle"
    )
    return {"summary": summary, "plots": plots}

def international_sales_and_currency_exchange_analysis(df):
    expected = ['currency_pair', 'exchange_rate', 'converted_amount', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['exchange_rate', 'converted_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    summary = df.groupby('currency_pair').agg(
        total_value=('converted_amount', 'sum'),
        num_transactions=('transaction_id', 'count')
    ).reset_index()
    plots = {}
    plots['transaction_value_by_currency'] = px.bar(
        summary, x='currency_pair', y='total_value', color='num_transactions',
        title="Total Transaction Value by Currency Pair"
    )
    return {"summary": summary, "plots": plots}

def sales_contract_and_renewal_analysis(df):
    expected = ['customer_id', 'contract_value', 'renewal_option', 'renewal_probability_perc', 'contract_status', 'signed_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['contract_value', 'renewal_probability_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    status_counts = df['contract_status'].value_counts().reset_index()
    plots = {}
    plots['contract_status_distribution'] = px.pie(
        status_counts, names='index', values='contract_status',
        title="Distribution of Contract Statuses"
    )
    plots['renewal_vs_value'] = px.scatter(
        df, x='contract_value', y='renewal_probability_perc', color='renewal_option',
        title="Renewal Probability vs. Contract Value"
    )
    return {"plots": plots}

def e_commerce_sales_funnel_and_conversion_analysis(df):
    expected = ['session_id', 'add_to_cart_date', 'purchase_date', 'cart_abandon_flag', 'revenue']
    matched = fuzzy_match_column(df, expected)
    if not matched.get('cart_abandon_flag'):
        return {
            "missing_columns": show_missing_columns_warning(['cart_abandon_flag'], matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    total_sessions = df['session_id'].nunique()
    purchased_sessions = df[df['cart_abandon_flag'] == False]['session_id'].nunique()
    conversion_rate = (purchased_sessions / total_sessions) * 100
    metrics = {"session_to_purchase_conversion_rate": conversion_rate}
    funnel_values = [total_sessions, purchased_sessions]
    funnel_stages = ['Total Sessions', 'Purchased']
    fig = go.Figure(go.Funnel(
        y=funnel_stages,
        x=funnel_values,
        textinfo="value+percent previous"
    ))
    fig.update_layout(title_text="Simple Sales Funnel (Sessions to Purchase)")
    plots = {"sales_funnel": fig}
    return {"metrics": metrics, "plots": plots}
def sales_order_fulfillment_and_status_analysis(df):
    expected = ['order_id', 'customer_id', 'order_date', 'order_status', 'order_value', 'fulfillment_days']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        } 

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce')
    df['fulfillment_days'] = pd.to_numeric(df['fulfillment_days'], errors='coerce')
    df.dropna(inplace=True)

    plots = {}
    plots['order_status_distribution'] = px.pie(df, names='order_status', title="Distribution of Order Statuses")
    avg_fulfillment = df.groupby('order_status')['fulfillment_days'].mean().reset_index()
    plots['avg_fulfillment'] = px.bar(avg_fulfillment, x='order_status', y='fulfillment_days', title="Average Fulfillment Days by Order Status")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    revenue_trend = df.groupby(df['order_date'].dt.to_period("M"))['order_value'].sum().reset_index()
    revenue_trend['order_date'] = revenue_trend['order_date'].astype(str)
    plots['monthly_revenue'] = px.line(revenue_trend, x='order_date', y='order_value', markers=True, title="Monthly Revenue Trend")

    return {"plots": plots}

def sales_invoice_and_payment_reconciliation_analysis(df):
    expected = ['invoice_id', 'invoice_amount', 'payment_amount', 'payment_date', 'invoicestatus']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['invoice_amount', 'payment_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['balance_due'] = df['invoice_amount'] - df['payment_amount']
    df.dropna(inplace=True)

    summary = df.groupby('invoicestatus').agg(
        num_invoices=('invoice_id', 'nunique'),
        total_balance=('balance_due', 'sum')
    ).reset_index()

    plots = {}
    plots['invoice_status_distribution'] = px.pie(summary, names='invoicestatus', values='num_invoices', title="Distribution of Invoice Statuses")
    plots['balance_due_by_status'] = px.bar(summary, x='invoicestatus', y='total_balance', title="Total Balance Due by Invoice Status")

    return {"summary": summary, "plots": plots}

def sales_transaction_and_profit_margin_analysis(df):
    expected = ['transaction_id', 'product_id', 'sales_price', 'cost_of_goods_sold']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['sales_price', 'cost_of_goods_sold']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['profit_margin'] = df['sales_price'] - df['cost_of_goods_sold']
    
    product_summary = df.groupby('product_id').agg(
        total_revenue=('sales_price', 'sum'),
        total_profit=('profit_margin', 'sum'),
        avg_profit_margin=('profit_margin', 'mean')
    ).nlargest(15, 'total_revenue').reset_index()

    plots = {}
    plots['top_products_revenue_profit'] = px.bar(product_summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 15 Products by Revenue and Profit")
    plots['profit_margin_distribution'] = px.histogram(df, x='profit_margin', title="Distribution of Profit Margins")

    return {"summary": product_summary, "plots": plots}

def sales_representative_performance_and_revenue_analysis(df):
    expected = ['sales_rep_id', 'revenue', 'transactions_count', 'deal_size']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['revenue', 'transactions_count', 'deal_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    rep_summary = df.groupby('sales_rep_id').agg(
        total_revenue=('revenue', 'sum'),
        total_transactions=('transactions_count', 'sum'),
        avg_deal_size=('deal_size', 'mean')
    ).nlargest(10, 'total_revenue').reset_index()

    plots = {}
    plots['revenue_by_rep'] = px.bar(rep_summary, x='sales_rep_id', y='total_revenue', title="Total Revenue by Sales Representative")
    plots['deal_size_by_rep'] = px.bar(rep_summary, x='sales_rep_id', y='avg_deal_size', title="Average Deal Size by Sales Representative")

    return {"summary": rep_summary, "plots": plots}

def sales_channel_and_customer_segment_performance_analysis(df):
    expected = ['sales_channel', 'customer_segment', 'revenue', 'transactions']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['revenue', 'transactions']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    summary = df.groupby(['sales_channel', 'customer_segment']).agg(
        total_revenue=('revenue', 'sum'),
        total_transactions=('transactions', 'sum')
    ).reset_index()

    plots = {}
    plots['revenue_by_channel_segment'] = px.bar(summary, x='sales_channel', y='total_revenue', color='customer_segment', barmode='group', title="Revenue by Sales Channel and Customer Segment")
    plots['transactions_by_channel_segment'] = px.bar(summary, x='sales_channel', y='total_transactions', color='customer_segment', barmode='group', title="Transactions by Sales Channel and Customer Segment")

    return {"summary": summary, "plots": plots}

def sales_opportunity_and_pipeline_analysis(df):
    expected = ['opportunity_id', 'deal_stage', 'deal_value', 'deal_close_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['deal_value'] = pd.to_numeric(df['deal_value'], errors='coerce')
    df.dropna(inplace=True)

    pipeline_summary = df.groupby('deal_stage')['deal_value'].sum().reset_index()

    funnel_stages = ['Prospecting', 'Qualifying', 'Closing', 'Won', 'Lost']
    funnel_values = [pipeline_summary[pipeline_summary['deal_stage'] == stage]['deal_value'].sum() for stage in funnel_stages]

    plots = {}
    plots['sales_funnel'] = go.Figure(go.Funnel(y=funnel_stages, x=funnel_values, textinfo="value+percent previous"))
    plots['sales_funnel'].update_layout(title_text="Sales Pipeline Funnel")

    return {"summary": pipeline_summary, "plots": plots}

def sales_quote_conversion_and_pricing_analysis(df):
    expected = ['quote_id', 'quote_amount', 'conversion_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quote_amount'] = pd.to_numeric(df['quote_amount'], errors='coerce')
    df['conversion_flag'] = df['conversion_flag'].astype(bool)
    df.dropna(inplace=True)

    conversion_rate = df['conversion_flag'].mean() * 100
    metrics = {"overall_conversion_rate": conversion_rate}

    df['quote_amount_bin'] = pd.cut(df['quote_amount'], bins=10)
    conversion_by_amount = df.groupby('quote_amount_bin')['conversion_flag'].mean().mul(100).reset_index()

    plots = {}
    plots['conversion_rate_by_amount'] = px.bar(conversion_by_amount, x='quote_amount_bin', y='conversion_flag', title="Quote Conversion Rate by Quote Amount")

    return {"metrics": metrics, "plots": plots}

def sales_return_and_refund_analysis(df):
    expected = ['return_id', 'transaction_id', 'return_amount', 'return_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['return_amount'] = pd.to_numeric(df['return_amount'], errors='coerce')
    df.dropna(inplace=True)

    return_reason_summary = df.groupby('return_reason').agg(
        total_returns=('return_id', 'nunique'),
        total_refund_amount=('return_amount', 'sum')
    ).reset_index()
    
    plots = {}
    plots['return_reasons_count'] = px.pie(return_reason_summary, names='return_reason', values='total_returns', title="Distribution of Return Reasons by Count")
    plots['return_reasons_amount'] = px.bar(return_reason_summary, x='return_reason', y='total_refund_amount', title="Total Refund Amount by Return Reason")

    return {"summary": return_reason_summary, "plots": plots}

def sales_lead_and_opportunity_conversion_analysis(df):
    expected = ['lead_id', 'lead_source', 'is_converted', 'deal_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['is_converted'] = df['is_converted'].astype(bool)
    df.dropna(inplace=True)

    conversion_by_source = df.groupby('lead_source').agg(
        total_leads=('lead_id', 'nunique'),
        converted_leads=('is_converted', 'sum'),
        total_deal_value=('deal_value', 'sum')
    ).reset_index()
    conversion_by_source['conversion_rate'] = (conversion_by_source['converted_leads'] / conversion_by_source['total_leads']) * 100

    plots = {}
    plots['conversion_rate_by_source'] = px.bar(conversion_by_source, x='lead_source', y='conversion_rate', title="Lead Conversion Rate by Source")
    plots['deal_value_by_source'] = px.bar(conversion_by_source, x='lead_source', y='total_deal_value', title="Total Deal Value by Lead Source")

    return {"summary": conversion_by_source, "plots": plots}

def customer_payment_and_reconciliation_analysis(df):
    expected = ['payment_id', 'customer_id', 'payment_amount', 'invoice_id', 'payment_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('payment_status').agg(
        total_payments=('payment_amount', 'sum'),
        num_payments=('payment_id', 'nunique')
    ).reset_index()

    plots = {}
    plots['payments_by_status'] = px.bar(summary, x='payment_status', y='total_payments', title="Total Payments by Status")
    
    return {"summary": summary, "plots": plots}

def lead_management_and_conversion_funnel_analysis(df):
    expected = ['lead_id', 'funnel_stage', 'conversion_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    
    stage_counts = df.groupby('funnel_stage')['lead_id'].nunique().reset_index()
    # Order stages for a meaningful funnel chart
    funnel_stages = ['Contact', 'Qualified', 'Proposal', 'Conversion']
    funnel_data = pd.DataFrame({'stage': funnel_stages, 'count': [stage_counts[stage_counts['funnel_stage'] == s]['lead_id'].sum() for s in funnel_stages]})
    
    plots = {}
    plots['funnel_chart'] = go.Figure(go.Funnel(y=funnel_data['stage'], x=funnel_data['count'], textinfo="value+percent previous"))
    plots['funnel_chart'].update_layout(title_text="Lead Conversion Funnel")
    
    return {"summary": funnel_data, "plots": plots}

def field_sales_visit_effectiveness_analysis(df):
    expected = ['visit_id', 'sales_rep_id', 'visit_duration', 'outcome']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['visit_duration'] = pd.to_numeric(df['visit_duration'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['sales_rep_id', 'outcome']).agg(
        num_visits=('visit_id', 'nunique'),
        avg_duration=('visit_duration', 'mean')
    ).reset_index()
    
    # Calculate success rate (assuming 'Success' is a possible outcome)
    rep_outcomes = df.groupby('sales_rep_id')['outcome'].value_counts(normalize=True).unstack(fill_value=0)
    
    plots = {}
    plots['visits_by_rep_outcome'] = px.bar(summary, x='sales_rep_id', y='num_visits', color='outcome', title="Number of Visits by Sales Rep and Outcome")
    if 'Success' in rep_outcomes.columns:
        rep_outcomes.reset_index(inplace=True)
        plots['success_rate_by_rep'] = px.bar(rep_outcomes, x='sales_rep_id', y='Success', title="Success Rate by Sales Rep")
    
    return {"summary": summary, "plots": plots}

def sales_key_performance_indicator_kpi_trend_analysis(df):
    expected = ['date', 'kpi_name', 'kpi_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['kpi_value'] = pd.to_numeric(df['kpi_value'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['date', 'kpi_name'])['kpi_value'].sum().reset_index()
    
    plots = {}
    plots['kpi_trend_lines'] = px.line(summary, x='date', y='kpi_value', color='kpi_name', title="KPI Trends Over Time")
    
    return {"summary": summary, "plots": plots}

def sales_refund_and_reason_code_analysis(df):
    expected = ['refund_id', 'refund_amount', 'refund_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['refund_amount'] = pd.to_numeric(df['refund_amount'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('refund_reason').agg(
        num_refunds=('refund_id', 'nunique'),
        total_refund_amount=('refund_amount', 'sum')
    ).reset_index()
    
    plots = {}
    plots['refunds_by_reason_count'] = px.bar(summary, x='refund_reason', y='num_refunds', title="Number of Refunds by Reason")
    plots['refunds_by_reason_amount'] = px.bar(summary, x='refund_reason', y='total_refund_amount', title="Total Refund Amount by Reason")
    
    return {"summary": summary, "plots": plots}

def lead_nurturing_campaign_effectiveness_analysis(df):
    expected = ['campaign_id', 'leads_generated', 'qualified_leads', 'converted_leads']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['leads_generated', 'qualified_leads', 'converted_leads']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df_long = df.melt(id_vars='campaign_id', value_vars=['leads_generated', 'qualified_leads', 'converted_leads'],
                      var_name='metric', value_name='count')
    
    plots = {}
    plots['campaign_performance'] = px.bar(df_long, x='campaign_id', y='count', color='metric', barmode='group', title="Campaign Performance (Leads, Qualified, Converted)")
    
    return {"df_long": df_long, "plots": plots}
def sales_performance_analysis(df):
    expected = ['date', 'revenue', 'transactions', 'product_id', 'sales_rep_id', 'sales_channel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'revenue'], inplace=True)
    
    total_revenue = df['revenue'].sum()
    total_transactions = df['transactions'].sum() if 'transactions' in df else df.shape[0]
    
    metrics = {
        "total_revenue": total_revenue,
        "total_transactions": total_transactions,
        "avg_transaction_value": total_revenue / total_transactions if total_transactions > 0 else 0
    }
    
    plots = {}
    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()
    plots['revenue_by_channel'] = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title="Revenue Distribution by Sales Channel")
    
    revenue_by_rep = df.groupby('sales_rep_id')['revenue'].sum().nlargest(10).reset_index()
    plots['top_reps_by_revenue'] = px.bar(revenue_by_rep, x='sales_rep_id', y='revenue', title="Top 10 Sales Reps by Revenue")
    
    return {"metrics": metrics, "plots": plots}

def time_series_analysis(df):
    expected = ['date', 'time_series_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'time_series_value'], inplace=True)
    df.set_index('date', inplace=True)
    
    monthly_trend = df['time_series_value'].resample('M').sum().reset_index()
    monthly_trend['date'] = monthly_trend['date'].dt.strftime('%Y-%m')
    
    plots = {}
    plots['monthly_trend'] = px.line(monthly_trend, x='date', y='time_series_value', title="Monthly Time Series Trend")
    
    return {"summary": monthly_trend, "plots": plots}

def regional_analysis(df):
    expected = ['customer_id', 'revenue', 'country', 'region']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['revenue', 'country'], inplace=True)
    
    summary = df.groupby('country')['revenue'].sum().reset_index()
    
    plots = {}
    plots['revenue_by_country'] = px.bar(summary.nlargest(10, 'revenue'), x='country', y='revenue', title="Top 10 Countries by Revenue")
    
    if 'region' in df.columns:
        summary_region = df.groupby('region')['revenue'].sum().reset_index()
        plots['revenue_by_region'] = px.pie(summary_region, names='region', values='revenue', title="Revenue Distribution by Region")
    
    return {"summary": summary, "plots": plots}
    
def sales_channel_analysis(df):
    expected = ['sales_channel', 'revenue', 'transactions', 'customer_segment']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

    summary = df.groupby('sales_channel').agg(
        total_revenue=('revenue', 'sum'),
        total_transactions=('transactions', 'sum') if 'transactions' in df else ('revenue', 'count')
    ).reset_index()
    
    plots = {}
    plots['revenue_by_channel'] = px.bar(summary, x='sales_channel', y='total_revenue', title="Total Revenue by Sales Channel")
    if 'customer_segment' in df.columns:
        revenue_by_channel_segment = df.groupby(['sales_channel', 'customer_segment'])['revenue'].sum().reset_index()
        plots['revenue_by_channel_segment'] = px.bar(revenue_by_channel_segment, x='sales_channel', y='revenue', color='customer_segment', title="Revenue by Channel and Customer Segment")

    return {"summary": summary, "plots": plots}

def campaign_analysis(df):
    expected = ['campaign_id', 'leads_generated', 'revenue', 'cost', 'conversion_rate', 'roi']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(inplace=True)
    
    total_campaigns = df['campaign_id'].nunique()
    total_revenue = df['revenue'].sum()
    total_cost = df['cost'].sum()
    overall_roi = (total_revenue - total_cost) / total_cost if total_cost > 0 else 0
    
    metrics = {
        "total_campaigns": total_campaigns,
        "total_revenue": total_revenue,
        "overall_roi": overall_roi
    }
    
    summary = df.groupby('campaign_id').agg(
        revenue=('revenue', 'sum'),
        cost=('cost', 'sum')
    ).reset_index()
    summary['roi'] = (summary['revenue'] - summary['cost']) / summary['cost'] if summary['cost'].any() else 0
    
    plots = {}
    plots['roi_by_campaign'] = px.bar(summary, x='campaign_id', y='roi', title="ROI by Campaign")
    
    return {"metrics": metrics, "summary": summary, "plots": plots}

def sales_forecasting(df):
    expected = ['date', 'forecasted_sales', 'actual_sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'forecasted_sales', 'actual_sales'], inplace=True)
    
    df['error'] = df['actual_sales'] - df['forecasted_sales']
    mape = (np.abs(df['error']) / df['actual_sales']).mean() * 100
    
    metrics = {"mean_absolute_percentage_error": mape}
    
    df_long = df.melt(id_vars='date', value_vars=['forecasted_sales', 'actual_sales'], var_name='type', value_name='sales')
    plots = {}
    plots['forecast_vs_actual'] = px.line(df_long, x='date', y='sales', color='type', title="Forecast vs. Actual Sales")
    
    return {"metrics": metrics, "plots": plots}

def profit_analysis(df):
    expected = ['product_id', 'revenue', 'cost_of_goods_sold', 'gross_profit', 'profit_margin']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"missing_columns": show_missing_columns_warning(missing, matched),
                "insights": show_general_insights(df, "General Analysis")}
    
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(inplace=True)
    
    df['profit_margin_perc'] = (df['gross_profit'] / df['revenue']) * 100 if df['revenue'].any() else 0
    
    summary = df.groupby('product_id').agg(
        total_revenue=('revenue', 'sum'),
        total_profit=('gross_profit', 'sum'),
        avg_margin_perc=('profit_margin_perc', 'mean')
    ).nlargest(10, 'total_profit').reset_index()
    
    plots = {}
    plots['revenue_and_profit_by_product'] = px.bar(summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 10 Products by Revenue and Profit")
    plots['margin_distribution'] = px.histogram(df, x='profit_margin_perc', title="Distribution of Profit Margins (%)")
    
    return {"summary": summary, "plots": plots}


def sales_order_fulfillment_and_status_analysis(df):
    expected = ['order_id', 'customer_id', 'order_date', 'order_status', 'order_value', 'fulfillment_days']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['order_value'] = pd.to_numeric(df['order_value'], errors='coerce')
    df['fulfillment_days'] = pd.to_numeric(df['fulfillment_days'], errors='coerce')
    df.dropna(inplace=True)

    plots = {}
    plots['order_status_distribution'] = px.pie(df, names='order_status', title="Distribution of Order Statuses")
    avg_fulfillment = df.groupby('order_status')['fulfillment_days'].mean().reset_index()
    plots['avg_fulfillment'] = px.bar(avg_fulfillment, x='order_status', y='fulfillment_days', title="Average Fulfillment Days by Order Status")
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    revenue_trend = df.groupby(df['order_date'].dt.to_period("M"))['order_value'].sum().reset_index()
    revenue_trend['order_date'] = revenue_trend['order_date'].astype(str)
    plots['monthly_revenue'] = px.line(revenue_trend, x='order_date', y='order_value', markers=True, title="Monthly Revenue Trend")

    return {"plots": plots}

def sales_invoice_and_payment_reconciliation_analysis(df):
    expected = ['invoice_id', 'invoice_amount', 'payment_amount', 'payment_date', 'invoicestatus']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['invoice_amount', 'payment_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['balance_due'] = df['invoice_amount'] - df['payment_amount']
    df.dropna(inplace=True)

    summary = df.groupby('invoicestatus').agg(
        num_invoices=('invoice_id', 'nunique'),
        total_balance=('balance_due', 'sum')
    ).reset_index()

    plots = {}
    plots['invoice_status_distribution'] = px.pie(summary, names='invoicestatus', values='num_invoices', title="Distribution of Invoice Statuses")
    plots['balance_due_by_status'] = px.bar(summary, x='invoicestatus', y='total_balance', title="Total Balance Due by Invoice Status")

    return {"summary": summary, "plots": plots}

def sales_transaction_and_profit_margin_analysis(df):
    expected = ['transaction_id', 'product_id', 'sales_price', 'cost_of_goods_sold']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['sales_price', 'cost_of_goods_sold']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['profit_margin'] = df['sales_price'] - df['cost_of_goods_sold']
    
    product_summary = df.groupby('product_id').agg(
        total_revenue=('sales_price', 'sum'),
        total_profit=('profit_margin', 'sum'),
        avg_profit_margin=('profit_margin', 'mean')
    ).nlargest(15, 'total_revenue').reset_index()

    plots = {}
    plots['top_products_revenue_profit'] = px.bar(product_summary, x='product_id', y=['total_revenue', 'total_profit'], title="Top 15 Products by Revenue and Profit")
    plots['profit_margin_distribution'] = px.histogram(df, x='profit_margin', title="Distribution of Profit Margins")

    return {"summary": product_summary, "plots": plots}

def sales_representative_performance_and_revenue_analysis(df):
    expected = ['sales_rep_id', 'revenue', 'transactions_count', 'deal_size']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['revenue', 'transactions_count', 'deal_size']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    rep_summary = df.groupby('sales_rep_id').agg(
        total_revenue=('revenue', 'sum'),
        total_transactions=('transactions_count', 'sum'),
        avg_deal_size=('deal_size', 'mean')
    ).nlargest(10, 'total_revenue').reset_index()

    plots = {}
    plots['revenue_by_rep'] = px.bar(rep_summary, x='sales_rep_id', y='total_revenue', title="Total Revenue by Sales Representative")
    plots['deal_size_by_rep'] = px.bar(rep_summary, x='sales_rep_id', y='avg_deal_size', title="Average Deal Size by Sales Representative")

    return {"summary": rep_summary, "plots": plots}

def sales_channel_and_customer_segment_performance_analysis(df):
    expected = ['sales_channel', 'customer_segment', 'revenue', 'transactions']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['revenue', 'transactions']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    summary = df.groupby(['sales_channel', 'customer_segment']).agg(
        total_revenue=('revenue', 'sum'),
        total_transactions=('transactions', 'sum')
    ).reset_index()

    plots = {}
    plots['revenue_by_channel_segment'] = px.bar(summary, x='sales_channel', y='total_revenue', color='customer_segment', barmode='group', title="Revenue by Sales Channel and Customer Segment")
    plots['transactions_by_channel_segment'] = px.bar(summary, x='sales_channel', y='total_transactions', color='customer_segment', barmode='group', title="Transactions by Sales Channel and Customer Segment")

    return {"summary": summary, "plots": plots}

def sales_opportunity_and_pipeline_analysis(df):
    expected = ['opportunity_id', 'deal_stage', 'deal_value', 'deal_close_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['deal_value'] = pd.to_numeric(df['deal_value'], errors='coerce')
    df.dropna(inplace=True)

    pipeline_summary = df.groupby('deal_stage')['deal_value'].sum().reset_index()

    funnel_stages = ['Prospecting', 'Qualifying', 'Closing', 'Won', 'Lost']
    funnel_values = [pipeline_summary[pipeline_summary['deal_stage'] == stage]['deal_value'].sum() for stage in funnel_stages]

    plots = {}
    plots['sales_funnel'] = go.Figure(go.Funnel(y=funnel_stages, x=funnel_values, textinfo="value+percent previous"))
    plots['sales_funnel'].update_layout(title_text="Sales Pipeline Funnel")

    return {"summary": pipeline_summary, "plots": plots}

def sales_quote_conversion_and_pricing_analysis(df):
    expected = ['quote_id', 'quote_amount', 'conversion_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quote_amount'] = pd.to_numeric(df['quote_amount'], errors='coerce')
    df['conversion_flag'] = df['conversion_flag'].astype(bool)
    df.dropna(inplace=True)

    conversion_rate = df['conversion_flag'].mean() * 100
    metrics = {"overall_conversion_rate": conversion_rate}

    df['quote_amount_bin'] = pd.cut(df['quote_amount'], bins=10)
    conversion_by_amount = df.groupby('quote_amount_bin')['conversion_flag'].mean().mul(100).reset_index()

    plots = {}
    plots['conversion_rate_by_amount'] = px.bar(conversion_by_amount, x='quote_amount_bin', y='conversion_flag', title="Quote Conversion Rate by Quote Amount")

    return {"metrics": metrics, "plots": plots}

def sales_return_and_refund_analysis(df):
    expected = ['return_id', 'transaction_id', 'return_amount', 'return_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['return_amount'] = pd.to_numeric(df['return_amount'], errors='coerce')
    df.dropna(inplace=True)

    return_reason_summary = df.groupby('return_reason').agg(
        total_returns=('return_id', 'nunique'),
        total_refund_amount=('return_amount', 'sum')
    ).reset_index()
    
    plots = {}
    plots['return_reasons_count'] = px.pie(return_reason_summary, names='return_reason', values='total_returns', title="Distribution of Return Reasons by Count")
    plots['return_reasons_amount'] = px.bar(return_reason_summary, x='return_reason', y='total_refund_amount', title="Total Refund Amount by Return Reason")

    return {"summary": return_reason_summary, "plots": plots}

def sales_lead_and_opportunity_conversion_analysis(df):
    expected = ['lead_id', 'lead_source', 'is_converted', 'deal_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['is_converted'] = df['is_converted'].astype(bool)
    df.dropna(inplace=True)

    conversion_by_source = df.groupby('lead_source').agg(
        total_leads=('lead_id', 'nunique'),
        converted_leads=('is_converted', 'sum'),
        total_deal_value=('deal_value', 'sum')
    ).reset_index()
    conversion_by_source['conversion_rate'] = (conversion_by_source['converted_leads'] / conversion_by_source['total_leads']) * 100

    plots = {}
    plots['conversion_rate_by_source'] = px.bar(conversion_by_source, x='lead_source', y='conversion_rate', title="Lead Conversion Rate by Source")
    plots['deal_value_by_source'] = px.bar(conversion_by_source, x='lead_source', y='total_deal_value', title="Total Deal Value by Lead Source")

    return {"summary": conversion_by_source, "plots": plots}

def customer_payment_and_reconciliation_analysis(df):
    expected = ['payment_id', 'customer_id', 'payment_amount', 'invoice_id', 'payment_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['payment_amount'] = pd.to_numeric(df['payment_amount'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('payment_status').agg(
        total_payments=('payment_amount', 'sum'),
        num_payments=('payment_id', 'nunique')
    ).reset_index()

    plots = {}
    plots['payments_by_status'] = px.bar(summary, x='payment_status', y='total_payments', title="Total Payments by Status")
    
    return {"summary": summary, "plots": plots}

def lead_management_and_conversion_funnel_analysis(df):
    expected = ['lead_id', 'funnel_stage', 'conversion_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    
    stage_counts = df.groupby('funnel_stage')['lead_id'].nunique().reset_index()
    # Order stages for a meaningful funnel chart
    funnel_stages = ['Contact', 'Qualified', 'Proposal', 'Conversion']
    funnel_data = pd.DataFrame({'stage': funnel_stages, 'count': [stage_counts[stage_counts['funnel_stage'] == s]['lead_id'].sum() for s in funnel_stages]})
    
    plots = {}
    plots['funnel_chart'] = go.Figure(go.Funnel(y=funnel_data['stage'], x=funnel_data['count'], textinfo="value+percent previous"))
    plots['funnel_chart'].update_layout(title_text="Lead Conversion Funnel")
    
    return {"summary": funnel_data, "plots": plots}

def field_sales_visit_effectiveness_analysis(df):
    expected = ['visit_id', 'sales_rep_id', 'visit_duration', 'outcome']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['visit_duration'] = pd.to_numeric(df['visit_duration'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['sales_rep_id', 'outcome']).agg(
        num_visits=('visit_id', 'nunique'),
        avg_duration=('visit_duration', 'mean')
    ).reset_index()
    
    # Calculate success rate (assuming 'Success' is a possible outcome)
    rep_outcomes = df.groupby('sales_rep_id')['outcome'].value_counts(normalize=True).unstack(fill_value=0)
    
    plots = {}
    plots['visits_by_rep_outcome'] = px.bar(summary, x='sales_rep_id', y='num_visits', color='outcome', title="Number of Visits by Sales Rep and Outcome")
    if 'Success' in rep_outcomes.columns:
        rep_outcomes.reset_index(inplace=True)
        plots['success_rate_by_rep'] = px.bar(rep_outcomes, x='sales_rep_id', y='Success', title="Success Rate by Sales Rep")
    
    return {"summary": summary, "plots": plots}

def sales_key_performance_indicator_kpi_trend_analysis(df):
    expected = ['date', 'kpi_name', 'kpi_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['kpi_value'] = pd.to_numeric(df['kpi_value'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['date', 'kpi_name'])['kpi_value'].sum().reset_index()
    
    plots = {}
    plots['kpi_trend_lines'] = px.line(summary, x='date', y='kpi_value', color='kpi_name', title="KPI Trends Over Time")
    
    return {"summary": summary, "plots": plots}

def sales_refund_and_reason_code_analysis(df):
    expected = ['refund_id', 'refund_amount', 'refund_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['refund_amount'] = pd.to_numeric(df['refund_amount'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('refund_reason').agg(
        num_refunds=('refund_id', 'nunique'),
        total_refund_amount=('refund_amount', 'sum')
    ).reset_index()
    
    plots = {}
    plots['refunds_by_reason_count'] = px.bar(summary, x='refund_reason', y='num_refunds', title="Number of Refunds by Reason")
    plots['refunds_by_reason_amount'] = px.bar(summary, x='refund_reason', y='total_refund_amount', title="Total Refund Amount by Reason")
    
    return {"summary": summary, "plots": plots}

def lead_nurturing_campaign_effectiveness_analysis(df):
    expected = ['campaign_id', 'leads_generated', 'qualified_leads', 'converted_leads']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis"),
        }
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['leads_generated', 'qualified_leads', 'converted_leads']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df_long = df.melt(id_vars='campaign_id', value_vars=['leads_generated', 'qualified_leads', 'converted_leads'],
                      var_name='metric', value_name='count')
    
    plots = {}
    plots['campaign_performance'] = px.bar(df_long, x='campaign_id', y='count', color='metric', barmode='group', title="Campaign Performance (Leads, Qualified, Converted)")
    
    return {"df_long": df_long, "plots": plots}



def main_backend(file, encoding='utf-8'):
    df = load_data(file, encoding)
    if df is None:
        return {"error": "Failed to load data"}

    specific_sales_function_mapping = {
        "Sales Order Fulfillment and Status Analysis": sales_order_fulfillment_and_status_analysis,
        "Sales Invoice and Payment Reconciliation Analysis": sales_invoice_and_payment_reconciliation_analysis,
        "Sales Transaction and Profit Margin Analysis": sales_transaction_and_profit_margin_analysis,
        "Sales Representative Performance and Revenue Analysis": sales_representative_performance_and_revenue_analysis,
        "Sales Channel and Customer Segment Performance Analysis": sales_channel_and_customer_segment_performance_analysis,
        "Sales Opportunity and Pipeline Analysis": sales_opportunity_and_pipeline_analysis,
        "Sales Quote Conversion and Pricing Analysis": sales_quote_conversion_and_pricing_analysis,
        "Sales Return and Refund Analysis": sales_return_and_refund_analysis,
        "Sales Lead and Opportunity Conversion Analysis": sales_lead_and_opportunity_conversion_analysis,
        "Customer Payment and Reconciliation Analysis": customer_payment_and_reconciliation_analysis,
        "Lead Management and Conversion Funnel Analysis": lead_management_and_conversion_funnel_analysis,
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
        "Field Sales Visit Effectiveness Analysis": field_sales_visit_effectiveness_analysis,
        "Sales Key Performance Indicator (KPI) Trend Analysis": sales_key_performance_indicator_kpi_trend_analysis,
        "Sales Refund and Reason Code Analysis": sales_refund_and_reason_code_analysis,
        "Lead Nurturing Campaign Effectiveness Analysis": lead_nurturing_campaign_effectiveness_analysis,
    }

    
    category = None  # e.g., "General Sales Analysis"
    analysis = None  # e.g., "Sales Performance"
    specific_analysis_name = None  # e.g., "Sales Order Fulfillment and Status Analysis"

    result = None

    if category == "General Sales Analysis":
        if analysis == "Sales Performance":
            result = sales_performance_analysis(df)
        elif analysis == "Customer Analysis":
            result = customer_analysis(df)
        elif analysis == "Product Analysis":
            result = product_analysis(df)
        elif analysis == "Time Series Analysis":
            result = time_series_analysis(df)
        elif analysis == "Regional Analysis":
            result = regional_analysis(df)
        elif analysis == "Sales Channel":
            result = sales_channel_analysis(df)
        elif analysis == "Campaign Analysis":
            result = campaign_analysis(df)
        elif analysis == "Sales Forecasting":
            result = sales_forecasting(df)
        elif analysis == "Profit Analysis":
            result = profit_analysis(df)
        else:
            result = show_general_insights(df)

    elif category == "Specific Sales Analysis" and specific_analysis_name:
        func = specific_sales_function_mapping.get(specific_analysis_name)
        if func:
            try:
                result = func(df)
            except Exception as e:
                result = {"error": str(e), "message": f"Error running analysis '{specific_analysis_name}'"}
        else:
            result = {"error": f"Function not found for analysis '{specific_analysis_name}'"}

    else:
        result = show_general_insights(df)

    return result
