import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "sales_summary",
    "top_products",
    "customer_analysis",
    "revenue_trends",
    "marketing_analysis",
    "regional_channel_analysis",
    "conversion_analysis",
    "mobile_phone_feature_and_customer_rating_analysis",
    "customer_geodemographic_segmentation_analysis",
    "clothing_review_sentiment_and_recommendation_analysis",
    "customer_purchase_behavior_and_profile_analysis",
    "customer_lifetime_value_prediction_analysis",
    "ecommerce_customer_segmentation_and_behavior_analysis",
    "sales_revenue_analysis_by_product_variant",
    "product_sales_performance_analysis_by_category",
    "ecommerce_customer_churn_prediction_analysis",
    "product_inventory_and_pricing_strategy_analysis",
    "order_fulfillment_and_shipping_analysis",
    "customer_product_review_analysis",
    "customer_search_behavior_and_clickstream_analysis",
    "shopping_cart_abandonment_analysis",
    "customer_wishlist_analysis",
    "shipping_and_delivery_performance_analysis",
    "payment_transaction_status_analysis",
    "user_preference_and_localization_analysis",
    "product_specification_and_attribute_analysis",
    "user_event_tracking_and_funnel_analysis",
    "product_return_and_refund_analysis",
    "inventory_management_and_stock_level_analysis",
    "supplier_performance_and_lead_time_analysis",
    "promotional_campaign_effectiveness_analysis",
    "coupon_usage_and_discount_strategy_analysis",
    "customer_feedback_and_support_ticket_analysis",
    "subscription_service_management_and_churn_analysis",
    "user_device_and_platform_usage_analysis",
    "product_media_asset_management_analysis",
    "user_session_and_onsite_behavior_analysis",
    "customer_service_ticket_resolution_analysis",
    "affiliate_marketing_performance_analysis",
    "digital_advertising_campaign_performance_analysis",
    "gift_card_issuance_and_redemption_analysis",
    "customer_survey_response_analysis",
    "user_notification_engagement_analysis",
    "product_faq_analysis",
    "product_catalog_management_analysis",
    "special_offer_performance_analysis",
    "user_login_and_session_duration_analysis",
    "product_price_change_history_analysis",
    "order_packaging_and_sustainability_analysis",
    "vendor_performance_and_contract_management_analysis",
    "product_sustainability_and_eco_certification_analysis",
    "customer_loyalty_and_rewards_program_analysis",
    "product_stock_status_and_availability_analysis",
    "beacon_based_location_and_interaction_analysis",
    "augmented_reality_feature_engagement_analysis",
    "mobile_app_feature_usage_analysis",
]
def show_missing_columns_warning(missing_cols, matched_cols=None):
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
    return df.rename(columns={v: k for k, v in matched.items() if v is not None})

def get_key_metrics(df):
    return {
        "total_orders": df['order_id'].nunique() if 'order_id' in df.columns else len(df),
        "total_revenue": df['revenue'].sum() if 'revenue' in df.columns else 0,
        "total_customers": df['customer_id'].nunique() if 'customer_id' in df.columns else 0,
        "total_products": df['product_id'].nunique() if 'product_id' in df.columns else 0
    }
def sales_summary(df):
    expected = ['order_id', 'order_date', 'customer_id', 'product_id', 'quantity', 'price', 'revenue', 'channel', 'country']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"warning": f"Missing columns for Sales Summary: {missing}",
                "metrics": get_key_metrics(df)}
    df = safe_rename(df, matched)
    if not pd.api.types.is_datetime64_any_dtype(df['order_date']):
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['revenue'] = df['revenue'].fillna(df['quantity'] * df['price'])
    df['month'] = df['order_date'].dt.to_period('M').astype(str)
    monthly = df.groupby('month')['revenue'].sum().reset_index()
    fig = px.line(monthly, x='month', y='revenue', title='Monthly Revenue')
    return {"metrics": get_key_metrics(df), "plots": {"monthly_revenue": fig}}

def top_products(df):
    expected = ['product_id', 'product_name', 'quantity', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {"warning": f"Missing columns for Top Products: {missing}",
                "metrics": get_key_metrics(df)}
    df = safe_rename(df, matched)
    prod_rev = df.groupby(['product_id', 'product_name']).agg({'quantity':'sum','revenue':'sum'}).reset_index()
    top_by_rev = prod_rev.sort_values('revenue', ascending=False).head(20)
    fig_rev = px.bar(top_by_rev, x='product_name', y='revenue', title='Top Products by Revenue')
    return {"metrics": get_key_metrics(df), "plots": {"top_revenue_products": fig_rev}}
def show_general_insights(df, analysis_name):
    return f"General insights for {analysis_name}: Dataset has {len(df)} rows and {len(df.columns)} columns."

# --- Sales Analysis Functions ---

def sales_order_fulfillment_and_status_analysis(df):
    expected = ['order_id', 'order_date', 'delivery_date', 'order_status', 'customer_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Order Fulfillment and Status Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    df.dropna(subset=['order_id', 'order_status'], inplace=True)

    # Order status distribution
    order_status_counts = df['order_status'].value_counts(normalize=True).reset_index()
    order_status_counts.columns = ['order_status', 'proportion']

    # Average fulfillment time (assuming 'Completed' status has both dates)
    df_completed = df[df['order_status'].astype(str).str.lower() == 'completed'].copy()
    if not df_completed.empty:
        df_completed['fulfillment_time_days'] = (df_completed['delivery_date'] - df_completed['order_date']).dt.days
        avg_fulfillment_time = df_completed['fulfillment_time_days'].mean()
        fig_fulfillment_time_dist = px.histogram(df_completed, x='fulfillment_time_days', nbins=50, title='Distribution of Order Fulfillment Times (Days)')
    else:
        avg_fulfillment_time = 'N/A'
        fig_fulfillment_time_dist = go.Figure().add_annotation(text="No completed orders or date data for fulfillment time.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_order_status_pie = px.pie(order_status_counts, names='order_status', values='proportion', title='Order Status Distribution')

    plots = {
        'order_status_distribution': fig_order_status_pie,
        'fulfillment_time_distribution': fig_fulfillment_time_dist
    }

    metrics = {
        "total_orders": df['order_id'].nunique(),
        "avg_fulfillment_time_days": avg_fulfillment_time
    }

    return {"metrics": metrics, "plots": plots}

def sales_invoice_and_payment_reconciliation_analysis(df):
    expected = ['invoice_id', 'order_id', 'invoice_amount', 'payment_received_amount', 'payment_status', 'invoice_date', 'payment_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Invoice and Payment Reconciliation Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['invoice_id', 'invoice_amount', 'payment_status'], inplace=True)

    df['outstanding_amount'] = df['invoice_amount'] - df.get('payment_received_amount', 0)
    df['reconciliation_status'] = df.apply(lambda x: 'Fully Paid' if x['outstanding_amount'] <= 0.01 else 'Outstanding' if x['outstanding_amount'] > 0 else 'Overpaid', axis=1) # Handle floating point

    # Payment status distribution
    payment_status_counts = df['payment_status'].value_counts(normalize=True).reset_index()
    payment_status_counts.columns = ['payment_status', 'proportion']

    # Total outstanding amount by reconciliation status
    outstanding_by_status = df.groupby('reconciliation_status')['outstanding_amount'].sum().reset_index()

    fig_payment_status = px.pie(payment_status_counts, names='payment_status', values='proportion', title='Payment Status Distribution')
    fig_outstanding_amount = px.bar(outstanding_by_status, x='reconciliation_status', y='outstanding_amount', title='Total Outstanding Amount by Reconciliation Status')

    plots = {
        'payment_status_distribution': fig_payment_status,
        'outstanding_amount_by_reconciliation_status': fig_outstanding_amount
    }

    metrics = {
        "total_invoices": df['invoice_id'].nunique(),
        "total_billed_amount": df['invoice_amount'].sum(),
        "total_outstanding_amount": df['outstanding_amount'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def sales_transaction_and_profit_margin_analysis(df):
    expected = ['transaction_id', 'sale_amount', 'cost_of_goods_sold', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Transaction and Profit Margin Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['transaction_id', 'sale_amount', 'cost_of_goods_sold'], inplace=True)

    df['gross_profit'] = df['sale_amount'] - df['cost_of_goods_sold']
    df['profit_margin_percent'] = (df['gross_profit'] / df['sale_amount']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero for profit margin

    # Distribution of profit margins
    fig_profit_margin_dist = px.histogram(df['profit_margin_percent'].dropna(), nbins=50, title='Distribution of Profit Margins (%)')

    # Top 10 products by gross profit
    if 'product_id' in df.columns:
        top_profit_products = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()
        fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')
    else:
        fig_top_profit_products = go.Figure().add_annotation(text="Product ID data not available for top products by profit.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'profit_margin_distribution': fig_profit_margin_dist,
        'top_products_by_gross_profit': fig_top_profit_products
    }

    metrics = {
        "total_sales_amount": df['sale_amount'].sum(),
        "total_gross_profit": df['gross_profit'].sum(),
        "overall_profit_margin_percent": (df['gross_profit'].sum() / df['sale_amount'].sum()) * 100 if df['sale_amount'].sum() > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}

def sales_representative_performance_and_revenue_analysis(df):
    expected = ['sales_representative_id', 'revenue', 'number_of_deals', 'customer_id', 'region']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Representative Performance and Revenue Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_representative_id', 'revenue'], inplace=True)

    # Top 10 sales representatives by revenue
    revenue_by_rep = df.groupby('sales_representative_id')['revenue'].sum().nlargest(10).reset_index()

    # Average revenue per deal by representative (if 'number_of_deals' is suitable as deal count per record)
    # A more robust approach might be to group by deal_id first if available.
    if 'number_of_deals' in df.columns:
        df['avg_revenue_per_deal'] = df['revenue'] / df['number_of_deals']
        avg_revenue_per_deal_by_rep = df.groupby('sales_representative_id')['avg_revenue_per_deal'].mean().nlargest(10).reset_index()
        fig_avg_revenue_per_deal = px.bar(avg_revenue_per_deal_by_rep, x='sales_representative_id', y='avg_revenue_per_deal', title='Top 10 Sales Reps by Average Revenue Per Deal')
    else:
        fig_avg_revenue_per_deal = go.Figure().add_annotation(text="Number of deals data not available for average revenue per deal.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_revenue_by_rep = px.bar(revenue_by_rep, x='sales_representative_id', y='revenue', title='Top 10 Sales Representatives by Total Revenue')

    plots = {
        'revenue_by_sales_representative': fig_revenue_by_rep,
        'average_revenue_per_deal_by_representative': fig_avg_revenue_per_deal
    }

    metrics = {
        "total_revenue_overall": df['revenue'].sum(),
        "num_unique_sales_reps": df['sales_representative_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_channel_and_customer_segment_performance_analysis(df):
    expected = ['sales_channel', 'customer_segment', 'revenue', 'transaction_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Channel and Customer Segment Performance Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'customer_segment', 'revenue'], inplace=True)

    # Total revenue by sales channel
    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()

    # Revenue by customer segment for each channel (pivot table visualization)
    revenue_channel_segment = df.groupby(['sales_channel', 'customer_segment'])['revenue'].sum().unstack(fill_value=0)
    fig_revenue_channel_segment = px.bar(revenue_channel_segment, x=revenue_channel_segment.index, y=revenue_channel_segment.columns,
                                          title='Revenue by Customer Segment per Sales Channel', barmode='group')

    fig_revenue_by_channel_pie = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title='Revenue Distribution by Sales Channel')

    plots = {
        'revenue_distribution_by_sales_channel': fig_revenue_by_channel_pie,
        'revenue_by_customer_segment_per_channel': fig_revenue_channel_segment
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "num_unique_sales_channels": df['sales_channel'].nunique(),
        "num_unique_customer_segments": df['customer_segment'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_opportunity_and_pipeline_analysis(df):
    expected = ['opportunity_id', 'stage', 'expected_revenue', 'close_date', 'sales_representative_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Opportunity and Pipeline Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['close_date'] = pd.to_datetime(df['close_date'], errors='coerce')
    df.dropna(subset=['opportunity_id', 'stage', 'expected_revenue'], inplace=True)

    # Opportunity count by stage
    opportunity_by_stage = df['stage'].value_counts().reset_index()
    opportunity_by_stage.columns = ['stage', 'count']

    # Total expected revenue by stage
    revenue_by_stage = df.groupby('stage')['expected_revenue'].sum().reset_index()
    fig_revenue_by_stage = px.bar(revenue_by_stage, x='stage', y='expected_revenue', title='Total Expected Revenue by Opportunity Stage')

    fig_opportunity_by_stage = px.pie(opportunity_by_stage, names='stage', values='count', title='Opportunity Count by Stage')

    plots = {
        'opportunity_count_by_stage': fig_opportunity_by_stage,
        'expected_revenue_by_stage': fig_revenue_by_stage
    }

    metrics = {
        "total_opportunities": df['opportunity_id'].nunique(),
        "total_expected_pipeline_revenue": df['expected_revenue'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def sales_quote_conversion_and_pricing_analysis(df):
    expected = ['quote_id', 'customer_id', 'quoted_price', 'conversion_status', 'conversion_date', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Quote Conversion and Pricing Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['quote_id', 'quoted_price', 'conversion_status'], inplace=True)

    # Conversion rate
    converted_quotes = df[df['conversion_status'].astype(str).str.lower() == 'converted']
    conversion_rate = (len(converted_quotes) / len(df)) * 100 if len(df) > 0 else 0

    # Distribution of quoted prices for converted vs. unconverted quotes
    fig_quoted_price_dist = px.histogram(df, x='quoted_price', color='conversion_status', barmode='overlay',
                                         title='Distribution of Quoted Prices by Conversion Status')

    # Average quoted price by conversion status
    avg_quoted_price_by_status = df.groupby('conversion_status')['quoted_price'].mean().reset_index()
    fig_avg_quoted_price = px.bar(avg_quoted_price_by_status, x='conversion_status', y='quoted_price', title='Average Quoted Price by Conversion Status')

    plots = {
        'quoted_price_distribution_by_conversion_status': fig_quoted_price_dist,
        'average_quoted_price_by_conversion_status': fig_avg_quoted_price
    }

    metrics = {
        "total_quotes": len(df),
        "total_converted_quotes": len(converted_quotes),
        "conversion_rate_percent": conversion_rate
    }

    return {"metrics": metrics, "plots": plots}

def sales_return_and_refund_analysis(df):
    expected = ['return_id', 'order_id', 'refund_amount', 'return_reason', 'return_date', 'sale_amount']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Return and Refund Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
    df.dropna(subset=['return_id', 'refund_amount'], inplace=True)

    # Total refund amount
    total_refund_amount = df['refund_amount'].sum()

    # Top 10 return reasons
    if 'return_reason' in df.columns:
        return_reasons_counts = df['return_reason'].value_counts().nlargest(10).reset_index()
        return_reasons_counts.columns = ['reason', 'count']
        fig_return_reasons = px.bar(return_reasons_counts, x='reason', y='count', title='Top 10 Sales Return Reasons')
    else:
        fig_return_reasons = go.Figure().add_annotation(text="Return reason data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Monthly refund trend
    monthly_refunds = df.groupby(df['return_date'].dt.to_period('M').dt.start_time)['refund_amount'].sum().reset_index()
    monthly_refunds.columns = ['month_year', 'refund_amount']
    monthly_refunds = monthly_refunds.sort_values('month_year')

    fig_monthly_refunds = px.line(monthly_refunds, x='month_year', y='refund_amount', title='Monthly Sales Refund Trend')

    plots = {
        'monthly_refund_trend': fig_monthly_refunds,
        'top_return_reasons': fig_return_reasons
    }

    metrics = {
        "total_returns": len(df),
        "total_refund_amount": total_refund_amount
    }

    return {"metrics": metrics, "plots": plots}

def sales_lead_and_opportunity_conversion_analysis(df):
    expected = ['lead_id', 'opportunity_id', 'conversion_status', 'lead_source', 'deal_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Lead and Opportunity Conversion Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['lead_id', 'conversion_status'], inplace=True)

    # Overall conversion rate (lead to opportunity/deal)
    total_leads = len(df)
    converted_leads = df[df['conversion_status'].astype(str).str.lower() == 'converted']
    conversion_rate = (len(converted_leads) / total_leads) * 100 if total_leads > 0 else 0

    # Conversion rate by lead source
    if 'lead_source' in df.columns:
        conversion_by_source = df.groupby('lead_source')['conversion_status'].apply(
            lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
        ).reset_index(name='conversion_rate_percent')
        fig_conversion_by_source = px.bar(conversion_by_source, x='lead_source', y='conversion_rate_percent', title='Lead Conversion Rate by Lead Source (%)')
    else:
        fig_conversion_by_source = go.Figure().add_annotation(text="Lead source data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of conversion statuses
    conversion_status_counts = df['conversion_status'].value_counts(normalize=True).reset_index()
    conversion_status_counts.columns = ['status', 'proportion']

    fig_conversion_status_pie = px.pie(conversion_status_counts, names='status', values='proportion', title='Lead Conversion Status Distribution')

    plots = {
        'lead_conversion_status_distribution': fig_conversion_status_pie,
        'lead_conversion_rate_by_source': fig_conversion_by_source
    }

    metrics = {
        "total_leads": total_leads,
        "total_converted_leads": len(converted_leads),
        "overall_conversion_rate_percent": conversion_rate
    }

    return {"metrics": metrics, "plots": plots}

def customer_payment_and_reconciliation_analysis(df):
    expected = ['customer_id', 'payment_id', 'payment_amount', 'invoice_id', 'payment_status', 'payment_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Payment and Reconciliation Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
    df.dropna(subset=['payment_id', 'payment_amount', 'payment_status'], inplace=True)

    # Total payment amount received
    total_payment_received = df['payment_amount'].sum()

    # Payment status distribution
    payment_status_dist = df['payment_status'].value_counts(normalize=True).reset_index()
    payment_status_dist.columns = ['status', 'proportion']

    # Monthly payment received trend
    monthly_payments = df.groupby(df['payment_date'].dt.to_period('M').dt.start_time)['payment_amount'].sum().reset_index()
    monthly_payments.columns = ['month_year', 'payment_amount']
    monthly_payments = monthly_payments.sort_values('month_year')

    fig_payment_status_pie = px.pie(payment_status_dist, names='status', values='proportion', title='Customer Payment Status Distribution')
    fig_monthly_payments = px.line(monthly_payments, x='month_year', y='payment_amount', title='Monthly Customer Payments Received Trend')

    plots = {
        'payment_status_distribution': fig_payment_status_pie,
        'monthly_payments_trend': fig_monthly_payments
    }

    metrics = {
        "total_payments_received": total_payment_received,
        "num_unique_customers_paying": df['customer_id'].nunique() if 'customer_id' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def lead_management_and_conversion_funnel_analysis(df):
    expected = ['lead_id', 'lead_status', 'lead_source', 'conversion_date', 'deal_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Lead Management and Conversion Funnel Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['lead_id', 'lead_status'], inplace=True)

    # Lead funnel stages (count of leads in each stage)
    # Assuming lead_status represents sequential stages like 'New', 'Qualified', 'Proposal', 'Converted', 'Lost'
    funnel_stages_counts = df['lead_status'].value_counts().reset_index()
    funnel_stages_counts.columns = ['stage', 'count']

    # Conversion rate by lead source
    if 'lead_source' in df.columns:
        # Assuming 'Converted' is the final successful status
        conversion_by_source = df.groupby('lead_source')['lead_status'].apply(
            lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
        ).reset_index(name='conversion_rate_percent')
        fig_conversion_by_source = px.bar(conversion_by_source, x='lead_source', y='conversion_rate_percent', title='Lead Conversion Rate by Lead Source (%)')
    else:
        fig_conversion_by_source = go.Figure().add_annotation(text="Lead source data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_funnel_stages = px.bar(funnel_stages_counts, x='stage', y='count', title='Lead Conversion Funnel Stages',
                               category_orders={"stage": ["New", "Qualified", "Proposal", "Converted", "Lost"]}) # Example order

    plots = {
        'lead_funnel_stages': fig_funnel_stages,
        'lead_conversion_rate_by_source': fig_conversion_by_source
    }

    metrics = {
        "total_leads": len(df),
        "total_converted_leads": df[df['lead_status'].astype(str).str.lower() == 'converted'].shape[0],
        "overall_conversion_rate_percent": (df[df['lead_status'].astype(str).str.lower() == 'converted'].shape[0] / len(df)) * 100 if len(df) > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}

def customer_lifetime_value_and_churn_risk_analysis(df):
    expected = ['customer_id', 'total_revenue', 'number_of_purchases', 'first_purchase_date', 'last_purchase_date', 'churn_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Lifetime Value and Churn Risk Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_id', 'total_revenue', 'number_of_purchases'], inplace=True)

    # Simplified CLV calculation (can be more complex with predictive models)
    df['avg_purchase_value'] = df['total_revenue'] / df['number_of_purchases']
    # If customer lifetime in days is available
    if 'first_purchase_date' in df.columns and 'last_purchase_date' in df.columns:
        df['first_purchase_date'] = pd.to_datetime(df['first_purchase_date'], errors='coerce')
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'], errors='coerce')
        df['customer_lifetime_days'] = (df['last_purchase_date'] - df['first_purchase_date']).dt.days
        # For CLV calculation, also need average customer lifespan (from full dataset) and purchase frequency.
        # This is just average value per customer.
        avg_clv_by_customer = df.groupby('customer_id')['total_revenue'].sum().mean()
    else:
        avg_clv_by_customer = df['total_revenue'].mean() # Fallback to average total revenue per customer

    # Churn status distribution
    if 'churn_status' in df.columns:
        churn_status_counts = df['churn_status'].value_counts(normalize=True).reset_index()
        churn_status_counts.columns = ['status', 'proportion']
        fig_churn_status = px.pie(churn_status_counts, names='status', values='proportion', title='Customer Churn Status Distribution')
    else:
        fig_churn_status = go.Figure().add_annotation(text="Churn status data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of total revenue (as proxy for CLV)
    fig_total_revenue_dist = px.histogram(df, x='total_revenue', nbins=50, title='Distribution of Customer Total Revenue (Proxy for CLV)')

    plots = {
        'customer_total_revenue_distribution': fig_total_revenue_dist,
        'customer_churn_status_distribution': fig_churn_status
    }

    metrics = {
        "total_customers": df['customer_id'].nunique(),
        "avg_clv_estimate": avg_clv_by_customer,
        "churn_rate_percent": churn_status_counts[churn_status_counts['status'].astype(str).str.lower() == 'churned']['proportion'].sum() * 100 if 'churned' in churn_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def subscription_sales_and_renewal_analysis(df):
    expected = ['subscription_id', 'customer_id', 'subscription_type', 'start_date', 'end_date', 'renewal_status', 'revenue_per_subscription']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Subscription Sales and Renewal Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['subscription_id', 'renewal_status', 'revenue_per_subscription'], inplace=True)

    # Renewal rate
    total_subscriptions = len(df)
    renewed_subscriptions = df[df['renewal_status'].astype(str).str.lower() == 'renewed']
    renewal_rate_percent = (len(renewed_subscriptions) / total_subscriptions) * 100 if total_subscriptions > 0 else 0

    # Revenue by subscription type
    revenue_by_subscription_type = df.groupby('subscription_type')['revenue_per_subscription'].sum().reset_index()

    # Renewal status distribution
    renewal_status_counts = df['renewal_status'].value_counts(normalize=True).reset_index()
    renewal_status_counts.columns = ['status', 'proportion']

    fig_renewal_status = px.pie(renewal_status_counts, names='status', values='proportion', title='Subscription Renewal Status Distribution')
    fig_revenue_by_sub_type = px.bar(revenue_by_subscription_type, x='subscription_type', y='revenue_per_subscription', title='Revenue by Subscription Type')

    plots = {
        'renewal_status_distribution': fig_renewal_status,
        'revenue_by_subscription_type': fig_revenue_by_sub_type
    }

    metrics = {
        "total_subscriptions": total_subscriptions,
        "total_revenue_from_subscriptions": df['revenue_per_subscription'].sum(),
        "renewal_rate_percent": renewal_rate_percent
    }

    return {"metrics": metrics, "plots": plots}

def sales_channel_performance_and_conversion_analysis(df):
    expected = ['sales_channel', 'revenue', 'transactions', 'leads_generated', 'converted_customers']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Channel Performance and Conversion Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

    # Revenue by sales channel
    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()

    # Conversion rate by sales channel (if leads_generated and converted_customers are present per channel)
    if 'leads_generated' in df.columns and 'converted_customers' in df.columns:
        df['conversion_rate_percent'] = (df['converted_customers'] / df['leads_generated']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        avg_conversion_rate_by_channel = df.groupby('sales_channel')['conversion_rate_percent'].mean().reset_index()
        fig_conversion_rate_channel = px.bar(avg_conversion_rate_by_channel.dropna(), x='sales_channel', y='conversion_rate_percent', title='Average Conversion Rate by Sales Channel (%)')
    else:
        fig_conversion_rate_channel = go.Figure().add_annotation(text="Leads or converted customers data missing for conversion rate.",
                                                                  xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_revenue_by_channel_bar = px.bar(revenue_by_channel, x='sales_channel', y='revenue', title='Total Revenue by Sales Channel')

    plots = {
        'revenue_by_sales_channel': fig_revenue_by_channel_bar,
        'conversion_rate_by_sales_channel': fig_conversion_rate_channel
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "num_unique_sales_channels": df['sales_channel'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def cross_sell_and_upsell_opportunity_analysis(df):
    expected = ['customer_id', 'product_id', 'total_sales_value', 'cross_sell_potential', 'upsell_potential', 'customer_segment']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Cross-Sell and Up-Sell Opportunity Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_id'], inplace=True)

    # Number of customers with cross-sell potential vs. no cross-sell potential
    if 'cross_sell_potential' in df.columns:
        cross_sell_counts = df['cross_sell_potential'].value_counts(normalize=True).reset_index()
        cross_sell_counts.columns = ['potential', 'proportion']
        fig_cross_sell_pie = px.pie(cross_sell_counts, names='potential', values='proportion', title='Customer Cross-Sell Potential Distribution')
    else:
        fig_cross_sell_pie = go.Figure().add_annotation(text="Cross-sell potential data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Up-sell revenue potential by customer segment
    if 'upsell_potential' in df.columns and 'customer_segment' in df.columns:
        upsell_potential_by_segment = df.groupby('customer_segment')['upsell_potential'].sum().reset_index()
        fig_upsell_segment = px.bar(upsell_potential_by_segment, x='customer_segment', y='upsell_potential', title='Total Up-Sell Potential by Customer Segment')
    else:
        fig_upsell_segment = go.Figure().add_annotation(text="Up-sell potential or customer segment data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'customer_cross_sell_potential': fig_cross_sell_pie,
        'upsell_potential_by_customer_segment': fig_upsell_segment
    }

    metrics = {
        "total_customers_analyzed": df['customer_id'].nunique(),
        "total_cross_sell_potential_customers": df[df['cross_sell_potential'].astype(str).str.lower() == 'yes']['customer_id'].nunique() if 'cross_sell_potential' in df.columns else 'N/A',
        "total_upsell_potential_value": df['upsell_potential'].sum() if 'upsell_potential' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def sales_territory_performance_and_quota_achievement_analysis(df):
    expected = ['territory_id', 'sales_representative_id', 'revenue', 'quota', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Territory Performance and Quota Achievement Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['territory_id', 'revenue'], inplace=True)

    # Total revenue by territory
    revenue_by_territory = df.groupby('territory_id')['revenue'].sum().reset_index()

    # Quota achievement rate by territory
    if 'quota' in df.columns:
        territory_performance = df.groupby('territory_id').agg(
            total_revenue=('revenue', 'sum'),
            total_quota=('quota', 'sum')
        ).reset_index()
        territory_performance['achievement_rate_percent'] = (territory_performance['total_revenue'] / territory_performance['total_quota']) * 100
        territory_performance.replace([np.inf, -np.inf], np.nan, inplace=True)
        fig_quota_achievement = px.bar(territory_performance.dropna(), x='territory_id', y='achievement_rate_percent', title='Sales Territory Quota Achievement Rate (%)')
    else:
        fig_quota_achievement = go.Figure().add_annotation(text="Quota data not available for achievement rate analysis.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_revenue_by_territory = px.bar(revenue_by_territory, x='territory_id', y='revenue', title='Total Revenue by Sales Territory')

    plots = {
        'revenue_by_sales_territory': fig_revenue_by_territory,
        'quota_achievement_rate_by_territory': fig_quota_achievement
    }

    metrics = {
        "total_revenue_across_territories": df['revenue'].sum(),
        "num_unique_territories": df['territory_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def product_sales_performance_and_profitability_analysis(df):
    expected = ['product_id', 'product_category', 'sales_amount', 'cost_of_goods_sold', 'quantity_sold']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Sales Performance and Profitability Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'sales_amount', 'cost_of_goods_sold'], inplace=True)

    df['gross_profit'] = df['sales_amount'] - df['cost_of_goods_sold']
    df['profit_margin_percent'] = (df['gross_profit'] / df['sales_amount']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Top 10 products by gross profit
    top_profit_products = df.groupby('product_id')['gross_profit'].sum().nlargest(10).reset_index()

    # Total sales and profit by product category
    if 'product_category' in df.columns:
        category_profitability = df.groupby('product_category').agg(
            total_sales=('sales_amount', 'sum'),
            total_gross_profit=('gross_profit', 'sum')
        ).reset_index()
        fig_category_profitability = px.bar(category_profitability, x='product_category', y=['total_sales', 'total_gross_profit'],
                                            title='Total Sales and Gross Profit by Product Category', barmode='group')
    else:
        fig_category_profitability = go.Figure().add_annotation(text="Product category data not available for category profitability.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_top_profit_products = px.bar(top_profit_products, x='product_id', y='gross_profit', title='Top 10 Products by Gross Profit')

    plots = {
        'top_products_by_gross_profit': fig_top_profit_products,
        'sales_and_profit_by_product_category': fig_category_profitability
    }

    metrics = {
        "total_sales_amount": df['sales_amount'].sum(),
        "total_gross_profit": df['gross_profit'].sum(),
        "overall_product_profit_margin_percent": (df['gross_profit'].sum() / df['sales_amount'].sum()) * 100 if df['sales_amount'].sum() > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}

def product_pricing_strategy_and_tier_analysis(df):
    expected = ['product_id', 'price_tier', 'unit_price', 'sales_volume', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Pricing Strategy and Tier Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'unit_price', 'sales_volume'], inplace=True)

    # Total revenue by price tier
    if 'price_tier' in df.columns and 'revenue' in df.columns:
        revenue_by_tier = df.groupby('price_tier')['revenue'].sum().reset_index()
        fig_revenue_by_tier = px.pie(revenue_by_tier, names='price_tier', values='revenue', title='Revenue Distribution by Price Tier')
    else:
        fig_revenue_by_tier = go.Figure().add_annotation(text="Price tier or revenue data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average unit price vs. sales volume
    if 'sales_volume' in df.columns and 'unit_price' in df.columns:
        # Aggregate to product level for scatter plot if not already per product
        product_agg = df.groupby('product_id').agg(
            avg_unit_price=('unit_price', 'mean'),
            total_sales_volume=('sales_volume', 'sum')
        ).reset_index()
        fig_price_vs_volume = px.scatter(product_agg, x='avg_unit_price', y='total_sales_volume',
                                         title='Average Unit Price vs. Total Sales Volume',
                                         hover_name='product_id')
    else:
        fig_price_vs_volume = go.Figure().add_annotation(text="Sales volume or unit price data not available for scatter plot.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'revenue_distribution_by_price_tier': fig_revenue_by_tier,
        'average_unit_price_vs_total_sales_volume': fig_price_vs_volume
    }

    metrics = {
        "total_revenue_overall": df['revenue'].sum() if 'revenue' in df.columns else 'N/A',
        "num_unique_price_tiers": df['price_tier'].nunique() if 'price_tier' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def sales_forecasting_accuracy_analysis(df):
    expected = ['period', 'actual_sales', 'forecasted_sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Forecasting Accuracy Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['period', 'actual_sales', 'forecasted_sales'], inplace=True)

    df['forecast_error'] = df['actual_sales'] - df['forecasted_sales']
    df['absolute_forecast_error'] = df['forecast_error'].abs()
    df['percentage_error'] = (df['forecast_error'] / df['actual_sales']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Mean Absolute Percentage Error (MAPE)
    mape = df['percentage_error'].abs().mean() if df['actual_sales'].sum() > 0 else 0

    # Actual vs. Forecasted Sales over time
    fig_actual_vs_forecast = px.line(df, x='period', y=['actual_sales', 'forecasted_sales'],
                                     title='Actual vs. Forecasted Sales Over Time')

    # Distribution of forecast errors
    fig_error_distribution = px.histogram(df['forecast_error'].dropna(), nbins=50, title='Distribution of Forecast Errors')

    plots = {
        'actual_vs_forecasted_sales_trend': fig_actual_vs_forecast,
        'forecast_error_distribution': fig_error_distribution
    }

    metrics = {
        "total_actual_sales": df['actual_sales'].sum(),
        "total_forecasted_sales": df['forecasted_sales'].sum(),
        "mean_absolute_percentage_error": mape
    }

    return {"metrics": metrics, "plots": plots}

def channel_promotion_performance_and_roi_analysis(df):
    expected = ['promotion_id', 'sales_channel', 'revenue', 'promotion_cost', 'start_date', 'end_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Channel Promotion Performance and ROI Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['promotion_id', 'revenue', 'promotion_cost'], inplace=True)

    df['promotion_roi'] = ((df['revenue'] - df['promotion_cost']) / df['promotion_cost']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle division by zero

    # Top 10 promotions by ROI
    top_roi_promotions = df.groupby('promotion_id')['promotion_roi'].mean().nlargest(10).reset_index()

    # Total revenue generated by sales channel during promotions
    if 'sales_channel' in df.columns:
        revenue_by_channel_promo = df.groupby('sales_channel')['revenue'].sum().reset_index()
        fig_revenue_by_channel_promo = px.bar(revenue_by_channel_promo, x='sales_channel', y='revenue', title='Total Promotional Revenue by Sales Channel')
    else:
        fig_revenue_by_channel_promo = go.Figure().add_annotation(text="Sales channel data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_top_roi_promotions = px.bar(top_roi_promotions.dropna(), x='promotion_id', y='promotion_roi', title='Top 10 Promotions by ROI (%)')

    plots = {
        'top_promotions_by_roi': fig_top_roi_promotions,
        'promotional_revenue_by_sales_channel': fig_revenue_by_channel_promo
    }

    metrics = {
        "total_promotional_revenue": df['revenue'].sum(),
        "total_promotion_cost": df['promotion_cost'].sum(),
        "avg_promotion_roi_percent": df['promotion_roi'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def customer_service_impact_on_sales_analysis(df):
    expected = ['customer_id', 'sales_amount', 'customer_service_interaction_count', 'satisfaction_score', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Customer Service Impact on Sales Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['customer_id', 'sales_amount'], inplace=True)

    # Sales amount vs. customer service interaction count
    if 'customer_service_interaction_count' in df.columns:
        avg_sales_by_interactions = df.groupby('customer_service_interaction_count')['sales_amount'].mean().reset_index()
        fig_sales_vs_interactions = px.bar(avg_sales_by_interactions, x='customer_service_interaction_count', y='sales_amount',
                                            title='Average Sales Amount by Customer Service Interaction Count')
    else:
        fig_sales_vs_interactions = go.Figure().add_annotation(text="Customer service interaction count data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Sales by customer satisfaction score
    if 'satisfaction_score' in df.columns:
        sales_by_satisfaction = df.groupby('satisfaction_score')['sales_amount'].sum().reset_index()
        fig_sales_by_satisfaction = px.bar(sales_by_satisfaction, x='satisfaction_score', y='sales_amount', title='Total Sales by Customer Satisfaction Score')
    else:
        fig_sales_by_satisfaction = go.Figure().add_annotation(text="Customer satisfaction score data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_sales_by_interactions': fig_sales_vs_interactions,
        'total_sales_by_customer_satisfaction_score': fig_sales_by_satisfaction
    }

    metrics = {
        "total_sales": df['sales_amount'].sum(),
        "num_customers_with_interactions": df['customer_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_call_outcome_and_effectiveness_analysis(df):
    expected = ['call_id', 'sales_representative_id', 'call_outcome', 'call_duration_minutes', 'deal_closed']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Call Outcome and Effectiveness Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['call_id', 'call_outcome'], inplace=True)

    # Call outcome distribution
    call_outcome_counts = df['call_outcome'].value_counts(normalize=True).reset_index()
    call_outcome_counts.columns = ['outcome', 'proportion']

    # Conversion rate from calls (if 'deal_closed' indicates successful conversion)
    if 'deal_closed' in df.columns:
        total_calls = len(df)
        converted_calls = df[df['deal_closed'].astype(bool) == True]
        call_conversion_rate = (len(converted_calls) / total_calls) * 100 if total_calls > 0 else 0

        # Average call duration for converted vs. non-converted calls
        if 'call_duration_minutes' in df.columns:
            avg_duration_by_conversion = df.groupby('deal_closed')['call_duration_minutes'].mean().reset_index()
            avg_duration_by_conversion['deal_closed'] = avg_duration_by_conversion['deal_closed'].map({True: 'Deal Closed', False: 'No Deal'})
            fig_avg_call_duration = px.bar(avg_duration_by_conversion, x='deal_closed', y='call_duration_minutes', title='Average Call Duration for Deal Closed vs. No Deal (Minutes)')
        else:
            fig_avg_call_duration = go.Figure().add_annotation(text="Call duration data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    else:
        call_conversion_rate = 'N/A'
        fig_avg_call_duration = go.Figure().add_annotation(text="Deal closed data not available for conversion analysis.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))


    fig_call_outcome_pie = px.pie(call_outcome_counts, names='outcome', values='proportion', title='Sales Call Outcome Distribution')

    plots = {
        'call_outcome_distribution': fig_call_outcome_pie,
        'average_call_duration_by_conversion': fig_avg_call_duration
    }

    metrics = {
        "total_sales_calls": len(df),
        "call_conversion_rate_percent": call_conversion_rate
    }

    return {"metrics": metrics, "plots": plots}

def market_segment_revenue_and_profitability_analysis(df):
    expected = ['market_segment', 'revenue', 'cost_of_goods_sold', 'customer_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Market Segment Revenue and Profitability Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['market_segment', 'revenue', 'cost_of_goods_sold'], inplace=True)

    df['gross_profit'] = df['revenue'] - df['cost_of_goods_sold']
    df['profit_margin_percent'] = (df['gross_profit'] / df['revenue']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Revenue and profitability by market segment
    segment_performance = df.groupby('market_segment').agg(
        total_revenue=('revenue', 'sum'),
        total_gross_profit=('gross_profit', 'sum'),
        avg_profit_margin=('profit_margin_percent', 'mean')
    ).reset_index()

    fig_segment_revenue = px.bar(segment_performance, x='market_segment', y='total_revenue', title='Total Revenue by Market Segment')
    fig_segment_profit_margin = px.bar(segment_performance.dropna(), x='market_segment', y='avg_profit_margin', title='Average Profit Margin by Market Segment (%)')

    plots = {
        'revenue_by_market_segment': fig_segment_revenue,
        'profit_margin_by_market_segment': fig_segment_profit_margin
    }

    metrics = {
        "total_revenue_overall": df['revenue'].sum(),
        "total_gross_profit_overall": df['gross_profit'].sum(),
        "num_unique_market_segments": df['market_segment'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def competitor_pricing_and_feature_analysis(df):
    expected = ['product_name', 'our_price', 'competitor_price_1', 'competitor_price_2', 'feature_advantage_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Competitor Pricing and Feature Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_name', 'our_price'], inplace=True)

    # Compare our price vs. competitor prices (for a few example products)
    if 'competitor_price_1' in df.columns and 'competitor_price_2' in df.columns:
        # Select top 5 products for comparison
        sample_products = df.head(5).copy()
        price_comparison_data = []
        for index, row in sample_products.iterrows():
            price_comparison_data.append({'Product': row['product_name'], 'Source': 'Our Price', 'Price': row['our_price']})
            price_comparison_data.append({'Product': row['product_name'], 'Source': 'Competitor 1', 'Price': row['competitor_price_1']})
            price_comparison_data.append({'Product': row['product_name'], 'Source': 'Competitor 2', 'Price': row['competitor_price_2']})
        price_comparison_df = pd.DataFrame(price_comparison_data)
        fig_price_comparison = px.bar(price_comparison_df, x='Product', y='Price', color='Source',
                                      barmode='group', title='Price Comparison for Sample Products')
    else:
        fig_price_comparison = go.Figure().add_annotation(text="Competitor pricing data not available for comparison.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of feature advantage scores
    if 'feature_advantage_score' in df.columns:
        fig_feature_advantage_dist = px.histogram(df['feature_advantage_score'].dropna(), nbins=50, title='Distribution of Feature Advantage Scores')
    else:
        fig_feature_advantage_dist = go.Figure().add_annotation(text="Feature advantage score data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'price_comparison_with_competitors': fig_price_comparison,
        'feature_advantage_score_distribution': fig_feature_advantage_dist
    }

    metrics = {
        "num_products_analyzed": len(df),
        "avg_our_price": df['our_price'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def product_bundle_sales_performance_analysis(df):
    expected = ['bundle_id', 'bundle_name', 'total_sales_revenue', 'quantity_sold_bundles', 'product_ids_in_bundle']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Bundle Sales Performance Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['bundle_id', 'total_sales_revenue', 'quantity_sold_bundles'], inplace=True)

    # Top 10 selling bundles by revenue
    top_revenue_bundles = df.groupby('bundle_name')['total_sales_revenue'].sum().nlargest(10).reset_index()

    # Average revenue per bundle sold
    df['avg_revenue_per_bundle'] = df['total_sales_revenue'] / df['quantity_sold_bundles']
    avg_revenue_per_bundle_overall = df['avg_revenue_per_bundle'].mean()

    fig_top_revenue_bundles = px.bar(top_revenue_bundles, x='bundle_name', y='total_sales_revenue', title='Top 10 Product Bundles by Total Sales Revenue')

    # Distribution of bundles sold (if quantity_sold_bundles is granular per transaction)
    fig_quantity_sold_bundles_dist = px.histogram(df['quantity_sold_bundles'], nbins=20, title='Distribution of Quantity Sold for Bundles')

    plots = {
        'top_revenue_bundles': fig_top_revenue_bundles,
        'quantity_sold_bundles_distribution': fig_quantity_sold_bundles_dist
    }

    metrics = {
        "total_bundle_revenue": df['total_sales_revenue'].sum(),
        "total_bundles_sold": df['quantity_sold_bundles'].sum(),
        "avg_revenue_per_bundle_sold": avg_revenue_per_bundle_overall
    }

    return {"metrics": metrics, "plots": plots}

def international_sales_and_currency_exchange_analysis(df):
    expected = ['transaction_id', 'country', 'sales_amount_local_currency', 'exchange_rate_to_usd', 'sales_amount_usd', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "International Sales and Currency Exchange Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['transaction_id', 'country', 'sales_amount_local_currency'], inplace=True)

    # Calculate USD sales if not directly provided, assuming exchange_rate_to_usd is available
    if 'sales_amount_usd' not in df.columns and 'exchange_rate_to_usd' in df.columns:
        df['sales_amount_usd'] = df['sales_amount_local_currency'] * df['exchange_rate_to_usd']
    elif 'sales_amount_usd' not in df.columns:
        df['sales_amount_usd'] = df['sales_amount_local_currency'] # Fallback if no exchange rate to treat as USD

    # Total sales in USD by country
    sales_usd_by_country = df.groupby('country')['sales_amount_usd'].sum().reset_index()

    # Average exchange rate trend over time (if transaction_date and exchange_rate_to_usd available)
    if 'transaction_date' in df.columns and 'exchange_rate_to_usd' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        monthly_avg_exchange_rate = df.groupby(df['transaction_date'].dt.to_period('M').dt.start_time)['exchange_rate_to_usd'].mean().reset_index()
        monthly_avg_exchange_rate.columns = ['month_year', 'avg_exchange_rate']
        monthly_avg_exchange_rate = monthly_avg_exchange_rate.sort_values('month_year')
        fig_exchange_rate_trend = px.line(monthly_avg_exchange_rate, x='month_year', y='avg_exchange_rate', title='Monthly Average Exchange Rate to USD Trend')
    else:
        fig_exchange_rate_trend = go.Figure().add_annotation(text="Transaction date or exchange rate data not available for trend.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_sales_usd_by_country = px.choropleth(sales_usd_by_country, locations='country', locationmode='country names',
                                             color='sales_amount_usd', hover_name='country',
                                             color_continuous_scale=px.colors.sequential.Plasma,
                                             title='Total Sales in USD by Country')

    plots = {
        'sales_in_usd_by_country_map': fig_sales_usd_by_country,
        'average_exchange_rate_trend': fig_exchange_rate_trend
    }

    metrics = {
        "total_international_sales_usd": df['sales_amount_usd'].sum(),
        "num_unique_countries": df['country'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_contract_and_renewal_analysis(df):
    expected = ['contract_id', 'customer_id', 'contract_value', 'start_date', 'end_date', 'renewal_status', 'sales_representative_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Contract and Renewal Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['contract_id', 'contract_value', 'renewal_status'], inplace=True)

    # Renewal status distribution
    renewal_status_counts = df['renewal_status'].value_counts(normalize=True).reset_index()
    renewal_status_counts.columns = ['status', 'proportion']

    # Total value of renewed vs. non-renewed contracts
    contract_value_by_renewal_status = df.groupby('renewal_status')['contract_value'].sum().reset_index()
    fig_contract_value_renewal = px.bar(contract_value_by_renewal_status, x='renewal_status', y='contract_value', title='Total Contract Value by Renewal Status')

    fig_renewal_status_pie = px.pie(renewal_status_counts, names='status', values='proportion', title='Contract Renewal Status Distribution')

    plots = {
        'contract_renewal_status_distribution': fig_renewal_status_pie,
        'total_contract_value_by_renewal_status': fig_contract_value_renewal
    }

    metrics = {
        "total_contracts": len(df),
        "total_contract_value_overall": df['contract_value'].sum(),
        "renewal_rate_percent": renewal_status_counts[renewal_status_counts['status'].astype(str).str.lower() == 'renewed']['proportion'].sum() * 100 if 'renewed' in renewal_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def e_commerce_sales_funnel_and_conversion_analysis(df):
    expected = ['session_id', 'product_view_count', 'add_to_cart_count', 'checkout_initiated_count', 'purchase_completed_count', 'revenue']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "E-commerce Sales Funnel and Conversion Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['session_id'], inplace=True)

    # Sum up counts for each stage across all sessions
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

    fig_sales_funnel = px.funnel(funnel_df, x='Count', y='Stage', title='E-commerce Sales Funnel')

    # Overall purchase conversion rate (purchases completed / product views or sessions)
    overall_conversion_rate = 'N/A'
    if 'purchase_completed_count' in df.columns and 'product_view_count' in df.columns and df['product_view_count'].sum() > 0:
        overall_conversion_rate = (df['purchase_completed_count'].sum() / df['product_view_count'].sum()) * 100
    elif 'purchase_completed_count' in df.columns and len(df) > 0: # Fallback to sessions if views not available
        overall_conversion_rate = (df['purchase_completed_count'].sum() / len(df)) * 100

    # Revenue distribution by conversion status (assuming conversion_status indicates purchase completion)
    if 'revenue' in df.columns and 'purchase_completed_count' in df.columns:
        # Create a proxy 'conversion_status' based on purchase_completed_count > 0
        df['is_purchased'] = (df['purchase_completed_count'] > 0).astype(str)
        revenue_by_purchase_status = df.groupby('is_purchased')['revenue'].sum().reset_index()
        revenue_by_purchase_status['is_purchased'] = revenue_by_purchase_status['is_purchased'].map({'True': 'Purchased', 'False': 'Not Purchased'})
        fig_revenue_conversion_status = px.pie(revenue_by_purchase_status, names='is_purchased', values='revenue', title='Revenue by Purchase Completion Status')
    else:
        fig_revenue_conversion_status = go.Figure().add_annotation(text="Revenue or purchase completion data not available.",
                                                                    xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'e_commerce_sales_funnel': fig_sales_funnel,
        'revenue_by_purchase_completion_status': fig_revenue_conversion_status
    }

    metrics = {
        "total_sessions": len(df),
        "overall_purchase_conversion_rate_percent": overall_conversion_rate,
        "total_e_commerce_revenue": df['revenue'].sum() if 'revenue' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def field_sales_visit_effectiveness_analysis(df):
    expected = ['visit_id', 'sales_representative_id', 'visit_date', 'outcome_status', 'deal_value_closed', 'customer_id', 'visit_duration_minutes']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Field Sales Visit Effectiveness Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['visit_id', 'outcome_status'], inplace=True)

    # Visit outcome distribution
    visit_outcome_counts = df['outcome_status'].value_counts(normalize=True).reset_index()
    visit_outcome_counts.columns = ['outcome', 'proportion']

    # Average deal value closed per visit (for successful visits)
    if 'deal_value_closed' in df.columns:
        successful_visits = df[df['outcome_status'].astype(str).str.lower() == 'deal closed'].copy()
        if not successful_visits.empty:
            avg_deal_value_per_visit = successful_visits['deal_value_closed'].mean()
            fig_deal_value_dist = px.histogram(successful_visits, x='deal_value_closed', nbins=50, title='Distribution of Deal Values Closed from Visits')
        else:
            avg_deal_value_per_visit = 'N/A'
            fig_deal_value_dist = go.Figure().add_annotation(text="No 'Deal Closed' visits found or deal value data missing.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    else:
        avg_deal_value_per_visit = 'N/A'
        fig_deal_value_dist = go.Figure().add_annotation(text="Deal value data not available for effectiveness analysis.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_visit_outcome_pie = px.pie(visit_outcome_counts, names='outcome', values='proportion', title='Field Sales Visit Outcome Distribution')

    plots = {
        'visit_outcome_distribution': fig_visit_outcome_pie,
        'deal_value_closed_distribution_from_visits': fig_deal_value_dist
    }

    metrics = {
        "total_field_visits": len(df),
        "success_rate_percent": visit_outcome_counts[visit_outcome_counts['outcome'].astype(str).str.lower() == 'deal closed']['proportion'].sum() * 100 if 'deal closed' in visit_outcome_counts['outcome'].astype(str).str.lower().values else 0,
        "avg_deal_value_per_successful_visit": avg_deal_value_per_visit
    }

    return {"metrics": metrics, "plots": plots}

def sales_key_performance_indicator_kpi_trend_analysis(df):
    expected = ['date', 'kpi_name', 'kpi_value', 'sales_representative_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Key Performance Indicator (KPI) Trend Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'kpi_name', 'kpi_value'], inplace=True)

    # Monthly trend for a specific KPI (e.g., 'Revenue' if kpi_name contains it, or the most common KPI)
    if not df['kpi_name'].empty:
        target_kpi = 'Revenue' if 'revenue' in df['kpi_name'].astype(str).str.lower().values else df['kpi_name'].mode()[0]
        kpi_trend = df[df['kpi_name'] == target_kpi].groupby(df['date'].dt.to_period('M').dt.start_time)['kpi_value'].sum().reset_index()
        kpi_trend.columns = ['month_year', 'kpi_value']
        kpi_trend = kpi_trend.sort_values('month_year')
        fig_kpi_trend = px.line(kpi_trend, x='month_year', y='kpi_value', title=f'Monthly Trend for {target_kpi} KPI')
    else:
        fig_kpi_trend = go.Figure().add_annotation(text="KPI name data not available for trend analysis.",
                                                   xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # KPI values distribution by KPI name (boxplot for comparison)
    fig_kpi_distribution = px.box(df, x='kpi_name', y='kpi_value', title='KPI Value Distribution by KPI Name')

    plots = {
        'kpi_monthly_trend': fig_kpi_trend,
        'kpi_value_distribution_by_kpi_name': fig_kpi_distribution
    }

    metrics = {
        "total_kpi_records": len(df),
        "num_unique_kpis": df['kpi_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_refund_and_reason_code_analysis(df):
    expected = ['refund_id', 'order_id', 'refund_amount', 'reason_code', 'refund_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Refund and Reason Code Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['refund_date'] = pd.to_datetime(df['refund_date'], errors='coerce')
    df.dropna(subset=['refund_id', 'refund_amount'], inplace=True)

    # Total refund amount by reason code (top 10)
    if 'reason_code' in df.columns:
        refund_amount_by_reason = df.groupby('reason_code')['refund_amount'].sum().nlargest(10).reset_index()
        fig_refund_amount_by_reason = px.bar(refund_amount_by_reason, x='reason_code', y='refund_amount', title='Total Refund Amount by Top 10 Reason Codes')
    else:
        fig_refund_amount_by_reason = go.Figure().add_annotation(text="Reason code data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Monthly total refund amount trend
    monthly_refunds = df.groupby(df['refund_date'].dt.to_period('M').dt.start_time)['refund_amount'].sum().reset_index()
    monthly_refunds.columns = ['month_year', 'refund_amount']
    monthly_refunds = monthly_refunds.sort_values('month_year')

    fig_monthly_refunds_trend = px.line(monthly_refunds, x='month_year', y='refund_amount', title='Monthly Sales Refund Amount Trend')

    plots = {
        'total_refund_amount_by_reason_code': fig_refund_amount_by_reason,
        'monthly_sales_refund_amount_trend': fig_monthly_refunds_trend
    }

    metrics = {
        "total_refunds_processed": len(df),
        "total_refund_amount_overall": df['refund_amount'].sum()
    }

    return {"metrics": metrics, "plots": plots}

def lead_nurturing_campaign_effectiveness_analysis(df):
    expected = ['campaign_id', 'lead_id', 'engagement_score', 'conversion_status', 'channel', 'cost_per_lead']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Lead Nurturing Campaign Effectiveness Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['campaign_id', 'lead_id', 'conversion_status'], inplace=True)

    # Conversion rate by campaign
    conversion_by_campaign = df.groupby('campaign_id')['conversion_status'].apply(
        lambda x: (x.astype(str).str.lower() == 'converted').sum() / len(x) * 100
    ).reset_index(name='conversion_rate_percent')

    # Average engagement score by campaign channel (if available)
    if 'engagement_score' in df.columns and 'channel' in df.columns:
        avg_engagement_by_channel = df.groupby('channel')['engagement_score'].mean().reset_index()
        fig_avg_engagement_channel = px.bar(avg_engagement_by_channel, x='channel', y='engagement_score', title='Average Engagement Score by Campaign Channel')
    else:
        fig_avg_engagement_channel = go.Figure().add_annotation(text="Engagement score or channel data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_conversion_by_campaign = px.bar(conversion_by_campaign, x='campaign_id', y='conversion_rate_percent', title='Lead Conversion Rate by Nurturing Campaign (%)')

    plots = {
        'conversion_rate_by_campaign': fig_conversion_by_campaign,
        'average_engagement_by_campaign_channel': fig_avg_engagement_channel
    }

    metrics = {
        "total_leads_in_campaigns": len(df),
        "num_unique_campaigns": df['campaign_id'].nunique(),
        "overall_conversion_rate_percent": (df[df['conversion_status'].astype(str).str.lower() == 'converted'].shape[0] / len(df)) * 100 if len(df) > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}

# --- General Sales Analysis Functions ---
# Note: These might be redundant if your specific functions cover these aspects.
# I am including them as per your original request's "general" analysis section.

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

def product_analysis(df):
    expected = ['product_id', 'product_name', 'sales_amount', 'quantity_sold', 'product_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Product Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['product_id', 'sales_amount', 'quantity_sold'], inplace=True)

    # Top 10 products by sales
    top_selling_products = df.groupby('product_id')['sales_amount'].sum().nlargest(10).reset_index()

    # Sales by product category
    if 'product_category' in df.columns:
        sales_by_category = df.groupby('product_category')['sales_amount'].sum().reset_index()
    else:
        sales_by_category = pd.DataFrame(columns=['product_category', 'sales_amount'])

    fig_top_selling_products = px.bar(top_selling_products, x='product_id', y='sales_amount', title='Top 10 Selling Products by Sales')

    plots = {
        'top_selling_products': fig_top_selling_products,
    }

    if not sales_by_category.empty:
        fig_sales_by_category = px.pie(sales_by_category, names='product_category', values='sales_amount', title='Sales Distribution by Product Category')
        plots['sales_by_category'] = fig_sales_by_category
    else:
        plots['sales_by_category_warning'] = "Product category data not available for sales distribution."

    metrics = {
        "total_sales": df['sales_amount'].sum(),
        "total_quantity_sold": df['quantity_sold'].sum(),
        "num_unique_products": df['product_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def time_series_analysis(df):
    expected = ['date', 'value'] # Generic time series, assuming 'value' is what's being measured (e.g., sales, revenue)
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Time Series Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'value'], inplace=True)
    df.sort_values('date', inplace=True)

    # Daily trend
    daily_trend = df.groupby(df['date'].dt.date)['value'].sum().reset_index()
    daily_trend.columns = ['date', 'value']

    # Monthly trend
    monthly_trend = df.groupby(df['date'].dt.to_period('M').dt.start_time)['value'].sum().reset_index()
    monthly_trend.columns = ['month_year', 'value']
    monthly_trend = monthly_trend.sort_values('month_year')

    fig_daily_trend = px.line(daily_trend, x='date', y='value', title='Daily Trend Over Time')
    fig_monthly_trend = px.line(monthly_trend, x='month_year', y='value', title='Monthly Trend Over Time')

    plots = {
        'daily_trend': fig_daily_trend,
        'monthly_trend': fig_monthly_trend
    }

    metrics = {
        "total_value": df['value'].sum(),
        "start_date": df['date'].min(),
        "end_date": df['date'].max()
    }

    return {"metrics": metrics, "plots": plots}

def regional_analysis(df):
    expected = ['region', 'sales_amount', 'customer_id', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Regional Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['region', 'sales_amount'], inplace=True)

    # Total sales by region
    sales_by_region = df.groupby('region')['sales_amount'].sum().reset_index()

    # Number of unique customers by region
    if 'customer_id' in df.columns:
        customers_by_region = df.groupby('region')['customer_id'].nunique().reset_index(name='unique_customers')
        fig_customers_by_region = px.bar(customers_by_region, x='region', y='unique_customers', title='Number of Unique Customers by Region')
    else:
        fig_customers_by_region = go.Figure().add_annotation(text="Customer ID data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_sales_by_region = px.bar(sales_by_region, x='region', y='sales_amount', title='Total Sales by Region')

    plots = {
        'sales_by_region': fig_sales_by_region,
        'customers_by_region': fig_customers_by_region
    }

    metrics = {
        "total_sales_overall": df['sales_amount'].sum(),
        "num_unique_regions": df['region'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_channel_analysis(df):
    expected = ['sales_channel', 'revenue', 'transaction_count', 'customer_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Channel Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['sales_channel', 'revenue'], inplace=True)

    # Revenue distribution by sales channel
    revenue_by_channel = df.groupby('sales_channel')['revenue'].sum().reset_index()

    # Average transaction value per channel
    if 'transaction_count' in df.columns:
        df_channel_agg = df.groupby('sales_channel').agg(
            total_revenue=('revenue', 'sum'),
            total_transactions=('transaction_count', 'sum')
        ).reset_index()
        df_channel_agg['avg_transaction_value'] = df_channel_agg['total_revenue'] / df_channel_agg['total_transactions']
        df_channel_agg.replace([np.inf, -np.inf], np.nan, inplace=True)
        fig_avg_transaction_value = px.bar(df_channel_agg.dropna(), x='sales_channel', y='avg_transaction_value', title='Average Transaction Value by Sales Channel')
    else:
        fig_avg_transaction_value = go.Figure().add_annotation(text="Transaction count data not available for average transaction value.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_revenue_by_channel_pie = px.pie(revenue_by_channel, names='sales_channel', values='revenue', title='Revenue Distribution by Sales Channel')

    plots = {
        'revenue_distribution_by_channel': fig_revenue_by_channel_pie,
        'average_transaction_value_by_channel': fig_avg_transaction_value
    }

    metrics = {
        "total_revenue_overall": df['revenue'].sum(),
        "num_unique_sales_channels": df['sales_channel'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def campaign_analysis(df):
    expected = ['campaign_id', 'revenue', 'cost', 'conversion_rate', 'leads_generated']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Campaign Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['campaign_id', 'revenue', 'cost'], inplace=True)

    df['roi'] = ((df['revenue'] - df['cost']) / df['cost']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Top 10 campaigns by ROI
    top_roi_campaigns = df.groupby('campaign_id')['roi'].mean().nlargest(10).reset_index()

    # Total revenue by campaign
    revenue_by_campaign = df.groupby('campaign_id')['revenue'].sum().reset_index()
    fig_revenue_by_campaign = px.bar(revenue_by_campaign.nlargest(10, 'revenue'), x='campaign_id', y='revenue', title='Top 10 Campaigns by Total Revenue')

    plots = {
        'top_campaigns_by_roi': px.bar(top_roi_campaigns.dropna(), x='campaign_id', y='roi', title='Top 10 Campaigns by ROI (%)'),
        'total_revenue_by_campaign': fig_revenue_by_campaign
    }

    metrics = {
        "total_campaign_revenue": df['revenue'].sum(),
        "total_campaign_cost": df['cost'].sum(),
        "avg_campaign_roi_percent": df['roi'].mean(),
        "num_unique_campaigns": df['campaign_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def sales_forecasting(df):
    expected = ['date', 'actual_sales', 'forecasted_sales']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Sales Forecasting")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date', 'actual_sales', 'forecasted_sales'], inplace=True)
    df.sort_values('date', inplace=True)

    # Actual vs. Forecasted Sales over time
    fig_actual_vs_forecast = px.line(df, x='date', y=['actual_sales', 'forecasted_sales'],
                                     title='Actual vs. Forecasted Sales Over Time')

    # Forecast error distribution
    df['forecast_error'] = df['actual_sales'] - df['forecasted_sales']
    fig_error_distribution = px.histogram(df['forecast_error'], nbins=50, title='Distribution of Sales Forecast Errors')

    plots = {
        'actual_vs_forecasted_sales_trend': fig_actual_vs_forecast,
        'sales_forecast_error_distribution': fig_error_distribution
    }

    metrics = {
        "total_actual_sales": df['actual_sales'].sum(),
        "total_forecasted_sales": df['forecasted_sales'].sum(),
        "mean_absolute_error": df['forecast_error'].abs().mean()
    }

    return {"metrics": metrics, "plots": plots}

def profit_analysis(df):
    expected = ['date', 'revenue', 'cost_of_goods_sold', 'product_id', 'sales_channel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Profit Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['revenue', 'cost_of_goods_sold'], inplace=True)

    df['gross_profit'] = df['revenue'] - df['cost_of_goods_sold']
    df['profit_margin_percent'] = (df['gross_profit'] / df['revenue']) * 100
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Total gross profit by sales channel
    if 'sales_channel' in df.columns:
        profit_by_channel = df.groupby('sales_channel')['gross_profit'].sum().reset_index()
        fig_profit_by_channel = px.bar(profit_by_channel, x='sales_channel', y='gross_profit', title='Total Gross Profit by Sales Channel')
    else:
        fig_profit_by_channel = go.Figure().add_annotation(text="Sales channel data not available for profit breakdown.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of profit margins
    fig_profit_margin_dist = px.histogram(df['profit_margin_percent'].dropna(), nbins=50, title='Distribution of Profit Margins (%)')

    plots = {
        'total_gross_profit_by_channel': fig_profit_by_channel,
        'profit_margin_distribution': fig_profit_margin_dist
    }

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "total_cost_of_goods_sold": df['cost_of_goods_sold'].sum(),
        "total_gross_profit": df['gross_profit'].sum(),
        "overall_profit_margin_percent": (df['gross_profit'].sum() / df['revenue'].sum()) * 100 if df['revenue'].sum() > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}
def revenue_trends(df):
    df = df.copy()
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    else:
        return {"error": "Missing 'order_date' column"}

    if 'revenue' not in df.columns:
        return {"error": "Missing 'revenue' column"}

    df['week'] = df['order_date'].dt.to_period('W').astype(str)
    weekly_revenue = df.groupby('week')['revenue'].sum().reset_index()

    fig_line = px.line(weekly_revenue, x='week', y='revenue', title='Weekly Revenue Trends')
    fig_bar = px.bar(weekly_revenue, x='week', y='revenue', title='Weekly Revenue')

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "average_weekly_revenue": weekly_revenue['revenue'].mean()
    }

    return {"metrics": metrics, "plots": {"line_chart": fig_line, "bar_chart": fig_bar}}


def marketing_analysis(df):
    df = df.copy()
    required_cols = ['order_id', 'order_date', 'revenue', 'utm_source']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    summary = df.groupby('utm_source').agg(
        orders=('order_id', 'nunique'),
        total_revenue=('revenue', 'sum')
    ).reset_index()
    summary['avg_order_value'] = summary['total_revenue'] / summary['orders']

    fig_bar_revenue = px.bar(summary, x='utm_source', y='total_revenue', title='Revenue by UTM Source')
    fig_bar_orders = px.bar(summary, x='utm_source', y='orders', title='Orders by UTM Source')

    metrics = {
        "total_revenue": summary['total_revenue'].sum(),
        "total_orders": summary['orders'].sum(),
        "average_order_value": summary['avg_order_value'].mean()
    }

    return {"metrics": metrics, "plots": {"revenue_by_source": fig_bar_revenue, "orders_by_source": fig_bar_orders}}


def regional_channel_analysis(df):
    df = df.copy()
    required_cols = ['revenue', 'country', 'region', 'channel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        return {"error": f"Missing columns: {missing_cols}"}

    country_revenue = df.groupby('country')['revenue'].sum().reset_index()
    region_revenue = df.groupby('region')['revenue'].sum().reset_index()
    channel_revenue = df.groupby('channel')['revenue'].sum().reset_index()

    fig_country = px.bar(country_revenue.sort_values('revenue', ascending=False).head(10),
                         x='country', y='revenue', title='Top 10 Countries by Revenue')
    fig_region = px.bar(region_revenue.sort_values('revenue', ascending=False).head(10),
                        x='region', y='revenue', title='Top 10 Regions by Revenue')
    fig_channel = px.pie(channel_revenue, names='channel', values='revenue', title='Revenue by Channel')

    metrics = {
        "total_revenue": df['revenue'].sum(),
        "unique_countries": df['country'].nunique(),
        "unique_regions": df['region'].nunique(),
        "unique_channels": df['channel'].nunique()
    }

    return {
        "metrics": metrics,
        "plots": {
            "top_countries": fig_country,
            "top_regions": fig_region,
            "channel_share": fig_channel
        }
    }
def conversion_analysis(df):
    df = df.copy()
    stages = ['impressions', 'clicks', 'add_to_cart', 'order_id']
    present_stages = [col for col in stages if col in df.columns]
    if not present_stages:
        return {"error": "No conversion funnel columns found (impressions, clicks, add_to_cart, order_id)."}

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

    return {
        "metrics": {**funnel, **conversion_rates},
        "plots": {
            "funnel_counts": fig_bar,
            "conversion_rates": fig_rate
        }
    }


analysis_function_mapping = {
    "sales_summary": sales_summary,
    "top_products": top_products,
    "customer_analysis": customer_analysis,  # You must provide these funcs
    "revenue_trends": revenue_trends,
    "marketing_analysis": marketing_analysis,
    "regional_channel_analysis": regional_channel_analysis,
    "conversion_analysis": conversion_analysis,}
     

def run_analysis(df, analysis_name):
    func = analysis_function_mapping.get(analysis_name)
    if func is None:
        return {"error": f"No analysis function found for '{analysis_name}'"}
    try:
        return func(df)
    except Exception as e:
        return {"error": str(e), "message": f"Error running analysis '{analysis_name}'"}
# Example of loading and using the backend analytics:
def load_data(file_path, encoding='utf-8'):
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path, encoding=encoding)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        return pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type")
def main_backend(file, encoding='utf-8', category=None, analysis=None, specific_analysis_name=None):
    # Load data
    df = load_data(file, encoding)
    if df is None:
        return {"error": "Failed to load data"}

    # Mapping of specific sales analyses to functions
    specific_sales_function_mapping = {
        "Sales Order Fulfillment and Status Analysis": sales_order_fulfillment_and_status_analysis,
        "Sales Invoice and Payment Reconciliation": sales_invoice_and_payment_reconciliation_analysis,
        "Sales Transaction and Profit Margin": sales_transaction_and_profit_margin_analysis,
        "Sales Representative Performance and Revenue": sales_representative_performance_and_revenue_analysis,
        "Sales Channel and Customer Segment Performance": sales_channel_and_customer_segment_performance_analysis,
        "Sales Opportunity and Pipeline": sales_opportunity_and_pipeline_analysis,
        "Sales Quote Conversion and Pricing": sales_quote_conversion_and_pricing_analysis,
        "Sales Return and Refund": sales_return_and_refund_analysis,
        "Sales Lead and Opportunity Conversion": sales_lead_and_opportunity_conversion_analysis,
        "Customer Payment and Reconciliation": customer_payment_and_reconciliation_analysis,
        "Lead Management and Conversion Funnel": lead_management_and_conversion_funnel_analysis,
        "Customer Lifetime Value and Churn Risk": customer_lifetime_value_and_churn_risk_analysis,
        "Subscription Sales and Renewal": subscription_sales_and_renewal_analysis,
        "Sales Channel Performance and Conversion": sales_channel_performance_and_conversion_analysis,
        "Cross-Sell and Upsell Opportunity": cross_sell_and_upsell_opportunity_analysis,
        "Sales Territory Performance and Quota Achievement": sales_territory_performance_and_quota_achievement_analysis,
        "Product Sales Performance and Profitability": product_sales_performance_and_profitability_analysis,
        "Product Pricing Strategy and Tier": product_pricing_strategy_and_tier_analysis,
        "Sales Forecasting Accuracy": sales_forecasting_accuracy_analysis,
        "Channel Promotion Performance and ROI": channel_promotion_performance_and_roi_analysis,
        "Customer Service Impact on Sales": customer_service_impact_on_sales_analysis,
        "Sales Call Outcome and Effectiveness": sales_call_outcome_and_effectiveness_analysis,
        "Market Segment Revenue and Profitability": market_segment_revenue_and_profitability_analysis,
        "Competitor Pricing and Feature": competitor_pricing_and_feature_analysis,
        "Product Bundle Sales Performance": product_bundle_sales_performance_analysis,
        "International Sales and Currency Exchange": international_sales_and_currency_exchange_analysis,
        "Sales Contract and Renewal": sales_contract_and_renewal_analysis,
        "E-commerce Sales Funnel and Conversion": e_commerce_sales_funnel_and_conversion_analysis,
        "Field Sales Visit Effectiveness": field_sales_visit_effectiveness_analysis,
        "Sales KPI Trend": sales_key_performance_indicator_kpi_trend_analysis,
        "Sales Refund and Reason Code": sales_refund_and_reason_code_analysis,
        "Lead Nurturing Campaign Effectiveness": lead_nurturing_campaign_effectiveness_analysis,
    }

    result = None

    # Dispatch based on the category and analysis type
    if category == "General":
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

    elif category == "Specific" and specific_analysis_name:
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


