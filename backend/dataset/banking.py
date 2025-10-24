import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process
import warnings
warnings.filterwarnings('ignore')

# This is a list of potential analysis types. It is not used directly in the code,
# but serves as a reference for what the script can do.
analysis_options = [
    "financial_statements", "profitability_analysis", "cash_flow_analysis",
    "financial_ratios", "budget_vs_actual", "investment_analysis",
    "financial_transaction_categorization_and_analysis",
    "general_ledger_journal_entry_audit_analysis",
    "accounts_receivable_and_invoice_payment_analysis",
    "accounts_payable_and_vendor_payment_analysis",
    "chart_of_accounts_and_balance_management_analysis",
    "general_ledger_reconciliation_analysis",
    "departmental_budget_vs._actual_variance_analysis",
    "employee_expense_report_and_reimbursement_analysis",
    "payroll_processing_and_compensation_analysis",
    "loan_portfolio_and_risk_management_analysis",
    "credit_card_transaction_fraud_detection_analysis",
    "investment_portfolio_performance_analysis",
    "mortgage_portfolio_and_prepayment_risk_analysis",
    "securities_trading_and_settlement_analysis",
    "foreign_exchange_(fx)_trading_analysis",
    "financial_risk_assessment_and_mitigation_analysis",
    "regulatory_compliance_and_audit_findings_analysis",
    "corporate_cash_flow_statement_analysis",
    "company_financial_position_(balance_sheet)_analysis",
    "company_financial_performance_(income_statement)_analysis",
    "corporate_tax_compliance_and_filing_analysis",
    "insurance_policy_underwriting_and_management_analysis",
    "insurance_claim_processing_and_fraud_analysis",
    "stock_dividend_payment_and_tax_analysis",
    "monthly_budget_variance_reporting_and_analysis",
    "financial_forecasting_accuracy_analysis",
    "corporate_liquidity_risk_monitoring_analysis",
    "capital_expenditure_(capex)_project_analysis",
    "corporate_debt_issuance_and_structure_analysis",
    "securities_lending_transaction_analysis",
    "treasury_operations_and_trading_analysis",
]

# ========== UTILITY FUNCTIONS ==========
def show_key_metrics(df):
    """Display key metrics about the dataset"""
    print("\n--- Key Metrics ---")
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    print(f"Total Records: {total_records}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

def show_general_insights(df, title="General Insights"):
    """Show general data visualizations"""
    print(f"\n--- {title} ---")
    show_key_metrics(df)
    
    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\nNumeric Features Analysis")
        # For a non-interactive environment, we'll pick the first numeric column
        selected_num_col = numeric_cols[0]
        print(f"Analyzing numeric feature: {selected_num_col}")

        fig1 = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
        fig1.show()

        fig2 = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
        fig2.show()
    else:
        print("[WARNING] No numeric columns found for analysis.")

    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        print("\nFeature Correlations:")
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
        fig3.show()

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Features Analysis")
        # For a non-interactive environment, we'll pick the first categorical column
        selected_cat_col = categorical_cols[0]
        print(f"Analyzing categorical feature: {selected_cat_col}")

        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        fig4 = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Distribution of {selected_cat_col}")
        fig4.show()
    else:
        print("[WARNING] No categorical columns found for analysis.")

# ========== DATA LOADING ==========
def load_data(file_path, encoding='utf-8'):
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            print("[ERROR] Failed to decode file. Try another encoding.")
            return None
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            print("[ERROR] Unsupported file format")
            return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

def fuzzy_match_column(df, target_columns):
    matched_columns = {}
    available_columns = df.columns.tolist()
    for target in target_columns:
        if target in available_columns:
            matched_columns[target] = target
            continue
        match, score = process.extractOne(target, available_columns)
        if score >= 70:
            matched_columns[target] = match
        else:
            matched_columns[target] = None
    return matched_columns

def show_missing_columns_warning(missing_cols, matched_cols=None):
    print("WARNING: The following columns are needed for this analysis but weren't found in your data:")
    for col in missing_cols:
        print(f"- {col}" + (f" (matched to: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""))
    print("Showing general data insights instead.")

# ========== ANALYSIS FUNCTIONS ==========
def credit_risk_analysis(df):
    print("General Credit Risk Analysis")
    expected = ['default', 'risk', 'risk_flag', 'loan_status', 'not_fully_paid', 'credit_score', 'fico', 'income', 'loan_amount', 'credit_amount', 'dti', 'age', 'housing', 'purpose']
    matched = fuzzy_match_column(df, expected)
    found_cols = {k: v for k, v in matched.items() if v}
    if not found_cols:
        print("WARNING: Could not find any standard credit risk columns like 'default', 'risk_flag', 'credit_score', or 'income'.")
        show_general_insights(df, "General Analysis")
        return
    risk_col = matched.get('default') or matched.get('risk') or matched.get('risk_flag') or matched.get('not_fully_paid')
    if risk_col:
        print("Default/Risk Rate Analysis")
        if df[risk_col].dtype == 'object':
            df['risk_numeric'] = df[risk_col].apply(lambda x: 1 if str(x).lower() in ['bad', 'default', 'yes', '1'] else 0)
        else:
            df['risk_numeric'] = pd.to_numeric(df[risk_col], errors='coerce')
        default_rate = df['risk_numeric'].mean() * 100
        print(f"Overall Default / Bad Risk Rate: {default_rate:.2f}%")
        score_col = matched.get('credit_score') or matched.get('fico')
        if score_col:
            df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
            fig = px.box(df, x=df['risk_numeric'].astype(str), y=score_col, title=f"Distribution of {score_col.title()} by Risk Status")
            fig.show()
        purpose_col = matched.get('purpose')
        if purpose_col:
            risk_by_purpose = df.groupby(purpose_col)['risk_numeric'].mean().mul(100).sort_values(ascending=False).reset_index()
            fig2 = px.bar(risk_by_purpose, x=purpose_col, y='risk_numeric', title="Bad Risk Rate by Loan Purpose")
            fig2.show()

def fraud_detection_analysis(df):
    print("General Fraud Detection Analysis")
    expected = ['fraud', 'class', 'risk_flag', 'amount', 'transaction_type', 'variance', 'skewness', 'entropy']
    matched = fuzzy_match_column(df, expected)
    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("WARNING: Could not find fraud-related columns like 'fraud', 'class', or 'amount'.")
        show_general_insights(df, "General Analysis")
        return
    fraud_col = matched.get('fraud') or matched.get('class') or matched.get('risk_flag')
    if fraud_col:
        print("Fraudulent Transaction Overview")
        if df[fraud_col].dtype == 'object':
            df['is_fraud'] = df[fraud_col].apply(lambda x: 1 if str(x).lower() in ['fraud', '1', 'yes'] else 0)
        else:
            df['is_fraud'] = pd.to_numeric(df[fraud_col], errors='coerce')
        fraud_rate = df['is_fraud'].mean() * 100
        print(f"Fraud Rate: {fraud_rate:.3f}%")
        fraud_dist = df['is_fraud'].value_counts().reset_index()
        fraud_dist.columns = ['is_fraud_label', 'count'] # Rename columns to avoid issues
        fraud_dist['is_fraud_label'] = fraud_dist['is_fraud_label'].map({1: 'Fraud', 0: 'Not Fraud'})
        fig = px.pie(fraud_dist, names='is_fraud_label', values='count', title="Distribution of Fraudulent Transactions")
        fig.show()
        amount_col = matched.get('amount')
        if amount_col:
            df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
            fig2 = px.box(df, x='is_fraud', y=amount_col, title="Transaction Amount by Fraud Status")
            fig2.show()

def customer_segmentation_analysis(df):
    print("General Customer Segmentation Analysis")
    expected = ['age', 'gender', 'income', 'balance', 'numofproducts', 'segment', 'geography']
    matched = fuzzy_match_column(df, expected)
    found_cols = {k: v for k, v in matched.items() if v}
    if not found_cols:
        print("WARNING: Could not find customer columns like 'age', 'income', or 'balance'.")
        show_general_insights(df, "General Analysis")
        return
    age_col, income_col, balance_col = matched.get('age'), matched.get('income'), matched.get('balance')
    if not matched.get('segment'):
        print("No 'segment' column found. Creating a simple segmentation based on available data.")
        if age_col and income_col:
            df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
            df[income_col] = pd.to_numeric(df[income_col], errors='coerce')
            age_bins = [0, 30, 50, 100]
            age_labels = ['Young', 'Adult', 'Senior']
            income_bins = df[income_col].quantile([0, 0.33, 0.66, 1]).tolist()
            income_labels = ['Low Income', 'Mid Income', 'High Income']
            df['age_group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels, right=False)
            df['income_group'] = pd.cut(df[income_col], bins=income_bins, labels=income_labels, include_lowest=True)
            df['segment'] = df['income_group'].astype(str) + " - " + df['age_group'].astype(str)
            segment_col = 'segment'
        else:
            print("WARNING: Could not create segments. Need 'age' and 'income' columns.")
            return
    else:
        segment_col = matched.get('segment')
    print("Segment Distribution")
    segment_counts = df[segment_col].value_counts().reset_index()
    segment_counts.columns = ['Segment', 'Count']
    fig1 = px.pie(segment_counts, names='Segment', values='Count', title="Customer Segment Distribution")
    fig1.show()
    if balance_col:
        df[balance_col] = pd.to_numeric(df[balance_col], errors='coerce')
        fig2 = px.box(df, x=segment_col, y=balance_col, title="Account Balance by Customer Segment")
        fig2.show()

def churn_prediction_analysis(df):
    print("General Churn Prediction Analysis")
    expected = ['churn', 'exited', 'attrition', 'credit_score', 'age', 'tenure', 'balance', 'numofproducts', 'isactivemember', 'geography']
    matched = fuzzy_match_column(df, expected)
    churn_col = matched.get('churn') or matched.get('exited') or matched.get('attrition')
    if not churn_col:
        print("WARNING: Could not find a churn indicator column like 'churn', 'exited', or 'attrition'.")
        show_general_insights(df, "General Analysis")
        return
    df['is_churn'] = pd.to_numeric(df[churn_col], errors='coerce')
    churn_rate = df['is_churn'].mean() * 100
    print(f"Overall Churn Rate: {churn_rate:.2f}%")
    products_col = matched.get('numofproducts')
    if products_col:
        df[products_col] = pd.to_numeric(df[products_col], errors='coerce')
        churn_by_products = df.groupby(products_col)['is_churn'].mean().mul(100).reset_index()
        fig1 = px.bar(churn_by_products, x=products_col, y='is_churn', title="Churn Rate by Number of Products")
        fig1.show()
    balance_col, age_col = matched.get('balance'), matched.get('age')
    if balance_col and age_col:
        df[balance_col] = pd.to_numeric(df[balance_col], errors='coerce')
        df[age_col] = pd.to_numeric(df[age_col], errors='coerce')
        fig2 = px.scatter(df, x=age_col, y=balance_col, color='is_churn', title="Churn Status by Age and Account Balance")
        fig2.show()

def liquidity_and_cash_flow_analysis(df):
    print("Liquidity & Cash Flow Analysis")
    expected = ['date', 'current_assets', 'current_liabilities', 'operating_cf', 'investing_cf', 'financing_cf', 'net_cash_flow']
    matched = fuzzy_match_column(df, expected)
    found_cols = {k: v for k, v in matched.items() if v}
    if not found_cols:
        print("WARNING: Could not find liquidity or cash flow columns.")
        show_general_insights(df, "General Analysis")
        return
    date_col = matched.get('date')
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.sort_values(date_col)
    if matched.get('current_assets') and matched.get('current_liabilities'):
        print("Liquidity Ratios")
        df['current_ratio'] = df[matched['current_assets']] / df[matched['current_liabilities']]
        latest_ratio = df['current_ratio'].iloc[-1]
        print(f"Latest Current Ratio: {latest_ratio:.2f}")
        if date_col:
            fig1 = px.line(df, x=date_col, y='current_ratio', title="Current Ratio Over Time")
            fig1.show()
    cf_cols = ['operating_cf', 'investing_cf', 'financing_cf']
    found_cf_cols = [matched[c] for c in cf_cols if matched.get(c)]
    if len(found_cf_cols) > 1:
        print("Cash Flow Components")
        df_cf = df[found_cf_cols].sum().reset_index()
        df_cf.columns = ['Cash Flow Type', 'Amount']
        fig2 = px.bar(df_cf, x='Cash Flow Type', y='Amount', title="Total Cash Flow by Component")
        fig2.show()

def transaction_trend_analysis(df):
    print("Transaction Trend Analysis")
    expected = ['transaction_date', 'date', 'amount', 'transaction_type']
    matched = fuzzy_match_column(df, expected)
    date_col = matched.get('transaction_date') or matched.get('date')
    amount_col = matched.get('amount')
    if not date_col or not amount_col:
        print("WARNING: Could not find necessary columns 'date' and 'amount' for trend analysis.")
        show_general_insights(df, "General Analysis")
        return
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    df = df.sort_values(date_col).dropna(subset=[date_col, amount_col])
    df.set_index(date_col, inplace=True)
    monthly_sum = df[amount_col].resample('M').sum().reset_index()
    monthly_count = df[amount_col].resample('M').count().reset_index()
    print("Transaction Volume and Value Over Time")
    fig1 = px.line(monthly_sum, x=date_col, y=amount_col, title="Total Transaction Value (Monthly)")
    fig1.show()
    fig2 = px.line(monthly_count, x=date_col, y=amount_col, title="Total Transaction Count (Monthly)")
    fig2.show()

def aml_analysis(df):
    print("Anti-Money Laundering (AML) Analysis")
    expected = ['customer_id', 'amount', 'date', 'source_country', 'destination_country', 'transaction_type']
    matched = fuzzy_match_column(df, expected)
    if not matched.get('amount') or not matched.get('customer_id'):
        print("WARNING: Could not find 'amount' and 'customer_id' for AML analysis.")
        show_general_insights(df, "General Analysis")
        return
    amount_col = matched['amount']
    cust_col = matched['customer_id']
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    print("High-Value Transaction Monitoring")
    threshold = 10000.0
    high_value_txns = df[df[amount_col] > threshold]
    print(f"Number of Transactions Above Threshold: {len(high_value_txns)}")
    print(high_value_txns.head())
    print("High-Frequency Transaction Monitoring")
    customer_txn_counts = df[cust_col].value_counts().reset_index()
    customer_txn_counts.columns = [cust_col, 'transaction_count']
    fig = px.histogram(customer_txn_counts, x='transaction_count', title="Distribution of Transaction Frequency per Customer")
    fig.show()

def profitability_analysis(df):
    print("Profitability Analysis")
    expected = ['date', 'product', 'revenue', 'cost', 'net_income', 'interest_income', 'interest_expense', 'roa', 'roe']
    matched = fuzzy_match_column(df, expected)
    if matched.get('revenue') and matched.get('cost'):
        print("Product Profitability")
        df['revenue'] = pd.to_numeric(df[matched['revenue']], errors='coerce')
        df['cost'] = pd.to_numeric(df[matched['cost']], errors='coerce')
        df['profit'] = df['revenue'] - df['cost']
        df['profit_margin'] = (df['profit'] / df['revenue']) * 100
        avg_margin = df['profit_margin'].mean()
        print(f"Average Profit Margin: {avg_margin:.2f}%")
        profit_by_product = df.groupby(matched.get('product', 'product'))[['revenue', 'cost', 'profit']].sum().reset_index()
        fig = px.bar(profit_by_product, x=matched.get('product', 'product'), y=['revenue', 'cost'], title="Revenue and Cost by Product")
        fig.show()
    elif matched.get('roa') and matched.get('roe'):
        print("Bank-Level Profitability Ratios")
        df['roa'] = pd.to_numeric(df[matched['roa']], errors='coerce')
        df['roe'] = pd.to_numeric(df[matched['roe']], errors='coerce')
        if matched.get('date'):
            df[matched['date']] = pd.to_datetime(df[matched['date']], errors='coerce')
            fig = px.line(df, x=matched['date'], y=['roa', 'roe'], title="ROA & ROE Over Time")
            fig.show()
        else:
            print("Average ROA:", df['roa'].mean())
            print("Average ROE:", df['roe'].mean())

def loan_portfolio_stress_testing(df):
    print("Loan Portfolio Stress Testing (Simplified)")
    expected = ['loan_amount', 'ltv', 'dti', 'fico', 'credit_score', 'default_status']
    matched = fuzzy_match_column(df, expected)
    score_col = matched.get('fico') or matched.get('credit_score')
    amount_col = matched.get('loan_amount')
    if not score_col or not amount_col:
        print("WARNING: Requires 'fico'/'credit_score' and 'loan_amount' columns for stress testing.")
        show_general_insights(df, "General Analysis")
        return
    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
    df.dropna(subset=[score_col, amount_col], inplace=True)
    print("Portfolio Distribution by Credit Score")
    fig = px.histogram(df, x=score_col, title=f"Loan Portfolio Distribution by {score_col.title()}")
    fig.show()
    print("Stress Test Simulation")
    score_threshold = 620
    default_rate_increase = 10
    low_score_portfolio = df[df[score_col] < score_threshold]
    portfolio_at_risk = low_score_portfolio[amount_col].sum()
    potential_loss = portfolio_at_risk * (default_rate_increase / 100.0)
    print(f"Portfolio Amount At Risk (Below Threshold): ${portfolio_at_risk:,.0f}")
    print(f"Simulated Potential Loss: ${potential_loss:,.0f}")

def sentiment_analysis(df):
    print("Customer Sentiment Analysis")
    expected = ['text', 'review', 'feedback', 'date', 'rating', 'sentiment']
    matched = fuzzy_match_column(df, expected)
    text_col = matched.get('text') or matched.get('review') or matched.get('feedback')
    if not text_col:
        print("WARNING: Requires a text column ('text', 'review', 'feedback') for sentiment analysis.")
        show_general_insights(df, "General Analysis")
        return
    if matched.get('sentiment'):
        print("Sentiment Distribution (from data)")
        sentiment_counts = df[matched['sentiment']].value_counts().reset_index()
        fig = px.pie(sentiment_counts, names='index', values=matched['sentiment'], title="Customer Sentiment Distribution")
        fig.show()
    else:
        print("Sentiment Analysis (Calculated)")
        try:
            from textblob import TextBlob
            print("Performing live sentiment analysis with TextBlob...")
            df['polarity'] = df[text_col].dropna().apply(lambda text: TextBlob(str(text)).sentiment.polarity)
            def get_sentiment(polarity):
                if polarity > 0.1: return 'Positive'
                elif polarity < -0.1: return 'Negative'
                else: return 'Neutral'
            df['sentiment_calculated'] = df['polarity'].apply(get_sentiment)
            avg_polarity = df['polarity'].mean()
            print(f"Average Sentiment Polarity: {avg_polarity:.3f}")
            sentiment_counts = df['sentiment_calculated'].value_counts().reset_index()
            fig = px.pie(sentiment_counts, names='index', values='sentiment_calculated', title="Calculated Customer Sentiment")
            fig.show()
        except ImportError:
            print("ERROR: This analysis requires the `textblob` library. Please install it (`pip install textblob`) and download corpora (`python -m textblob.download_corpora`).")

def bank_marketing_campaign_analysis(df):
    print("Bank Marketing Campaign Effectiveness Analysis")
    expected = ['age', 'job', 'marital', 'education', 'balance', 'loan', 'poutcome', 'y']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() in ['yes', '1', 'true'] else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df['balance'] = pd.to_numeric(df['balance'], errors='coerce')
    df.dropna(inplace=True)
    conversion_rate = df['subscribed'].mean() * 100
    avg_age_subscribed = df[df['subscribed'] == 1]['age'].mean()
    top_job_subscribed = df[df['subscribed'] == 1]['job'].mode()[0]
    print(f"Overall Conversion Rate: {conversion_rate:.2f}%")
    print(f"Avg. Age of Subscribers: {avg_age_subscribed:.1f}")
    print(f"Top Job for Subscribers: {top_job_subscribed}")
    conversion_by_poutcome = df.groupby('poutcome')['subscribed'].mean().mul(100).reset_index()
    fig1 = px.bar(conversion_by_poutcome, x='poutcome', y='subscribed', title="Conversion Rate by Previous Campaign Outcome", labels={'poutcome': 'Previous Outcome', 'subscribed': 'Conversion Rate (%)'})
    fig1.show()
    fig2 = px.box(df, x='y', y='balance', title="Account Balance by Subscription Status")
    fig2.show()

def loan_default_risk_prediction_analysis(df):
    print("Loan Default Risk Prediction Analysis")
    expected = ['income', 'age', 'loan', 'default']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['default_status'] = df['default'].map({1: 'Default', 0: 'No Default'})
    default_rate = df['default'].mean() * 100
    avg_income_default = df[df['default'] == 1]['income'].mean()
    avg_income_no_default = df[df['default'] == 0]['income'].mean()
    print(f"Overall Default Rate: {default_rate:.2f}%")
    print(f"Avg. Income (Default): ${avg_income_default:,.0f}")
    print(f"Avg. Income (No Default): ${avg_income_no_default:,.0f}")
    fig1 = px.box(df, x='default_status', y='income', color='default_status', title="Income Distribution by Default Status")
    fig1.show()
    fig2 = px.scatter(df, x='age', y='loan', color='default_status', title="Loan Amount vs. Age by Default Status", labels={'age': 'Age', 'loan': 'Loan Amount'})
    fig2.show()

def bank_institution_financial_and_structural_analysis(df):
    print("Bank Institution Financial and Structural Analysis")
    expected = ['stname', 'name', 'asset', 'dep', 'offices', 'roa', 'roe']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['asset', 'dep', 'offices', 'roa', 'roe']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    total_assets = df['asset'].sum()
    total_deposits = df['dep'].sum()
    avg_roa = df['roa'].mean()
    print(f"Total Assets: ${total_assets:,.0f}")
    print(f"Total Deposits: ${total_deposits:,.0f}")
    print(f"Average ROA: {avg_roa:.2f}%")
    assets_by_state = df.groupby('stname')['asset'].sum().nlargest(10).reset_index()
    fig1 = px.bar(assets_by_state, x='stname', y='asset', title="Top 10 States by Total Bank Assets")
    fig1.show()
    fig2 = px.scatter(df, x='asset', y='dep', size='offices', hover_name='name', title="Assets vs. Deposits", log_x=True, log_y=True)
    fig2.show()

def bank_branch_geospatial_distribution_analysis(df):
    print("Bank Branch Geospatial Distribution Analysis")
    expected = ['name', 'address', 'latitude', 'longitude', 'ward']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['latitude', 'longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    num_branches = len(df)
    top_ward = df['ward'].mode()[0]
    print(f"Total Number of Branches: {num_branches}")
    print(f"Ward with Most Branches: {top_ward}")
    print("Cannot display map without a suitable environment.")
    branches_by_ward = df['ward'].value_counts().nlargest(15).reset_index()
    branches_by_ward.columns = ['ward', 'count']
    fig = px.bar(branches_by_ward, x='ward', y='count', title="Top 15 Wards by Number of Bank Branches")
    fig.show()

def bank_office_and_service_type_analysis(df):
    print("Bank Office and Service Type Analysis")
    expected = ['name', 'servtype', 'city', 'stalp']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(inplace=True)
    num_offices = len(df)
    top_service_type = df['servtype'].mode()[0]
    top_state = df['stalp'].mode()[0]
    print(f"Total Offices Listed: {num_offices:,}")
    print(f"Most Common Service Type: {top_service_type}")
    print(f"State with Most Offices: {top_state}")
    service_type_counts = df['servtype'].value_counts().reset_index()
    service_type_counts.columns = ['servtype', 'count']
    fig1 = px.pie(service_type_counts, names='servtype', values='count', title="Distribution of Bank Service Types")
    fig1.show()
    offices_by_state = df['stalp'].value_counts().nlargest(20).reset_index()
    offices_by_state.columns = ['state', 'count']
    fig2 = px.bar(offices_by_state, x='state', y='count', title="Top 20 States by Number of Bank Offices")
    fig2.show()

def financial_institution_geolocation_analysis(df):
    print("Financial Institution Geolocation Analysis")
    expected = ['name_of_institution', 'city', 'county', 'georeference']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    try:
        coords = df['georeference'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)')
        df['longitude'] = pd.to_numeric(coords[0], errors='coerce')
        df['latitude'] = pd.to_numeric(coords[1], errors='coerce')
    except Exception:
        print("ERROR: Could not parse latitude/longitude from 'georeference' column.")
        return
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    print(f"Total Institutions Mapped: {len(df)}")
    print("Cannot display map without a suitable environment.")
    inst_by_county = df['county'].value_counts().nlargest(20).reset_index()
    inst_by_county.columns = ['county', 'count']
    fig = px.bar(inst_by_county, x='county', y='count', title="Top 20 Counties by Number of Financial Institutions")
    fig.show()

def consumer_banking_habits_survey_analysis(df):
    print("Consumer Banking Habits Survey Analysis")
    expected = ['banking_status', 'age_group', 'income_group', 'q4', 'q12', 'q27']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    has_checking = (df['q4'].astype(str).str.lower() == '1').mean() * 100
    has_credit_card = (df['q12'].astype(str).str.lower() == '1').mean() * 100
    uses_mobile = (df['q27'].astype(str).str.lower() == '1').mean() * 100
    print(f"Has Checking Account: {has_checking:.1f}%")
    print(f"Has Credit Card: {has_credit_card:.1f}%")
    print(f"Uses Mobile Banking: {uses_mobile:.1f}%")
    banking_dist = df['banking_status'].value_counts().reset_index()
    banking_dist.columns = ['status', 'count']
    fig1 = px.pie(banking_dist, names='status', values='count', title="Distribution of Banking Status")
    fig1.show()
    mobile_by_income = df.groupby('income_group')['q27'].apply(lambda x: (x == '1').mean()).mul(100).reset_index()
    mobile_by_income.columns = ['income_group', 'rate']
    fig2 = px.bar(mobile_by_income, x='income_group', y='rate', title="Mobile Banking Adoption by Income Group")
    fig2.show()

def financial_service_provider_accessibility_analysis(df):
    print("Financial Service Provider Accessibility Analysis")
    expected = ['provider', 'borough', 'days_open', 'language_s', 'latitude', 'longitude']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    num_providers = len(df)
    top_borough = df['borough'].mode()[0]
    weekend_availability = df['days_open'].str.contains('Sat|Sun', case=False, na=False).mean() * 100
    print(f"Total Providers: {num_providers:,}")
    print(f"Borough with Most Providers: {top_borough}")
    print(f"Weekend Availability: {weekend_availability:.1f}%")
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    print("Cannot display map without a suitable environment.")
    try:
        languages = df['language_s'].str.split(',').explode().str.strip().value_counts().nlargest(10)
        fig2 = px.bar(languages, x=languages.index, y=languages.values, title="Top 10 Languages Offered")
        fig2.show()
    except Exception:
        print("WARNING: Could not analyze 'language_s' column.")

def socio_economic_analysis_of_unbanked_populations(df):
    print("Socio-Economic Analysis of Unbanked Populations")
    expected = ['sub_boro_name', 'unbanked_2013', 'underbanked_2013', 'perc_poor_2013', 'median_income_2013', 'unemployment_2013']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if 'name' not in col:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace('%', ''), errors='coerce')
    df.dropna(inplace=True)
    avg_unbanked_rate = df['unbanked_2013'].mean()
    avg_poverty_rate = df['perc_poor_2013'].mean()
    avg_median_income = df['median_income_2013'].mean()
    print(f"Average Unbanked Rate: {avg_unbanked_rate:.1f}%")
    print(f"Average Poverty Rate: {avg_poverty_rate:.1f}%")
    print(f"Average Median Income: ${avg_median_income:,.0f}")
    fig1 = px.scatter(df, x='median_income_2013', y='unbanked_2013', size='unemployment_2013', hover_name='sub_boro_name', title="Unbanked Rate vs. Median Income", trendline='ols')
    fig1.show()
    top_unbanked = df.nlargest(10, 'unbanked_2013')
    fig2 = px.bar(top_unbanked, x='sub_boro_name', y='unbanked_2013', title="Top 10 Areas by Unbanked Population Rate")
    fig2.show()

def automated_teller_machine_atm_location_analysis(df):
    print("Automated Teller Machine (ATM) Location Analysis")
    expected = ['name', 'address', 'latitude', 'longitude', 'ward']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['latitude', 'longitude'], inplace=True)
    print(f"Total ATMs Mapped: {len(df)}")
    print("Cannot display map without a suitable environment.")
    atms_by_ward = df['ward'].value_counts().nlargest(20).reset_index()
    atms_by_ward.columns = ['ward', 'count']
    fig = px.bar(atms_by_ward, x='ward', y='count', title="Top 20 Wards by Number of ATMs")
    fig.show()

def transaction_fraud_risk_flagging_analysis(df):
    print("Transaction Fraud Risk Flagging Analysis")
    expected = ['id', 'risk_flag']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['risk_flag'] = pd.to_numeric(df['risk_flag'], errors='coerce')
    df.dropna(inplace=True)
    total_transactions = len(df)
    risky_transactions = df['risk_flag'].sum()
    risk_rate = (risky_transactions / total_transactions) * 100
    print(f"Total Transactions: {total_transactions:,}")
    print(f"Flagged as Risky: {risky_transactions:,}")
    print(f"Risk Rate: {risk_rate:.2f}%")
    risk_dist = df['risk_flag'].value_counts().reset_index()
    risk_dist.columns = ['risk_label', 'count']
    risk_dist['risk_label'] = risk_dist['risk_label'].map({1: 'Risky', 0: 'Not Risky'})
    fig = px.pie(risk_dist, names='risk_label', values='count', title="Distribution of Risky Transactions")
    fig.show()

def customer_churn_prediction_analysis(df):
    print("Customer Churn Prediction Analysis")
    expected = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'isactivemember', 'estimatedsalary', 'exited']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'exited']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    churn_rate = df['exited'].mean() * 100
    avg_credit_score_churn = df[df['exited'] == 1]['creditscore'].mean()
    avg_balance_churn = df[df['exited'] == 1]['balance'].mean()
    print(f"Overall Churn Rate: {churn_rate:.2f}%")
    print(f"Avg. Credit Score (Churned): {avg_credit_score_churn:.0f}")
    print(f"Avg. Balance (Churned): ${avg_balance_churn:,.0f}")
    fig1 = px.histogram(df, x='creditscore', color='exited', barmode='overlay', title="Credit Score Distribution by Churn Status")
    fig1.show()
    churn_by_products = df.groupby('numofproducts')['exited'].mean().mul(100).reset_index()
    fig2 = px.bar(churn_by_products, x='numofproducts', y='exited', title="Churn Rate by Number of Products Held")
    fig2.show()

def customer_loan_risk_assessment_analysis(df):
    print("Customer Loan Risk Assessment Analysis")
    expected = ['income', 'age', 'experience', 'house_ownership', 'car_ownership', 'profession', 'current_job_yrs', 'current_house_yrs', 'risk_flag']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['income', 'age', 'experience', 'current_job_yrs', 'current_house_yrs', 'risk_flag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    high_risk_rate = df['risk_flag'].mean() * 100
    avg_income_high_risk = df[df['risk_flag'] == 1]['income'].mean()
    print(f"High Risk Rate: {high_risk_rate:.2f}%")
    print(f"Avg. Income of High-Risk Applicants: ${avg_income_high_risk:,.0f}")
    fig1 = px.box(df, x='house_ownership', y='income', color='risk_flag', title="Income Distribution by House Ownership and Risk Flag")
    fig1.show()
    risk_by_profession = df.groupby('profession')['risk_flag'].mean().mul(100).sort_values(ascending=False).reset_index()
    fig2 = px.bar(risk_by_profession, x='profession', y='risk_flag', title="High Risk Rate by Profession")
    fig2.show()

def stock_index_time_series_analysis(df):
    print("Stock Index Time Series Analysis")
    expected = ['date', 'open', 'high', 'low', 'close', 'volume']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('date').dropna()
    latest_close = df['close'].iloc[-1]
    change = latest_close - df['close'].iloc[-2]
    perc_change = (change / df['close'].iloc[-2]) * 100
    print(f"Latest Close Price: ${latest_close:,.2f}")
    print(f"Day Change: ${change:,.2f}")
    print(f"% Change: {perc_change:.2f}%")
    fig1 = px.line(df, x='date', y='close', title="Closing Price Over Time")
    fig1.show()
    fig2 = px.bar(df, x='date', y='volume', title="Trading Volume Over Time")
    fig2.show()

def global_stock_market_index_performance_analysis(df):
    print("Global Stock Market Index Performance Analysis")
    expected = ['region', 'exchange', 'index', 'currency']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    num_indices = df['index'].nunique()
    num_regions = df['region'].nunique()
    top_currency = df['currency'].mode()[0]
    print(f"Number of Indices: {num_indices}")
    print(f"Number of Regions: {num_regions}")
    print(f"Most Common Currency: {top_currency}")
    indices_by_region = df.groupby('region')['index'].count().reset_index()
    indices_by_region.columns = ['region', 'count']
    fig1 = px.bar(indices_by_region, x='region', y='count', title="Number of Stock Indices by Region")
    fig1.show()
    currency_dist = df['currency'].value_counts().reset_index()
    currency_dist.columns = ['currency', 'count']
    fig2 = px.pie(currency_dist, names='currency', values='count', title="Distribution of Index Currencies")
    fig2.show()

def stock_index_performance_and_currency_conversion_analysis(df):
    print("Stock Index Performance and Currency Conversion Analysis")
    expected = ['date', 'close', 'closeusd', 'volume']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['close', 'closeusd', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['usd_to_local_rate'] = df['closeusd'] / df['close']
    latest_rate = df['usd_to_local_rate'].iloc[-1]
    avg_rate = df['usd_to_local_rate'].mean()
    print(f"Latest Implied USD/Local Rate: {latest_rate:.4f}")
    print(f"Average Implied Rate: {avg_rate:.4f}")
    fig1 = px.line(df, x='date', y=['close', 'closeusd'], title="Closing Price in Local Currency vs. USD")
    fig1.show()
    fig2 = px.line(df, x='date', y='usd_to_local_rate', title="Implied USD to Local Currency Exchange Rate Over Time")
    fig2.show()

def customer_credit_score_factor_analysis(df):
    print("Customer Credit Score Factor Analysis")
    expected = ['age', 'gender', 'income', 'education', 'marital_status', 'number_of_children', 'home_ownership', 'credit_score']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['age', 'income', 'number_of_children', 'credit_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    avg_credit_score = df['credit_score'].mean()
    income_corr = df['income'].corr(df['credit_score'])
    age_corr = df['age'].corr(df['credit_score'])
    print(f"Average Credit Score: {avg_credit_score:.0f}")
    print(f"Income/Score Correlation: {income_corr:.2f}")
    print(f"Age/Score Correlation: {age_corr:.2f}")
    fig1 = px.scatter(df, x='income', y='credit_score', color='home_ownership', title="Credit Score vs. Income by Home Ownership")
    fig1.show()
    fig2 = px.box(df, x='education', y='credit_score', title="Credit Score Distribution by Education Level")
    fig2.show()

def loan_application_approval_prediction_analysis(df):
    print("Loan Application Approval Prediction Analysis")
    expected = ['loan_amount', 'property_value', 'income', 'credit_score', 'ltv', 'dtir1', 'status']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['loan_amount', 'property_value', 'income', 'credit_score', 'ltv', 'dtir1']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if df['status'].dtype != 'object':
        df['status'] = df['status'].map({1: 'Approved', 0: 'Denied'})
    df.dropna(inplace=True)
    approval_rate = (df['status'] == 'Approved').mean() * 100
    avg_credit_score_approved = df[df['status'] == 'Approved']['credit_score'].mean()
    avg_ltv_approved = df[df['status'] == 'Approved']['ltv'].mean()
    print(f"Overall Approval Rate: {approval_rate:.2f}%")
    print(f"Avg. Credit Score (Approved): {avg_credit_score_approved:.0f}")
    print(f"Avg. LTV (Approved): {avg_ltv_approved:.1f}%")
    fig1 = px.box(df, x='status', y='credit_score', color='status', title="Credit Score Distribution by Loan Status")
    fig1.show()
    fig2 = px.scatter(df, x='income', y='loan_amount', color='status', title="Loan Amount vs. Applicant Income by Status", labels={'income': 'Income', 'loan_amount': 'Loan Amount'})
    fig2.show()

def banknote_authentication_analysis(df):
    print("Banknote Authentication Analysis")
    expected = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['class_label'] = df['class'].map({0: 'Genuine', 1: 'Forged'})
    forgery_rate = df['class'].mean() * 100
    avg_variance_forged = df[df['class'] == 1]['variance'].mean()
    avg_entropy_genuine = df[df['class'] == 0]['entropy'].mean()
    print(f"Forgery Rate: {forgery_rate:.2f}%")
    print(f"Avg. Variance (Forged): {avg_variance_forged:.2f}")
    print(f"Avg. Entropy (Genuine): {avg_entropy_genuine:.2f}")
    fig1 = px.scatter(df, x='skewness', y='curtosis', color='class_label', title="Skewness vs. Curtosis of Wavelet Transforms")
    fig1.show()
    fig2 = px.scatter_matrix(df, dimensions=['variance', 'skewness', 'curtosis', 'entropy'], color='class_label', title="Scatter Matrix of Banknote Features")
    fig2.show()

def customer_response_prediction_for_marketing_campaign(df):
    print("Customer Response Prediction for Marketing Campaign")
    expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'poutcome', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)
    conversion_rate = df['subscribed'].mean() * 100
    print(f"Overall Subscription Rate: {conversion_rate:.2f}%")
    conversion_by_job = df.groupby('job')['subscribed'].mean().mul(100).sort_values().reset_index()
    fig1 = px.bar(conversion_by_job, x='subscribed', y='job', orientation='h', title="Subscription Rate by Job Title")
    fig1.show()
    conversion_by_marital = df.groupby('marital')['subscribed'].mean().mul(100).reset_index()
    fig2 = px.pie(conversion_by_marital, names='marital', values='subscribed', title="Subscription Rate by Marital Status", hole=0.4)
    fig2.show()

def international_banking_statistics_and_cross_border_claims_analysis(df):
    print("International Banking Statistics and Cross-Border Claims Analysis")
    expected = ['l_rep_cty', 'l_cp_country', '2022_q1', '2022_q2', '2022_q3', '2022_q4']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    id_vars = [col for col in df.columns if not col.startswith(('1', '2'))]
    value_vars = [col for col in df.columns if col.startswith(('1', '2'))]
    df_long = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='quarter', value_name='value')
    df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce').dropna()
    total_value = df_long['value'].sum()
    top_reporting_country = df_long.groupby('l_rep_cty')['value'].sum().idxmax()
    top_counterparty = df_long.groupby('l_cp_country')['value'].sum().idxmax()
    print(f"Total Value of Claims: ${total_value:,.0f}M")
    print(f"Top Reporting Country: {top_reporting_country}")
    print(f"Top Counterparty Country: {top_counterparty}")
    print("Cannot plot time series without converting `quarter` to datetime objects.")
    top_reporting = df_long.groupby('l_rep_cty')['value'].sum().nlargest(10).reset_index()
    top_reporting.columns = ['country', 'value']
    fig2 = px.bar(top_reporting, x='country', y='value', title="Top 10 Reporting Countries by Claim Value")
    fig2.show()

def customer_account_transaction_pattern_analysis(df):
    print("Customer Account Transaction Pattern Analysis")
    expected = ['account_id', 'transaction_type', 'amount', 'balance', 'transaction_date']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['amount', 'balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    avg_txn_amount = df['amount'].mean()
    most_freq_txn = df['transaction_type'].mode()[0]
    num_accounts = df['account_id'].nunique()
    print(f"Average Transaction Amount: ${avg_txn_amount:,.2f}")
    print(f"Most Frequent Txn Type: {most_freq_txn}")
    print(f"Number of Accounts: {num_accounts:,}")
    print("Cannot plot time series without converting a date column to datetime objects.")
    txn_type_dist = df['transaction_type'].value_counts().reset_index()
    txn_type_dist.columns = ['type', 'count']
    fig2 = px.bar(txn_type_dist, x='type', y='count', title="Distribution of Transaction Types")
    fig2.show()

def telemarketing_campaign_outcome_analysis(df):
    print("Telemarketing Campaign Outcome Analysis")
    expected = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['campaign'] = pd.to_numeric(df['campaign'], errors='coerce')
    df.dropna(inplace=True)
    conversion_rate = df['subscribed'].mean() * 100
    avg_duration_subscribed = df[df['subscribed'] == 1]['duration'].mean()
    avg_duration_not_subscribed = df[df['subscribed'] == 0]['duration'].mean()
    print(f"Conversion Rate: {conversion_rate:.2f}%")
    print(f"Avg. Call Duration (Subscribed): {avg_duration_subscribed:.0f}s")
    print(f"Avg. Call Duration (Not Subscribed): {avg_duration_not_subscribed:.0f}s")
    fig1 = px.box(df, x='subscribed', y='duration', title="Call Duration by Subscription Outcome")
    fig1.show()
    conversion_by_month = df.groupby('month')['subscribed'].mean().mul(100).reset_index()
    month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    conversion_by_month['month'] = pd.Categorical(conversion_by_month['month'], categories=month_order, ordered=True)
    conversion_by_month = conversion_by_month.sort_values('month')
    fig2 = px.bar(conversion_by_month, x='month', y='subscribed', title="Conversion Rate by Month")
    fig2.show()

def bank_term_deposit_subscription_analysis(df):
    print("Bank Term Deposit Subscription Analysis")
    expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)
    print(f"Subscription Rate: {df['subscribed'].mean()*100:.2f}%")
    rate_by_housing = df.groupby('housing')['subscribed'].mean().mul(100).reset_index()
    fig1 = px.pie(rate_by_housing, names='housing', values='subscribed', title="Subscription Rate by Housing Loan Status")
    fig1.show()
    fig2 = px.histogram(df, x='age', color='y', barmode='overlay', title="Age Distribution by Subscription Status")
    fig2.show()

def millennial_banking_preferences_survey_analysis(df):
    print("Millennial Banking Preferences Survey Analysis")
    expected = ['how_old_are_you', 'i_know_the_difference_between_a_bank_and_a_credit_union', 'what_makes_you_consider_a_financial_institution_worthwhile', 'what_are_your_top_2_preferences_when_choosing_a_financial_institution']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    knows_difference = (df['i_know_the_difference_between_a_bank_and_a_credit_union'].str.lower() == 'yes').mean() * 100
    print(f"Knows Difference Between Bank & Credit Union: {knows_difference:.1f}%")
    age_dist = df['how_old_are_you'].value_counts().reset_index()
    age_dist.columns = ['age_group', 'count']
    fig1 = px.bar(age_dist, x='age_group', y='count', title="Age Distribution of Survey Respondents")
    fig1.show()
    print("What Makes a Financial Institution Worthwhile?")
    print("Cannot generate word cloud as 'wordcloud' and 'matplotlib' are not allowed imports.")
    print("Raw Responses:", df['what_makes_you_consider_a_financial_institution_worthwhile'].dropna().to_string())

def bank_direct_marketing_success_prediction_analysis(df):
    print("Bank Direct Marketing Success Prediction Analysis")
    expected = ['age', 'job', 'marital', 'education', 'loan', 'contact', 'month', 'duration', 'campaign', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    for col in ['emp_var_rate', 'cons_price_idx', 'cons_conf_idx']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    print(f"Subscription Rate: {df['subscribed'].mean()*100:.2f}%")
    fig1 = px.box(df, x='y', y='cons_conf_idx', title="Consumer Confidence Index by Subscription Outcome")
    fig1.show()
    fig2 = px.scatter(df, x='cons_price_idx', y='emp_var_rate', color='y', title="Subscription Outcome by Economic Indicators", labels={'cons_price_idx': 'Consumer Price Index', 'emp_var_rate': 'Employment Variation Rate'})
    fig2.show()

def customer_segmentation_and_product_limit_analysis(df):
    print("Customer Segmentation and Product Limit Analysis")
    expected = ['age', 'city', 'product', 'limit', 'company', 'segment']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['limit'] = pd.to_numeric(df['limit'], errors='coerce')
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df.dropna(inplace=True)
    avg_limit = df['limit'].mean()
    top_segment = df['segment'].mode()[0]
    print(f"Average Product Limit: ${avg_limit:,.0f}")
    print(f"Largest Customer Segment: {top_segment}")
    fig1 = px.box(df, x='segment', y='limit', color='product', title="Product Limit by Customer Segment and Product")
    fig1.show()
    segment_dist = df['segment'].value_counts().reset_index()
    segment_dist.columns = ['segment', 'count']
    fig2 = px.pie(segment_dist, names='segment', values='count', title="Customer Segment Distribution")
    fig2.show()

def credit_risk_classification_analysis(df):
    print("Credit Risk Classification Analysis")
    expected = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'age', 'housing', 'class']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['class'].dtype == 'object':
        df['risk_label'] = df['class'].apply(lambda x: 'Bad' if x.lower() in ['bad', '2'] else 'Good')
    else:
        df['risk_label'] = df['class'].apply(lambda x: 'Bad' if x == 2 else 'Good')
    for col in ['duration', 'credit_amount', 'age']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    bad_risk_rate = (df['risk_label'] == 'Bad').mean() * 100
    avg_credit_amount_bad = df[df['risk_label'] == 'Bad']['credit_amount'].mean()
    print(f"Bad Risk Rate: {bad_risk_rate:.2f}%")
    print(f"Avg. Credit Amount (Bad Risk): €{avg_credit_amount_bad:,.0f}")
    fig1 = px.box(df, x='risk_label', y='credit_amount', color='housing', title="Credit Amount by Risk Status and Housing Type")
    fig1.show()
    risk_by_purpose = df.groupby('purpose')['class'].apply(lambda x: (x == 2).mean()).mul(100).sort_values(ascending=False).reset_index()
    risk_by_purpose.columns = ['purpose', 'bad_risk_rate']
    fig2 = px.bar(risk_by_purpose, x='purpose', y='bad_risk_rate', title="Bad Risk Rate by Loan Purpose")
    fig2.show()

def loan_application_status_prediction_analysis(df):
    print("Loan Application Status Prediction Analysis")
    expected = ['gender', 'married', 'dependents', 'education', 'self_employed', 'applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term', 'credit_history', 'property_area', 'loan_status']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.items()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['loan_status'].dtype == 'object':
        df['approved'] = df['loan_status'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
    else:
        df['approved'] = pd.to_numeric(df['loan_status'], errors='coerce')
    for col in ['applicantincome', 'coapplicantincome', 'loanamount', 'credit_history']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    approval_rate = df['approved'].mean() * 100
    approval_with_credit_hist = df[df['credit_history'] == 1.0]['approved'].mean() * 100
    approval_no_credit_hist = df[df['credit_history'] == 0.0]['approved'].mean() * 100
    print(f"Overall Approval Rate: {approval_rate:.1f}%")
    print(f"Approval Rate (with History): {approval_with_credit_hist:.1f}%")
    print(f"Approval Rate (no History): {approval_no_credit_hist:.1f}%")
    fig1 = px.histogram(df, x='applicantincome', color='loan_status', barmode='overlay', title="Applicant Income Distribution by Loan Status")
    fig1.show()
    approval_by_prop_area = df.groupby('property_area')['approved'].mean().mul(100).reset_index()
    fig2 = px.bar(approval_by_prop_area, x='property_area', y='approved', title="Approval Rate by Property Area")
    fig2.show()

def bank_customer_attrition_analysis(df):
    print("Bank Customer Attrition Analysis")
    expected = ['customerid', 'surname', 'creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary', 'exited']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'exited']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    attrition_rate = df['exited'].mean() * 100
    avg_balance_leavers = df[df['exited'] == 1]['balance'].mean()
    avg_salary_leavers = df[df['exited'] == 1]['estimatedsalary'].mean()
    print(f"Attrition Rate: {attrition_rate:.2f}%")
    print(f"Avg. Balance (Leavers): ${avg_balance_leavers:,.0f}")
    print(f"Avg. Salary (Leavers): ${avg_salary_leavers:,.0f}")
    fig1 = px.violin(df, x='geography', y='estimatedsalary', color='exited', box=True, points="all", title="Salary Distribution by Geography and Attrition Status")
    fig1.show()
    fig2 = px.density_heatmap(df, x="age", y="balance", z="exited", histfunc="avg", title="Heatmap of Attrition Rate by Age and Balance")
    fig2.show()

def loan_approval_status_prediction_analysis(df):
    print("Loan Approval Status Prediction Analysis")
    expected = ['applicant_name', 'loan_amount_usd', 'annual_income_usd', 'credit_score', 'approval_status']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['loan_amount_usd', 'annual_income_usd', 'credit_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    if df['approval_status'].dtype == 'object':
        approved_val = df['approval_status'].mode()[0]
        approval_rate = (df['approval_status'] == approved_val).mean() * 100
    else:
        approval_rate = (df['approval_status'] == 1).mean() * 100
    print(f"Overall Approval Rate: {approval_rate:.1f}%")
    fig1 = px.box(df, x='approval_status', y='credit_score', color='approval_status', title="Credit Score Distribution by Approval Status")
    fig1.show()
    fig2 = px.scatter(df, x='annual_income_usd', y='loan_amount_usd', color='approval_status', title="Loan Amount vs. Annual Income by Approval Status", log_x=True, log_y=True)
    fig2.show()

def credit_risk_and_loan_repayment_analysis(df):
    print("Credit Risk and Loan Repayment Analysis")
    expected = ['credit_policy', 'purpose', 'int_rate', 'installment', 'log_annual_inc', 'dti', 'fico', 'revol_bal', 'inq_last_6mths', 'not_fully_paid']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col != 'purpose':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    not_paid_rate = df['not_fully_paid'].mean() * 100
    avg_fico_not_paid = df[df['not_fully_paid'] == 1]['fico'].mean()
    avg_fico_paid = df[df['not_fully_paid'] == 0]['fico'].mean()
    print(f"Not Fully Paid Rate: {not_paid_rate:.2f}%")
    print(f"Avg. FICO (Not Paid): {avg_fico_not_paid:.0f}")
    print(f"Avg. FICO (Paid): {avg_fico_paid:.0f}")
    fig1 = px.histogram(df, x='fico', color='not_fully_paid', barmode='overlay', title="FICO Score Distribution by Repayment Status")
    fig1.show()
    not_paid_by_purpose = df.groupby('purpose')['not_fully_paid'].mean().mul(100).sort_values(ascending=False).reset_index()
    fig2 = px.bar(not_paid_by_purpose, x='purpose', y='not_fully_paid', title="Default Rate by Loan Purpose")
    fig2.show()

def direct_marketing_campaign_outcome_analysis(df):
    print("Direct Marketing Campaign Outcome Analysis")
    expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'poutcome']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    success_rate = (df['poutcome'] == 'success').mean() * 100
    failure_rate = (df['poutcome'] == 'failure').mean() * 100
    print(f"Success Rate of Previous Campaign: {success_rate:.1f}%")
    print(f"Failure Rate of Previous Campaign: {failure_rate:.1f}%")
    outcome_dist = df['poutcome'].value_counts().reset_index()
    outcome_dist.columns = ['outcome', 'count']
    fig1 = px.pie(outcome_dist, names='outcome', values='count', title="Distribution of Previous Campaign Outcomes")
    fig1.show()
    fig2 = px.density_heatmap(df, x="age", y="balance", z="poutcome", histfunc="count", facet_col="poutcome", title="Age-Balance Distribution by Previous Outcome")
    fig2.show()

def loan_default_prediction_analysis_based_on_employment_and_balance(df):
    print("Loan Default Prediction Analysis (Employment & Balance)")
    expected = ['employed', 'bank_balance', 'annual_salary', 'defaulted']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    default_rate = df['defaulted'].mean() * 100
    print(f"Overall Default Rate: {default_rate:.2f}%")
    fig1 = px.scatter(df, x='bank_balance', y='annual_salary', color='defaulted', title="Default Status by Bank Balance and Annual Salary")
    fig1.show()
    fig2 = px.box(df, x='employed', y='bank_balance', color='defaulted', title="Bank Balance by Employment and Default Status")
    fig2.show()

def customer_churn_analysis_based_on_credit_score_and_demographics(df):
    print("Customer Churn Analysis (Credit Score & Demographics)")
    expected = ['customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'churn']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['credit_score', 'age', 'tenure', 'balance', 'churn']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    churn_rate = df['churn'].mean() * 100
    print(f"Overall Churn Rate: {churn_rate:.2f}%")
    churn_by_country = df.groupby('country')['churn'].mean().mul(100).reset_index()
    fig1 = px.bar(churn_by_country, x='country', y='churn', title="Churn Rate by Country")
    fig1.show()
    fig2 = px.violin(df, x='gender', y='age', color='churn', box=True, title="Age Distribution by Gender and Churn Status")
    fig2.show()

def customer_creditworthiness_assessment_analysis(df):
    print("Customer Creditworthiness Assessment Analysis")
    expected = ['creditability', 'account_balance', 'duration_of_credit_monthly', 'payment_status_of_previous_credit', 'purpose', 'credit_amount', 'value_savings_stocks', 'age_years', 'occupation']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysist")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['creditability', 'duration_of_credit_monthly', 'credit_amount', 'age_years']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    good_credit_rate = df['creditability'].mean() * 100
    print(f"Good Creditability Rate: {good_credit_rate:.1f}%")
    fig1 = px.box(df, x='creditability', y='credit_amount', title="Credit Amount by Creditability")
    fig1.show()
    credit_by_purpose = df.groupby('purpose')['creditability'].mean().mul(100).sort_values().reset_index()
    fig2 = px.bar(credit_by_purpose, x='purpose', y='creditability', title="Good Credit Rate by Loan Purpose")
    fig2.show()

def financial_product_subscription_likelihood_analysis(df):
    print("Financial Product Subscription Likelihood Analysis")
    expected = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df.dropna(inplace=True)
    print(f"Subscription Rate: {df['subscribed'].mean()*100:.2f}%")
    rate_by_education = df.groupby('education')['subscribed'].mean().mul(100).reset_index()
    fig1 = px.bar(rate_by_education, x='education', y='subscribed', title="Subscription Rate by Education Level")
    fig1.show()
    pivot = df.pivot_table(index='housing', columns='loan', values='subscribed', aggfunc='mean')
    fig2 = px.imshow(pivot, text_auto=True, aspect="auto", title="Subscription Rate by Housing and Personal Loan Status")
    fig2.show()

def debt_recovery_strategy_effectiveness_analysis(df):
    print("Debt Recovery Strategy Effectiveness Analysis")
    expected = ['expected_recovery_amount', 'actual_recovery_amount', 'recovery_strategy', 'age', 'sex']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['expected_recovery_amount', 'actual_recovery_amount', 'age']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['recovery_rate'] = df['actual_recovery_amount'] / df['expected_recovery_amount']
    avg_recovery_rate = df['recovery_rate'].mean() * 100
    best_strategy = df.groupby('recovery_strategy')['recovery_rate'].mean().idxmax()
    print(f"Average Recovery Rate: {avg_recovery_rate:.2f}%")
    print(f"Best Performing Strategy: {best_strategy}")
    fig1 = px.box(df, x='recovery_strategy', y='recovery_rate', title="Recovery Rate by Strategy")
    fig1.show()
    fig2 = px.scatter(df, x='age', y='recovery_rate', color='sex', title="Recovery Rate by Age and Sex", trendline='ols')
    fig2.show()

def general_ledger_journal_voucher_transaction_analysis(df):
    print("General Ledger Journal Voucher Transaction Analysis")
    expected = ['journal_voucher_item_amount', 'credit_debit_code', 'fiscal_month', 'departmentnumber', 'general_ledger_account_code']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['journal_voucher_item_amount'] = pd.to_numeric(df['journal_voucher_item_amount'], errors='coerce')
    df.dropna(inplace=True)
    total_debit = df[df['credit_debit_code'].str.upper() == 'D']['journal_voucher_item_amount'].sum()
    total_credit = df[df['credit_debit_code'].str.upper() == 'C']['journal_voucher_item_amount'].sum()
    print(f"Total Debit Amount: ${total_debit:,.2f}")
    print(f"Total Credit Amount: ${total_credit:,.2f}")
    print(f"Net Difference (Debit - Credit): ${total_debit - total_credit:,.2f}")
    amount_by_dept = df.groupby('departmentnumber')['journal_voucher_item_amount'].sum().nlargest(15).reset_index()
    amount_by_dept.columns = ['department', 'amount']
    fig1 = px.bar(amount_by_dept, x='department', y='amount', title="Top 15 Departments by Transaction Amount")
    fig1.show()
    amount_by_gl = df.groupby('general_ledger_account_code')['journal_voucher_item_amount'].sum().nlargest(15).reset_index()
    amount_by_gl.columns = ['gl_account', 'amount']
    fig2 = px.bar(amount_by_gl, x='gl_account', y='amount', title="Top 15 GL Accounts by Transaction Amount")
    fig2.show()

def customer_credit_risk_assessment(df):
    print("Customer Credit Risk Assessment")
    expected = ['age', 'sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'credit_amount', 'duration', 'purpose', 'risk']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['age', 'credit_amount', 'duration']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    bad_risk_rate = (df['risk'] == 'bad').mean() * 100
    print(f"Bad Risk Rate: {bad_risk_rate:.2f}%")
    risk_by_job = df.groupby('job')['risk'].apply(lambda x: (x == 'bad').mean()).mul(100).reset_index()
    risk_by_job.columns = ['job', 'risk_rate']
    fig1 = px.bar(risk_by_job, x='job', y='risk_rate', title="Bad Risk Rate by Job Type")
    fig1.show()
    fig2 = px.scatter(df, x='age', y='credit_amount', color='risk', facet_col='sex', title="Credit Amount vs. Age by Risk and Sex")
    fig2.show()

def call_center_campaign_effectiveness_analysis(df):
    print("Call Center Campaign Effectiveness Analysis")
    expected = ['age', 'job', 'marital', 'education', 'loan', 'month', 'duration', 'campaign', 'poutcome', 'y']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['y'].dtype == 'object':
        df['subscribed'] = df['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['y'], errors='coerce')
    df['duration'] = pd.to_numeric(df['duration'], errors='coerce')
    df['campaign'] = pd.to_numeric(df['campaign'], errors='coerce')
    df.dropna(inplace=True)
    conversion_rate = df['subscribed'].mean() * 100
    avg_contacts = df['campaign'].mean()
    print(f"Conversion Rate: {conversion_rate:.2f}%")
    print(f"Average Contacts per Person: {avg_contacts:.2f}")
    conversion_by_contacts = df.groupby('campaign')['subscribed'].mean().mul(100).reset_index()
    fig1 = px.bar(conversion_by_contacts, x='campaign', y='subscribed', title="Conversion Rate by Number of Contacts in this Campaign")
    fig1.show()
    fig2 = px.box(df, x='poutcome', y='duration', color='y', title="Call Duration by Subscription Outcome and Previous Outcome")
    fig2.show()

def bank_deposit_subscription_prediction_analysis(df):
    print("Bank Deposit Subscription Prediction Analysis")
    expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit']
    matched = fuzzy_match_column(df, expected)
    if any(v is None for v in matched.values()):
        show_missing_columns_warning([k for k,v in matched.items() if v is None], matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['deposit'].dtype == 'object':
        df['subscribed'] = df['deposit'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
    else:
        df['subscribed'] = pd.to_numeric(df['deposit'], errors='coerce')
    df.dropna(inplace=True)
    print(f"Deposit Subscription Rate: {df['subscribed'].mean()*100:.2f}%")
    fig1 = px.density_heatmap(df, x="age", y="balance", z="subscribed", histfunc="avg", title="Heatmap of Subscription Rate by Age and Balance", labels={'balance': 'Account Balance'})
    fig1.show()
    conversion_by_contact = df.groupby('contact')['subscribed'].mean().mul(100).reset_index()
    fig2 = px.pie(conversion_by_contact, names='contact', values='subscribed', title="Subscription Rate by Contact Method", hole=0.4)
    fig2.show()

# ========== Main driver for all analysis functions ==========
def run_analysis(df, analysis_name):
    """
    Run a chosen analysis function on the dataframe.
    """
    available_functions = {
        "credit_risk_analysis": credit_risk_analysis,
        "fraud_detection_analysis": fraud_detection_analysis,
        "customer_segmentation_analysis": customer_segmentation_analysis,
        "churn_prediction_analysis": churn_prediction_analysis,
        "liquidity_&_cash_flow_analysis": liquidity_and_cash_flow_analysis,
        "transaction_trend_analysis": transaction_trend_analysis,
        "aml_analysis": aml_analysis,
        "profitability_analysis": profitability_analysis,
        "loan_portfolio_stress_testing": loan_portfolio_stress_testing,
        "sentiment_analysis": sentiment_analysis,
        "bank_marketing_campaign_effectiveness_analysis": bank_marketing_campaign_analysis,
        "loan_default_risk_prediction_analysis": loan_default_risk_prediction_analysis,
        "bank_institution_financial_and_structural_analysis": bank_institution_financial_and_structural_analysis,
        "bank_branch_geospatial_distribution_analysis": bank_branch_geospatial_distribution_analysis,
        "bank_office_and_service_type_analysis": bank_office_and_service_type_analysis,
        "financial_institution_geolocation_analysis": financial_institution_geolocation_analysis,
        "consumer_banking_habits_survey_analysis": consumer_banking_habits_survey_analysis,
        "financial_service_provider_accessibility_analysis": financial_service_provider_accessibility_analysis,
        "socio-economic_analysis_of_unbanked_populations": socio_economic_analysis_of_unbanked_populations,
        "automated_teller_machine_(atm)_location_analysis": automated_teller_machine_atm_location_analysis,
        "transaction_fraud_risk_flagging_analysis": transaction_fraud_risk_flagging_analysis,
        "customer_churn_prediction_analysis": customer_churn_prediction_analysis,
        "customer_loan_risk_assessment_analysis": customer_loan_risk_assessment_analysis,
        "stock_index_time_series_analysis": stock_index_time_series_analysis,
        "global_stock_market_index_performance_analysis": global_stock_market_index_performance_analysis,
        "stock_index_performance_and_currency_conversion_analysis": stock_index_performance_and_currency_conversion_analysis,
        "customer_credit_score_factor_analysis": customer_credit_score_factor_analysis,
        "loan_application_approval_prediction_analysis": loan_application_approval_prediction_analysis,
        "banknote_authentication_analysis": banknote_authentication_analysis,
        "customer_response_prediction_for_marketing_campaign": customer_response_prediction_for_marketing_campaign,
        "international_banking_statistics_and_cross-border_claims_analysis": international_banking_statistics_and_cross_border_claims_analysis,
        "customer_account_transaction_pattern_analysis": customer_account_transaction_pattern_analysis,
        "telemarketing_campaign_outcome_analysis": telemarketing_campaign_outcome_analysis,
        "bank_term_deposit_subscription_analysis": bank_term_deposit_subscription_analysis,
        "millennial_banking_preferences_survey_analysis": millennial_banking_preferences_survey_analysis,
        "bank_direct_marketing_success_prediction_analysis": bank_direct_marketing_success_prediction_analysis,
        "customer_segmentation_and_product_limit_analysis": customer_segmentation_and_product_limit_analysis,
        "credit_risk_classification_analysis": credit_risk_classification_analysis,
        "loan_application_status_prediction_analysis": loan_application_status_prediction_analysis,
        "bank_customer_attrition_analysis": bank_customer_attrition_analysis,
        "loan_approval_status_prediction_analysis": loan_approval_status_prediction_analysis,
        "credit_risk_and_loan_repayment_analysis": credit_risk_and_loan_repayment_analysis,
        "direct_marketing_campaign_outcome_analysis": direct_marketing_campaign_outcome_analysis,
        "loan_default_prediction_analysis_based_on_employment_and_balance": loan_default_prediction_analysis_based_on_employment_and_balance,
        "customer_churn_analysis_based_on_credit_score_and_demographics": customer_churn_analysis_based_on_credit_score_and_demographics,
        "customer_creditworthiness_assessment_analysis": customer_creditworthiness_assessment_analysis,
        "financial_product_subscription_likelihood_analysis": financial_product_subscription_likelihood_analysis,
        "debt_recovery_strategy_effectiveness_analysis": debt_recovery_strategy_effectiveness_analysis,
        "general_ledger_journal_voucher_transaction_analysis": general_ledger_journal_voucher_transaction_analysis,
        "customer_credit_risk_assessment": customer_credit_risk_assessment,
        "call_center_campaign_effectiveness_analysis": call_center_campaign_effectiveness_analysis,
        "bank_deposit_subscription_prediction_analysis": bank_deposit_subscription_prediction_analysis,
    }

    key = analysis_name.strip().lower().replace(' ', '_').replace('_-', '_').replace('-', '_').replace('.', '_')
    matched_func = available_functions.get(key, None)

    if matched_func:
        try:
            matched_func(df)
        except Exception as e:
            print(f"ERROR running {analysis_name}: {e}")
            show_general_insights(df, f"Fallback for {analysis_name}")
    else:
        print(f"WARNING: Analysis function for '{analysis_name}' not found.")
        show_general_insights(df, f"Fallback for {analysis_name}")

def main():
    """
    Main function to load data and run the analysis.
    User can change the file path and analysis type here.
    """
    print("🏦 Banking & Financial Analytics Dashboard")
    # --- CHANGE THESE VARIABLES TO MATCH YOUR FILE AND DESIRED ANALYSIS ---
    user_data_path = "your_data.csv"  # <--- Change this to your file path
    analysis_to_run = "loan_default_risk_prediction_analysis"  # <--- Change this to one of the analysis_options
    user_encoding = "utf-8"  # <--- Change this if your file has a different encoding

    # Check if user has updated the file path
    if user_data_path == "your_data.csv":
        print("\nPlease update the `user_data_path` variable with your data file's location.")
        return

    df = load_data(user_data_path, encoding=user_encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("✅ Data loaded successfully!")
    run_analysis(df, analysis_to_run)

# This is the standard entry point for a Python script.
if __name__ == "__main__":
    main()
