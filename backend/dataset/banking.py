import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process
import warnings
import json

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

# ========== NEW/REFACTORED UTILITY FUNCTIONS ==========

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


def create_missing_columns_message(missing_cols, matched_cols=None):
    """Generates a warning message for missing columns."""
    message = "⚠️ The following columns are needed for this analysis but weren't found: "
    col_messages = []
    for col in missing_cols:
        match_info = f" (attempted match: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""
        col_messages.append(f"'{col}'{match_info}")
    message += ", ".join(col_messages) + ". Showing general data insights instead."
    return message


# ========== DATA LOADING (Unchanged) ==========
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


# ========== REFACTORED ANALYSIS FUNCTIONS ==========

def credit_risk_analysis(df):
    analysis_name = "credit_risk_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["General Credit Risk Analysis"]
        
        expected = ['default', 'risk', 'risk_flag', 'loan_status', 'not_fully_paid', 'credit_score', 'fico', 'income', 'loan_amount', 'credit_amount', 'dti', 'age', 'housing', 'purpose']
        matched = fuzzy_match_column(df, expected)
        
        found_cols = {k: v for k, v in matched.items() if v}
        if not found_cols:
            message = "Could not find any standard credit risk columns like 'default', 'risk_flag', 'credit_score', or 'income'."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        
        risk_col_name = 'risk_numeric'
        risk_col_original = matched.get('default') or matched.get('risk') or matched.get('risk_flag') or matched.get('not_fully_paid')
        
        if risk_col_original:
            insights.append("Default/Risk Rate Analysis")
            if df_analysis[risk_col_original].dtype == 'object':
                df_analysis[risk_col_name] = df_analysis[risk_col_original].apply(lambda x: 1 if str(x).lower() in ['bad', 'default', 'yes', '1'] else 0)
            else:
                df_analysis[risk_col_name] = pd.to_numeric(df_analysis[risk_col_original], errors='coerce')
                
            metrics['default_rate'] = df_analysis[risk_col_name].mean() * 100
            insights.append(f"Overall Default / Bad Risk Rate: {metrics['default_rate']:.2f}%")
            
            score_col_original = matched.get('credit_score') or matched.get('fico')
            if score_col_original:
                df_analysis[score_col_original] = pd.to_numeric(df_analysis[score_col_original], errors='coerce')
                fig = px.box(df_analysis, x=df_analysis[risk_col_name].astype(str), y=score_col_original, title=f"Distribution of {score_col_original.title()} by Risk Status")
                visualizations['credit_score_by_risk'] = fig.to_json()

            purpose_col_original = matched.get('purpose')
            if purpose_col_original:
                risk_by_purpose = df_analysis.groupby(purpose_col_original)[risk_col_name].mean().mul(100).sort_values(ascending=False).reset_index()
                metrics['risk_by_purpose'] = risk_by_purpose.to_dict('records')
                fig2 = px.bar(risk_by_purpose, x=purpose_col_original, y=risk_col_name, title="Bad Risk Rate by Loan Purpose")
                visualizations['risk_by_purpose_bar'] = fig2.to_json()
        
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def fraud_detection_analysis(df):
    analysis_name = "fraud_detection_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["General Fraud Detection Analysis"]
        
        expected = ['fraud', 'class', 'risk_flag', 'amount', 'transaction_type', 'variance', 'skewness', 'entropy']
        matched = fuzzy_match_column(df, expected)
        
        found_cols = {k: v for k, v in matched.items() if v}
        if not found_cols:
            message = "Could not find fraud-related columns like 'fraud', 'class', or 'amount'."
            return get_general_insights(df, analysis_name, message)

        df_analysis = df.copy()
        fraud_col_name = 'is_fraud'
        fraud_col_original = matched.get('fraud') or matched.get('class') or matched.get('risk_flag')
        
        if fraud_col_original:
            insights.append("Fraudulent Transaction Overview")
            if df_analysis[fraud_col_original].dtype == 'object':
                df_analysis[fraud_col_name] = df_analysis[fraud_col_original].apply(lambda x: 1 if str(x).lower() in ['fraud', '1', 'yes'] else 0)
            else:
                df_analysis[fraud_col_name] = pd.to_numeric(df_analysis[fraud_col_original], errors='coerce')
                
            metrics['fraud_rate'] = df_analysis[fraud_col_name].mean() * 100
            insights.append(f"Fraud Rate: {metrics['fraud_rate']:.3f}%")
            
            fraud_dist = df_analysis[fraud_col_name].value_counts().reset_index()
            fraud_dist.columns = ['is_fraud_label', 'count']
            fraud_dist['is_fraud_label'] = fraud_dist['is_fraud_label'].map({1: 'Fraud', 0: 'Not Fraud'})
            metrics['fraud_distribution'] = fraud_dist.to_dict('records')
            
            fig = px.pie(fraud_dist, names='is_fraud_label', values='count', title="Distribution of Fraudulent Transactions")
            visualizations['fraud_distribution_pie'] = fig.to_json()
            
            amount_col_original = matched.get('amount')
            if amount_col_original:
                df_analysis[amount_col_original] = pd.to_numeric(df_analysis[amount_col_original], errors='coerce')
                fig2 = px.box(df_analysis, x=fraud_col_name, y=amount_col_original, title="Transaction Amount by Fraud Status")
                visualizations['amount_by_fraud_status'] = fig2.to_json()
        
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_segmentation_analysis(df):
    analysis_name = "customer_segmentation_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["General Customer Segmentation Analysis"]
        
        expected = ['age', 'gender', 'income', 'balance', 'numofproducts', 'segment', 'geography']
        matched = fuzzy_match_column(df, expected)
        
        found_cols = {k: v for k, v in matched.items() if v}
        if not found_cols:
            message = "Could not find customer columns like 'age', 'income', or 'balance'."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        age_col, income_col, balance_col = matched.get('age'), matched.get('income'), matched.get('balance')
        
        if not matched.get('segment'):
            insights.append("No 'segment' column found. Creating simple segmentation.")
            if age_col and income_col:
                df_analysis[age_col] = pd.to_numeric(df_analysis[age_col], errors='coerce')
                df_analysis[income_col] = pd.to_numeric(df_analysis[income_col], errors='coerce')
                
                age_bins = [0, 30, 50, 100]
                age_labels = ['Young', 'Adult', 'Senior']
                income_bins = df_analysis[income_col].quantile([0, 0.33, 0.66, 1]).tolist()
                income_labels = ['Low Income', 'Mid Income', 'High Income']
                
                df_analysis['age_group'] = pd.cut(df_analysis[age_col], bins=age_bins, labels=age_labels, right=False)
                df_analysis['income_group'] = pd.cut(df_analysis[income_col], bins=income_bins, labels=income_labels, include_lowest=True)
                df_analysis['segment'] = df_analysis['income_group'].astype(str) + " - " + df_analysis['age_group'].astype(str)
                segment_col = 'segment'
            else:
                insights.append("Could not create segments. Need 'age' and 'income' columns.")
                segment_col = None
        else:
            segment_col = matched.get('segment')
            
        if segment_col:
            insights.append("Segment Distribution Analysis")
            segment_counts = df_analysis[segment_col].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            metrics['segment_distribution'] = segment_counts.to_dict('records')
            
            fig1 = px.pie(segment_counts, names='Segment', values='Count', title="Customer Segment Distribution")
            visualizations['segment_distribution_pie'] = fig1.to_json()
            
            if balance_col:
                df_analysis[balance_col] = pd.to_numeric(df_analysis[balance_col], errors='coerce')
                fig2 = px.box(df_analysis, x=segment_col, y=balance_col, title="Account Balance by Customer Segment")
                visualizations['balance_by_segment'] = fig2.to_json()
        
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }
        
    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def churn_prediction_analysis(df):
    analysis_name = "churn_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["General Churn Prediction Analysis"]
        
        expected = ['churn', 'exited', 'attrition', 'credit_score', 'age', 'tenure', 'balance', 'numofproducts', 'isactivemember', 'geography']
        matched = fuzzy_match_column(df, expected)
        
        churn_col_original = matched.get('churn') or matched.get('exited') or matched.get('attrition')
        if not churn_col_original:
            message = "Could not find a churn indicator column like 'churn', 'exited', or 'attrition'."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        churn_col_name = 'is_churn'
        df_analysis[churn_col_name] = pd.to_numeric(df_analysis[churn_col_original], errors='coerce')
        
        metrics['churn_rate'] = df_analysis[churn_col_name].mean() * 100
        insights.append(f"Overall Churn Rate: {metrics['churn_rate']:.2f}%")
        
        products_col_original = matched.get('numofproducts')
        if products_col_original:
            df_analysis[products_col_original] = pd.to_numeric(df_analysis[products_col_original], errors='coerce')
            churn_by_products = df_analysis.groupby(products_col_original)[churn_col_name].mean().mul(100).reset_index()
            metrics['churn_by_products'] = churn_by_products.to_dict('records')
            fig1 = px.bar(churn_by_products, x=products_col_original, y=churn_col_name, title="Churn Rate by Number of Products")
            visualizations['churn_by_products_bar'] = fig1.to_json()
            
        balance_col_original, age_col_original = matched.get('balance'), matched.get('age')
        if balance_col_original and age_col_original:
            df_analysis[balance_col_original] = pd.to_numeric(df_analysis[balance_col_original], errors='coerce')
            df_analysis[age_col_original] = pd.to_numeric(df_analysis[age_col_original], errors='coerce')
            fig2 = px.scatter(df_analysis, x=age_col_original, y=balance_col_original, color=churn_col_name, title="Churn Status by Age and Account Balance")
            visualizations['churn_by_age_balance_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def liquidity_and_cash_flow_analysis(df):
    analysis_name = "liquidity_and_cash_flow_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Liquidity & Cash Flow Analysis"]
        
        expected = ['date', 'current_assets', 'current_liabilities', 'operating_cf', 'investing_cf', 'financing_cf', 'net_cash_flow']
        matched = fuzzy_match_column(df, expected)
        
        found_cols = {k: v for k, v in matched.items() if v}
        if not found_cols:
            message = "Could not find liquidity or cash flow columns."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        date_col = matched.get('date')
        if date_col:
            df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors='coerce')
            df_analysis = df_analysis.sort_values(date_col)
            
        if matched.get('current_assets') and matched.get('current_liabilities'):
            insights.append("Liquidity Ratios")
            df_analysis['current_ratio'] = df_analysis[matched['current_assets']] / df_analysis[matched['current_liabilities']]
            metrics['latest_current_ratio'] = df_analysis['current_ratio'].iloc[-1]
            insights.append(f"Latest Current Ratio: {metrics['latest_current_ratio']:.2f}")
            if date_col:
                fig1 = px.line(df_analysis, x=date_col, y='current_ratio', title="Current Ratio Over Time")
                visualizations['current_ratio_over_time'] = fig1.to_json()
                
        cf_cols = ['operating_cf', 'investing_cf', 'financing_cf']
        found_cf_cols = [matched[c] for c in cf_cols if matched.get(c)]
        if len(found_cf_cols) > 1:
            insights.append("Cash Flow Components")
            df_cf = df_analysis[found_cf_cols].sum().reset_index()
            df_cf.columns = ['Cash Flow Type', 'Amount']
            metrics['cash_flow_components'] = df_cf.to_dict('records')
            fig2 = px.bar(df_cf, x='Cash Flow Type', y='Amount', title="Total Cash Flow by Component")
            visualizations['cash_flow_components_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def transaction_trend_analysis(df):
    analysis_name = "transaction_trend_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Transaction Trend Analysis"]
        
        expected = ['transaction_date', 'date', 'amount', 'transaction_type']
        matched = fuzzy_match_column(df, expected)
        
        date_col = matched.get('transaction_date') or matched.get('date')
        amount_col = matched.get('amount')
        
        if not date_col or not amount_col:
            message = "Could not find necessary columns 'date' and 'amount' for trend analysis."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        df_analysis[date_col] = pd.to_datetime(df_analysis[date_col], errors='coerce')
        df_analysis[amount_col] = pd.to_numeric(df_analysis[amount_col], errors='coerce')
        df_analysis = df_analysis.sort_values(date_col).dropna(subset=[date_col, amount_col])
        df_analysis.set_index(date_col, inplace=True)
        
        monthly_sum = df_analysis[amount_col].resample('M').sum().reset_index()
        monthly_count = df_analysis[amount_col].resample('M').count().reset_index()
        
        metrics['monthly_value'] = monthly_sum.to_dict('records')
        metrics['monthly_count'] = monthly_count.to_dict('records')
        insights.append("Transaction Volume and Value Over Time")
        
        fig1 = px.line(monthly_sum, x=date_col, y=amount_col, title="Total Transaction Value (Monthly)")
        visualizations['transaction_value_over_time'] = fig1.to_json()
        
        fig2 = px.line(monthly_count, x=date_col, y=amount_col, title="Total Transaction Count (Monthly)")
        visualizations['transaction_count_over_time'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def aml_analysis(df):
    analysis_name = "aml_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Anti-Money Laundering (AML) Analysis"]
        
        expected = ['customer_id', 'amount', 'date', 'source_country', 'destination_country', 'transaction_type']
        matched = fuzzy_match_column(df, expected)
        
        if not matched.get('amount') or not matched.get('customer_id'):
            message = "Could not find 'amount' and 'customer_id' for AML analysis."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        amount_col = matched['amount']
        cust_col = matched['customer_id']
        df_analysis[amount_col] = pd.to_numeric(df_analysis[amount_col], errors='coerce')
        
        insights.append("High-Value Transaction Monitoring")
        threshold = 10000.0
        high_value_txns = df_analysis[df_analysis[amount_col] > threshold]
        metrics['high_value_txn_count'] = len(high_value_txns)
        metrics['high_value_txn_threshold'] = threshold
        insights.append(f"Number of Transactions Above ${threshold:,.0f}: {metrics['high_value_txn_count']}")
        
        insights.append("High-Frequency Transaction Monitoring")
        customer_txn_counts = df_analysis[cust_col].value_counts().reset_index()
        customer_txn_counts.columns = [cust_col, 'transaction_count']
        metrics['transaction_frequency_per_customer'] = customer_txn_counts.head(10).to_dict('records')
        
        fig = px.histogram(customer_txn_counts, x='transaction_count', title="Distribution of Transaction Frequency per Customer")
        visualizations['transaction_frequency_histogram'] = fig.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def profitability_analysis(df):
    analysis_name = "profitability_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Profitability Analysis"]
        
        expected = ['date', 'product', 'revenue', 'cost', 'net_income', 'interest_income', 'interest_expense', 'roa', 'roe']
        matched = fuzzy_match_column(df, expected)
        df_analysis = df.copy()

        if matched.get('revenue') and matched.get('cost'):
            insights.append("Product Profitability")
            df_analysis['revenue'] = pd.to_numeric(df_analysis[matched['revenue']], errors='coerce')
            df_analysis['cost'] = pd.to_numeric(df_analysis[matched['cost']], errors='coerce')
            df_analysis['profit'] = df_analysis['revenue'] - df_analysis['cost']
            df_analysis['profit_margin'] = (df_analysis['profit'] / df_analysis['revenue']) * 100
            
            metrics['average_profit_margin'] = df_analysis['profit_margin'].mean()
            insights.append(f"Average Profit Margin: {metrics['average_profit_margin']:.2f}%")
            
            product_col = matched.get('product', 'product')
            if product_col not in df_analysis.columns:
                product_col = 'product' # Create a dummy if not found
                df_analysis[product_col] = 'Overall'
                
            profit_by_product = df_analysis.groupby(product_col)[['revenue', 'cost', 'profit']].sum().reset_index()
            metrics['profit_by_product'] = profit_by_product.to_dict('records')
            
            fig = px.bar(profit_by_product, x=product_col, y=['revenue', 'cost'], title="Revenue and Cost by Product")
            visualizations['revenue_cost_by_product'] = fig.to_json()
            
        elif matched.get('roa') and matched.get('roe'):
            insights.append("Bank-Level Profitability Ratios")
            df_analysis['roa'] = pd.to_numeric(df_analysis[matched['roa']], errors='coerce')
            df_analysis['roe'] = pd.to_numeric(df_analysis[matched['roe']], errors='coerce')
            
            if matched.get('date'):
                df_analysis[matched['date']] = pd.to_datetime(df_analysis[matched['date']], errors='coerce')
                fig = px.line(df_analysis, x=matched['date'], y=['roa', 'roe'], title="ROA & ROE Over Time")
                visualizations['roa_roe_over_time'] = fig.to_json()
            else:
                metrics['average_roa'] = df_analysis['roa'].mean()
                metrics['average_roe'] = df_analysis['roe'].mean()
                insights.append(f"Average ROA: {metrics['average_roa']:.2f}%")
                insights.append(f"Average ROE: {metrics['average_roe']:.2f}%")
        else:
            message = "Could not find required columns for profitability (e.g., 'revenue'/'cost' or 'roa'/'roe')."
            return get_general_insights(df, analysis_name, message)

        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_portfolio_stress_testing(df):
    analysis_name = "loan_portfolio_stress_testing"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Portfolio Stress Testing (Simplified)"]
        
        expected = ['loan_amount', 'ltv', 'dti', 'fico', 'credit_score', 'default_status']
        matched = fuzzy_match_column(df, expected)
        
        score_col = matched.get('fico') or matched.get('credit_score')
        amount_col = matched.get('loan_amount')
        
        if not score_col or not amount_col:
            message = "Requires 'fico'/'credit_score' and 'loan_amount' columns for stress testing."
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        df_analysis[score_col] = pd.to_numeric(df_analysis[score_col], errors='coerce')
        df_analysis[amount_col] = pd.to_numeric(df_analysis[amount_col], errors='coerce')
        df_analysis.dropna(subset=[score_col, amount_col], inplace=True)
        
        insights.append("Portfolio Distribution by Credit Score")
        fig = px.histogram(df_analysis, x=score_col, title=f"Loan Portfolio Distribution by {score_col.title()}")
        visualizations['portfolio_distribution_histogram'] = fig.to_json()
        
        insights.append("Stress Test Simulation")
        score_threshold = 620
        default_rate_increase = 10  # 10%
        
        low_score_portfolio = df_analysis[df_analysis[score_col] < score_threshold]
        metrics['portfolio_at_risk_amount'] = low_score_portfolio[amount_col].sum()
        metrics['potential_loss_simulation'] = metrics['portfolio_at_risk_amount'] * (default_rate_increase / 100.0)
        metrics['stress_test_score_threshold'] = score_threshold
        metrics['stress_test_default_rate_increase_perc'] = default_rate_increase
        
        insights.append(f"Portfolio Amount At Risk (Below {score_threshold}): ${metrics['portfolio_at_risk_amount']:,.0f}")
        insights.append(f"Simulated Potential Loss (at {default_rate_increase}%): ${metrics['potential_loss_simulation']:,.0f}")
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def sentiment_analysis(df):
    analysis_name = "sentiment_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Sentiment Analysis"]
        
        expected = ['text', 'review', 'feedback', 'date', 'rating', 'sentiment']
        matched = fuzzy_match_column(df, expected)
        df_analysis = df.copy()
        
        text_col = matched.get('text') or matched.get('review') or matched.get('feedback')
        
        if not text_col and not matched.get('sentiment'):
            message = "Requires a 'sentiment' column or a text column ('text', 'review', 'feedback') for analysis."
            return get_general_insights(df, analysis_name, message)
            
        if matched.get('sentiment'):
            insights.append("Sentiment Distribution (from data)")
            sentiment_col = matched['sentiment']
            sentiment_counts = df_analysis[sentiment_col].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            metrics['sentiment_distribution'] = sentiment_counts.to_dict('records')
            fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Customer Sentiment Distribution")
            visualizations['sentiment_distribution_pie'] = fig.to_json()
        
        elif text_col:
            insights.append("Sentiment Analysis (Calculated)")
            try:
                from textblob import TextBlob
                insights.append("Performing live sentiment analysis with TextBlob...")
                
                df_analysis['polarity'] = df_analysis[text_col].dropna().apply(lambda text: TextBlob(str(text)).sentiment.polarity)
                
                def get_sentiment(polarity):
                    if polarity > 0.1: return 'Positive'
                    elif polarity < -0.1: return 'Negative'
                    else: return 'Neutral'
                    
                df_analysis['sentiment_calculated'] = df_analysis['polarity'].apply(get_sentiment)
                metrics['average_polarity'] = df_analysis['polarity'].mean()
                insights.append(f"Average Sentiment Polarity: {metrics['average_polarity']:.3f}")
                
                sentiment_counts = df_analysis['sentiment_calculated'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentiment', 'Count']
                metrics['calculated_sentiment_distribution'] = sentiment_counts.to_dict('records')
                
                fig = px.pie(sentiment_counts, names='Sentiment', values='Count', title="Calculated Customer Sentiment")
                visualizations['calculated_sentiment_pie'] = fig.to_json()
                
            except ImportError:
                return {
                    "analysis_type": analysis_name,
                    "status": "error",
                    "message": "This analysis requires the `textblob` library. Please install it (`pip install textblob`) and download corpora (`python -m textblob.download_corpora`).",
                    "matched_columns": matched,
                    "metrics": {}, "visualizations": {}, "insights": []
                }
        
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_marketing_campaign_analysis(df):
    analysis_name = "bank_marketing_campaign_effectiveness_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Marketing Campaign Effectiveness Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'balance', 'loan', 'poutcome', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() in ['yes', '1', 'true'] else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
            
        df_analysis['age'] = pd.to_numeric(df_analysis['age'], errors='coerce')
        df_analysis['balance'] = pd.to_numeric(df_analysis['balance'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['conversion_rate'] = df_analysis['subscribed'].mean() * 100
        metrics['avg_age_subscribed'] = df_analysis[df_analysis['subscribed'] == 1]['age'].mean()
        metrics['top_job_subscribed'] = df_analysis[df_analysis['subscribed'] == 1]['job'].mode()[0]
        
        insights.append(f"Overall Conversion Rate: {metrics['conversion_rate']:.2f}%")
        insights.append(f"Avg. Age of Subscribers: {metrics['avg_age_subscribed']:.1f}")
        insights.append(f"Top Job for Subscribers: {metrics['top_job_subscribed']}")
        
        conversion_by_poutcome = df_analysis.groupby('poutcome')['subscribed'].mean().mul(100).reset_index()
        metrics['conversion_by_poutcome'] = conversion_by_poutcome.to_dict('records')
        fig1 = px.bar(conversion_by_poutcome, x='poutcome', y='subscribed', title="Conversion Rate by Previous Campaign Outcome", labels={'poutcome': 'Previous Outcome', 'subscribed': 'Conversion Rate (%)'})
        visualizations['conversion_by_poutcome_bar'] = fig1.to_json()
        
        fig2 = px.box(df_analysis, x='y', y='balance', title="Account Balance by Subscription Status")
        visualizations['balance_by_subscription_status'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_default_risk_prediction_analysis(df):
    analysis_name = "loan_default_risk_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Default Risk Prediction Analysis"]
        
        expected = ['income', 'age', 'loan', 'default']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in expected:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        df_analysis['default_status'] = df_analysis['default'].map({1: 'Default', 0: 'No Default'})
        
        metrics['default_rate'] = df_analysis['default'].mean() * 100
        metrics['avg_income_default'] = df_analysis[df_analysis['default'] == 1]['income'].mean()
        metrics['avg_income_no_default'] = df_analysis[df_analysis['default'] == 0]['income'].mean()
        
        insights.append(f"Overall Default Rate: {metrics['default_rate']:.2f}%")
        insights.append(f"Avg. Income (Default): ${metrics['avg_income_default']:,.0f}")
        insights.append(f"Avg. Income (No Default): ${metrics['avg_income_no_default']:,.0f}")
        
        fig1 = px.box(df_analysis, x='default_status', y='income', color='default_status', title="Income Distribution by Default Status")
        visualizations['income_distribution_by_default'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='age', y='loan', color='default_status', title="Loan Amount vs. Age by Default Status", labels={'age': 'Age', 'loan': 'Loan Amount'})
        visualizations['loan_vs_age_by_default'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_institution_financial_and_structural_analysis(df):
    analysis_name = "bank_institution_financial_and_structural_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Institution Financial and Structural Analysis"]
        
        expected = ['stname', 'name', 'asset', 'dep', 'offices', 'roa', 'roe']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['asset', 'dep', 'offices', 'roa', 'roe']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['total_assets'] = df_analysis['asset'].sum()
        metrics['total_deposits'] = df_analysis['dep'].sum()
        metrics['average_roa'] = df_analysis['roa'].mean()
        
        insights.append(f"Total Assets: ${metrics['total_assets']:,.0f}")
        insights.append(f"Total Deposits: ${metrics['total_deposits']:,.0f}")
        insights.append(f"Average ROA: {metrics['average_roa']:.2f}%")
        
        assets_by_state = df_analysis.groupby('stname')['asset'].sum().nlargest(10).reset_index()
        metrics['top_10_states_by_assets'] = assets_by_state.to_dict('records')
        fig1 = px.bar(assets_by_state, x='stname', y='asset', title="Top 10 States by Total Bank Assets")
        visualizations['top_states_by_assets_bar'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='asset', y='dep', size='offices', hover_name='name', title="Assets vs. Deposits", log_x=True, log_y=True)
        visualizations['assets_vs_deposits_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_branch_geospatial_distribution_analysis(df):
    analysis_name = "bank_branch_geospatial_distribution_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Branch Geospatial Distribution Analysis"]
        
        expected = ['name', 'address', 'latitude', 'longitude', 'ward']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['latitude', 'longitude']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        metrics['total_branches'] = len(df_analysis)
        metrics['top_ward'] = df_analysis['ward'].mode()[0]
        
        insights.append(f"Total Number of Branches: {metrics['total_branches']}")
        insights.append(f"Ward with Most Branches: {metrics['top_ward']}")
        
        branches_by_ward = df_analysis['ward'].value_counts().nlargest(15).reset_index()
        branches_by_ward.columns = ['ward', 'count']
        metrics['top_15_wards_by_branches'] = branches_by_ward.to_dict('records')
        
        fig = px.bar(branches_by_ward, x='ward', y='count', title="Top 15 Wards by Number of Bank Branches")
        visualizations['branches_by_ward_bar'] = fig.to_json()
        
        # Geospatial plot (e.g., Mapbox scatter)
        fig_map = px.scatter_mapbox(df_analysis, lat="latitude", lon="longitude", hover_name="name",
                                    hover_data=["address", "ward"],
                                    color_discrete_sequence=["blue"], zoom=10, height=500,
                                    title="Bank Branch Locations")
        fig_map.update_layout(mapbox_style="open-street-map")
        visualizations['branch_location_map'] = fig_map.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_office_and_service_type_analysis(df):
    analysis_name = "bank_office_and_service_type_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Office and Service Type Analysis"]
        
        expected = ['name', 'servtype', 'city', 'stalp']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        df_analysis.dropna(inplace=True)
        
        metrics['total_offices'] = len(df_analysis)
        metrics['most_common_service_type'] = df_analysis['servtype'].mode()[0]
        metrics['state_with_most_offices'] = df_analysis['stalp'].mode()[0]
        
        insights.append(f"Total Offices Listed: {metrics['total_offices']:,}")
        insights.append(f"Most Common Service Type: {metrics['most_common_service_type']}")
        insights.append(f"State with Most Offices: {metrics['state_with_most_offices']}")
        
        service_type_counts = df_analysis['servtype'].value_counts().reset_index()
        service_type_counts.columns = ['servtype', 'count']
        metrics['service_type_distribution'] = service_type_counts.to_dict('records')
        fig1 = px.pie(service_type_counts, names='servtype', values='count', title="Distribution of Bank Service Types")
        visualizations['service_type_pie'] = fig1.to_json()
        
        offices_by_state = df_analysis['stalp'].value_counts().nlargest(20).reset_index()
        offices_by_state.columns = ['state', 'count']
        metrics['top_20_states_by_offices'] = offices_by_state.to_dict('records')
        fig2 = px.bar(offices_by_state, x='state', y='count', title="Top 20 States by Number of Bank Offices")
        visualizations['offices_by_state_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def financial_institution_geolocation_analysis(df):
    analysis_name = "financial_institution_geolocation_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Financial Institution Geolocation Analysis"]
        
        expected = ['name_of_institution', 'city', 'county', 'georeference']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        try:
            coords = df_analysis['georeference'].str.extract(r'POINT \(([-\d\.]+) ([-\d\.]+)\)')
            df_analysis['longitude'] = pd.to_numeric(coords[0], errors='coerce')
            df_analysis['latitude'] = pd.to_numeric(coords[1], errors='coerce')
        except Exception as e:
            insights.append(f"ERROR: Could not parse latitude/longitude from 'georeference' column: {e}")
            return get_general_insights(df, analysis_name, f"Could not parse 'georeference' column: {e}")
            
        df_analysis.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        metrics['total_institutions_mapped'] = len(df_analysis)
        insights.append(f"Total Institutions Mapped: {metrics['total_institutions_mapped']}")
        
        inst_by_county = df_analysis['county'].value_counts().nlargest(20).reset_index()
        inst_by_county.columns = ['county', 'count']
        metrics['top_20_counties_by_institutions'] = inst_by_county.to_dict('records')
        
        fig = px.bar(inst_by_county, x='county', y='count', title="Top 20 Counties by Number of Financial Institutions")
        visualizations['institutions_by_county_bar'] = fig.to_json()
        
        # Geospatial plot
        fig_map = px.scatter_mapbox(df_analysis, lat="latitude", lon="longitude", hover_name="name_of_institution",
                                    hover_data=["city", "county"],
                                    color_discrete_sequence=["green"], zoom=6, height=500,
                                    title="Financial Institution Locations")
        fig_map.update_layout(mapbox_style="open-street-map")
        visualizations['institution_location_map'] = fig_map.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def consumer_banking_habits_survey_analysis(df):
    analysis_name = "consumer_banking_habits_survey_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Consumer Banking Habits Survey Analysis"]
        
        expected = ['banking_status', 'age_group', 'income_group', 'q4', 'q12', 'q27']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        metrics['has_checking_account_perc'] = (df_analysis['q4'].astype(str).str.lower() == '1').mean() * 100
        metrics['has_credit_card_perc'] = (df_analysis['q12'].astype(str).str.lower() == '1').mean() * 100
        metrics['uses_mobile_banking_perc'] = (df_analysis['q27'].astype(str).str.lower() == '1').mean() * 100
        
        insights.append(f"Has Checking Account: {metrics['has_checking_account_perc']:.1f}%")
        insights.append(f"Has Credit Card: {metrics['has_credit_card_perc']:.1f}%")
        insights.append(f"Uses Mobile Banking: {metrics['uses_mobile_banking_perc']:.1f}%")
        
        banking_dist = df_analysis['banking_status'].value_counts().reset_index()
        banking_dist.columns = ['status', 'count']
        metrics['banking_status_distribution'] = banking_dist.to_dict('records')
        fig1 = px.pie(banking_dist, names='status', values='count', title="Distribution of Banking Status")
        visualizations['banking_status_pie'] = fig1.to_json()
        
        mobile_by_income = df_analysis.groupby('income_group')['q27'].apply(lambda x: (x.astype(str).str.lower() == '1').mean()).mul(100).reset_index()
        mobile_by_income.columns = ['income_group', 'rate']
        metrics['mobile_adoption_by_income'] = mobile_by_income.to_dict('records')
        fig2 = px.bar(mobile_by_income, x='income_group', y='rate', title="Mobile Banking Adoption by Income Group")
        visualizations['mobile_adoption_by_income_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def financial_service_provider_accessibility_analysis(df):
    analysis_name = "financial_service_provider_accessibility_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Financial Service Provider Accessibility Analysis"]
        
        expected = ['provider', 'borough', 'days_open', 'language_s', 'latitude', 'longitude']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        metrics['total_providers'] = len(df_analysis)
        metrics['top_borough'] = df_analysis['borough'].mode()[0]
        metrics['weekend_availability_perc'] = df_analysis['days_open'].str.contains('Sat|Sun', case=False, na=False).mean() * 100
        
        insights.append(f"Total Providers: {metrics['total_providers']:,}")
        insights.append(f"Borough with Most Providers: {metrics['top_borough']}")
        insights.append(f"Weekend Availability: {metrics['weekend_availability_perc']:.1f}%")
        
        df_analysis.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        try:
            languages = df_analysis['language_s'].str.split(',').explode().str.strip().value_counts().nlargest(10)
            languages_df = languages.reset_index()
            languages_df.columns = ['language', 'count']
            metrics['top_10_languages'] = languages_df.to_dict('records')
            fig2 = px.bar(languages_df, x='language', y='count', title="Top 10 Languages Offered")
            visualizations['top_languages_bar'] = fig2.to_json()
        except Exception as lang_e:
            insights.append(f"WARNING: Could not analyze 'language_s' column: {lang_e}")

        # Geospatial plot
        fig_map = px.scatter_mapbox(df_analysis, lat="latitude", lon="longitude", hover_name="provider",
                                    hover_data=["borough", "days_open", "language_s"],
                                    color="borough", zoom=10, height=500,
                                    title="Financial Service Provider Locations")
        fig_map.update_layout(mapbox_style="open-street-map")
        visualizations['provider_location_map'] = fig_map.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def socio_economic_analysis_of_unbanked_populations(df):
    analysis_name = "socio_economic_analysis_of_unbanked_populations"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Socio-Economic Analysis of Unbanked Populations"]
        
        expected = ['sub_boro_name', 'unbanked_2013', 'underbanked_2013', 'perc_poor_2013', 'median_income_2013', 'unemployment_2013']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in expected:
            if 'name' not in col:
                df_analysis[col] = pd.to_numeric(df_analysis[col].astype(str).str.replace('%', ''), errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['avg_unbanked_rate'] = df_analysis['unbanked_2013'].mean()
        metrics['avg_poverty_rate'] = df_analysis['perc_poor_2013'].mean()
        metrics['avg_median_income'] = df_analysis['median_income_2013'].mean()
        
        insights.append(f"Average Unbanked Rate: {metrics['avg_unbanked_rate']:.1f}%")
        insights.append(f"Average Poverty Rate: {metrics['avg_poverty_rate']:.1f}%")
        insights.append(f"Average Median Income: ${metrics['avg_median_income']:,.0f}")
        
        fig1 = px.scatter(df_analysis, x='median_income_2013', y='unbanked_2013', size='unemployment_2013', hover_name='sub_boro_name', title="Unbanked Rate vs. Median Income", trendline='ols')
        visualizations['unbanked_vs_income_scatter'] = fig1.to_json()
        
        top_unbanked = df_analysis.nlargest(10, 'unbanked_2013')
        metrics['top_10_unbanked_areas'] = top_unbanked.to_dict('records')
        fig2 = px.bar(top_unbanked, x='sub_boro_name', y='unbanked_2013', title="Top 10 Areas by Unbanked Population Rate")
        visualizations['top_unbanked_areas_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def automated_teller_machine_atm_location_analysis(df):
    analysis_name = "automated_teller_machine_atm_location_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Automated Teller Machine (ATM) Location Analysis"]
        
        expected = ['name', 'address', 'latitude', 'longitude', 'ward']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['latitude'] = pd.to_numeric(df_analysis['latitude'], errors='coerce')
        df_analysis['longitude'] = pd.to_numeric(df_analysis['longitude'], errors='coerce')
        df_analysis.dropna(subset=['latitude', 'longitude'], inplace=True)
        
        metrics['total_atms_mapped'] = len(df_analysis)
        insights.append(f"Total ATMs Mapped: {metrics['total_atms_mapped']}")
        
        atms_by_ward = df_analysis['ward'].value_counts().nlargest(20).reset_index()
        atms_by_ward.columns = ['ward', 'count']
        metrics['top_20_wards_by_atms'] = atms_by_ward.to_dict('records')
        
        fig = px.bar(atms_by_ward, x='ward', y='count', title="Top 20 Wards by Number of ATMs")
        visualizations['atms_by_ward_bar'] = fig.to_json()

        # Geospatial plot
        fig_map = px.scatter_mapbox(df_analysis, lat="latitude", lon="longitude", hover_name="name",
                                    hover_data=["address", "ward"],
                                    color_discrete_sequence=["red"], zoom=10, height=500,
                                    title="ATM Locations")
        fig_map.update_layout(mapbox_style="open-street-map")
        visualizations['atm_location_map'] = fig_map.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def transaction_fraud_risk_flagging_analysis(df):
    analysis_name = "transaction_fraud_risk_flagging_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Transaction Fraud Risk Flagging Analysis"]
        
        expected = ['id', 'risk_flag']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['risk_flag'] = pd.to_numeric(df_analysis['risk_flag'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['total_transactions'] = len(df_analysis)
        metrics['risky_transactions'] = int(df_analysis['risk_flag'].sum())
        metrics['risk_rate_perc'] = (metrics['risky_transactions'] / metrics['total_transactions']) * 100
        
        insights.append(f"Total Transactions: {metrics['total_transactions']:,}")
        insights.append(f"Flagged as Risky: {metrics['risky_transactions']:,}")
        insights.append(f"Risk Rate: {metrics['risk_rate_perc']:.2f}%")
        
        risk_dist = df_analysis['risk_flag'].value_counts().reset_index()
        risk_dist.columns = ['risk_label', 'count']
        risk_dist['risk_label'] = risk_dist['risk_label'].map({1: 'Risky', 0: 'Not Risky'})
        metrics['risk_distribution'] = risk_dist.to_dict('records')
        
        fig = px.pie(risk_dist, names='risk_label', values='count', title="Distribution of Risky Transactions")
        visualizations['risk_distribution_pie'] = fig.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_churn_prediction_analysis(df):
    analysis_name = "customer_churn_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Churn Prediction Analysis"]
        
        expected = ['creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'isactivemember', 'estimatedsalary', 'exited']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'exited']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['churn_rate_perc'] = df_analysis['exited'].mean() * 100
        metrics['avg_credit_score_churned'] = df_analysis[df_analysis['exited'] == 1]['creditscore'].mean()
        metrics['avg_balance_churned'] = df_analysis[df_analysis['exited'] == 1]['balance'].mean()
        
        insights.append(f"Overall Churn Rate: {metrics['churn_rate_perc']:.2f}%")
        insights.append(f"Avg. Credit Score (Churned): {metrics['avg_credit_score_churned']:.0f}")
        insights.append(f"Avg. Balance (Churned): ${metrics['avg_balance_churned']:,.0f}")
        
        fig1 = px.histogram(df_analysis, x='creditscore', color='exited', barmode='overlay', title="Credit Score Distribution by Churn Status")
        visualizations['credit_score_by_churn_histogram'] = fig1.to_json()
        
        churn_by_products = df_analysis.groupby('numofproducts')['exited'].mean().mul(100).reset_index()
        metrics['churn_by_products'] = churn_by_products.to_dict('records')
        fig2 = px.bar(churn_by_products, x='numofproducts', y='exited', title="Churn Rate by Number of Products Held")
        visualizations['churn_by_products_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_loan_risk_assessment_analysis(df):
    analysis_name = "customer_loan_risk_assessment_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Loan Risk Assessment Analysis"]
        
        expected = ['income', 'age', 'experience', 'house_ownership', 'car_ownership', 'profession', 'current_job_yrs', 'current_house_yrs', 'risk_flag']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['income', 'age', 'experience', 'current_job_yrs', 'current_house_yrs', 'risk_flag']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['high_risk_rate_perc'] = df_analysis['risk_flag'].mean() * 100
        metrics['avg_income_high_risk'] = df_analysis[df_analysis['risk_flag'] == 1]['income'].mean()
        
        insights.append(f"High Risk Rate: {metrics['high_risk_rate_perc']:.2f}%")
        insights.append(f"Avg. Income of High-Risk Applicants: ${metrics['avg_income_high_risk']:,.0f}")
        
        fig1 = px.box(df_analysis, x='house_ownership', y='income', color='risk_flag', title="Income Distribution by House Ownership and Risk Flag")
        visualizations['income_by_housing_risk_boxplot'] = fig1.to_json()
        
        risk_by_profession = df_analysis.groupby('profession')['risk_flag'].mean().mul(100).sort_values(ascending=False).reset_index()
        metrics['risk_by_profession'] = risk_by_profession.to_dict('records')
        fig2 = px.bar(risk_by_profession, x='profession', y='risk_flag', title="High Risk Rate by Profession")
        visualizations['risk_by_profession_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def stock_index_time_series_analysis(df):
    analysis_name = "stock_index_time_series_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Stock Index Time Series Analysis"]
        
        expected = ['date', 'open', 'high', 'low', 'close', 'volume']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['date'] = pd.to_datetime(df_analysis['date'], errors='coerce')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis = df_analysis.sort_values('date').dropna()
        
        if len(df_analysis) < 2:
            insights.append("Not enough data for time series analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "error",
                "message": "Not enough data (less than 2 rows) for time series analysis.",
                "matched_columns": matched,
                "metrics": {}, "visualizations": {}, "insights": insights
            }

        metrics['latest_close_price'] = df_analysis['close'].iloc[-1]
        metrics['day_change'] = metrics['latest_close_price'] - df_analysis['close'].iloc[-2]
        metrics['day_change_perc'] = (metrics['day_change'] / df_analysis['close'].iloc[-2]) * 100
        
        insights.append(f"Latest Close Price: ${metrics['latest_close_price']:,.2f}")
        insights.append(f"Day Change: ${metrics['day_change']:,.2f}")
        insights.append(f"% Change: {metrics['day_change_perc']:.2f}%")
        
        fig1 = px.line(df_analysis, x='date', y='close', title="Closing Price Over Time")
        visualizations['closing_price_over_time_line'] = fig1.to_json()
        
        fig2 = px.bar(df_analysis, x='date', y='volume', title="Trading Volume Over Time")
        visualizations['volume_over_time_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def global_stock_market_index_performance_analysis(df):
    analysis_name = "global_stock_market_index_performance_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Global Stock Market Index Performance Analysis"]
        
        expected = ['region', 'exchange', 'index', 'currency']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        metrics['number_of_indices'] = df_analysis['index'].nunique()
        metrics['number_of_regions'] = df_analysis['region'].nunique()
        metrics['most_common_currency'] = df_analysis['currency'].mode()[0]
        
        insights.append(f"Number of Indices: {metrics['number_of_indices']}")
        insights.append(f"Number of Regions: {metrics['number_of_regions']}")
        insights.append(f"Most Common Currency: {metrics['most_common_currency']}")
        
        indices_by_region = df_analysis.groupby('region')['index'].count().reset_index()
        indices_by_region.columns = ['region', 'count']
        metrics['indices_by_region'] = indices_by_region.to_dict('records')
        fig1 = px.bar(indices_by_region, x='region', y='count', title="Number of Stock Indices by Region")
        visualizations['indices_by_region_bar'] = fig1.to_json()
        
        currency_dist = df_analysis['currency'].value_counts().reset_index()
        currency_dist.columns = ['currency', 'count']
        metrics['currency_distribution'] = currency_dist.to_dict('records')
        fig2 = px.pie(currency_dist, names='currency', values='count', title="Distribution of Index Currencies")
        visualizations['currency_distribution_pie'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def stock_index_performance_and_currency_conversion_analysis(df):
    analysis_name = "stock_index_performance_and_currency_conversion_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Stock Index Performance and Currency Conversion Analysis"]
        
        expected = ['date', 'close', 'closeusd', 'volume']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['date'] = pd.to_datetime(df_analysis['date'], errors='coerce')
        for col in ['close', 'closeusd', 'volume']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        df_analysis['usd_to_local_rate'] = df_analysis['closeusd'] / df_analysis['close']
        
        metrics['latest_implied_usd_rate'] = df_analysis['usd_to_local_rate'].iloc[-1]
        metrics['average_implied_usd_rate'] = df_analysis['usd_to_local_rate'].mean()
        
        insights.append(f"Latest Implied USD/Local Rate: {metrics['latest_implied_usd_rate']:.4f}")
        insights.append(f"Average Implied Rate: {metrics['average_implied_usd_rate']:.4f}")
        
        fig1 = px.line(df_analysis, x='date', y=['close', 'closeusd'], title="Closing Price in Local Currency vs. USD")
        visualizations['price_local_vs_usd_line'] = fig1.to_json()
        
        fig2 = px.line(df_analysis, x='date', y='usd_to_local_rate', title="Implied USD to Local Currency Exchange Rate Over Time")
        visualizations['implied_exchange_rate_line'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_credit_score_factor_analysis(df):
    analysis_name = "customer_credit_score_factor_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Credit Score Factor Analysis"]
        
        expected = ['age', 'gender', 'income', 'education', 'marital_status', 'number_of_children', 'home_ownership', 'credit_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['age', 'income', 'number_of_children', 'credit_score']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['average_credit_score'] = df_analysis['credit_score'].mean()
        metrics['income_score_correlation'] = df_analysis['income'].corr(df_analysis['credit_score'])
        metrics['age_score_correlation'] = df_analysis['age'].corr(df_analysis['credit_score'])
        
        insights.append(f"Average Credit Score: {metrics['average_credit_score']:.0f}")
        insights.append(f"Income/Score Correlation: {metrics['income_score_correlation']:.2f}")
        insights.append(f"Age/Score Correlation: {metrics['age_score_correlation']:.2f}")
        
        fig1 = px.scatter(df_analysis, x='income', y='credit_score', color='home_ownership', title="Credit Score vs. Income by Home Ownership")
        visualizations['score_vs_income_scatter'] = fig1.to_json()
        
        fig2 = px.box(df_analysis, x='education', y='credit_score', title="Credit Score Distribution by Education Level")
        visualizations['score_by_education_boxplot'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_application_approval_prediction_analysis(df):
    analysis_name = "loan_application_approval_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Application Approval Prediction Analysis"]
        
        expected = ['loan_amount', 'property_value', 'income', 'credit_score', 'ltv', 'dtir1', 'status']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['loan_amount', 'property_value', 'income', 'credit_score', 'ltv', 'dtir1']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        if df_analysis['status'].dtype != 'object':
             # Assuming 1 is approved, 0 is denied
            df_analysis['status_label'] = df_analysis['status'].map({1: 'Approved', 0: 'Denied'})
        else:
             # Find the most common approval string
            approve_str = df_analysis['status'].mode()[0]
            df_analysis['status_label'] = df_analysis['status'].apply(lambda x: 'Approved' if x == approve_str else 'Denied')

        df_analysis.dropna(inplace=True)

        metrics['approval_rate_perc'] = (df_analysis['status_label'] == 'Approved').mean() * 100
        metrics['avg_credit_score_approved'] = df_analysis[df_analysis['status_label'] == 'Approved']['credit_score'].mean()
        metrics['avg_ltv_approved'] = df_analysis[df_analysis['status_label'] == 'Approved']['ltv'].mean()
        
        insights.append(f"Overall Approval Rate: {metrics['approval_rate_perc']:.2f}%")
        insights.append(f"Avg. Credit Score (Approved): {metrics['avg_credit_score_approved']:.0f}")
        insights.append(f"Avg. LTV (Approved): {metrics['avg_ltv_approved']:.1f}%")
        
        fig1 = px.box(df_analysis, x='status_label', y='credit_score', color='status_label', title="Credit Score Distribution by Loan Status")
        visualizations['score_by_status_boxplot'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='income', y='loan_amount', color='status_label', title="Loan Amount vs. Applicant Income by Status", labels={'income': 'Income', 'loan_amount': 'Loan Amount'})
        visualizations['loan_vs_income_by_status_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def banknote_authentication_analysis(df):
    analysis_name = "banknote_authentication_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Banknote Authentication Analysis"]
        
        expected = ['variance', 'skewness', 'curtosis', 'entropy', 'class']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in expected:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        df_analysis['class_label'] = df_analysis['class'].map({0: 'Genuine', 1: 'Forged'})
        
        metrics['forgery_rate_perc'] = df_analysis['class'].mean() * 100
        metrics['avg_variance_forged'] = df_analysis[df_analysis['class'] == 1]['variance'].mean()
        metrics['avg_entropy_genuine'] = df_analysis[df_analysis['class'] == 0]['entropy'].mean()
        
        insights.append(f"Forgery Rate: {metrics['forgery_rate_perc']:.2f}%")
        insights.append(f"Avg. Variance (Forged): {metrics['avg_variance_forged']:.2f}")
        insights.append(f"Avg. Entropy (Genuine): {metrics['avg_entropy_genuine']:.2f}")
        
        fig1 = px.scatter(df_analysis, x='skewness', y='curtosis', color='class_label', title="Skewness vs. Curtosis of Wavelet Transforms")
        visualizations['skewness_vs_curtosis_scatter'] = fig1.to_json()
        
        fig2 = px.scatter_matrix(df_analysis, dimensions=['variance', 'skewness', 'curtosis', 'entropy'], color='class_label', title="Scatter Matrix of Banknote Features")
        visualizations['feature_scatter_matrix'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_response_prediction_for_marketing_campaign(df):
    analysis_name = "customer_response_prediction_for_marketing_campaign"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Response Prediction for Marketing Campaign"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'poutcome', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['subscription_rate_perc'] = df_analysis['subscribed'].mean() * 100
        insights.append(f"Overall Subscription Rate: {metrics['subscription_rate_perc']:.2f}%")
        
        conversion_by_job = df_analysis.groupby('job')['subscribed'].mean().mul(100).sort_values().reset_index()
        metrics['subscription_by_job'] = conversion_by_job.to_dict('records')
        fig1 = px.bar(conversion_by_job, x='subscribed', y='job', orientation='h', title="Subscription Rate by Job Title")
        visualizations['subscription_by_job_bar'] = fig1.to_json()
        
        conversion_by_marital = df_analysis.groupby('marital')['subscribed'].mean().mul(100).reset_index()
        metrics['subscription_by_marital'] = conversion_by_marital.to_dict('records')
        fig2 = px.pie(conversion_by_marital, names='marital', values='subscribed', title="Subscription Rate by Marital Status", hole=0.4)
        visualizations['subscription_by_marital_pie'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def international_banking_statistics_and_cross_border_claims_analysis(df):
    analysis_name = "international_banking_statistics_and_cross_border_claims_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["International Banking Statistics and Cross-Border Claims Analysis"]
        
        expected = ['l_rep_cty', 'l_cp_country', '2022_q1', '2022_q2', '2022_q3', '2022_q4']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        id_vars = [col for col in df_analysis.columns if not col.startswith(('1', '2')) and col in rename_map.values()]
        value_vars = [col for col in df_analysis.columns if col.startswith(('1', '2'))]
        
        df_long = df_analysis.melt(id_vars=id_vars, value_vars=value_vars, var_name='quarter', value_name='value')
        df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce').dropna()
        
        metrics['total_claim_value_millions'] = df_long['value'].sum()
        metrics['top_reporting_country'] = df_long.groupby('l_rep_cty')['value'].sum().idxmax()
        metrics['top_counterparty_country'] = df_long.groupby('l_cp_country')['value'].sum().idxmax()
        
        insights.append(f"Total Value of Claims: ${metrics['total_claim_value_millions']:,.0f}M")
        insights.append(f"Top Reporting Country: {metrics['top_reporting_country']}")
        insights.append(f"Top Counterparty Country: {metrics['top_counterparty_country']}")
        
        value_by_quarter = df_long.groupby('quarter')['value'].sum().reset_index()
        metrics['value_by_quarter'] = value_by_quarter.to_dict('records')
        fig1 = px.bar(value_by_quarter, x='quarter', y='value', title="Total Claim Value by Quarter")
        visualizations['value_by_quarter_bar'] = fig1.to_json()
        
        top_reporting = df_long.groupby('l_rep_cty')['value'].sum().nlargest(10).reset_index()
        top_reporting.columns = ['country', 'value']
        metrics['top_10_reporting_countries'] = top_reporting.to_dict('records')
        fig2 = px.bar(top_reporting, x='country', y='value', title="Top 10 Reporting Countries by Claim Value")
        visualizations['top_reporting_countries_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_account_transaction_pattern_analysis(df):
    analysis_name = "customer_account_transaction_pattern_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Account Transaction Pattern Analysis"]
        
        expected = ['account_id', 'transaction_type', 'amount', 'balance', 'transaction_date']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['amount', 'balance']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['average_transaction_amount'] = df_analysis['amount'].mean()
        metrics['most_frequent_txn_type'] = df_analysis['transaction_type'].mode()[0]
        metrics['number_of_accounts'] = df_analysis['account_id'].nunique()
        
        insights.append(f"Average Transaction Amount: ${metrics['average_transaction_amount']:,.2f}")
        insights.append(f"Most Frequent Txn Type: {metrics['most_frequent_txn_type']}")
        insights.append(f"Number of Accounts: {metrics['number_of_accounts']:,}")
        
        if 'transaction_date' in df_analysis.columns:
            df_analysis['transaction_date'] = pd.to_datetime(df_analysis['transaction_date'], errors='coerce')
            df_analysis = df_analysis.sort_values('transaction_date')
            monthly_sum = df_analysis.resample('M', on='transaction_date')['amount'].sum().reset_index()
            fig1 = px.line(monthly_sum, x='transaction_date', y='amount', title="Total Transaction Amount Over Time")
            visualizations['amount_over_time_line'] = fig1.to_json()
        else:
            insights.append("No 'transaction_date' column found for time series analysis.")
            
        txn_type_dist = df_analysis['transaction_type'].value_counts().reset_index()
        txn_type_dist.columns = ['type', 'count']
        metrics['transaction_type_distribution'] = txn_type_dist.to_dict('records')
        fig2 = px.bar(txn_type_dist, x='type', y='count', title="Distribution of Transaction Types")
        visualizations['transaction_type_distribution_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def telemarketing_campaign_outcome_analysis(df):
    analysis_name = "telemarketing_campaign_outcome_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Telemarketing Campaign Outcome Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'duration', 'campaign', 'poutcome', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
            
        df_analysis['duration'] = pd.to_numeric(df_analysis['duration'], errors='coerce')
        df_analysis['campaign'] = pd.to_numeric(df_analysis['campaign'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['conversion_rate_perc'] = df_analysis['subscribed'].mean() * 100
        metrics['avg_duration_subscribed_sec'] = df_analysis[df_analysis['subscribed'] == 1]['duration'].mean()
        metrics['avg_duration_not_subscribed_sec'] = df_analysis[df_analysis['subscribed'] == 0]['duration'].mean()
        
        insights.append(f"Conversion Rate: {metrics['conversion_rate_perc']:.2f}%")
        insights.append(f"Avg. Call Duration (Subscribed): {metrics['avg_duration_subscribed_sec']:.0f}s")
        insights.append(f"Avg. Call Duration (Not Subscribed): {metrics['avg_duration_not_subscribed_sec']:.0f}s")
        
        fig1 = px.box(df_analysis, x='subscribed', y='duration', title="Call Duration by Subscription Outcome")
        visualizations['duration_by_subscription_boxplot'] = fig1.to_json()
        
        conversion_by_month = df_analysis.groupby('month')['subscribed'].mean().mul(100).reset_index()
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        conversion_by_month['month'] = pd.Categorical(conversion_by_month['month'], categories=month_order, ordered=True)
        conversion_by_month = conversion_by_month.sort_values('month')
        metrics['conversion_by_month'] = conversion_by_month.to_dict('records')
        
        fig2 = px.bar(conversion_by_month, x='month', y='subscribed', title="Conversion Rate by Month")
        visualizations['conversion_by_month_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_term_deposit_subscription_analysis(df):
    analysis_name = "bank_term_deposit_subscription_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Term Deposit Subscription Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['subscription_rate_perc'] = df_analysis['subscribed'].mean() * 100
        insights.append(f"Subscription Rate: {metrics['subscription_rate_perc']:.2f}%")
        
        rate_by_housing = df_analysis.groupby('housing')['subscribed'].mean().mul(100).reset_index()
        metrics['subscription_by_housing_loan'] = rate_by_housing.to_dict('records')
        fig1 = px.pie(rate_by_housing, names='housing', values='subscribed', title="Subscription Rate by Housing Loan Status")
        visualizations['subscription_by_housing_pie'] = fig1.to_json()
        
        fig2 = px.histogram(df_analysis, x='age', color='y', barmode='overlay', title="Age Distribution by Subscription Status")
        visualizations['age_distribution_by_subscription_histogram'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def millennial_banking_preferences_survey_analysis(df):
    analysis_name = "millennial_banking_preferences_survey_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Millennial Banking Preferences Survey Analysis"]
        
        expected = ['how_old_are_you', 'i_know_the_difference_between_a_bank_and_a_credit_union', 'what_makes_you_consider_a_financial_institution_worthwhile', 'what_are_your_top_2_preferences_when_choosing_a_financial_institution']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        metrics['knows_difference_perc'] = (df_analysis['i_know_the_difference_between_a_bank_and_a_credit_union'].str.lower() == 'yes').mean() * 100
        insights.append(f"Knows Difference Between Bank & Credit Union: {metrics['knows_difference_perc']:.1f}%")
        
        age_dist = df_analysis['how_old_are_you'].value_counts().reset_index()
        age_dist.columns = ['age_group', 'count']
        metrics['age_distribution'] = age_dist.to_dict('records')
        fig1 = px.bar(age_dist, x='age_group', y='count', title="Age Distribution of Survey Respondents")
        visualizations['age_distribution_bar'] = fig1.to_json()
        
        insights.append("What Makes a Financial Institution Worthwhile? (Top 5 Responses)")
        worthwhile_responses = df_analysis['what_makes_you_consider_a_financial_institution_worthwhile'].value_counts().nlargest(5).reset_index()
        worthwhile_responses.columns = ['response', 'count']
        metrics['top_worthwhile_responses'] = worthwhile_responses.to_dict('records')
        
        insights.append("Cannot generate word cloud as 'wordcloud' and 'matplotlib' are not allowed imports.")
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_direct_marketing_success_prediction_analysis(df):
    analysis_name = "bank_direct_marketing_success_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Direct Marketing Success Prediction Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'loan', 'contact', 'month', 'duration', 'campaign', 'emp_var_rate', 'cons_price_idx', 'cons_conf_idx', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
            
        for col in ['emp_var_rate', 'cons_price_idx', 'cons_conf_idx']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['subscription_rate_perc'] = df_analysis['subscribed'].mean() * 100
        insights.append(f"Subscription Rate: {metrics['subscription_rate_perc']:.2f}%")
        
        fig1 = px.box(df_analysis, x='y', y='cons_conf_idx', title="Consumer Confidence Index by Subscription Outcome")
        visualizations['consumer_confidence_by_subscription_boxplot'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='cons_price_idx', y='emp_var_rate', color='y', title="Subscription Outcome by Economic Indicators", labels={'cons_price_idx': 'Consumer Price Index', 'emp_var_rate': 'Employment Variation Rate'})
        visualizations['economic_indicators_by_subscription_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_segmentation_and_product_limit_analysis(df):
    analysis_name = "customer_segmentation_and_product_limit_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Segmentation and Product Limit Analysis"]
        
        expected = ['age', 'city', 'product', 'limit', 'company', 'segment']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['limit'] = pd.to_numeric(df_analysis['limit'], errors='coerce')
        df_analysis['age'] = pd.to_numeric(df_analysis['age'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['average_product_limit'] = df_analysis['limit'].mean()
        metrics['largest_customer_segment'] = df_analysis['segment'].mode()[0]
        
        insights.append(f"Average Product Limit: ${metrics['average_product_limit']:,.0f}")
        insights.append(f"Largest Customer Segment: {metrics['largest_customer_segment']}")
        
        fig1 = px.box(df_analysis, x='segment', y='limit', color='product', title="Product Limit by Customer Segment and Product")
        visualizations['limit_by_segment_product_boxplot'] = fig1.to_json()
        
        segment_dist = df_analysis['segment'].value_counts().reset_index()
        segment_dist.columns = ['segment', 'count']
        metrics['segment_distribution'] = segment_dist.to_dict('records')
        fig2 = px.pie(segment_dist, names='segment', values='count', title="Customer Segment Distribution")
        visualizations['segment_distribution_pie'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def credit_risk_classification_analysis(df):
    analysis_name = "credit_risk_classification_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Credit Risk Classification Analysis"]
        
        expected = ['checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount', 'savings_status', 'employment', 'age', 'housing', 'class']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['class'].dtype == 'object':
            df_analysis['risk_label'] = df_analysis['class'].apply(lambda x: 'Bad' if x.lower() in ['bad', '2'] else 'Good')
        else:
            df_analysis['risk_label'] = df_analysis['class'].apply(lambda x: 'Bad' if x == 2 else 'Good') # Assuming German Credit: 1=Good, 2=Bad
            
        for col in ['duration', 'credit_amount', 'age']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['bad_risk_rate_perc'] = (df_analysis['risk_label'] == 'Bad').mean() * 100
        metrics['avg_credit_amount_bad_risk'] = df_analysis[df_analysis['risk_label'] == 'Bad']['credit_amount'].mean()
        
        insights.append(f"Bad Risk Rate: {metrics['bad_risk_rate_perc']:.2f}%")
        insights.append(f"Avg. Credit Amount (Bad Risk): {metrics['avg_credit_amount_bad_risk']:,.0f}")
        
        fig1 = px.box(df_analysis, x='risk_label', y='credit_amount', color='housing', title="Credit Amount by Risk Status and Housing Type")
        visualizations['credit_amount_by_risk_housing_boxplot'] = fig1.to_json()
        
        risk_by_purpose = df_analysis.groupby('purpose')['risk_label'].apply(lambda x: (x == 'Bad').mean()).mul(100).sort_values(ascending=False).reset_index()
        risk_by_purpose.columns = ['purpose', 'bad_risk_rate']
        metrics['risk_by_purpose'] = risk_by_purpose.to_dict('records')
        fig2 = px.bar(risk_by_purpose, x='purpose', y='bad_risk_rate', title="Bad Risk Rate by Loan Purpose")
        visualizations['risk_by_purpose_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_application_status_prediction_analysis(df):
    analysis_name = "loan_application_status_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Application Status Prediction Analysis"]
        
        expected = ['gender', 'married', 'dependents', 'education', 'self_employed', 'applicantincome', 'coapplicantincome', 'loanamount', 'loan_amount_term', 'credit_history', 'property_area', 'loan_status']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['loan_status'].dtype == 'object':
            df_analysis['approved'] = df_analysis['loan_status'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
        else:
            df_analysis['approved'] = pd.to_numeric(df_analysis['loan_status'], errors='coerce')
            
        for col in ['applicantincome', 'coapplicantincome', 'loanamount', 'credit_history']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['overall_approval_rate_perc'] = df_analysis['approved'].mean() * 100
        metrics['approval_with_credit_history_perc'] = df_analysis[df_analysis['credit_history'] == 1.0]['approved'].mean() * 100
        metrics['approval_no_credit_history_perc'] = df_analysis[df_analysis['credit_history'] == 0.0]['approved'].mean() * 100
        
        insights.append(f"Overall Approval Rate: {metrics['overall_approval_rate_perc']:.1f}%")
        insights.append(f"Approval Rate (with History): {metrics['approval_with_credit_history_perc']:.1f}%")
        insights.append(f"Approval Rate (no History): {metrics['approval_no_credit_history_perc']:.1f}%")
        
        fig1 = px.histogram(df_analysis, x='applicantincome', color='loan_status', barmode='overlay', title="Applicant Income Distribution by Loan Status")
        visualizations['income_distribution_by_status_histogram'] = fig1.to_json()
        
        approval_by_prop_area = df_analysis.groupby('property_area')['approved'].mean().mul(100).reset_index()
        metrics['approval_by_property_area'] = approval_by_prop_area.to_dict('records')
        fig2 = px.bar(approval_by_prop_area, x='property_area', y='approved', title="Approval Rate by Property Area")
        visualizations['approval_by_property_area_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_customer_attrition_analysis(df):
    analysis_name = "bank_customer_attrition_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Customer Attrition Analysis"]
        
        expected = ['customerid', 'surname', 'creditscore', 'geography', 'gender', 'age', 'tenure', 'balance', 'numofproducts', 'hascrcard', 'isactivemember', 'estimatedsalary', 'exited']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['creditscore', 'age', 'tenure', 'balance', 'numofproducts', 'estimatedsalary', 'exited']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['attrition_rate_perc'] = df_analysis['exited'].mean() * 100
        metrics['avg_balance_leavers'] = df_analysis[df_analysis['exited'] == 1]['balance'].mean()
        metrics['avg_salary_leavers'] = df_analysis[df_analysis['exited'] == 1]['estimatedsalary'].mean()
        
        insights.append(f"Attrition Rate: {metrics['attrition_rate_perc']:.2f}%")
        insights.append(f"Avg. Balance (Leavers): ${metrics['avg_balance_leavers']:,.0f}")
        insights.append(f"Avg. Salary (Leavers): ${metrics['avg_salary_leavers']:,.0f}")
        
        fig1 = px.violin(df_analysis, x='geography', y='estimatedsalary', color='exited', box=True, points="all", title="Salary Distribution by Geography and Attrition Status")
        visualizations['salary_by_geography_attrition_violin'] = fig1.to_json()
        
        fig2 = px.density_heatmap(df_analysis, x="age", y="balance", z="exited", histfunc="avg", title="Heatmap of Attrition Rate by Age and Balance")
        visualizations['attrition_by_age_balance_heatmap'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_approval_status_prediction_analysis(df):
    analysis_name = "loan_approval_status_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Approval Status Prediction Analysis"]
        
        expected = ['applicant_name', 'loan_amount_usd', 'annual_income_usd', 'credit_score', 'approval_status']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['loan_amount_usd', 'annual_income_usd', 'credit_score']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        if df_analysis['approval_status'].dtype == 'object':
            approved_val = df_analysis['approval_status'].mode()[0]
            metrics['approval_rate_perc'] = (df_analysis['approval_status'] == approved_val).mean() * 100
        else:
            metrics['approval_rate_perc'] = (df_analysis['approval_status'] == 1).mean() * 100 # Assuming 1 = Approved
            
        insights.append(f"Overall Approval Rate: {metrics['approval_rate_perc']:.1f}%")
        
        fig1 = px.box(df_analysis, x='approval_status', y='credit_score', color='approval_status', title="Credit Score Distribution by Approval Status")
        visualizations['score_by_approval_status_boxplot'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='annual_income_usd', y='loan_amount_usd', color='approval_status', title="Loan Amount vs. Annual Income by Approval Status", log_x=True, log_y=True)
        visualizations['loan_vs_income_by_status_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def credit_risk_and_loan_repayment_analysis(df):
    analysis_name = "credit_risk_and_loan_repayment_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Credit Risk and Loan Repayment Analysis"]
        
        expected = ['credit_policy', 'purpose', 'int_rate', 'installment', 'log_annual_inc', 'dti', 'fico', 'revol_bal', 'inq_last_6mths', 'not_fully_paid']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in expected:
            if col != 'purpose':
                df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['not_fully_paid_rate_perc'] = df_analysis['not_fully_paid'].mean() * 100
        metrics['avg_fico_not_paid'] = df_analysis[df_analysis['not_fully_paid'] == 1]['fico'].mean()
        metrics['avg_fico_paid'] = df_analysis[df_analysis['not_fully_paid'] == 0]['fico'].mean()
        
        insights.append(f"Not Fully Paid Rate: {metrics['not_fully_paid_rate_perc']:.2f}%")
        insights.append(f"Avg. FICO (Not Paid): {metrics['avg_fico_not_paid']:.0f}")
        insights.append(f"Avg. FICO (Paid): {metrics['avg_fico_paid']:.0f}")
        
        fig1 = px.histogram(df_analysis, x='fico', color='not_fully_paid', barmode='overlay', title="FICO Score Distribution by Repayment Status")
        visualizations['fico_distribution_by_repayment_histogram'] = fig1.to_json()
        
        not_paid_by_purpose = df_analysis.groupby('purpose')['not_fully_paid'].mean().mul(100).sort_values(ascending=False).reset_index()
        metrics['default_by_purpose'] = not_paid_by_purpose.to_dict('records')
        fig2 = px.bar(not_paid_by_purpose, x='purpose', y='not_fully_paid', title="Default Rate by Loan Purpose")
        visualizations['default_by_purpose_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def direct_marketing_campaign_outcome_analysis(df):
    analysis_name = "direct_marketing_campaign_outcome_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Direct Marketing Campaign Outcome Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'poutcome']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        metrics['previous_success_rate_perc'] = (df_analysis['poutcome'] == 'success').mean() * 100
        metrics['previous_failure_rate_perc'] = (df_analysis['poutcome'] == 'failure').mean() * 100
        
        insights.append(f"Success Rate of Previous Campaign: {metrics['previous_success_rate_perc']:.1f}%")
        insights.append(f"Failure Rate of Previous Campaign: {metrics['previous_failure_rate_perc']:.1f}%")
        
        outcome_dist = df_analysis['poutcome'].value_counts().reset_index()
        outcome_dist.columns = ['outcome', 'count']
        metrics['previous_outcome_distribution'] = outcome_dist.to_dict('records')
        
        fig1 = px.pie(outcome_dist, names='outcome', values='count', title="Distribution of Previous Campaign Outcomes")
        visualizations['previous_outcome_pie'] = fig1.to_json()
        
        df_analysis['age'] = pd.to_numeric(df_analysis['age'], errors='coerce')
        df_analysis['balance'] = pd.to_numeric(df_analysis['balance'], errors='coerce')
        
        fig2 = px.density_heatmap(df_analysis, x="age", y="balance", z="poutcome", histfunc="count", facet_col="poutcome", title="Age-Balance Distribution by Previous Outcome")
        visualizations['age_balance_by_outcome_heatmap'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def loan_default_prediction_analysis_based_on_employment_and_balance(df):
    analysis_name = "loan_default_prediction_analysis_based_on_employment_and_balance"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Loan Default Prediction Analysis (Employment & Balance)"]
        
        expected = ['employed', 'bank_balance', 'annual_salary', 'defaulted']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in expected:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['overall_default_rate_perc'] = df_analysis['defaulted'].mean() * 100
        insights.append(f"Overall Default Rate: {metrics['overall_default_rate_perc']:.2f}%")
        
        fig1 = px.scatter(df_analysis, x='bank_balance', y='annual_salary', color='defaulted', title="Default Status by Bank Balance and Annual Salary")
        visualizations['balance_vs_salary_by_default_scatter'] = fig1.to_json()
        
        fig2 = px.box(df_analysis, x='employed', y='bank_balance', color='defaulted', title="Bank Balance by Employment and Default Status")
        visualizations['balance_by_employment_default_boxplot'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_churn_analysis_based_on_credit_score_and_demographics(df):
    analysis_name = "customer_churn_analysis_based_on_credit_score_and_demographics"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Churn Analysis (Credit Score & Demographics)"]
        
        expected = ['customer_id', 'credit_score', 'country', 'gender', 'age', 'tenure', 'balance', 'churn']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['credit_score', 'age', 'tenure', 'balance', 'churn']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['overall_churn_rate_perc'] = df_analysis['churn'].mean() * 100
        insights.append(f"Overall Churn Rate: {metrics['overall_churn_rate_perc']:.2f}%")
        
        churn_by_country = df_analysis.groupby('country')['churn'].mean().mul(100).reset_index()
        metrics['churn_by_country'] = churn_by_country.to_dict('records')
        fig1 = px.bar(churn_by_country, x='country', y='churn', title="Churn Rate by Country")
        visualizations['churn_by_country_bar'] = fig1.to_json()
        
        fig2 = px.violin(df_analysis, x='gender', y='age', color='churn', box=True, title="Age Distribution by Gender and Churn Status")
        visualizations['age_by_gender_churn_violin'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_creditworthiness_assessment_analysis(df):
    analysis_name = "customer_creditworthiness_assessment_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Creditworthiness Assessment Analysis"]
        
        expected = ['creditability', 'account_balance', 'duration_of_credit_monthly', 'payment_status_of_previous_credit', 'purpose', 'credit_amount', 'value_savings_stocks', 'age_years', 'occupation']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['creditability', 'duration_of_credit_monthly', 'credit_amount', 'age_years']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        # Assuming 1 = Good Credit
        metrics['good_creditability_rate_perc'] = df_analysis['creditability'].mean() * 100
        insights.append(f"Good Creditability Rate: {metrics['good_creditability_rate_perc']:.1f}%")
        
        fig1 = px.box(df_analysis, x='creditability', y='credit_amount', title="Credit Amount by Creditability")
        visualizations['amount_by_creditability_boxplot'] = fig1.to_json()
        
        credit_by_purpose = df_analysis.groupby('purpose')['creditability'].mean().mul(100).sort_values().reset_index()
        metrics['creditability_by_purpose'] = credit_by_purpose.to_dict('records')
        fig2 = px.bar(credit_by_purpose, x='purpose', y='creditability', title="Good Credit Rate by Loan Purpose")
        visualizations['creditability_by_purpose_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def financial_product_subscription_likelihood_analysis(df):
    analysis_name = "financial_product_subscription_likelihood_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Financial Product Subscription Likelihood Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['subscription_rate_perc'] = df_analysis['subscribed'].mean() * 100
        insights.append(f"Subscription Rate: {metrics['subscription_rate_perc']:.2f}%")
        
        rate_by_education = df_analysis.groupby('education')['subscribed'].mean().mul(100).reset_index()
        metrics['subscription_by_education'] = rate_by_education.to_dict('records')
        fig1 = px.bar(rate_by_education, x='education', y='subscribed', title="Subscription Rate by Education Level")
        visualizations['subscription_by_education_bar'] = fig1.to_json()
        
        pivot = df_analysis.pivot_table(index='housing', columns='loan', values='subscribed', aggfunc='mean') * 100
        metrics['subscription_by_housing_loan_pivot'] = pivot.to_dict()
        fig2 = px.imshow(pivot, text_auto=True, aspect="auto", title="Subscription Rate (%) by Housing and Personal Loan Status")
        visualizations['subscription_by_housing_loan_heatmap'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def debt_recovery_strategy_effectiveness_analysis(df):
    analysis_name = "debt_recovery_strategy_effectiveness_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Debt Recovery Strategy Effectiveness Analysis"]
        
        expected = ['expected_recovery_amount', 'actual_recovery_amount', 'recovery_strategy', 'age', 'sex']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['expected_recovery_amount', 'actual_recovery_amount', 'age']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        df_analysis['recovery_rate'] = (df_analysis['actual_recovery_amount'] / df_analysis['expected_recovery_amount']).clip(0, 2) # Cap rate at 200%
        
        metrics['average_recovery_rate_perc'] = df_analysis['recovery_rate'].mean() * 100
        strategy_performance = df_analysis.groupby('recovery_strategy')['recovery_rate'].mean().sort_values(ascending=False)
        metrics['best_strategy'] = strategy_performance.idxmax()
        metrics['strategy_performance'] = strategy_performance.to_dict()
        
        insights.append(f"Average Recovery Rate: {metrics['average_recovery_rate_perc']:.2f}%")
        insights.append(f"Best Performing Strategy: {metrics['best_strategy']}")
        
        fig1 = px.box(df_analysis, x='recovery_strategy', y='recovery_rate', title="Recovery Rate by Strategy")
        visualizations['recovery_rate_by_strategy_boxplot'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='age', y='recovery_rate', color='sex', title="Recovery Rate by Age and Sex", trendline='ols')
        visualizations['recovery_rate_by_age_sex_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def general_ledger_journal_voucher_transaction_analysis(df):
    analysis_name = "general_ledger_journal_voucher_transaction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["General Ledger Journal Voucher Transaction Analysis"]
        
        expected = ['journal_voucher_item_amount', 'credit_debit_code', 'fiscal_month', 'departmentnumber', 'general_ledger_account_code']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        df_analysis['journal_voucher_item_amount'] = pd.to_numeric(df_analysis['journal_voucher_item_amount'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['total_debit_amount'] = df_analysis[df_analysis['credit_debit_code'].str.upper() == 'D']['journal_voucher_item_amount'].sum()
        metrics['total_credit_amount'] = df_analysis[df_analysis['credit_debit_code'].str.upper() == 'C']['journal_voucher_item_amount'].sum()
        metrics['net_difference'] = metrics['total_debit_amount'] - metrics['total_credit_amount']
        
        insights.append(f"Total Debit Amount: ${metrics['total_debit_amount']:,.2f}")
        insights.append(f"Total Credit Amount: ${metrics['total_credit_amount']:,.2f}")
        insights.append(f"Net Difference (Debit - Credit): ${metrics['net_difference']:,.2f}")
        
        amount_by_dept = df_analysis.groupby('departmentnumber')['journal_voucher_item_amount'].sum().nlargest(15).reset_index()
        amount_by_dept.columns = ['department', 'amount']
        metrics['top_15_departments_by_amount'] = amount_by_dept.to_dict('records')
        fig1 = px.bar(amount_by_dept, x='department', y='amount', title="Top 15 Departments by Transaction Amount")
        visualizations['top_departments_bar'] = fig1.to_json()
        
        amount_by_gl = df_analysis.groupby('general_ledger_account_code')['journal_voucher_item_amount'].sum().nlargest(15).reset_index()
        amount_by_gl.columns = ['gl_account', 'amount']
        metrics['top_15_gl_accounts_by_amount'] = amount_by_gl.to_dict('records')
        fig2 = px.bar(amount_by_gl, x='gl_account', y='amount', title="Top 15 GL Accounts by Transaction Amount")
        visualizations['top_gl_accounts_bar'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def customer_credit_risk_assessment(df):
    analysis_name = "customer_credit_risk_assessment"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Customer Credit Risk Assessment"]
        
        expected = ['age', 'sex', 'job', 'housing', 'saving_accounts', 'checking_account', 'credit_amount', 'duration', 'purpose', 'risk']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        for col in ['age', 'credit_amount', 'duration']:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['bad_risk_rate_perc'] = (df_analysis['risk'] == 'bad').mean() * 100
        insights.append(f"Bad Risk Rate: {metrics['bad_risk_rate_perc']:.2f}%")
        
        risk_by_job = df_analysis.groupby('job')['risk'].apply(lambda x: (x == 'bad').mean()).mul(100).reset_index()
        risk_by_job.columns = ['job', 'risk_rate']
        metrics['risk_by_job'] = risk_by_job.to_dict('records')
        fig1 = px.bar(risk_by_job, x='job', y='risk_rate', title="Bad Risk Rate by Job Type")
        visualizations['risk_by_job_bar'] = fig1.to_json()
        
        fig2 = px.scatter(df_analysis, x='age', y='credit_amount', color='risk', facet_col='sex', title="Credit Amount vs. Age by Risk and Sex")
        visualizations['amount_vs_age_by_risk_sex_scatter'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def call_center_campaign_effectiveness_analysis(df):
    analysis_name = "call_center_campaign_effectiveness_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Call Center Campaign Effectiveness Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'loan', 'month', 'duration', 'campaign', 'poutcome', 'y']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['y'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['y'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['y'], errors='coerce')
            
        df_analysis['duration'] = pd.to_numeric(df_analysis['duration'], errors='coerce')
        df_analysis['campaign'] = pd.to_numeric(df_analysis['campaign'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['conversion_rate_perc'] = df_analysis['subscribed'].mean() * 100
        metrics['average_contacts_per_person'] = df_analysis['campaign'].mean()
        
        insights.append(f"Conversion Rate: {metrics['conversion_rate_perc']:.2f}%")
        insights.append(f"Average Contacts per Person: {metrics['average_contacts_per_person']:.2f}")
        
        conversion_by_contacts = df_analysis.groupby('campaign')['subscribed'].mean().mul(100).reset_index()
        metrics['conversion_by_contacts'] = conversion_by_contacts.to_dict('records')
        fig1 = px.bar(conversion_by_contacts, x='campaign', y='subscribed', title="Conversion Rate by Number of Contacts in this Campaign")
        visualizations['conversion_by_contacts_bar'] = fig1.to_json()
        
        fig2 = px.box(df_analysis, x='poutcome', y='duration', color='y', title="Call Duration by Subscription Outcome and Previous Outcome")
        visualizations['duration_by_subscription_poutcome_boxplot'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }

def bank_deposit_subscription_prediction_analysis(df):
    analysis_name = "bank_deposit_subscription_prediction_analysis"
    try:
        metrics = {}
        visualizations = {}
        insights = ["Bank Deposit Subscription Prediction Analysis"]
        
        expected = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'deposit']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched[col] is None]
        if missing:
            message = create_missing_columns_message(missing, matched)
            return get_general_insights(df, analysis_name, message)
            
        df_analysis = df.copy()
        rename_map = {v: k for k, v in matched.items() if v}
        df_analysis = df_analysis.rename(columns=rename_map)
        
        if df_analysis['deposit'].dtype == 'object':
            df_analysis['subscribed'] = df_analysis['deposit'].apply(lambda x: 1 if x.lower() == 'yes' else 0)
        else:
            df_analysis['subscribed'] = pd.to_numeric(df_analysis['deposit'], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        metrics['subscription_rate_perc'] = df_analysis['subscribed'].mean() * 100
        insights.append(f"Deposit Subscription Rate: {metrics['subscription_rate_perc']:.2f}%")
        
        df_analysis['age'] = pd.to_numeric(df_analysis['age'], errors='coerce')
        df_analysis['balance'] = pd.to_numeric(df_analysis['balance'], errors='coerce')

        fig1 = px.density_heatmap(df_analysis, x="age", y="balance", z="subscribed", histfunc="avg", title="Heatmap of Subscription Rate by Age and Balance", labels={'balance': 'Account Balance'})
        visualizations['subscription_by_age_balance_heatmap'] = fig1.to_json()
        
        conversion_by_contact = df_analysis.groupby('contact')['subscribed'].mean().mul(100).reset_index()
        metrics['subscription_by_contact_method'] = conversion_by_contact.to_dict('records')
        fig2 = px.pie(conversion_by_contact, names='contact', values='subscribed', title="Subscription Rate by Contact Method", hole=0.4)
        visualizations['subscription_by_contact_pie'] = fig2.to_json()
            
        return {
            "analysis_type": analysis_name,
            "status": "success",
            "matched_columns": matched,
            "metrics": clean_metrics(metrics),
            "visualizations": visualizations,
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "message": str(e),
            "matched_columns": matched if 'matched' in locals() else {},
            "metrics": {}, "visualizations": {}, "insights": [str(e)]
        }


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
            return matched_func(df)
        except Exception as e:
            return {
                "analysis_type": analysis_name,
                "status": "error",
                "message": f"An unexpected error occurred in {analysis_name}: {e}",
                "matched_columns": {},
                "metrics": {},
                "visualizations": {},
                "insights": [f"An unexpected error occurred: {str(e)}"]
            }
    else:
        fallback_message = f"WARNING: Analysis function for '{analysis_name}' not found. Showing general insights instead."
        return get_general_insights(df, "unknown_analysis", fallback_message)

def main():
    """
    Main function to load data and run the analysis.
    User can change the file path and analysis type here.
    This function will now print the JSON output, simulating an API response.
    """
    print("🏦 Banking & Financial Analytics Dashboard")
    # --- CHANGE THESE VARIABLES TO MATCH YOUR FILE AND DESIRED ANALYSIS ---
    user_data_path = "your_data.csv"  # <--- Change this to your file path
    analysis_to_run = "loan_default_risk_prediction_analysis"  # <--- Change this to one of the analysis_options
    user_encoding = "utf-8"  # <--- Change this if your file has a different encoding

    # Check if user has updated the file path
    if user_data_path == "your_data.csv":
        print("\nPlease update the `user_data_path` variable with your data file's location.")
        print("Using a placeholder DataFrame for demonstration.")
        # Create a sample DataFrame for demonstration
        demo_data = {
            'income': np.random.randint(30000, 150000, 100),
            'age': np.random.randint(25, 65, 100),
            'loan': np.random.randint(10000, 500000, 100),
            'default': np.random.choice([0, 1], 100, p=[0.85, 0.15])
        }
        df = pd.DataFrame(demo_data)
        print("✅ Using demo data with loan default risk columns")
    else:
        df = load_data(user_data_path, encoding=user_encoding)
        if df is None:
            print("❌ Failed to load data. Exiting.")
            return
        print("✅ Data loaded successfully!")

    print(f"\n📊 Running analysis: {analysis_to_run}")
    print("=" * 50)
    
    # Run the analysis and get the JSON result
    result = run_analysis(df, analysis_to_run)
    
    # Print the JSON output (simulating API response)
    print("\n🎯 ANALYSIS RESULTS (JSON Format):")
    print("=" * 50)
    print(json.dumps(result, indent=2))
    
    # Print a summary for quick viewing
    print("\n📋 QUICK SUMMARY:")
    print("=" * 50)
    print(f"Analysis Type: {result.get('analysis_type', 'Unknown')}")
    print(f"Status: {result.get('status', 'Unknown')}")
    
    if result.get('status') == 'success':
        print("✅ Analysis completed successfully!")
        insights = result.get('insights', [])
        if insights:
            print("\n💡 Key Insights:")
            for i, insight in enumerate(insights[:5], 1):  # Show first 5 insights
                print(f"  {i}. {insight}")
        
        metrics = result.get('metrics', {})
        if metrics:
            print("\n📈 Key Metrics:")
            for key, value in list(metrics.items())[:5]:  # Show first 5 metrics
                print(f"  {key}: {value}")
                
        visualizations = result.get('visualizations', {})
        if visualizations:
            print(f"\n📊 Visualizations Generated: {len(visualizations)}")
            
    elif result.get('status') == 'error':
        print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
    elif result.get('status') == 'fallback':
        print("⚠️  Using fallback general analysis (specific columns not found)")
        insights = result.get('insights', [])
        if insights:
            print("\n💡 General Insights:")
            for i, insight in enumerate(insights[:3], 1):
                print(f"  {i}. {insight}")

    print("\n" + "=" * 50)
    print("🏁 Analysis complete!")

# This is the standard entry point for a Python script.
if __name__ == "__main__":
    main()