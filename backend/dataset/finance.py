import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go


# Analysis options (these are just string names, not directly used in the logic below this section)
analysis_options = [
    "financial_statements",
    "profitability_analysis",
    "cash_flow_analysis",
    "financial_ratios",
    "budget_vs_actual",
    "investment_analysis",
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


def show_missing_columns_warning(missing_cols, matched_cols=None):
    print("\n⚠️ Required Columns Not Found")
    print("The following columns are needed for this analysis but weren't found in your data:")
    for col in missing_cols:
        match_info = f" (matched to: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""
        print(f" - {col}{match_info}")

def show_general_insights(df, title="General Insights"):
    """Show general data visualizations"""
    print(f"\n--- {title} ---")

    show_key_metrics(df)

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\nNumeric Features Analysis")
        # For a non-Streamlit app, you might ask for user input or pick a default
        print("Available numeric features:")
        for i, col in enumerate(numeric_cols):
            print(f"{i}: {col}")
        selected_num_col_idx = int(input("Select numeric feature to analyze (enter index): "))
        selected_num_col = numeric_cols[selected_num_col_idx]

        fig1 = px.histogram(df, x=selected_num_col,
                                title=f"Distribution of {selected_num_col}")
        fig1.show()

        fig2 = px.box(df, y=selected_num_col,
                              title=f"Box Plot of {selected_num_col}")
        fig2.show()
    else:
        print("[WARNING] No numeric columns found for analysis.")

    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        print("\nFeature Correlations:")
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                                 title="Correlation Between Numeric Features")
        fig3.show()

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Features Analysis")
        print("Available categorical features:")
        for i, col in enumerate(categorical_cols):
            print(f"{i}: {col}")
        selected_cat_col_idx = int(input("Select categorical feature to analyze (enter index): "))
        selected_cat_col = categorical_cols[selected_cat_col_idx]

        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']

        fig4 = px.bar(value_counts.head(10), x='Value', y='Count',
                              title=f"Distribution of {selected_cat_col}")
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
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available)
        matched[target] = match if score >= 70 else None
    return matched

# ========== ANALYSIS FUNCTIONS ==========

def financial_statements(df):
    print("\n--- Financial Statements Analysis ---")
    expected = ['account', 'period', 'revenue', 'expenses',
                'profit', 'assets', 'liabilities', 'equity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates if needed
    if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

    # Calculate metrics
    total_revenue = df['revenue'].sum()
    total_expenses = df['expenses'].sum()
    net_profit = df['profit'].sum()
    current_ratio = df['assets'].sum() / df['liabilities'].sum() if df['liabilities'].sum() != 0 else np.nan

    print(f"Total Revenue: ${total_revenue:,.0f}")
    print(f"Total Expenses: ${total_expenses:,.0f}")

    profit_status = "Profit" if net_profit >= 0 else "Loss"
    print(f"Net Profit ({profit_status}): ${net_profit:,.0f}")

    print(f"Current Ratio: {current_ratio:.2f}" if not pd.isna(current_ratio) else "Current Ratio: N/A")

    # Financial trends
    if 'period' in df:
        financial_trends = df.groupby('period').sum(numeric_only=True).reset_index()

        fig1 = px.line(financial_trends, x='period', y=['revenue', 'expenses', 'profit'],
                        title="Financial Performance Over Time")
        fig1.show()

    # Balance sheet breakdown
    if all(col in df for col in ['assets', 'liabilities', 'equity']):
        balance_sheet = df[['assets', 'liabilities', 'equity']].sum().reset_index()
        balance_sheet.columns = ['Category', 'Amount']

        fig2 = px.pie(balance_sheet, names='Category', values='Amount',
                        title="Balance Sheet Composition")
        fig2.show()

def profitability_analysis(df):
    print("\n--- Profitability Analysis ---")
    expected = ['segment', 'revenue', 'cost_of_goods_sold', 'gross_profit',
                'operating_expenses', 'net_profit', 'profit_margin']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Calculate metrics
    total_revenue = df['revenue'].sum()
    total_profit = df['net_profit'].sum()
    avg_margin = df['profit_margin'].mean()

    print(f"Total Revenue: ${total_revenue:,.0f}")
    print(f"Total Profit: ${total_profit:,.0f}")
    print(f"Avg Profit Margin: {avg_margin:.1f}%")

    # Profitability by segment
    if 'segment' in df and 'net_profit' in df:
        segment_profit = df.groupby('segment')['net_profit'].sum().reset_index()

        fig1 = px.bar(segment_profit, x='segment', y='net_profit',
                        title="Profit by Business Segment")
        fig1.show()

    # Margin analysis
    if 'segment' in df and 'profit_margin' in df:
        fig2 = px.box(df, x='segment', y='profit_margin',
                        title="Profit Margin Distribution by Segment")
        fig2.show()

    # Cost structure analysis
    if all(col in df for col in ['revenue', 'cost_of_goods_sold', 'operating_expenses']):
        cost_structure = df[['revenue', 'cost_of_goods_sold', 'operating_expenses']].sum().reset_index()
        cost_structure.columns = ['Category', 'Amount']
        cost_structure['Percentage'] = cost_structure['Amount'] / cost_structure.loc[0, 'Amount'] * 100

        fig3 = px.bar(cost_structure[1:], x='Category', y='Percentage',
                        title="Cost Structure as % of Revenue")
        fig3.show()

def cash_flow_analysis(df): # Renamed from cashflow_analysis to avoid conflict with `corporate_cash_flow_statement_analysis`
    print("\n--- Cash Flow Analysis ---")
    expected = ['period', 'operating_cashflow', 'investing_cashflow',
                'financing_cashflow', 'net_cashflow', 'free_cashflow']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates if needed
    if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

    # Calculate metrics
    total_operating = df['operating_cashflow'].sum()
    total_investing = df['investing_cashflow'].sum()
    total_financing = df['financing_cashflow'].sum()
    net_cashflow = df['net_cashflow'].sum()

    print(f"Operating Cash Flow: ${total_operating:,.0f}")
    print(f"Investing Cash Flow: ${total_investing:,.0f}")
    print(f"Financing Cash Flow: ${total_financing:,.0f}")

    cash_status = "Positive" if net_cashflow >= 0 else "Negative"
    print(f"Net Cash Flow ({cash_status}): ${net_cashflow:,.0f}")

    # Cash flow trends
    if 'period' in df:
        cashflow_trends = df.groupby('period').sum(numeric_only=True).reset_index()

        fig1 = px.line(cashflow_trends, x='period',
                            y=['operating_cashflow', 'investing_cashflow', 'financing_cashflow'],
                            title="Cash Flow Trends Over Time")
        fig1.show()

    # Cash flow composition
    if all(col in df for col in ['operating_cashflow', 'investing_cashflow', 'financing_cashflow']):
        cash_composition = df[['operating_cashflow', 'investing_cashflow', 'financing_cashflow']].sum().reset_index()
        cash_composition.columns = ['Type', 'Amount']

        fig2 = px.pie(cash_composition, names='Type', values='Amount',
                            title="Cash Flow Composition")
        fig2.show()

def financial_ratios(df):
    print("\n--- Financial Ratios Analysis ---")
    expected = ['period', 'current_ratio', 'quick_ratio', 'debt_to_equity',
                'return_on_assets', 'return_on_equity', 'gross_margin']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates if needed
    if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

    # Latest period ratios
    latest = df.sort_values('period').iloc[-1]

    print(f"Current Ratio: {latest['current_ratio']:.2f}")
    print(f"Quick Ratio: {latest['quick_ratio']:.2f}")
    print(f"Debt-to-Equity: {latest['debt_to_equity']:.2f}")
    print(f"ROA: {latest['return_on_assets']:.1f}%")
    print(f"ROE: {latest['return_on_equity']:.1f}%")
    print(f"Gross Margin: {latest['gross_margin']:.1f}%")

    # Ratio trends
    if 'period' in df:
        ratio_trends = df.melt(id_vars='period',
                                 value_vars=['current_ratio', 'quick_ratio', 'debt_to_equity'],
                                 var_name='Ratio', value_name='Value')

        fig1 = px.line(ratio_trends, x='period', y='Value', color='Ratio',
                            title="Liquidity and Leverage Ratios Over Time")
        fig1.show()

        profitability_trends = df.melt(id_vars='period',
                                         value_vars=['return_on_assets', 'return_on_equity', 'gross_margin'],
                                         var_name='Ratio', value_name='Value')

        fig2 = px.line(profitability_trends, x='period', y='Value', color='Ratio',
                            title="Profitability Ratios Over Time")
        fig2.show()

def budget_vs_actual(df):
    print("\n--- Budget vs Actual Analysis ---")
    expected = ['category', 'budget', 'actual', 'variance', 'period']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates if needed
    if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
        df['period'] = pd.to_datetime(df['period'])

    # Calculate metrics
    total_budget = df['budget'].sum()
    total_actual = df['actual'].sum()
    total_variance = total_actual - total_budget
    variance_pct = (total_variance / total_budget) * 100

    print(f"Total Budget: ${total_budget:,.0f}")
    print(f"Total Actual: ${total_actual:,.0f}")
    print(f"Variance: ${total_variance:,.0f} ({variance_pct:.1f}%)")

    # Budget performance by category
    if 'category' in df:
        budget_comparison = df.groupby('category').sum(numeric_only=True).reset_index()
        budget_comparison = budget_comparison.melt(id_vars='category',
                                                     value_vars=['budget', 'actual'],
                                                     var_name='Type', value_name='Amount')

        fig1 = px.bar(budget_comparison, x='category', y='Amount', color='Type',
                            barmode='group', title="Budget vs Actual by Category")
        fig1.show()

    # Variance analysis
    if 'category' in df and 'variance' in df:
        df['variance_pct'] = (df['variance'] / df['budget']) * 100
        fig2 = px.bar(df, x='category', y='variance_pct',
                            title="Variance Percentage by Category")
        fig2.add_hline(y=0, line_color="black")
        fig2.show()

def investment_analysis(df):
    print("\n--- Investment Analysis ---")
    expected = ['investment_id', 'type', 'amount', 'date',
                'current_value', 'return_pct', 'duration']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates if needed
    if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    # Calculate metrics
    total_invested = df['amount'].sum()
    current_portfolio_value = df['current_value'].sum()
    total_return = current_portfolio_value - total_invested
    avg_return = df['return_pct'].mean()

    print(f"Total Invested: ${total_invested:,.0f}")
    print(f"Current Value: ${current_portfolio_value:,.0f}")

    return_status = "Positive" if total_return >= 0 else "Negative"
    print(f"Total Return ({return_status}): ${total_return:,.0f}")

    print(f"Avg Annual Return: {avg_return:.1f}%")

    # Portfolio composition
    if 'type' in df:
        portfolio_comp = df.groupby('type')['current_value'].sum().reset_index()

        fig1 = px.pie(portfolio_comp, names='type', values='current_value',
                            title="Portfolio Composition by Investment Type")
        fig1.show()

    # Return distribution
    if 'type' in df and 'return_pct' in df:
        fig2 = px.box(df, x='type', y='return_pct',
                            title="Return Distribution by Investment Type")
        fig2.show()

    # Performance over time
    if 'date' in df and 'current_value' in df:
        performance = df.groupby('date')['current_value'].sum().reset_index()

        fig3 = px.line(performance, x='date', y='current_value',
                            title="Portfolio Value Over Time")
        fig3.show()

#extra functions
def financial_transaction_categorization_and_analysis(df):
    print("\n--- Financial Transaction Categorization and Analysis ---")
    expected = ['transaction_type', 'subcategory', 'debit_usd', 'credit_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['debit_usd'] = pd.to_numeric(df['debit_usd'], errors='coerce').fillna(0)
    df['credit_usd'] = pd.to_numeric(df['credit_usd'], errors='coerce').fillna(0)
    df.dropna(subset=['transaction_type', 'subcategory'], inplace=True)

    # Metrics
    total_debit = df['debit_usd'].sum()
    total_credit = df['credit_usd'].sum()
    top_subcategory = df.groupby('subcategory')['debit_usd'].sum().idxmax()

    print(f"Total Debit: ${total_debit:,.2f}")
    print(f"Total Credit: ${total_credit:,.2f}")
    print(f"Top Expense Subcategory: {top_subcategory}")

    # Visualizations
    debits_by_sub = df.groupby('subcategory')['debit_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(debits_by_sub, x='subcategory', y='debit_usd', title="Top 15 Expense Subcategories by Debit Amount")
    fig1.show()

    txn_type_counts = df['transaction_type'].value_counts().reset_index()
    fig2 = px.pie(txn_type_counts, names='index', values='transaction_type', title="Distribution of Transaction Types")
    fig2.show()

def general_ledger_journal_entry_audit_analysis(df):
    print("\n--- General Ledger Journal Entry Audit Analysis ---")
    expected = ['entry_date', 'account_debit', 'account_credit', 'amount_usd', 'approved_by']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_value = df['amount_usd'].sum()
    num_entries = len(df)
    top_approver = df['approved_by'].mode()[0]

    print(f"Total Value of Entries: ${total_value:,.2f}")
    print(f"Number of Journal Entries: {num_entries:,}")
    print(f"Most Frequent Approver: {top_approver}")

    # Visualizations
    entries_over_time = df.groupby(df['entry_date'].dt.date)['amount_usd'].sum().reset_index()
    fig1 = px.line(entries_over_time, x='entry_date', y='amount_usd', title="Journal Entry Value Over Time")
    fig1.show()

    entries_by_approver = df.groupby('approved_by')['amount_usd'].sum().nlargest(10).reset_index()
    fig2 = px.bar(entries_by_approver, x='approved_by', y='amount_usd', title="Top 10 Approvers by Journal Value")
    fig2.show()

def accounts_receivable_and_invoice_payment_analysis(df):
    print("\n--- Accounts Receivable and Invoice Payment Analysis ---")
    expected = ['invoice_date', 'due_date', 'payment_date', 'total_usd', 'customer_name']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['invoice_date', 'due_date', 'payment_date']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df['total_usd'] = pd.to_numeric(df['total_usd'], errors='coerce')
    df.dropna(subset=['invoice_date', 'due_date', 'total_usd'], inplace=True)

    # Metrics
    df['days_to_pay'] = (df['payment_date'] - df['invoice_date']).dt.days
    df['days_overdue'] = (df['payment_date'] - df['due_date']).dt.days.clip(lower=0)

    avg_dso = df['days_to_pay'].mean()
    total_overdue_value = df[df['days_overdue'] > 0]['total_usd'].sum()

    print(f"Average Days Sales Outstanding (DSO): {avg_dso:.1f} days")
    print(f"Total Value of Overdue Invoices: ${total_overdue_value:,.2f}")

    # Visualizations
    fig1 = px.histogram(df, x='days_to_pay', title="Distribution of Days to Pay")
    fig1.show()

    overdue_by_customer = df.groupby('customer_name')['days_overdue'].mean().nlargest(10).reset_index()
    fig2 = px.bar(overdue_by_customer, x='customer_name', y='days_overdue', title="Top 10 Customers by Average Days Overdue")
    fig2.show()

def accounts_payable_and_vendor_payment_analysis(df):
    print("\n--- Accounts Payable and Vendor Payment Analysis ---")
    expected = ['invoice_id', 'vendor_name', 'amount_usd', 'payment_date', 'payment_method']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_paid = df['amount_usd'].sum()
    top_vendor = df.groupby('vendor_name')['amount_usd'].sum().idxmax()
    top_method = df['payment_method'].mode()[0]

    print(f"Total Amount Paid: ${total_paid:,.2f}")
    print(f"Top Vendor by Payment: {top_vendor}")
    print(f"Most Common Payment Method: {top_method}")

    # Visualizations
    paid_by_vendor = df.groupby('vendor_name')['amount_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(paid_by_vendor, x='vendor_name', y='amount_usd', title="Top 15 Vendors by Total Payments")
    fig1.show()

    paid_over_time = df.groupby(df['payment_date'].dt.to_period('M').astype(str))['amount_usd'].sum().reset_index()
    fig2 = px.line(paid_over_time, x='payment_date', y='amount_usd', title="Total Payments Over Time (Monthly)")
    fig2.show()

def chart_of_accounts_and_balance_management_analysis(df):
    print("\n--- Chart of Accounts and Balance Management Analysis ---")
    expected = ['account_name', 'account_type', 'opening_balance_usd', 'current_balance_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['opening_balance_usd', 'current_balance_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_balance = df['current_balance_usd'].sum()
    print(f"Total Current Balance Across All Accounts: ${total_balance:,.2f}")

    # Visualizations
    balance_by_type = df.groupby('account_type')['current_balance_usd'].sum().reset_index()
    fig1 = px.pie(balance_by_type, names='account_type', values='current_balance_usd', title="Balance Distribution by Account Type")
    fig1.show()

    df['balance_change'] = df['current_balance_usd'] - df['opening_balance_usd']
    change_by_account = df.nlargest(10, 'balance_change', keep='all').sort_values('balance_change', ascending=False)
    fig2 = px.bar(change_by_account, x='account_name', y='balance_change', title="Top 10 Accounts by Balance Increase")
    fig2.show()

def departmental_budget_vs_actual_variance_analysis(df):
    print("\n--- Departmental Budget vs. Actual Variance Analysis ---")
    expected = ['department_id', 'budgeted_amount_usd', 'actual_amount_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['budgeted_amount_usd', 'actual_amount_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    df['variance'] = df['budgeted_amount_usd'] - df['actual_amount_usd']

    # Metrics
    total_budget = df['budgeted_amount_usd'].sum()
    total_actual = df['actual_amount_usd'].sum()
    total_variance = total_budget - total_actual

    print(f"Total Budget: ${total_budget:,.0f}")
    print(f"Total Actual Spend: ${total_actual:,.0f}")
    print(f"Overall Variance: ${total_variance:,.0f} (Positive=Under Budget, Negative=Over Budget)")

    # Visualizations
    df_grouped = df.groupby('department_id')[['budgeted_amount_usd', 'actual_amount_usd']].sum().reset_index()
    fig1 = px.bar(df_grouped, x='department_id', y=['budgeted_amount_usd', 'actual_amount_usd'], barmode='group', title="Budget vs. Actual Spend by Department")
    fig1.show()

    variance_by_dept = df.groupby('department_id')['variance'].sum().sort_values().reset_index()
    fig2 = px.bar(variance_by_dept, x='department_id', y='variance', title="Variance by Department", color='variance', color_continuous_scale='RdBu')
    fig2.show()

def employee_expense_report_and_reimbursement_analysis(df):
    print("\n--- Employee Expense Report and Reimbursement Analysis ---")
    expected = ['employee_id', 'report_date', 'expense_type', 'amount_usd', 'reimbursed_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_expenses = df['amount_usd'].sum()
    reimbursement_rate = (df['reimbursed_flag'] == True).mean() * 100
    top_spender = df.groupby('employee_id')['amount_usd'].sum().idxmax()

    print(f"Total Expenses Submitted: ${total_expenses:,.2f}")
    print(f"Reimbursement Rate: {reimbursement_rate:.1f}%")
    print(f"Top Spender (by ID): {top_spender}")

    # Visualizations
    expense_by_type = df.groupby('expense_type')['amount_usd'].sum().reset_index()
    fig1 = px.pie(expense_by_type, names='expense_type', values='amount_usd', title="Expense Distribution by Type")
    fig1.show()

    expense_by_employee = df.groupby('employee_id')['amount_usd'].sum().nlargest(15).reset_index()
    fig2 = px.bar(expense_by_employee, x='employee_id', y='amount_usd', title="Top 15 Employees by Total Expenses")
    fig2.show()

def payroll_processing_and_compensation_analysis(df):
    print("\n--- Payroll Processing and Compensation Analysis ---")
    expected = ['employee_id', 'gross_pay_usd', 'taxes_usd', 'net_pay_usd', 'deductions_usd', 'position']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['gross_pay_usd', 'taxes_usd', 'net_pay_usd', 'deductions_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_gross_pay = df['gross_pay_usd'].sum()
    total_taxes = df['taxes_usd'].sum()
    effective_tax_rate = (total_taxes / total_gross_pay) * 100

    print(f"Total Gross Pay: ${total_gross_pay:,.2f}")
    print(f"Total Taxes Paid: ${total_taxes:,.2f}")
    print(f"Effective Tax Rate: {effective_tax_rate:.2f}%")

    # Visualizations
    pay_by_position = df.groupby('position')[['gross_pay_usd', 'net_pay_usd']].mean().reset_index()
    fig1 = px.bar(pay_by_position, x='position', y=['gross_pay_usd', 'net_pay_usd'], barmode='group', title="Average Gross vs. Net Pay by Position")
    fig1.show()

    pay_components = df[['gross_pay_usd', 'taxes_usd', 'deductions_usd', 'net_pay_usd']].sum().reset_index()
    pay_components.columns = ['Component', 'Amount']
    fig2 = px.pie(pay_components, names='Component', values='Amount', title="Overall Payroll Component Distribution")
    fig2.show()

def loan_portfolio_and_risk_management_analysis(df):
    print("\n--- Loan Portfolio and Risk Management Analysis ---")
    expected = ['loan_type', 'principal_amount_usd', 'interest_rate_perc', 'term_months', 'outstanding_balance']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if 'type' not in col: # Exclude 'loan_type' from numeric conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_outstanding = df['outstanding_balance'].sum()
    avg_interest_rate = df['interest_rate_perc'].mean()

    print(f"Total Outstanding Balance: ${total_outstanding:,.2f}")
    print(f"Average Interest Rate: {avg_interest_rate:.2f}%")

    # Visualizations
    outstanding_by_type = df.groupby('loan_type')['outstanding_balance'].sum().reset_index()
    fig1 = px.pie(outstanding_by_type, names='loan_type', values='outstanding_balance', title="Outstanding Balance by Loan Type")
    fig1.show()

    fig2 = px.scatter(df, x='term_months', y='principal_amount_usd', color='interest_rate_perc',
                      title="Loan Amount vs. Term (colored by Interest Rate)")
    fig2.show()

def credit_card_transaction_fraud_detection_analysis(df):
    print("\n--- Credit Card Transaction Fraud Detection Analysis ---")
    # This often requires a 'is_fraud' column, which we assume is missing for a general case
    # We will look for anomalies instead.
    expected = ['card_number', 'transaction_date', 'merchant_name', 'amount_usd', 'mcc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    df.dropna(inplace=True)

    # Analysis: Find potential outliers
    print("\nPotential Outlier Transactions (High Amount)")
    amount_threshold = df['amount_usd'].quantile(0.99)
    print(f"Displaying transactions above the 99th percentile (${amount_threshold:,.2f})")
    print(df[df['amount_usd'] > amount_threshold].to_string()) # Use to_string() for full DataFrame printing in console

    # Visualizations
    fig1 = px.histogram(df, x='amount_usd', title="Distribution of Transaction Amounts", log_y=True)
    fig1.show()

    amount_by_mcc = df.groupby('mcc')['amount_usd'].sum().nlargest(20).reset_index()
    fig2 = px.bar(amount_by_mcc, x='mcc', y='amount_usd', title="Top 20 Merchant Category Codes (MCC) by Transaction Value")
    fig2.show()

def investment_portfolio_performance_analysis(df):
    print("\n--- Investment Portfolio Performance Analysis ---")
    expected = ['instrument_type', 'ticker', 'market_value_usd', 'cost_basis_usd', 'unrealized_gain_loss_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['market_value_usd', 'cost_basis_usd', 'unrealized_gain_loss_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_market_value = df['market_value_usd'].sum()
    total_cost_basis = df['cost_basis_usd'].sum()
    total_unrealized_gain = df['unrealized_gain_loss_usd'].sum()

    print(f"Total Market Value: ${total_market_value:,.2f}")
    print(f"Total Cost Basis: ${total_cost_basis:,.2f}")
    print(f"Total Unrealized Gain/Loss: ${total_unrealized_gain:,.2f}")

    # Visualizations
    value_by_instrument = df.groupby('instrument_type')['market_value_usd'].sum().reset_index()
    fig1 = px.pie(value_by_instrument, names='instrument_type', values='market_value_usd', title="Portfolio Allocation by Instrument Type")
    fig1.show()

    df['percent_gain_loss'] = (df['unrealized_gain_loss_usd'] / df['cost_basis_usd']) * 100
    top_performers = df.nlargest(15, 'percent_gain_loss')
    fig2 = px.bar(top_performers, x='ticker', y='percent_gain_loss', title="Top 15 Holdings by % Unrealized Gain")
    fig2.show()

def mortgage_portfolio_and_prepayment_risk_analysis(df):
    print("\n--- Mortgage Portfolio and Prepayment Risk Analysis ---")
    expected = ['borrower_id', 'origination_date', 'loan_amount_usd', 'interest_rate_perc', 'term_years', 'outstanding_principal', 'prepayment_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if 'id' not in col and 'date' not in col and 'flag' not in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    prepayment_rate = (df['prepayment_flag'] == True).mean() * 100
    print(f"Overall Prepayment Rate: {prepayment_rate:.2f}%")

    # Visualizations
    fig1 = px.histogram(df, x='interest_rate_perc', color='prepayment_flag', barmode='overlay',
                        title="Interest Rate Distribution by Prepayment Status")
    fig1.show()

    prepayment_by_term = df.groupby('term_years')['prepayment_flag'].apply(lambda x: (x==True).mean()).mul(100).reset_index()
    fig2 = px.bar(prepayment_by_term, x='term_years', y='prepayment_flag', title="Prepayment Rate by Loan Term")
    fig2.show()

def corporate_cash_flow_statement_analysis(df):
    print("\n--- Corporate Cash Flow Statement Analysis ---")
    expected = ['period_end', 'operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd', 'net_cash_flow_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    for col in ['operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd', 'net_cash_flow_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('period_end').dropna()

    # Metrics
    latest_net_cf = df['net_cash_flow_usd'].iloc[-1]
    avg_operating_cf = df['operating_cf_usd'].mean()

    print(f"Latest Net Cash Flow: ${latest_net_cf:,.0f}")
    print(f"Average Operating Cash Flow: ${avg_operating_cf:,.0f}")

    # Visualizations
    fig = px.bar(df, x='period_end', y=['operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd'],
                    title="Cash Flow Components Over Time")
    fig.show()

def company_financial_position_balance_sheet_analysis(df):
    print("\n--- Company Financial Position (Balance Sheet) Analysis ---")
    expected = ['report_date', 'total_assets_usd', 'total_liabilities_usd', 'equity_usd', 'current_assets', 'current_liabilities']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
    for col in expected:
        if 'date' not in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('report_date').dropna()

    # Metrics
    df['debt_to_equity_ratio'] = df['total_liabilities_usd'] / df['equity_usd']
    df['current_ratio'] = df['current_assets'] / df['current_liabilities']

    latest_debt_equity = df['debt_to_equity_ratio'].iloc[-1]
    latest_current_ratio = df['current_ratio'].iloc[-1]

    print(f"Latest Debt-to-Equity Ratio: {latest_debt_equity:.2f}")
    print(f"Latest Current Ratio: {latest_current_ratio:.2f}")

    # Visualizations
    fig = px.area(df, x='report_date', y=['total_assets_usd', 'total_liabilities_usd', 'equity_usd'],
                    title="Balance Sheet Components Over Time")
    fig.show()

def company_financial_performance_income_statement_analysis(df):
    print("\n--- Company Financial Performance (Income Statement) Analysis ---")
    expected = ['period_end', 'revenue_usd', 'cost_of_goods_sold_usd', 'gross_profit_usd', 'operating_expenses_usd', 'operating_income_usd', 'net_income_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
    for col in expected:
        if 'period' not in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('period_end').dropna()

    # Metrics
    df['gross_margin_perc'] = (df['gross_profit_usd'] / df['revenue_usd']) * 100
    df['net_margin_perc'] = (df['net_income_usd'] / df['revenue_usd']) * 100

    latest_gross_margin = df['gross_margin_perc'].iloc[-1]
    latest_net_margin = df['net_margin_perc'].iloc[-1]

    print(f"Latest Gross Margin: {latest_gross_margin:.2f}%")
    print(f"Latest Net Margin: {latest_net_margin:.2f}%")

    # Visualizations
    fig = px.bar(df, x='period_end', y=['revenue_usd', 'gross_profit_usd', 'net_income_usd'],
                    title="Income Statement Key Figures Over Time")
    fig.show()

def insurance_claim_processing_and_fraud_analysis(df):
    print("\n--- Insurance Claim Processing and Fraud Analysis ---")
    expected = ['claim_date', 'claim_type', 'claim_amount_usd', 'approved_amount_usd', 'fraud_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
    for col in ['claim_amount_usd', 'approved_amount_usd', 'fraud_flag']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    fraud_rate = df['fraud_flag'].mean() * 100
    avg_claim_amount_fraud = df[df['fraud_flag'] == 1]['claim_amount_usd'].mean()

    print(f"Fraudulent Claim Rate: {fraud_rate:.2f}%")
    print(f"Avg. Amount of Fraudulent Claims: ${avg_claim_amount_fraud:,.2f}")

    # Visualizations
    fraud_by_type = df.groupby('claim_type')['fraud_flag'].mean().mul(100).reset_index()
    fig1 = px.bar(fraud_by_type, x='claim_type', y='fraud_flag', title="Fraud Rate by Claim Type")
    fig1.show()

    fig2 = px.box(df, x='fraud_flag', y='claim_amount_usd', title="Claim Amount by Fraud Status")
    fig2.show()

def financial_forecasting_accuracy_analysis(df):
    print("\n--- Financial Forecasting Accuracy Analysis ---")
    expected = ['forecast_date', 'metric', 'forecast_value_usd', 'actual_value_usd', 'forecast_error_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
    for col in ['forecast_value_usd', 'actual_value_usd', 'forecast_error_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    df['mape'] = (df['forecast_error_usd'].abs() / df['actual_value_usd'].abs()) * 100 # Mean Absolute Percentage Error
    avg_mape = df['mape'].mean()
    print(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape:.2f}%")

    # Visualizations
    df_long = df.melt(id_vars=['forecast_date', 'metric'], value_vars=['forecast_value_usd', 'actual_value_usd'],
                      var_name='value_type', value_name='amount')
    fig1 = px.line(df_long, x='forecast_date', y='amount', color='value_type', facet_row='metric',
                    title="Forecast vs. Actual Values Over Time")
    fig1.update_yaxes(matches=None) # Allow independent y-axes for facets
    fig1.show()

    fig2 = px.histogram(df, x='forecast_error_usd', color='metric', title="Distribution of Forecast Errors")
    fig2.show()

def general_ledger_reconciliation_analysis(df):
    print("\n--- General Ledger Reconciliation Analysis ---")
    expected = ['ledger_name', 'period_start', 'period_end', 'total_debits', 'total_credits', 'net_change', 'opening_balance', 'closing_balance']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['total_debits', 'total_credits', 'net_change', 'opening_balance', 'closing_balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Validation
    df['calculated_close'] = df['opening_balance'] + df['net_change']
    df['is_reconciled'] = (df['calculated_close'] - df['closing_balance']).abs() < 0.01 # Check for reconciliation

    # Metrics
    reconciliation_rate = df['is_reconciled'].mean() * 100
    print(f"Ledger Reconciliation Rate: {reconciliation_rate:.2f}%")

    # Visualizations
    unreconciled_ledgers = df[~df['is_reconciled']]
    print("\n--- Unreconciled Ledgers ---")
    print(unreconciled_ledgers.to_string()) # Use to_string() for full DataFrame printing in console

    df['debit_credit_ratio'] = df['total_debits'] / df['total_credits']
    fig = px.bar(df, x='ledger_name', y=['total_debits', 'total_credits'],
                    title="Total Debits vs. Credits by Ledger", barmode='group')
    fig.show()


def securities_trading_and_settlement_analysis(df):
    print("\n--- Securities Trading and Settlement Analysis ---")
    expected = ['trade_date', 'instrument', 'buy_sell', 'trade_value_usd', 'broker_id', 'settlement_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_trade_value = df['trade_value_usd'].sum()
    top_broker = df.groupby('broker_id')['trade_value_usd'].sum().idxmax()

    print(f"Total Trade Value: ${total_trade_value:,.2f}")
    print(f"Top Broker by Value: {top_broker}")

    # Visualizations
    trade_value_by_instrument = df.groupby('instrument')['trade_value_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(trade_value_by_instrument, x='instrument', y='trade_value_usd', title="Top 15 Instruments by Trade Value")
    fig1.show()

    buy_sell_value = df.groupby('buy_sell')['trade_value_usd'].sum().reset_index()
    fig2 = px.pie(buy_sell_value, names='buy_sell', values='trade_value_usd', title="Buy vs. Sell Volume")
    fig2.show()

def foreign_exchange_fx_trading_analysis(df):
    print("\n--- Foreign Exchange (FX) Trading Analysis ---")
    expected = ['trade_date', 'currency_pair', 'base_amount', 'quote_amount', 'exchange_rate', 'transaction_type', 'trader_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['base_amount', 'quote_amount', 'exchange_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_volume = df['base_amount'].sum()
    top_trader = df.groupby('trader_id')['base_amount'].sum().idxmax()
    top_pair = df.groupby('currency_pair')['base_amount'].sum().idxmax()

    print(f"Total Trading Volume (Base): ${total_volume:,.2f}")
    print(f"Top Trader by Volume: {top_trader}")
    print(f"Most Traded Currency Pair: {top_pair}")

    # Visualizations
    volume_by_pair = df.groupby('currency_pair')['base_amount'].sum().nlargest(10).reset_index()
    fig1 = px.bar(volume_by_pair, x='currency_pair', y='base_amount', title="Top 10 Currency Pairs by Volume")
    fig1.show()

    volume_by_type = df.groupby('transaction_type')['base_amount'].sum().reset_index()
    fig2 = px.pie(volume_by_type, names='transaction_type', values='base_amount', title="Trading Volume by Type (Spot vs. Forward)")
    fig2.show()

def financial_risk_assessment_and_mitigation_analysis(df):
    print("\n--- Financial Risk Assessment and Mitigation Analysis ---")
    expected = ['risk_type', 'probability_perc', 'impact_usd', 'risk_score', 'priority']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['probability_perc', 'impact_usd', 'risk_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_risk_score = df['risk_score'].mean()
    highest_impact_risk = df.loc[df['impact_usd'].idxmax()]['risk_type']

    print(f"Average Risk Score: {avg_risk_score:.2f}")
    print(f"Risk with Highest Potential Impact: {highest_impact_risk}")

    # Visualizations
    fig1 = px.scatter(df, x='probability_perc', y='impact_usd', size='risk_score', color='risk_type',
                      title="Risk Matrix (Probability vs. Impact)",
                      labels={'probability_perc': 'Probability (%)', 'impact_usd': 'Impact (USD)'})
    fig1.show()

    fig2 = px.treemap(df, path=[px.Constant("All Risks"), 'priority', 'risk_type'], values='impact_usd',
                      color='risk_score', color_continuous_scale='Reds',
                      title="Risk Treemap by Priority and Type (Sized by Impact)")
    fig2.show()

def regulatory_compliance_and_audit_findings_analysis(df):
    print("\n--- Regulatory Compliance and Audit Findings Analysis ---")
    expected = ['regulation', 'findings', 'severity', 'corrective_action', 'completed_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})

    # Metrics
    completion_rate = (df['completed_flag'] == True).mean() * 100
    print(f"Corrective Action Completion Rate: {completion_rate:.2f}%")

    # Visualizations
    findings_by_severity = df['severity'].value_counts().reset_index()
    fig1 = px.pie(findings_by_severity, names='index', values='severity', title="Distribution of Audit Findings by Severity")
    fig1.show()

    status_by_regulation = df.groupby(['regulation', 'completed_flag']).size().reset_index(name='count')
    fig2 = px.bar(status_by_regulation, x='regulation', y='count', color='completed_flag', barmode='group',
                    title="Corrective Action Status by Regulation")
    fig2.show()

def corporate_tax_compliance_and_filing_analysis(df):
    print("\n--- Corporate Tax Compliance and Filing Analysis ---")
    expected = ['filing_year', 'jurisdiction', 'taxable_income_usd', 'tax_due_usd', 'tax_paid_usd', 'refund_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if 'year' in col or 'usd' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    df['effective_tax_rate'] = (df['tax_due_usd'] / df['taxable_income_usd']) * 100
    avg_effective_rate = df['effective_tax_rate'].mean()
    total_tax_paid = df['tax_paid_usd'].sum()

    print(f"Total Tax Paid: ${total_tax_paid:,.2f}")
    print(f"Average Effective Tax Rate: {avg_effective_rate:.2f}%")

    # Visualizations
    tax_by_jurisdiction = df.groupby('jurisdiction')['tax_paid_usd'].sum().reset_index()
    fig1 = px.bar(tax_by_jurisdiction, x='jurisdiction', y='tax_paid_usd', title="Total Tax Paid by Jurisdiction")
    fig1.show()

    rate_over_time = df.groupby('filing_year')['effective_tax_rate'].mean().reset_index()
    fig2 = px.line(rate_over_time, x='filing_year', y='effective_tax_rate', title="Effective Tax Rate Over Time")
    fig2.show()

def insurance_policy_underwriting_and_management_analysis(df):
    print("\n--- Insurance Policy Underwriting and Management Analysis ---")
    expected = ['policy_type', 'premium_usd', 'coverage_amount_usd', 'agent_id', 'renewal_flag']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['premium_usd', 'coverage_amount_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_premiums = df['premium_usd'].sum()
    renewal_rate = (df['renewal_flag'] == True).mean() * 100

    print(f"Total Premiums Written: ${total_premiums:,.2f}")
    print(f"Policy Renewal Rate: {renewal_rate:.2f}%")

    # Visualizations
    premiums_by_type = df.groupby('policy_type')['premium_usd'].sum().reset_index()
    fig1 = px.pie(premiums_by_type, names='policy_type', values='premium_usd', title="Premium Distribution by Policy Type")
    fig1.show()

    fig2 = px.scatter(df, x='coverage_amount_usd', y='premium_usd', color='policy_type',
                      log_x=True, log_y=True, title="Premium vs. Coverage Amount")
    fig2.show()

def stock_dividend_payment_and_tax_analysis(df):
    print("\n--- Stock Dividend Payment and Tax Analysis ---")
    expected = ['payment_date', 'ticker', 'shares_held', 'dividend_per_share_usd', 'total_dividend_usd', 'tax_withheld_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
    for col in expected:
        if 'usd' in col or 'shares' in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_dividends = df['total_dividend_usd'].sum()
    total_tax = df['tax_withheld_usd'].sum()
    effective_tax_rate = (total_tax / total_dividends) * 100

    print(f"Total Dividends Received: ${total_dividends:,.2f}")
    print(f"Total Tax Withheld: ${total_tax:,.2f}")
    print(f"Effective Tax Rate: {effective_tax_rate:.2f}%")

    # Visualizations
    dividends_by_stock = df.groupby('ticker')['total_dividend_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(dividends_by_stock, x='ticker', y='total_dividend_usd', title="Top 15 Stocks by Dividend Paid")
    fig1.show()

    dividends_over_time = df.groupby(df['payment_date'].dt.to_period('Q').astype(str))['total_dividend_usd'].sum().reset_index()
    fig2 = px.line(dividends_over_time, x='payment_date', y='total_dividend_usd', title="Total Dividends Received per Quarter")
    fig2.show()

def monthly_budget_variance_reporting_and_analysis(df):
    print("\n--- Monthly Budget Variance Reporting and Analysis ---")
    expected = ['department', 'month', 'budget_usd', 'actual_usd', 'variance_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['month'] = pd.to_datetime(df['month'], errors='coerce')
    for col in ['budget_usd', 'actual_usd', 'variance_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_variance = df['variance_usd'].sum()
    print(f"Total Net Variance (All Months): ${total_variance:,.2f}")

    # Visualizations
    variance_over_time = df.groupby('month')['variance_usd'].sum().reset_index()
    fig1 = px.bar(variance_over_time, x='month', y='variance_usd', title="Net Budget Variance Over Time",
                    color='variance_usd', color_continuous_scale='RdBu_r')
    fig1.show()

    variance_by_dept = df.groupby('department')['variance_usd'].sum().reset_index()
    fig2 = px.bar(variance_by_dept, x='department', y='variance_usd', title="Total Net Variance by Department")
    fig2.show()

def corporate_liquidity_risk_monitoring_analysis(df):
    print("\n--- Corporate Liquidity Risk Monitoring Analysis ---")
    expected = ['metric_name', 'value_usd', 'threshold_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['value_usd', 'threshold_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    df['status'] = df.apply(lambda row: 'Breached' if row['value_usd'] < row['threshold_usd'] else 'Healthy', axis=1)
    breached_count = (df['status'] == 'Breached').sum()
    print(f"Number of Breached Liquidity Metrics: {breached_count}")

    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['metric_name'], y=df['value_usd'], name='Actual Value'))
    fig.add_trace(go.Scatter(x=df['metric_name'], y=df['threshold_usd'], name='Threshold', mode='lines+markers', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Liquidity Metrics vs. Thresholds")
    fig.show()

def capital_expenditure_capex_project_analysis(df):
    print("\n--- Capital Expenditure (CapEx) Project Analysis ---")
    expected = ['project_id', 'department', 'amount_usd', 'asset_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_capex = df['amount_usd'].sum()
    top_dept = df.groupby('department')['amount_usd'].sum().idxmax()

    print(f"Total CapEx: ${total_capex:,.2f}")
    print(f"Top Department by Spend: {top_dept}")

    # Visualizations
    capex_by_dept = df.groupby('department')['amount_usd'].sum().reset_index()
    fig1 = px.bar(capex_by_dept, x='department', y='amount_usd', title="Total CapEx by Department")
    fig1.show()

    capex_by_asset = df.groupby('asset_category')['amount_usd'].sum().reset_index()
    fig2 = px.pie(capex_by_asset, names='asset_category', values='amount_usd', title="CapEx Distribution by Asset Category")
    fig2.show()

def corporate_debt_issuance_and_structure_analysis(df):
    print("\n--- Corporate Debt Issuance and Structure Analysis ---")
    expected = ['issue_date', 'instrument_type', 'principal_usd', 'coupon_perc', 'maturity_date', 'rating']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
    for col in ['principal_usd', 'coupon_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_principal = df['principal_usd'].sum()
    avg_coupon = (df['principal_usd'] * df['coupon_perc']).sum() / total_principal

    print(f"Total Principal Issued: ${total_principal:,.2f}")
    print(f"Weighted Average Coupon: {avg_coupon:.2f}%")

    # Visualizations
    issuance_by_type = df.groupby('instrument_type')['principal_usd'].sum().reset_index()
    fig1 = px.pie(issuance_by_type, names='instrument_type', values='principal_usd', title="Debt Principal by Instrument Type")
    fig1.show()

    issuance_by_rating = df.groupby('rating')['principal_usd'].sum().reset_index()
    fig2 = px.bar(issuance_by_rating, x='rating', y='principal_usd', title="Debt Principal by Credit Rating")
    fig2.show()

def securities_lending_transaction_analysis(df):
    print("\n--- Securities Lending Transaction Analysis ---")
    expected = ['transaction_date', 'lender_id', 'borrower_id', 'instrument', 'fee_amount_usd', 'collateral_value_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['fee_amount_usd', 'collateral_value_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_fees = df['fee_amount_usd'].sum()
    total_collateral = df['collateral_value_usd'].sum()

    print(f"Total Lending Fees: ${total_fees:,.2f}")
    print(f"Total Collateral Value: ${total_collateral:,.2f}")

    # Visualizations
    fees_by_instrument = df.groupby('instrument')['fee_amount_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(fees_by_instrument, x='instrument', y='fee_amount_usd', title="Top 15 Instruments by Lending Fees Generated")
    fig1.show()

    fees_by_borrower = df.groupby('borrower_id')['fee_amount_usd'].sum().nlargest(15).reset_index()
    fig2 = px.bar(fees_by_borrower, x='borrower_id', y='fee_amount_usd', title="Top 15 Borrowers by Fees Paid")
    fig2.show()

def treasury_operations_and_trading_analysis(df):
    print("\n--- Treasury Operations and Trading Analysis ---")
    expected = ['trade_date', 'security', 'buy_sell', 'trade_value_usd', 'counterparty']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
    df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    buy_volume = df[df['buy_sell'].str.lower() == 'buy']['trade_value_usd'].sum()
    sell_volume = df[df['buy_sell'].str.lower() == 'sell']['trade_value_usd'].sum()
    net_volume = buy_volume - sell_volume

    print(f"Total Buy Volume: ${buy_volume:,.2f}")
    print(f"Total Sell Volume: ${sell_volume:,.2f}")
    print(f"Net Volume: ${net_volume:,.2f}")

    # Visualizations
    volume_by_security = df.groupby('security')['trade_value_usd'].sum().nlargest(15).reset_index()
    fig1 = px.bar(volume_by_security, x='security', y='trade_value_usd', title="Trade Volume by Security Type")
    fig1.show()

    net_flow_over_time = df.groupby('trade_date')['trade_value_usd'].sum().reset_index() # Simplified: assuming net_flow can be aggregated by date
    fig2 = px.line(net_flow_over_time, x='trade_date', y='trade_value_usd', title="Cumulative Net Trade Flow Over Time (Simplified)")
    fig2.show()


# ========== MAIN APP / EXECUTION LOGIC ==========
def main():
    """Main function to run the Financial Analytics script."""
    print("💰 Financial Analytics Script")

    # File path and encoding input
    file_path = input("Enter path to your data file (e.g., data.csv or data.xlsx): ")
    encoding = input("Enter file encoding (e.g., utf-8, latin1, cp1252, default=utf-8): ")
    if not encoding:
        encoding = 'utf-8'

    df = load_data(file_path, encoding)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")

    specific_finance_function_mapping = {
        "financial_statements": financial_statements,
        "profitability_analysis": profitability_analysis,
        "cash_flow_analysis": cash_flow_analysis,
        "financial_ratios": financial_ratios,
        "budget_vs_actual": budget_vs_actual,
        "investment_analysis": investment_analysis,
        "financial_transaction_categorization_and_analysis": financial_transaction_categorization_and_analysis,
        "general_ledger_journal_entry_audit_analysis": general_ledger_journal_entry_audit_analysis,
        "accounts_receivable_and_invoice_payment_analysis": accounts_receivable_and_invoice_payment_analysis,
        "accounts_payable_and_vendor_payment_analysis": accounts_payable_and_vendor_payment_analysis,
        "chart_of_accounts_and_balance_management_analysis": chart_of_accounts_and_balance_management_analysis,
        "general_ledger_reconciliation_analysis": general_ledger_reconciliation_analysis,
        "departmental_budget_vs._actual_variance_analysis": departmental_budget_vs_actual_variance_analysis,
        "employee_expense_report_and_reimbursement_analysis": employee_expense_report_and_reimbursement_analysis,
        "payroll_processing_and_compensation_analysis": payroll_processing_and_compensation_analysis,
        "loan_portfolio_and_risk_management_analysis": loan_portfolio_and_risk_management_analysis,
        "credit_card_transaction_fraud_detection_analysis": credit_card_transaction_fraud_detection_analysis,
        "investment_portfolio_performance_analysis": investment_portfolio_performance_analysis,
        "mortgage_portfolio_and_prepayment_risk_analysis": mortgage_portfolio_and_prepayment_risk_analysis,
        "securities_trading_and_settlement_analysis": securities_trading_and_settlement_analysis,
        "foreign_exchange_(fx)_trading_analysis": foreign_exchange_fx_trading_analysis,
        "financial_risk_assessment_and_mitigation_analysis": financial_risk_assessment_and_mitigation_analysis,
        "regulatory_compliance_and_audit_findings_analysis": regulatory_compliance_and_audit_findings_analysis,
        "corporate_cash_flow_statement_analysis": corporate_cash_flow_statement_analysis,
        "company_financial_position_(balance_sheet)_analysis": company_financial_position_balance_sheet_analysis,
        "company_financial_performance_(income_statement)_analysis": company_financial_performance_income_statement_analysis,
        "corporate_tax_compliance_and_filing_analysis": corporate_tax_compliance_and_filing_analysis,
        "insurance_policy_underwriting_and_management_analysis": insurance_policy_underwriting_and_management_analysis,
        "insurance_claim_processing_and_fraud_analysis": insurance_claim_processing_and_fraud_analysis,
        "stock_dividend_payment_and_tax_analysis": stock_dividend_payment_and_tax_analysis,
        "monthly_budget_variance_reporting_and_analysis": monthly_budget_variance_reporting_and_analysis,
        "financial_forecasting_accuracy_analysis": financial_forecasting_accuracy_analysis,
        "corporate_liquidity_risk_monitoring_analysis": corporate_liquidity_risk_monitoring_analysis,
        "capital_expenditure_(capex)_project_analysis": capital_expenditure_capex_project_analysis,
        "corporate_debt_issuance_and_structure_analysis": corporate_debt_issuance_and_structure_analysis,
        "securities_lending_transaction_analysis": securities_lending_transaction_analysis,
        "treasury_operations_and_trading_analysis": treasury_operations_and_trading_analysis,
    }

    # --- Analysis Selection ---
    print("\nSelect an Analysis to Perform:")
    all_analysis_names = list(specific_finance_function_mapping.keys())
    for i, name in enumerate(all_analysis_names):
        print(f"{i+1}: {name}")
    print(f"{len(all_analysis_names)+1}: General Insights")

    choice_str = input(f"Enter the number of your choice (1-{len(all_analysis_names)+1}): ")
    try:
        choice_idx = int(choice_str) - 1
        if 0 <= choice_idx < len(all_analysis_names):
            selected_analysis_name = all_analysis_names[choice_idx]
            selected_function = specific_finance_function_mapping.get(selected_analysis_name)
            if selected_function:
                try:
                    selected_function(df)
                except Exception as e:
                    print(f"\n[ERROR] An error occurred while running the analysis '{selected_analysis_name}':")
                    print(f"Error details: {e}")
            else:
                print(f"\n[ERROR] Function for '{selected_analysis_name}' not found. This should not happen.")
        elif choice_idx == len(all_analysis_names): # General Insights option
            show_general_insights(df, "Initial Data Overview")
        else:
            print("\nInvalid choice. Please enter a number within the given range.")
            show_general_insights(df, "Initial Data Overview")
    except ValueError:
        print("\nInvalid input. Please enter a number.")
        show_general_insights(df, "Initial Data Overview")


if __name__ == "__main__":
    main()