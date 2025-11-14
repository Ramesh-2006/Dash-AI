import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import json
import io

# Analysis options (for frontend/API to list)
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
    "general_insights" # Added for direct access
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
    """
    try:
        # Dump to string and reload to force type conversion
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception as e:
        print(f"Error in type conversion: {e}")
        # Fallback for complex un-serializable objects
        return str(data)

# --- Helper Functions ---

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
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
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
                    corr_matrix = df[numeric_cols].corr(numeric_only=True).round(2)
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
        
        # Add missing columns warning if provided (for fallback case)
        if missing_cols and len(missing_cols) > 0:
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
            "status": "success", # Always success for general insights
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
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred during General Insights analysis: {e}"],
            "matched_columns": matched_cols or {},
            "missing_columns": missing_cols or []
        }

def get_fallback_analysis(df, analysis_type, missing_columns, matched_cols={}):
    """
    Helper function to run the general insights analysis as a fallback
    and prepend a warning message.
    """
    general_result = show_general_insights(df, analysis_type, missing_columns, matched_cols)
    
    # Prepend a clear fallback warning to the insights from general_result
    warning_insights = [
        f"⚠️ REQUIRED COLUMNS NOT FOUND for '{analysis_type}'",
        "Showing General Analysis instead."
    ]
    
    # Add details about missing columns
    for col in missing_columns:
        match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
        warning_insights.append(f" - Missing: {col}{match_info}")
    
    # Combine insights
    general_result["insights"] = warning_insights + ["---"] + general_result.get("insights", [])
    
    # Set the status and original analysis type
    general_result["status"] = "fallback"
    general_result["analysis_type"] = analysis_type # Overwrite "General Insights" with the requested type
    general_result["missing_columns"] = missing_columns
    general_result["matched_columns"] = matched_cols
    
    return general_result


# ========== ANALYSIS FUNCTIONS (REFACTORED) ==========

def financial_statements(df):
    analysis_type = "Financial Statements"
    try:
        expected = ['account', 'period', 'revenue', 'expenses', 'profit', 'assets', 'liabilities', 'equity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)

        # Convert dates if needed
        if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
            df['period'] = pd.to_datetime(df['period'], errors='coerce')

        # Calculate metrics
        metrics = {}
        visualizations = {}
        insights = []

        total_revenue = df['revenue'].sum()
        total_expenses = df['expenses'].sum()
        net_profit = df['profit'].sum()
        current_ratio = df['assets'].sum() / df['liabilities'].sum() if df['liabilities'].sum() != 0 else np.nan

        metrics['total_revenue'] = total_revenue
        metrics['total_expenses'] = total_expenses
        metrics['net_profit'] = net_profit
        metrics['current_ratio'] = current_ratio

        insights.append(f"Total Revenue: ${total_revenue:,.0f}")
        insights.append(f"Total Expenses: ${total_expenses:,.0f}")
        profit_status = "Profit" if net_profit >= 0 else "Loss"
        insights.append(f"Net Profit ({profit_status}): ${net_profit:,.0f}")
        insights.append(f"Current Ratio: {current_ratio:.2f}" if not pd.isna(current_ratio) else "Current Ratio: N/A")

        # Financial trends
        if 'period' in df and not df['period'].isnull().all():
            financial_trends = df.groupby('period').sum(numeric_only=True).reset_index()
            fig1 = px.line(financial_trends, x='period', y=['revenue', 'expenses', 'profit'],
                           title="Financial Performance Over Time")
            visualizations['financial_performance_over_time'] = fig1.to_json()
            insights.append("Generated financial performance trend visualization.")

        # Balance sheet breakdown
        if all(col in df for col in ['assets', 'liabilities', 'equity']):
            balance_sheet = df[['assets', 'liabilities', 'equity']].sum().reset_index()
            balance_sheet.columns = ['Category', 'Amount']
            fig2 = px.pie(balance_sheet, names='Category', values='Amount',
                          title="Balance Sheet Composition")
            visualizations['balance_sheet_composition'] = fig2.to_json()
            insights.append("Generated balance sheet composition pie chart.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def profitability_analysis(df):
    analysis_type = "Profitability Analysis"
    try:
        expected = ['segment', 'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_expenses', 'net_profit', 'profit_margin']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)

        metrics = {}
        visualizations = {}
        insights = []

        # Calculate metrics
        total_revenue = df['revenue'].sum()
        total_profit = df['net_profit'].sum()
        avg_margin = df['profit_margin'].mean()

        metrics['total_revenue'] = total_revenue
        metrics['total_profit'] = total_profit
        metrics['avg_margin'] = avg_margin

        insights.append(f"Total Revenue: ${total_revenue:,.0f}")
        insights.append(f"Total Profit: ${total_profit:,.0f}")
        insights.append(f"Avg Profit Margin: {avg_margin:.1f}%")

        # Profitability by segment
        if 'segment' in df and 'net_profit' in df:
            segment_profit = df.groupby('segment')['net_profit'].sum().reset_index()
            fig1 = px.bar(segment_profit, x='segment', y='net_profit',
                          title="Profit by Business Segment")
            visualizations['profit_by_segment'] = fig1.to_json()
            insights.append("Analyzed profit by business segment.")

        # Margin analysis
        if 'segment' in df and 'profit_margin' in df:
            fig2 = px.box(df, x='segment', y='profit_margin',
                          title="Profit Margin Distribution by Segment")
            visualizations['margin_distribution_by_segment'] = fig2.to_json()
            insights.append("Analyzed profit margin distribution.")

        # Cost structure analysis
        if all(col in df for col in ['revenue', 'cost_of_goods_sold', 'operating_expenses']):
            cost_structure = df[['revenue', 'cost_of_goods_sold', 'operating_expenses']].sum().reset_index()
            cost_structure.columns = ['Category', 'Amount']
            cost_structure['Percentage'] = (cost_structure['Amount'] / cost_structure.loc[0, 'Amount']) * 100
            fig3 = px.bar(cost_structure[1:], x='Category', y='Percentage',
                          title="Cost Structure as % of Revenue")
            visualizations['cost_structure_percentage'] = fig3.to_json()
            insights.append("Analyzed cost structure as a percentage of revenue.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def cash_flow_analysis(df):
    analysis_type = "Cash Flow Analysis"
    try:
        expected = ['period', 'operating_cashflow', 'investing_cashflow', 'financing_cashflow', 'net_cashflow', 'free_cashflow']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)

        metrics = {}
        visualizations = {}
        insights = []

        # Convert dates if needed
        if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
            df['period'] = pd.to_datetime(df['period'], errors='coerce')

        # Calculate metrics
        total_operating = df['operating_cashflow'].sum()
        total_investing = df['investing_cashflow'].sum()
        total_financing = df['financing_cashflow'].sum()
        net_cashflow = df['net_cashflow'].sum()

        metrics['total_operating_cashflow'] = total_operating
        metrics['total_investing_cashflow'] = total_investing
        metrics['total_financing_cashflow'] = total_financing
        metrics['net_cashflow'] = net_cashflow

        insights.append(f"Operating Cash Flow: ${total_operating:,.0f}")
        insights.append(f"Investing Cash Flow: ${total_investing:,.0f}")
        insights.append(f"Financing Cash Flow: ${total_financing:,.0f}")
        cash_status = "Positive" if net_cashflow >= 0 else "Negative"
        insights.append(f"Net Cash Flow ({cash_status}): ${net_cashflow:,.0f}")

        # Cash flow trends
        if 'period' in df and not df['period'].isnull().all():
            cashflow_trends = df.groupby('period').sum(numeric_only=True).reset_index()
            fig1 = px.line(cashflow_trends, x='period',
                           y=['operating_cashflow', 'investing_cashflow', 'financing_cashflow'],
                           title="Cash Flow Trends Over Time")
            visualizations['cashflow_trends'] = fig1.to_json()
            insights.append("Generated cash flow trends visualization.")

        # Cash flow composition
        if all(col in df for col in ['operating_cashflow', 'investing_cashflow', 'financing_cashflow']):
            cash_composition = df[['operating_cashflow', 'investing_cashflow', 'financing_cashflow']].sum().reset_index()
            cash_composition.columns = ['Type', 'Amount']
            fig2 = px.pie(cash_composition, names='Type', values='Amount',
                          title="Cash Flow Composition")
            visualizations['cashflow_composition'] = fig2.to_json()
            insights.append("Generated cash flow composition pie chart.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def financial_ratios(df):
    analysis_type = "Financial Ratios"
    try:
        expected = ['period', 'current_ratio', 'quick_ratio', 'debt_to_equity', 'return_on_assets', 'return_on_equity', 'gross_margin']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)

        metrics = {}
        visualizations = {}
        insights = []

        # Convert dates if needed
        if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
            df['period'] = pd.to_datetime(df['period'], errors='coerce')
        
        df = df.dropna(subset=['period']).sort_values('period')
        
        if df.empty:
            insights.append("No data available after processing.")
            return {
                "analysis_type": analysis_type,
                "status": "error",
                "error_message": "No data available after processing dates.",
                "visualizations": {}, "metrics": {}, "insights": insights, "matched_columns": matched
            }

        # Latest period ratios
        latest = df.iloc[-1]
        metrics['latest_ratios'] = {
            'current_ratio': latest['current_ratio'],
            'quick_ratio': latest['quick_ratio'],
            'debt_to_equity': latest['debt_to_equity'],
            'return_on_assets': latest['return_on_assets'],
            'return_on_equity': latest['return_on_equity'],
            'gross_margin': latest['gross_margin']
        }
        
        insights.append(f"Latest Current Ratio: {latest['current_ratio']:.2f}")
        insights.append(f"Latest Quick Ratio: {latest['quick_ratio']:.2f}")
        insights.append(f"Latest Debt-to-Equity: {latest['debt_to_equity']:.2f}")
        insights.append(f"Latest ROA: {latest['return_on_assets']:.1f}%")
        insights.append(f"Latest ROE: {latest['return_on_equity']:.1f}%")
        insights.append(f"Latest Gross Margin: {latest['gross_margin']:.1f}%")

        # Ratio trends
        if 'period' in df:
            ratio_trends = df.melt(id_vars='period',
                                   value_vars=['current_ratio', 'quick_ratio', 'debt_to_equity'],
                                   var_name='Ratio', value_name='Value')
            fig1 = px.line(ratio_trends, x='period', y='Value', color='Ratio',
                           title="Liquidity and Leverage Ratios Over Time")
            visualizations['liquidity_leverage_trends'] = fig1.to_json()

            profitability_trends = df.melt(id_vars='period',
                                           value_vars=['return_on_assets', 'return_on_equity', 'gross_margin'],
                                           var_name='Ratio', value_name='Value')
            fig2 = px.line(profitability_trends, x='period', y='Value', color='Ratio',
                           title="Profitability Ratios Over Time")
            visualizations['profitability_trends'] = fig2.to_json()
            insights.append("Generated visualizations for ratio trends over time.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def budget_vs_actual(df):
    analysis_type = "Budget vs Actual"
    try:
        expected = ['category', 'budget', 'actual', 'variance', 'period']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)

        metrics = {}
        visualizations = {}
        insights = []

        # Convert dates if needed
        if 'period' in df and not pd.api.types.is_datetime64_any_dtype(df['period']):
            df['period'] = pd.to_datetime(df['period'], errors='coerce')

        # Calculate metrics
        total_budget = df['budget'].sum()
        total_actual = df['actual'].sum()
        total_variance = total_actual - total_budget
        variance_pct = (total_variance / total_budget) * 100 if total_budget != 0 else 0

        metrics['total_budget'] = total_budget
        metrics['total_actual'] = total_actual
        metrics['total_variance'] = total_variance
        metrics['variance_percentage'] = variance_pct

        insights.append(f"Total Budget: ${total_budget:,.0f}")
        insights.append(f"Total Actual: ${total_actual:,.0f}")
        insights.append(f"Variance: ${total_variance:,.0f} ({variance_pct:.1f}%)")

        # Budget performance by category
        if 'category' in df:
            budget_comparison = df.groupby('category').sum(numeric_only=True).reset_index()
            budget_comparison = budget_comparison.melt(id_vars='category',
                                                       value_vars=['budget', 'actual'],
                                                       var_name='Type', value_name='Amount')
            fig1 = px.bar(budget_comparison, x='category', y='Amount', color='Type',
                          barmode='group', title="Budget vs Actual by Category")
            visualizations['budget_vs_actual_by_category'] = fig1.to_json()
            insights.append("Generated budget vs. actual comparison by category.")

        # Variance analysis
        if 'category' in df and 'variance' in df:
            df['variance_pct'] = (df['variance'] / df['budget']) * 100
            fig2 = px.bar(df, x='category', y='variance_pct',
                          title="Variance Percentage by Category")
            fig2.add_hline(y=0, line_color="black")
            visualizations['variance_percentage_by_category'] = fig2.to_json()
            insights.append("Generated variance percentage by category.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def investment_analysis(df):
    analysis_type = "Investment Analysis"
    try:
        expected = ['investment_id', 'type', 'amount', 'date', 'current_value', 'return_pct', 'duration']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        
        metrics = {}
        visualizations = {}
        insights = []

        # Convert dates if needed
        if 'date' in df and not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Calculate metrics
        total_invested = df['amount'].sum()
        current_portfolio_value = df['current_value'].sum()
        total_return = current_portfolio_value - total_invested
        avg_return = df['return_pct'].mean()

        metrics['total_invested'] = total_invested
        metrics['current_portfolio_value'] = current_portfolio_value
        metrics['total_return'] = total_return
        metrics['avg_return_percentage'] = avg_return

        insights.append(f"Total Invested: ${total_invested:,.0f}")
        insights.append(f"Current Value: ${current_portfolio_value:,.0f}")
        return_status = "Positive" if total_return >= 0 else "Negative"
        insights.append(f"Total Return ({return_status}): ${total_return:,.0f}")
        insights.append(f"Avg Annual Return: {avg_return:.1f}%")

        # Portfolio composition
        if 'type' in df:
            portfolio_comp = df.groupby('type')['current_value'].sum().reset_index()
            fig1 = px.pie(portfolio_comp, names='type', values='current_value',
                          title="Portfolio Composition by Investment Type")
            visualizations['portfolio_composition'] = fig1.to_json()
            insights.append("Generated portfolio composition pie chart.")

        # Return distribution
        if 'type' in df and 'return_pct' in df:
            fig2 = px.box(df, x='type', y='return_pct',
                          title="Return Distribution by Investment Type")
            visualizations['return_distribution'] = fig2.to_json()
            insights.append("Generated return distribution box plot.")

        # Performance over time
        if 'date' in df and 'current_value' in df and not df['date'].isnull().all():
            performance = df.groupby('date')['current_value'].sum().reset_index()
            fig3 = px.line(performance, x='date', y='current_value',
                           title="Portfolio Value Over Time")
            visualizations['portfolio_performance_over_time'] = fig3.to_json()
            insights.append("Generated portfolio value over time.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def financial_transaction_categorization_and_analysis(df):
    analysis_type = "Financial Transaction Categorization and Analysis"
    try:
        expected = ['transaction_type', 'subcategory', 'debit_usd', 'credit_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['debit_usd'] = pd.to_numeric(df['debit_usd'], errors='coerce').fillna(0)
        df['credit_usd'] = pd.to_numeric(df['credit_usd'], errors='coerce').fillna(0)
        df.dropna(subset=['transaction_type', 'subcategory'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_debit = df['debit_usd'].sum()
        total_credit = df['credit_usd'].sum()
        top_subcategory = df.groupby('subcategory')['debit_usd'].sum().idxmax()

        metrics['total_debit'] = total_debit
        metrics['total_credit'] = total_credit
        metrics['top_expense_subcategory'] = top_subcategory

        insights.append(f"Total Debit: ${total_debit:,.2f}")
        insights.append(f"Total Credit: ${total_credit:,.2f}")
        insights.append(f"Top Expense Subcategory: {top_subcategory}")

        # Visualizations
        debits_by_sub = df.groupby('subcategory')['debit_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(debits_by_sub, x='subcategory', y='debit_usd', title="Top 15 Expense Subcategories by Debit Amount")
        visualizations['top_expense_subcategories'] = fig1.to_json()

        txn_type_counts = df['transaction_type'].value_counts().reset_index()
        txn_type_counts.columns = ['type', 'count']
        fig2 = px.pie(txn_type_counts, names='type', values='count', title="Distribution of Transaction Types")
        visualizations['transaction_type_distribution'] = fig2.to_json()
        
        insights.append("Generated visualizations for expense categories and transaction types.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def general_ledger_journal_entry_audit_analysis(df):
    analysis_type = "General Ledger Journal Entry Audit Analysis"
    try:
        expected = ['entry_date', 'account_debit', 'account_credit', 'amount_usd', 'approved_by']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['entry_date'] = pd.to_datetime(df['entry_date'], errors='coerce')
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df.dropna(inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_value = df['amount_usd'].sum()
        num_entries = len(df)
        top_approver = df['approved_by'].mode()[0] if not df['approved_by'].empty else 'N/A'

        metrics['total_value_of_entries'] = total_value
        metrics['number_of_journal_entries'] = num_entries
        metrics['most_frequent_approver'] = top_approver

        insights.append(f"Total Value of Entries: ${total_value:,.2f}")
        insights.append(f"Number of Journal Entries: {num_entries:,}")
        insights.append(f"Most Frequent Approver: {top_approver}")

        # Visualizations
        if not df['entry_date'].isnull().all():
            entries_over_time = df.groupby(df['entry_date'].dt.date)['amount_usd'].sum().reset_index()
            fig1 = px.line(entries_over_time, x='entry_date', y='amount_usd', title="Journal Entry Value Over Time")
            visualizations['journal_entry_value_over_time'] = fig1.to_json()

        entries_by_approver = df.groupby('approved_by')['amount_usd'].sum().nlargest(10).reset_index()
        fig2 = px.bar(entries_by_approver, x='approved_by', y='amount_usd', title="Top 10 Approvers by Journal Value")
        visualizations['top_approvers_by_value'] = fig2.to_json()

        insights.append("Generated visualizations for journal entry trends and approver activity.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def accounts_receivable_and_invoice_payment_analysis(df):
    analysis_type = "Accounts Receivable and Invoice Payment Analysis"
    try:
        expected = ['invoice_date', 'due_date', 'payment_date', 'total_usd', 'customer_name']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['invoice_date', 'due_date', 'payment_date']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        df['total_usd'] = pd.to_numeric(df['total_usd'], errors='coerce')
        df.dropna(subset=['invoice_date', 'due_date', 'total_usd'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        df['days_to_pay'] = (df['payment_date'] - df['invoice_date']).dt.days
        df['days_overdue'] = (df['payment_date'] - df['due_date']).dt.days.clip(lower=0)
        df['days_overdue'] = df['days_overdue'].fillna(0) # Assume not overdue if no payment date

        avg_dso = df['days_to_pay'].mean()
        total_overdue_value = df[df['days_overdue'] > 0]['total_usd'].sum()

        metrics['average_dso_days'] = avg_dso
        metrics['total_overdue_value'] = total_overdue_value

        insights.append(f"Average Days Sales Outstanding (DSO): {avg_dso:.1f} days")
        insights.append(f"Total Value of Overdue Invoices: ${total_overdue_value:,.2f}")

        # Visualizations
        fig1 = px.histogram(df.dropna(subset=['days_to_pay']), x='days_to_pay', title="Distribution of Days to Pay")
        visualizations['days_to_pay_distribution'] = fig1.to_json()

        if 'customer_name' in df.columns:
            overdue_by_customer = df.groupby('customer_name')['days_overdue'].mean().nlargest(10).reset_index()
            fig2 = px.bar(overdue_by_customer, x='customer_name', y='days_overdue', title="Top 10 Customers by Average Days Overdue")
            visualizations['top_customers_by_overdue_days'] = fig2.to_json()
        
        insights.append("Generated visualizations for DSO and overdue payments.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def accounts_payable_and_vendor_payment_analysis(df):
    analysis_type = "Accounts Payable and Vendor Payment Analysis"
    try:
        expected = ['invoice_id', 'vendor_name', 'amount_usd', 'payment_date', 'payment_method']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df.dropna(subset=['vendor_name', 'amount_usd', 'payment_date'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_paid = df['amount_usd'].sum()
        top_vendor = df.groupby('vendor_name')['amount_usd'].sum().idxmax()
        top_method = df['payment_method'].mode()[0] if 'payment_method' in df.columns and not df['payment_method'].empty else 'N/A'

        metrics['total_amount_paid'] = total_paid
        metrics['top_vendor_by_payment'] = top_vendor
        metrics['most_common_payment_method'] = top_method

        insights.append(f"Total Amount Paid: ${total_paid:,.2f}")
        insights.append(f"Top Vendor by Payment: {top_vendor}")
        insights.append(f"Most Common Payment Method: {top_method}")

        # Visualizations
        paid_by_vendor = df.groupby('vendor_name')['amount_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(paid_by_vendor, x='vendor_name', y='amount_usd', title="Top 15 Vendors by Total Payments")
        visualizations['top_vendors_by_payment'] = fig1.to_json()

        paid_over_time = df.groupby(df['payment_date'].dt.to_period('M').astype(str))['amount_usd'].sum().reset_index()
        fig2 = px.line(paid_over_time, x='payment_date', y='amount_usd', title="Total Payments Over Time (Monthly)")
        visualizations['payments_over_time'] = fig2.to_json()

        insights.append("Generated visualizations for vendor payments.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def chart_of_accounts_and_balance_management_analysis(df):
    analysis_type = "Chart of Accounts and Balance Management Analysis"
    try:
        expected = ['account_name', 'account_type', 'opening_balance_usd', 'current_balance_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        for col in ['opening_balance_usd', 'current_balance_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['account_name', 'account_type', 'current_balance_usd', 'opening_balance_usd'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_balance = df['current_balance_usd'].sum()
        metrics['total_current_balance'] = total_balance
        insights.append(f"Total Current Balance Across All Accounts: ${total_balance:,.2f}")

        # Visualizations
        balance_by_type = df.groupby('account_type')['current_balance_usd'].sum().reset_index()
        fig1 = px.pie(balance_by_type, names='account_type', values='current_balance_usd', title="Balance Distribution by Account Type")
        visualizations['balance_by_account_type'] = fig1.to_json()

        df['balance_change'] = df['current_balance_usd'] - df['opening_balance_usd']
        change_by_account = df.nlargest(10, 'balance_change', keep='all').sort_values('balance_change', ascending=False)
        fig2 = px.bar(change_by_account, x='account_name', y='balance_change', title="Top 10 Accounts by Balance Increase")
        visualizations['top_accounts_by_balance_increase'] = fig2.to_json()

        insights.append("Generated visualizations for account balances.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def departmental_budget_vs_actual_variance_analysis(df):
    analysis_type = "Departmental Budget vs. Actual Variance Analysis"
    try:
        expected = ['department_id', 'budgeted_amount_usd', 'actual_amount_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        for col in ['budgeted_amount_usd', 'actual_amount_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['department_id', 'budgeted_amount_usd', 'actual_amount_usd'], inplace=True)
        df['variance'] = df['budgeted_amount_usd'] - df['actual_amount_usd']

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_budget = df['budgeted_amount_usd'].sum()
        total_actual = df['actual_amount_usd'].sum()
        total_variance = total_budget - total_actual

        metrics['total_budget'] = total_budget
        metrics['total_actual'] = total_actual
        metrics['total_variance'] = total_variance

        insights.append(f"Total Budget: ${total_budget:,.0f}")
        insights.append(f"Total Actual Spend: ${total_actual:,.0f}")
        insights.append(f"Overall Variance: ${total_variance:,.0f} (Positive=Under Budget, Negative=Over Budget)")

        # Visualizations
        df_grouped = df.groupby('department_id')[['budgeted_amount_usd', 'actual_amount_usd']].sum().reset_index()
        fig1 = px.bar(df_grouped, x='department_id', y=['budgeted_amount_usd', 'actual_amount_usd'], barmode='group', title="Budget vs. Actual Spend by Department")
        visualizations['budget_vs_actual_by_dept'] = fig1.to_json()

        variance_by_dept = df.groupby('department_id')['variance'].sum().sort_values().reset_index()
        fig2 = px.bar(variance_by_dept, x='department_id', y='variance', title="Variance by Department", color='variance', color_continuous_scale='RdBu')
        visualizations['variance_by_dept'] = fig2.to_json()

        insights.append("Generated visualizations for departmental budget variance.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def employee_expense_report_and_reimbursement_analysis(df):
    analysis_type = "Employee Expense Report and Reimbursement Analysis"
    try:
        expected = ['employee_id', 'report_date', 'expense_type', 'amount_usd', 'reimbursed_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df.dropna(subset=['employee_id', 'expense_type', 'amount_usd', 'reimbursed_flag'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_expenses = df['amount_usd'].sum()
        reimbursement_rate = (df['reimbursed_flag'] == True).mean() * 100
        top_spender = df.groupby('employee_id')['amount_usd'].sum().idxmax()

        metrics['total_expenses_submitted'] = total_expenses
        metrics['reimbursement_rate_percent'] = reimbursement_rate
        metrics['top_spender_id'] = top_spender

        insights.append(f"Total Expenses Submitted: ${total_expenses:,.2f}")
        insights.append(f"Reimbursement Rate: {reimbursement_rate:.1f}%")
        insights.append(f"Top Spender (by ID): {top_spender}")

        # Visualizations
        expense_by_type = df.groupby('expense_type')['amount_usd'].sum().reset_index()
        fig1 = px.pie(expense_by_type, names='expense_type', values='amount_usd', title="Expense Distribution by Type")
        visualizations['expense_distribution_by_type'] = fig1.to_json()

        expense_by_employee = df.groupby('employee_id')['amount_usd'].sum().nlargest(15).reset_index()
        fig2 = px.bar(expense_by_employee, x='employee_id', y='amount_usd', title="Top 15 Employees by Total Expenses")
        visualizations['top_employees_by_expense'] = fig2.to_json()

        insights.append("Generated visualizations for employee expenses.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def payroll_processing_and_compensation_analysis(df):
    analysis_type = "Payroll Processing and Compensation Analysis"
    try:
        expected = ['employee_id', 'gross_pay_usd', 'taxes_usd', 'net_pay_usd', 'deductions_usd', 'position']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        for col in ['gross_pay_usd', 'taxes_usd', 'net_pay_usd', 'deductions_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['employee_id', 'gross_pay_usd', 'taxes_usd', 'net_pay_usd', 'position'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_gross_pay = df['gross_pay_usd'].sum()
        total_taxes = df['taxes_usd'].sum()
        effective_tax_rate = (total_taxes / total_gross_pay) * 100 if total_gross_pay > 0 else 0

        metrics['total_gross_pay'] = total_gross_pay
        metrics['total_taxes_paid'] = total_taxes
        metrics['effective_tax_rate_percent'] = effective_tax_rate

        insights.append(f"Total Gross Pay: ${total_gross_pay:,.2f}")
        insights.append(f"Total Taxes Paid: ${total_taxes:,.2f}")
        insights.append(f"Effective Tax Rate: {effective_tax_rate:.2f}%")

        # Visualizations
        pay_by_position = df.groupby('position')[['gross_pay_usd', 'net_pay_usd']].mean().reset_index()
        fig1 = px.bar(pay_by_position, x='position', y=['gross_pay_usd', 'net_pay_usd'], barmode='group', title="Average Gross vs. Net Pay by Position")
        visualizations['avg_pay_by_position'] = fig1.to_json()

        pay_components_cols = [col for col in ['gross_pay_usd', 'taxes_usd', 'deductions_usd', 'net_pay_usd'] if col in df.columns]
        pay_components = df[pay_components_cols].sum().reset_index()
        pay_components.columns = ['Component', 'Amount']
        fig2 = px.pie(pay_components, names='Component', values='Amount', title="Overall Payroll Component Distribution")
        visualizations['payroll_component_distribution'] = fig2.to_json()

        insights.append("Generated visualizations for payroll analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def loan_portfolio_and_risk_management_analysis(df):
    analysis_type = "Loan Portfolio and Risk Management Analysis"
    try:
        expected = ['loan_type', 'principal_amount_usd', 'interest_rate_perc', 'term_months', 'outstanding_balance']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        # Use matched.values() to get the actual column names from the df
        numeric_cols_to_convert = [matched[col] for col in expected if 'type' not in col and matched[col] is not None]
        for col in numeric_cols_to_convert:
             # Use the actual column name (v) for conversion
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Now rename to standard names
        df = safe_rename(df, matched)
        
        df.dropna(subset=['loan_type', 'principal_amount_usd', 'interest_rate_perc', 'outstanding_balance'], inplace=True)


        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_outstanding = df['outstanding_balance'].sum()
        avg_interest_rate = df['interest_rate_perc'].mean()

        metrics['total_outstanding_balance'] = total_outstanding
        metrics['average_interest_rate_percent'] = avg_interest_rate

        insights.append(f"Total Outstanding Balance: ${total_outstanding:,.2f}")
        insights.append(f"Average Interest Rate: {avg_interest_rate:.2f}%")

        # Visualizations
        outstanding_by_type = df.groupby('loan_type')['outstanding_balance'].sum().reset_index()
        fig1 = px.pie(outstanding_by_type, names='loan_type', values='outstanding_balance', title="Outstanding Balance by Loan Type")
        visualizations['outstanding_balance_by_loan_type'] = fig1.to_json()

        if 'term_months' in df.columns:
            fig2 = px.scatter(df, x='term_months', y='principal_amount_usd', color='interest_rate_perc',
                              title="Loan Amount vs. Term (colored by Interest Rate)")
            visualizations['loan_amount_vs_term'] = fig2.to_json()

        insights.append("Generated visualizations for loan portfolio analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def credit_card_transaction_fraud_detection_analysis(df):
    analysis_type = "Credit Card Transaction Fraud Detection Analysis"
    try:
        expected = ['card_number', 'transaction_date', 'merchant_name', 'amount_usd', 'mcc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df.dropna(subset=['amount_usd', 'mcc'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Analysis: Find potential outliers
        amount_threshold = df['amount_usd'].quantile(0.99)
        outliers = df[df['amount_usd'] > amount_threshold]

        metrics['outlier_threshold_99th_percentile'] = amount_threshold
        metrics['num_potential_outliers'] = len(outliers)
        
        insights.append(f"Transactions above 99th percentile (${amount_threshold:,.2f}) are flagged as potential outliers.")
        insights.append(f"Found {len(outliers)} potential outlier transactions.")
        
        # We cannot return the full outlier list in JSON, so we summarize
        if not outliers.empty:
            insights.append(f"Outliers include transactions up to ${outliers['amount_usd'].max():,.2f}.")
            metrics['max_outlier_amount'] = outliers['amount_usd'].max()

        # Visualizations
        fig1 = px.histogram(df, x='amount_usd', title="Distribution of Transaction Amounts", log_y=True)
        fig1.add_vline(x=amount_threshold, line_dash="dash", line_color="red", annotation_text="99th Percentile")
        visualizations['transaction_amount_distribution'] = fig1.to_json()

        amount_by_mcc = df.groupby('mcc')['amount_usd'].sum().nlargest(20).reset_index()
        fig2 = px.bar(amount_by_mcc, x='mcc', y='amount_usd', title="Top 20 Merchant Category Codes (MCC) by Transaction Value")
        visualizations['top_mcc_by_value'] = fig2.to_json()

        insights.append("Generated visualizations for transaction distribution and MCC analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def investment_portfolio_performance_analysis(df):
    analysis_type = "Investment Portfolio Performance Analysis"
    try:
        expected = ['instrument_type', 'ticker', 'market_value_usd', 'cost_basis_usd', 'unrealized_gain_loss_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        for col in ['market_value_usd', 'cost_basis_usd', 'unrealized_gain_loss_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['instrument_type', 'ticker', 'market_value_usd', 'cost_basis_usd'], inplace=True)
        
        # Recalculate gain/loss if not present or null
        if 'unrealized_gain_loss_usd' not in df.columns or df['unrealized_gain_loss_usd'].isnull().all():
             df['unrealized_gain_loss_usd'] = df['market_value_usd'] - df['cost_basis_usd']


        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_market_value = df['market_value_usd'].sum()
        total_cost_basis = df['cost_basis_usd'].sum()
        total_unrealized_gain = df['unrealized_gain_loss_usd'].sum()

        metrics['total_market_value'] = total_market_value
        metrics['total_cost_basis'] = total_cost_basis
        metrics['total_unrealized_gain_loss'] = total_unrealized_gain

        insights.append(f"Total Market Value: ${total_market_value:,.2f}")
        insights.append(f"Total Cost Basis: ${total_cost_basis:,.2f}")
        insights.append(f"Total Unrealized Gain/Loss: ${total_unrealized_gain:,.2f}")

        # Visualizations
        value_by_instrument = df.groupby('instrument_type')['market_value_usd'].sum().reset_index()
        fig1 = px.pie(value_by_instrument, names='instrument_type', values='market_value_usd', title="Portfolio Allocation by Instrument Type")
        visualizations['portfolio_allocation_by_instrument'] = fig1.to_json()

        df['percent_gain_loss'] = (df['unrealized_gain_loss_usd'] / df['cost_basis_usd']) * 100
        top_performers = df.nlargest(15, 'percent_gain_loss')
        fig2 = px.bar(top_performers, x='ticker', y='percent_gain_loss', title="Top 15 Holdings by % Unrealized Gain")
        visualizations['top_holdings_by_percent_gain'] = fig2.to_json()

        insights.append("Generated visualizations for portfolio allocation and top performers.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def mortgage_portfolio_and_prepayment_risk_analysis(df):
    analysis_type = "Mortgage Portfolio and Prepayment Risk Analysis"
    try:
        expected = ['borrower_id', 'origination_date', 'loan_amount_usd', 'interest_rate_perc', 'term_years', 'outstanding_principal', 'prepayment_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        
        numeric_cols_to_convert = [matched[col] for col in expected if 'id' not in col and 'date' not in col and 'flag' not in col and matched[col] is not None]
        for col in numeric_cols_to_convert:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = safe_rename(df, matched) # Rename again after conversion
        
        # Convert flag to boolean/int
        if 'prepayment_flag' in df.columns:
            if df['prepayment_flag'].dtype == 'object':
                 df['prepayment_flag'] = df['prepayment_flag'].apply(lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0)
            df['prepayment_flag'] = pd.to_numeric(df['prepayment_flag'], errors='coerce')

        df.dropna(subset=['interest_rate_perc', 'term_years', 'prepayment_flag'], inplace=True)


        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        prepayment_rate = df['prepayment_flag'].mean() * 100
        metrics['overall_prepayment_rate_percent'] = prepayment_rate
        insights.append(f"Overall Prepayment Rate: {prepayment_rate:.2f}%")

        # Visualizations
        fig1 = px.histogram(df, x='interest_rate_perc', color='prepayment_flag', barmode='overlay',
                            title="Interest Rate Distribution by Prepayment Status")
        visualizations['interest_rate_vs_prepayment'] = fig1.to_json()

        prepayment_by_term = df.groupby('term_years')['prepayment_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(prepayment_by_term, x='term_years', y='prepayment_flag', title="Prepayment Rate by Loan Term")
        visualizations['prepayment_rate_by_term'] = fig2.to_json()

        insights.append("Generated visualizations for prepayment risk analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def corporate_cash_flow_statement_analysis(df):
    analysis_type = "Corporate Cash Flow Statement Analysis"
    try:
        expected = ['period_end', 'operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd', 'net_cash_flow_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
        for col in ['operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd', 'net_cash_flow_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('period_end').dropna()
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        latest_net_cf = df['net_cash_flow_usd'].iloc[-1]
        avg_operating_cf = df['operating_cf_usd'].mean()

        metrics['latest_net_cash_flow'] = latest_net_cf
        metrics['average_operating_cash_flow'] = avg_operating_cf

        insights.append(f"Latest Net Cash Flow: ${latest_net_cf:,.0f}")
        insights.append(f"Average Operating Cash Flow: ${avg_operating_cf:,.0f}")

        # Visualizations
        fig = px.bar(df, x='period_end', y=['operating_cf_usd', 'investing_cf_usd', 'financing_cf_usd'],
                     title="Cash Flow Components Over Time")
        visualizations['cash_flow_components_over_time'] = fig.to_json()
        insights.append("Generated visualization for cash flow components.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def company_financial_position_balance_sheet_analysis(df):
    analysis_type = "Company Financial Position (Balance Sheet) Analysis"
    try:
        expected = ['report_date', 'total_assets_usd', 'total_liabilities_usd', 'equity_usd', 'current_assets', 'current_liabilities']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['report_date'] = pd.to_datetime(df['report_date'], errors='coerce')
        for col in expected:
            if 'date' not in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('report_date').dropna()

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        df['debt_to_equity_ratio'] = df['total_liabilities_usd'] / df['equity_usd']
        df['current_ratio'] = df['current_assets'] / df['current_liabilities']

        latest_debt_equity = df['debt_to_equity_ratio'].iloc[-1]
        latest_current_ratio = df['current_ratio'].iloc[-1]

        metrics['latest_debt_to_equity_ratio'] = latest_debt_equity
        metrics['latest_current_ratio'] = latest_current_ratio

        insights.append(f"Latest Debt-to-Equity Ratio: {latest_debt_equity:.2f}")
        insights.append(f"Latest Current Ratio: {latest_current_ratio:.2f}")

        # Visualizations
        fig = px.area(df, x='report_date', y=['total_assets_usd', 'total_liabilities_usd', 'equity_usd'],
                      title="Balance Sheet Components Over Time")
        visualizations['balance_sheet_components_over_time'] = fig.to_json()
        insights.append("Generated visualization for balance sheet components.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def company_financial_performance_income_statement_analysis(df):
    analysis_type = "Company Financial Performance (Income Statement) Analysis"
    try:
        expected = ['period_end', 'revenue_usd', 'cost_of_goods_sold_usd', 'gross_profit_usd', 'operating_expenses_usd', 'operating_income_usd', 'net_income_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['period_end'] = pd.to_datetime(df['period_end'], errors='coerce')
        for col in expected:
            if 'period' not in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.sort_values('period_end').dropna()
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        df['gross_margin_perc'] = (df['gross_profit_usd'] / df['revenue_usd']) * 100
        df['net_margin_perc'] = (df['net_income_usd'] / df['revenue_usd']) * 100

        latest_gross_margin = df['gross_margin_perc'].iloc[-1]
        latest_net_margin = df['net_margin_perc'].iloc[-1]

        metrics['latest_gross_margin_percent'] = latest_gross_margin
        metrics['latest_net_margin_percent'] = latest_net_margin

        insights.append(f"Latest Gross Margin: {latest_gross_margin:.2f}%")
        insights.append(f"Latest Net Margin: {latest_net_margin:.2f}%")

        # Visualizations
        fig = px.bar(df, x='period_end', y=['revenue_usd', 'gross_profit_usd', 'net_income_usd'],
                     title="Income Statement Key Figures Over Time")
        visualizations['income_statement_over_time'] = fig.to_json()
        insights.append("Generated visualization for income statement key figures.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def insurance_claim_processing_and_fraud_analysis(df):
    analysis_type = "Insurance Claim Processing and Fraud Analysis"
    try:
        expected = ['claim_date', 'claim_type', 'claim_amount_usd', 'approved_amount_usd', 'fraud_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['claim_date'] = pd.to_datetime(df['claim_date'], errors='coerce')
        for col in ['claim_amount_usd', 'approved_amount_usd', 'fraud_flag']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['claim_type', 'claim_amount_usd', 'fraud_flag'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        fraud_rate = df['fraud_flag'].mean() * 100
        avg_claim_amount_fraud = df[df['fraud_flag'] == 1]['claim_amount_usd'].mean()
        
        metrics['fraudulent_claim_rate_percent'] = fraud_rate
        metrics['avg_fraudulent_claim_amount'] = avg_claim_amount_fraud if not pd.isna(avg_claim_amount_fraud) else 0

        insights.append(f"Fraudulent Claim Rate: {fraud_rate:.2f}%")
        insights.append(f"Avg. Amount of Fraudulent Claims: ${avg_claim_amount_fraud:,.2f}")

        # Visualizations
        fraud_by_type = df.groupby('claim_type')['fraud_flag'].mean().mul(100).reset_index()
        fig1 = px.bar(fraud_by_type, x='claim_type', y='fraud_flag', title="Fraud Rate by Claim Type")
        visualizations['fraud_rate_by_claim_type'] = fig1.to_json()

        fig2 = px.box(df, x='fraud_flag', y='claim_amount_usd', title="Claim Amount by Fraud Status")
        visualizations['claim_amount_by_fraud_status'] = fig2.to_json()

        insights.append("Generated visualizations for fraud analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def financial_forecasting_accuracy_analysis(df):
    analysis_type = "Financial Forecasting Accuracy Analysis"
    try:
        expected = ['forecast_date', 'metric', 'forecast_value_usd', 'actual_value_usd', 'forecast_error_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['forecast_date'] = pd.to_datetime(df['forecast_date'], errors='coerce')
        for col in ['forecast_value_usd', 'actual_value_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate error if not present
        if 'forecast_error_usd' not in df.columns or df['forecast_error_usd'].isnull().all():
            df['forecast_error_usd'] = df['forecast_value_usd'] - df['actual_value_usd']

        df.dropna(subset=['forecast_date', 'metric', 'forecast_value_usd', 'actual_value_usd', 'forecast_error_usd'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        df['mape'] = (df['forecast_error_usd'].abs() / df['actual_value_usd'].abs()) * 100 # Mean Absolute Percentage Error
        avg_mape = df['mape'].mean()
        
        metrics['mean_absolute_percentage_error'] = avg_mape
        insights.append(f"Average Mean Absolute Percentage Error (MAPE): {avg_mape:.2f}%")

        # Visualizations
        df_long = df.melt(id_vars=['forecast_date', 'metric'], value_vars=['forecast_value_usd', 'actual_value_usd'],
                          var_name='value_type', value_name='amount')
        fig1 = px.line(df_long, x='forecast_date', y='amount', color='value_type', facet_row='metric',
                       title="Forecast vs. Actual Values Over Time")
        fig1.update_yaxes(matches=None) # Allow independent y-axes for facets
        visualizations['forecast_vs_actual_over_time'] = fig1.to_json()

        fig2 = px.histogram(df, x='forecast_error_usd', color='metric', title="Distribution of Forecast Errors")
        visualizations['forecast_error_distribution'] = fig2.to_json()

        insights.append("Generated visualizations for forecast accuracy.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def general_ledger_reconciliation_analysis(df):
    analysis_type = "General Ledger Reconciliation Analysis"
    try:
        expected = ['ledger_name', 'period_start', 'period_end', 'total_debits', 'total_credits', 'net_change', 'opening_balance', 'closing_balance']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['total_debits', 'total_credits', 'net_change', 'opening_balance', 'closing_balance']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Validation
        df['calculated_close'] = df['opening_balance'] + df['net_change']
        df['is_reconciled'] = (df['calculated_close'] - df['closing_balance']).abs() < 0.01 # Check for reconciliation

        # Metrics
        reconciliation_rate = df['is_reconciled'].mean() * 100
        metrics['ledger_reconciliation_rate_percent'] = reconciliation_rate
        insights.append(f"Ledger Reconciliation Rate: {reconciliation_rate:.2f}%")

        # Visualizations
        unreconciled_ledgers = df[~df['is_reconciled']]
        metrics['unreconciled_ledger_count'] = len(unreconciled_ledgers)
        insights.append(f"Found {len(unreconciled_ledgers)} unreconciled ledgers.")
        
        # Add summary of unreconciled ledgers instead of printing
        if not unreconciled_ledgers.empty:
            top_unreconciled = unreconciled_ledgers.nlargest(5, 'closing_balance')[['ledger_name', 'closing_balance', 'calculated_close']]
            insights.append("Top 5 unreconciled ledgers by closing balance:")
            for _, row in top_unreconciled.iterrows():
                insights.append(f" - {row['ledger_name']} (Expected: ${row['calculated_close']:,.2f}, Actual: ${row['closing_balance']:,.2f})")
            metrics['top_unreconciled_ledgers_sample'] = top_unreconciled.to_dict('records')


        fig = px.bar(df, x='ledger_name', y=['total_debits', 'total_credits'],
                     title="Total Debits vs. Credits by Ledger", barmode='group')
        visualizations['debits_vs_credits_by_ledger'] = fig.to_json()
        insights.append("Generated visualization for debits vs. credits.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def securities_trading_and_settlement_analysis(df):
    analysis_type = "Securities Trading and Settlement Analysis"
    try:
        expected = ['trade_date', 'instrument', 'buy_sell', 'trade_value_usd', 'broker_id', 'settlement_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
        df.dropna(subset=['instrument', 'buy_sell', 'trade_value_usd', 'broker_id'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_trade_value = df['trade_value_usd'].sum()
        top_broker = df.groupby('broker_id')['trade_value_usd'].sum().idxmax()

        metrics['total_trade_value'] = total_trade_value
        metrics['top_broker_by_value'] = top_broker

        insights.append(f"Total Trade Value: ${total_trade_value:,.2f}")
        insights.append(f"Top Broker by Value: {top_broker}")

        # Visualizations
        trade_value_by_instrument = df.groupby('instrument')['trade_value_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(trade_value_by_instrument, x='instrument', y='trade_value_usd', title="Top 15 Instruments by Trade Value")
        visualizations['top_instruments_by_trade_value'] = fig1.to_json()

        buy_sell_value = df.groupby('buy_sell')['trade_value_usd'].sum().reset_index()
        fig2 = px.pie(buy_sell_value, names='buy_sell', values='trade_value_usd', title="Buy vs. Sell Volume")
        visualizations['buy_vs_sell_volume'] = fig2.to_json()

        insights.append("Generated visualizations for trading analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def foreign_exchange_fx_trading_analysis(df):
    analysis_type = "Foreign Exchange (FX) Trading Analysis"
    try:
        expected = ['trade_date', 'currency_pair', 'base_amount', 'quote_amount', 'exchange_rate', 'transaction_type', 'trader_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['base_amount', 'quote_amount', 'exchange_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['currency_pair', 'base_amount', 'trader_id', 'transaction_type'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_volume = df['base_amount'].sum()
        top_trader = df.groupby('trader_id')['base_amount'].sum().idxmax()
        top_pair = df.groupby('currency_pair')['base_amount'].sum().idxmax()

        metrics['total_trading_volume_base'] = total_volume
        metrics['top_trader_by_volume'] = top_trader
        metrics['most_traded_currency_pair'] = top_pair

        insights.append(f"Total Trading Volume (Base): ${total_volume:,.2f}")
        insights.append(f"Top Trader by Volume: {top_trader}")
        insights.append(f"Most Traded Currency Pair: {top_pair}")

        # Visualizations
        volume_by_pair = df.groupby('currency_pair')['base_amount'].sum().nlargest(10).reset_index()
        fig1 = px.bar(volume_by_pair, x='currency_pair', y='base_amount', title="Top 10 Currency Pairs by Volume")
        visualizations['top_currency_pairs_by_volume'] = fig1.to_json()

        volume_by_type = df.groupby('transaction_type')['base_amount'].sum().reset_index()
        fig2 = px.pie(volume_by_type, names='transaction_type', values='base_amount', title="Trading Volume by Type (Spot vs. Forward)")
        visualizations['volume_by_transaction_type'] = fig2.to_json()

        insights.append("Generated visualizations for FX trading analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def financial_risk_assessment_and_mitigation_analysis(df):
    analysis_type = "Financial Risk Assessment and Mitigation Analysis"
    try:
        expected = ['risk_type', 'probability_perc', 'impact_usd', 'risk_score', 'priority']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['probability_perc', 'impact_usd', 'risk_score']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['risk_type', 'probability_perc', 'impact_usd', 'priority'], inplace=True)
        
        # Calculate score if not present
        if 'risk_score' not in df.columns or df['risk_score'].isnull().all():
            df['risk_score'] = df['probability_perc'] * df['impact_usd']

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        avg_risk_score = df['risk_score'].mean()
        highest_impact_risk = df.loc[df['impact_usd'].idxmax()]['risk_type']

        metrics['average_risk_score'] = avg_risk_score
        metrics['highest_impact_risk_type'] = highest_impact_risk

        insights.append(f"Average Risk Score: {avg_risk_score:.2f}")
        insights.append(f"Risk with Highest Potential Impact: {highest_impact_risk}")

        # Visualizations
        fig1 = px.scatter(df, x='probability_perc', y='impact_usd', size='risk_score', color='risk_type',
                          title="Risk Matrix (Probability vs. Impact)",
                          labels={'probability_perc': 'Probability (%)', 'impact_usd': 'Impact (USD)'})
        visualizations['risk_matrix'] = fig1.to_json()

        fig2 = px.treemap(df, path=[px.Constant("All Risks"), 'priority', 'risk_type'], values='impact_usd',
                          color='risk_score', color_continuous_scale='Reds',
                          title="Risk Treemap by Priority and Type (Sized by Impact)")
        visualizations['risk_treemap'] = fig2.to_json()

        insights.append("Generated risk matrix and treemap.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def regulatory_compliance_and_audit_findings_analysis(df):
    analysis_type = "Regulatory Compliance and Audit Findings Analysis"
    try:
        expected = ['regulation', 'findings', 'severity', 'corrective_action', 'completed_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        
        # Convert flag
        if 'completed_flag' in df.columns:
            if df['completed_flag'].dtype == 'object':
                 df['completed_flag'] = df['completed_flag'].apply(lambda x: True if str(x).lower() in ['true', '1', 'yes', 'completed'] else False)
            df['completed_flag'] = df['completed_flag'].astype(bool)

        df.dropna(subset=['regulation', 'severity', 'completed_flag'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        completion_rate = df['completed_flag'].mean() * 100
        metrics['corrective_action_completion_rate_percent'] = completion_rate
        insights.append(f"Corrective Action Completion Rate: {completion_rate:.2f}%")

        # Visualizations
        findings_by_severity = df['severity'].value_counts().reset_index()
        findings_by_severity.columns = ['severity', 'count']
        fig1 = px.pie(findings_by_severity, names='severity', values='count', title="Distribution of Audit Findings by Severity")
        visualizations['findings_by_severity'] = fig1.to_json()

        status_by_regulation = df.groupby(['regulation', 'completed_flag']).size().reset_index(name='count')
        fig2 = px.bar(status_by_regulation, x='regulation', y='count', color='completed_flag', barmode='group',
                      title="Corrective Action Status by Regulation")
        visualizations['action_status_by_regulation'] = fig2.to_json()

        insights.append("Generated visualizations for audit findings.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def corporate_tax_compliance_and_filing_analysis(df):
    analysis_type = "Corporate Tax Compliance and Filing Analysis"
    try:
        expected = ['filing_year', 'jurisdiction', 'taxable_income_usd', 'tax_due_usd', 'tax_paid_usd', 'refund_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in expected:
            if 'year' in col or 'usd' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['filing_year', 'jurisdiction', 'taxable_income_usd', 'tax_due_usd', 'tax_paid_usd'], inplace=True)

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        df['effective_tax_rate'] = (df['tax_due_usd'] / df['taxable_income_usd']) * 100
        avg_effective_rate = df['effective_tax_rate'].mean()
        total_tax_paid = df['tax_paid_usd'].sum()

        metrics['total_tax_paid'] = total_tax_paid
        metrics['average_effective_tax_rate_percent'] = avg_effective_rate

        insights.append(f"Total Tax Paid: ${total_tax_paid:,.2f}")
        insights.append(f"Average Effective Tax Rate: {avg_effective_rate:.2f}%")

        # Visualizations
        tax_by_jurisdiction = df.groupby('jurisdiction')['tax_paid_usd'].sum().reset_index()
        fig1 = px.bar(tax_by_jurisdiction, x='jurisdiction', y='tax_paid_usd', title="Total Tax Paid by Jurisdiction")
        visualizations['tax_paid_by_jurisdiction'] = fig1.to_json()

        rate_over_time = df.groupby('filing_year')['effective_tax_rate'].mean().reset_index()
        fig2 = px.line(rate_over_time, x='filing_year', y='effective_tax_rate', title="Effective Tax Rate Over Time")
        visualizations['effective_tax_rate_over_time'] = fig2.to_json()

        insights.append("Generated visualizations for tax analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def insurance_policy_underwriting_and_management_analysis(df):
    analysis_type = "Insurance Policy Underwriting and Management Analysis"
    try:
        expected = ['policy_type', 'premium_usd', 'coverage_amount_usd', 'agent_id', 'renewal_flag']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['premium_usd', 'coverage_amount_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert flag
        if 'renewal_flag' in df.columns:
            if df['renewal_flag'].dtype == 'object':
                 df['renewal_flag'] = df['renewal_flag'].apply(lambda x: True if str(x).lower() in ['true', '1', 'yes', 'renewed'] else False)
            df['renewal_flag'] = df['renewal_flag'].astype(bool)

        df.dropna(subset=['policy_type', 'premium_usd', 'coverage_amount_usd', 'renewal_flag'], inplace=True)

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }
        
        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_premiums = df['premium_usd'].sum()
        renewal_rate = df['renewal_flag'].mean() * 100

        metrics['total_premiums_written'] = total_premiums
        metrics['policy_renewal_rate_percent'] = renewal_rate

        insights.append(f"Total Premiums Written: ${total_premiums:,.2f}")
        insights.append(f"Policy Renewal Rate: {renewal_rate:.2f}%")

        # Visualizations
        premiums_by_type = df.groupby('policy_type')['premium_usd'].sum().reset_index()
        fig1 = px.pie(premiums_by_type, names='policy_type', values='premium_usd', title="Premium Distribution by Policy Type")
        visualizations['premium_distribution_by_policy_type'] = fig1.to_json()

        fig2 = px.scatter(df, x='coverage_amount_usd', y='premium_usd', color='policy_type',
                          log_x=True, log_y=True, title="Premium vs. Coverage Amount")
        visualizations['premium_vs_coverage_amount'] = fig2.to_json()

        insights.append("Generated visualizations for policy analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def stock_dividend_payment_and_tax_analysis(df):
    analysis_type = "Stock Dividend Payment and Tax Analysis"
    try:
        expected = ['payment_date', 'ticker', 'shares_held', 'dividend_per_share_usd', 'total_dividend_usd', 'tax_withheld_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['payment_date'] = pd.to_datetime(df['payment_date'], errors='coerce')
        for col in expected:
            if 'usd' in col or 'shares' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate total dividend if not present
        if ('total_dividend_usd' not in df.columns or df['total_dividend_usd'].isnull().all()) and \
           'shares_held' in df.columns and 'dividend_per_share_usd' in df.columns:
            df['total_dividend_usd'] = df['shares_held'] * df['dividend_per_share_usd']

        df.dropna(subset=['payment_date', 'ticker', 'total_dividend_usd', 'tax_withheld_usd'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_dividends = df['total_dividend_usd'].sum()
        total_tax = df['tax_withheld_usd'].sum()
        effective_tax_rate = (total_tax / total_dividends) * 100 if total_dividends > 0 else 0

        metrics['total_dividends_received'] = total_dividends
        metrics['total_tax_withheld'] = total_tax
        metrics['effective_tax_rate_percent'] = effective_tax_rate

        insights.append(f"Total Dividends Received: ${total_dividends:,.2f}")
        insights.append(f"Total Tax Withheld: ${total_tax:,.2f}")
        insights.append(f"Effective Tax Rate: {effective_tax_rate:.2f}%")

        # Visualizations
        dividends_by_stock = df.groupby('ticker')['total_dividend_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(dividends_by_stock, x='ticker', y='total_dividend_usd', title="Top 15 Stocks by Dividend Paid")
        visualizations['top_stocks_by_dividend'] = fig1.to_json()

        dividends_over_time = df.groupby(df['payment_date'].dt.to_period('Q').astype(str))['total_dividend_usd'].sum().reset_index()
        fig2 = px.line(dividends_over_time, x='payment_date', y='total_dividend_usd', title="Total Dividends Received per Quarter")
        visualizations['dividends_per_quarter'] = fig2.to_json()

        insights.append("Generated visualizations for dividend analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def monthly_budget_variance_reporting_and_analysis(df):
    analysis_type = "Monthly Budget Variance Reporting and Analysis"
    try:
        expected = ['department', 'month', 'budget_usd', 'actual_usd', 'variance_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['month'] = pd.to_datetime(df['month'], errors='coerce')
        for col in ['budget_usd', 'actual_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Calculate variance if not present
        if 'variance_usd' not in df.columns or df['variance_usd'].isnull().all():
            df['variance_usd'] = df['budget_usd'] - df['actual_usd'] # Budget - Actual (Positive = Under budget)

        df.dropna(subset=['department', 'month', 'budget_usd', 'actual_usd', 'variance_usd'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }
        
        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_variance = df['variance_usd'].sum()
        metrics['total_net_variance'] = total_variance
        insights.append(f"Total Net Variance (All Months): ${total_variance:,.2f} (Positive=Under Budget)")

        # Visualizations
        variance_over_time = df.groupby('month')['variance_usd'].sum().reset_index()
        fig1 = px.bar(variance_over_time, x='month', y='variance_usd', title="Net Budget Variance Over Time",
                      color='variance_usd', color_continuous_scale='RdBu_r')
        visualizations['net_variance_over_time'] = fig1.to_json()

        variance_by_dept = df.groupby('department')['variance_usd'].sum().reset_index()
        fig2 = px.bar(variance_by_dept, x='department', y='variance_usd', title="Total Net Variance by Department")
        visualizations['net_variance_by_department'] = fig2.to_json()

        insights.append("Generated visualizations for monthly budget variance.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def corporate_liquidity_risk_monitoring_analysis(df):
    analysis_type = "Corporate Liquidity Risk Monitoring Analysis"
    try:
        expected = ['metric_name', 'value_usd', 'threshold_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['value_usd', 'threshold_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Analysis
        df['status'] = df.apply(lambda row: 'Breached' if row['value_usd'] < row['threshold_usd'] else 'Healthy', axis=1)
        breached_count = (df['status'] == 'Breached').sum()
        
        metrics['breached_metric_count'] = breached_count
        metrics['breached_metrics'] = df[df['status'] == 'Breached'][['metric_name', 'value_usd', 'threshold_usd']].to_dict('records')
        
        insights.append(f"Number of Breached Liquidity Metrics: {breached_count}")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df['metric_name'], y=df['value_usd'], name='Actual Value', marker_color=df['status'].map({'Breached': 'red', 'Healthy': 'green'})))
        fig.add_trace(go.Scatter(x=df['metric_name'], y=df['threshold_usd'], name='Threshold', mode='lines+markers', line=dict(color='black', dash='dash')))
        fig.update_layout(title="Liquidity Metrics vs. Thresholds")
        visualizations['liquidity_metrics_vs_thresholds'] = fig.to_json()
        
        insights.append("Generated visualization for liquidity risk monitoring.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def capital_expenditure_capex_project_analysis(df):
    analysis_type = "Capital Expenditure (CapEx) Project Analysis"
    try:
        expected = ['project_id', 'department', 'amount_usd', 'asset_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df.dropna(subset=['department', 'amount_usd', 'asset_category'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_capex = df['amount_usd'].sum()
        top_dept = df.groupby('department')['amount_usd'].sum().idxmax()

        metrics['total_capex'] = total_capex
        metrics['top_department_by_spend'] = top_dept

        insights.append(f"Total CapEx: ${total_capex:,.2f}")
        insights.append(f"Top Department by Spend: {top_dept}")

        # Visualizations
        capex_by_dept = df.groupby('department')['amount_usd'].sum().reset_index()
        fig1 = px.bar(capex_by_dept, x='department', y='amount_usd', title="Total CapEx by Department")
        visualizations['capex_by_department'] = fig1.to_json()

        capex_by_asset = df.groupby('asset_category')['amount_usd'].sum().reset_index()
        fig2 = px.pie(capex_by_asset, names='asset_category', values='amount_usd', title="CapEx Distribution by Asset Category")
        visualizations['capex_by_asset_category'] = fig2.to_json()

        insights.append("Generated visualizations for CapEx analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def corporate_debt_issuance_and_structure_analysis(df):
    analysis_type = "Corporate Debt Issuance and Structure Analysis"
    try:
        expected = ['issue_date', 'instrument_type', 'principal_usd', 'coupon_perc', 'maturity_date', 'rating']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['issue_date'] = pd.to_datetime(df['issue_date'], errors='coerce')
        for col in ['principal_usd', 'coupon_perc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['instrument_type', 'principal_usd', 'coupon_perc', 'rating'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_principal = df['principal_usd'].sum()
        avg_coupon = (df['principal_usd'] * df['coupon_perc']).sum() / total_principal if total_principal > 0 else 0

        metrics['total_principal_issued'] = total_principal
        metrics['weighted_average_coupon_percent'] = avg_coupon

        insights.append(f"Total Principal Issued: ${total_principal:,.2f}")
        insights.append(f"Weighted Average Coupon: {avg_coupon:.2f}%")

        # Visualizations
        issuance_by_type = df.groupby('instrument_type')['principal_usd'].sum().reset_index()
        fig1 = px.pie(issuance_by_type, names='instrument_type', values='principal_usd', title="Debt Principal by Instrument Type")
        visualizations['debt_by_instrument_type'] = fig1.to_json()

        issuance_by_rating = df.groupby('rating')['principal_usd'].sum().reset_index()
        fig2 = px.bar(issuance_by_rating, x='rating', y='principal_usd', title="Debt Principal by Credit Rating")
        visualizations['debt_by_credit_rating'] = fig2.to_json()

        insights.append("Generated visualizations for debt structure analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def securities_lending_transaction_analysis(df):
    analysis_type = "Securities Lending Transaction Analysis"
    try:
        expected = ['transaction_date', 'lender_id', 'borrower_id', 'instrument', 'fee_amount_usd', 'collateral_value_usd']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        for col in ['fee_amount_usd', 'collateral_value_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['instrument', 'fee_amount_usd', 'collateral_value_usd', 'borrower_id'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        total_fees = df['fee_amount_usd'].sum()
        total_collateral = df['collateral_value_usd'].sum()

        metrics['total_lending_fees'] = total_fees
        metrics['total_collateral_value'] = total_collateral

        insights.append(f"Total Lending Fees: ${total_fees:,.2f}")
        insights.append(f"Total Collateral Value: ${total_collateral:,.2f}")

        # Visualizations
        fees_by_instrument = df.groupby('instrument')['fee_amount_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(fees_by_instrument, x='instrument', y='fee_amount_usd', title="Top 15 Instruments by Lending Fees Generated")
        visualizations['top_instruments_by_fees'] = fig1.to_json()

        fees_by_borrower = df.groupby('borrower_id')['fee_amount_usd'].sum().nlargest(15).reset_index()
        fig2 = px.bar(fees_by_borrower, x='borrower_id', y='fee_amount_usd', title="Top 15 Borrowers by Fees Paid")
        visualizations['top_borrowers_by_fees'] = fig2.to_json()

        insights.append("Generated visualizations for securities lending.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }

def treasury_operations_and_trading_analysis(df):
    analysis_type = "Treasury Operations and Trading Analysis"
    try:
        expected = ['trade_date', 'security', 'buy_sell', 'trade_value_usd', 'counterparty']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
        
        df = safe_rename(df, matched)
        df['trade_date'] = pd.to_datetime(df['trade_date'], errors='coerce')
        df['trade_value_usd'] = pd.to_numeric(df['trade_value_usd'], errors='coerce')
        df.dropna(subset=['trade_date', 'security', 'buy_sell', 'trade_value_usd'], inplace=True)
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No data available after processing.",
                "visualizations": {}, "metrics": {}, "insights": ["No valid data to analyze."], "matched_columns": matched
            }

        metrics = {}
        visualizations = {}
        insights = []

        # Metrics
        buy_volume = df[df['buy_sell'].astype(str).str.lower() == 'buy']['trade_value_usd'].sum()
        sell_volume = df[df['buy_sell'].astype(str).str.lower() == 'sell']['trade_value_usd'].sum()
        net_volume = buy_volume - sell_volume

        metrics['total_buy_volume'] = buy_volume
        metrics['total_sell_volume'] = sell_volume
        metrics['net_volume'] = net_volume

        insights.append(f"Total Buy Volume: ${buy_volume:,.2f}")
        insights.append(f"Total Sell Volume: ${sell_volume:,.2f}")
        insights.append(f"Net Volume: ${net_volume:,.2f}")

        # Visualizations
        volume_by_security = df.groupby('security')['trade_value_usd'].sum().nlargest(15).reset_index()
        fig1 = px.bar(volume_by_security, x='security', y='trade_value_usd', title="Trade Volume by Security Type")
        visualizations['trade_volume_by_security'] = fig1.to_json()

        # Create a net flow for visualization
        df['flow'] = df.apply(lambda row: row['trade_value_usd'] if row['buy_sell'].lower() == 'buy' else -row['trade_value_usd'], axis=1)
        net_flow_over_time = df.groupby('trade_date')['flow'].sum().reset_index()
        fig2 = px.line(net_flow_over_time, x='trade_date', y='flow', title="Net Trade Flow Over Time")
        visualizations['net_trade_flow_over_time'] = fig2.to_json()

        insights.append("Generated visualizations for treasury operations.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


# ========== API/BACKEND FUNCTIONS ==========

# Main analysis mapping dictionary
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
    "general_insights": show_general_insights # Add general insights as a choice
}

def run_analysis(df, analysis_name):
    """Main function to run any analysis by name"""
    func = specific_finance_function_mapping.get(analysis_name)
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
        # Pass df directly. If func is show_general_insights, it will handle it.
        if analysis_name == "general_insights":
             return func(df)
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

def load_data(file_path_or_buffer, encoding='utf-8'):
    """Load data from CSV or Excel file path or buffer"""
    try:
        # Check if it's a file path (string) or a buffer
        if isinstance(file_path_or_buffer, str):
            file_name = file_path_or_buffer.lower()
            if file_name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer, encoding=encoding)
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError("Unsupported file type. Please provide CSV or Excel file.")
        
        # If it's a buffer (like from a file upload)
        else:
            # We need to figure out the file type, often from the 'name' attribute
            file_name = getattr(file_path_or_buffer, 'name', '').lower()
            if file_name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer, encoding=encoding)
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # read_excel can handle buffers directly
                return pd.read_excel(file_path_or_buffer)
            else:
                # Fallback: Try reading as CSV, then Excel
                try:
                    # Reset buffer position
                    file_path_or_buffer.seek(0)
                    return pd.read_csv(file_path_or_buffer, encoding=encoding)
                except Exception:
                    try:
                        # Reset buffer position
                        file_path_or_buffer.seek(0)
                        return pd.read_excel(file_path_or_buffer)
                    except Exception:
                         raise ValueError("Could not determine file type from buffer. Please provide a .csv or .xlsx file.")
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def main_backend(file_path_or_buffer, analysis_name, encoding='utf-8'):
    """Main backend function to load data and run analysis"""
    # Load data
    df = load_data(file_path_or_buffer, encoding)
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
    # Run general insights
    file_path = "sample_finance_data.csv"
    result = main_backend(file_path)
    print("General Insights:", result.keys() if isinstance(result, dict) else "No result")
    
    # Run specific analysis
    result = main_backend(
        file_path, 
        category="Specific", 
        specific_analysis_name="Attrition Analysis"
    )
    print("Attrition Analysis completed:", "status" in result if isinstance(result, dict) else "No result")