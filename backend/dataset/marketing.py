import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, chi2_contingency, f_oneway
from statsmodels.stats.proportion import proportions_ztest
from datetime import datetime
import warnings
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

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
        # Using json.dumps and json.loads is a robust way to ensure conversion
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception:
        # Fallback for complex unhandled types
        return str(data)

def check_and_rename_columns(df, expected_cols_map):
    """
    Checks for potential column names and renames them to a standard.
    Returns the renamed DataFrame, a list of missing standard columns,
    and a dictionary of matched columns.
    """
    missing_cols = []
    renamed_df = df.copy()
    matched_cols = {}

    for standard_name, potential_names in expected_cols_map.items():
        found = False
        for p_name in potential_names:
            if p_name in renamed_df.columns:
                if p_name != standard_name:
                    renamed_df = renamed_df.rename(columns={p_name: standard_name})
                matched_cols[standard_name] = p_name
                found = True
                break
        if not found:
            # Check if the standard name itself is already present
            if standard_name in renamed_df.columns:
                 matched_cols[standard_name] = standard_name
                 found = True
            else:
                missing_cols.append(standard_name)
                matched_cols[standard_name] = None
                
    return renamed_df, missing_cols, matched_cols

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """
    Provides comprehensive general insights with visualizations and metrics.
    This is the new fallback function.
    """
    analysis_type = "General Insights"
    visualizations = {}
    metrics = {}
    insights = []

    try:
        # Basic dataset information
        total_rows, total_columns = df.shape
        
        # Data types
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
        other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]

        # Missing values
        missing_values = df.isnull().sum()
        columns_with_missing = missing_values[missing_values > 0]
        total_missing = int(columns_with_missing.sum())

        # Duplicate rows
        duplicate_rows = int(df.duplicated().sum())

        # Compile metrics
        metrics = {
            "dataset_overview": {
                "total_rows": total_rows,
                "total_columns": total_columns,
                "duplicate_rows": duplicate_rows,
                "total_missing_values": total_missing
            },
            "data_types_breakdown": {
                "numeric_columns": len(numeric_cols),
                "categorical_columns": len(categorical_cols),
                "datetime_columns": len(datetime_cols),
                "other_columns": len(other_cols)
            }
        }

        # Compile insights
        insights.append(f"Dataset contains {total_rows:,} rows and {total_columns} columns.")
        insights.append(f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns.")
        
        if duplicate_rows > 0:
            insights.append(f"Warning: Found {duplicate_rows:,} duplicate rows.")
        else:
            insights.append("No duplicate rows found.")
        
        if total_missing > 0:
            insights.append(f"Found {total_missing:,} total missing values across {len(columns_with_missing)} columns.")
        else:
            insights.append("No missing values found.")

        # Add missing columns warning if provided (from a fallback)
        if missing_cols and len(missing_cols) > 0:
            insights.append("---")
            insights.append(f"⚠️ Fallback Alert: The requested analysis '{analysis_name}' could not be run.")
            insights.append("The following required columns were not found:")
            for col in missing_cols:
                match_info = f" (Best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (No close match found)"
                insights.append(f"  - {col}{match_info}")
            insights.append("Showing General Dataset Insights instead.")

        # Visualizations
        # 1. Data types distribution
        try:
            dtype_counts = {
                'Numeric': len(numeric_cols),
                'Categorical': len(categorical_cols),
                'Datetime': len(datetime_cols),
                'Other': len(other_cols)
            }
            fig_dtypes = px.pie(values=list(dtype_counts.values()), names=list(dtype_counts.keys()),
                                title='Data Types Distribution')
            visualizations["data_types_distribution"] = fig_dtypes.to_json()
        except Exception as e:
            insights.append(f"Could not generate data type visualization: {e}")

        # 2. Missing values visualization
        try:
            if total_missing > 0:
                missing_df = columns_with_missing.reset_index().rename(columns={'index': 'column', 0: 'count'}).sort_values(by='count', ascending=False)
                fig_missing = px.bar(missing_df.head(10), x='column', y='count',
                                     title='Top 10 Columns with Missing Values')
                visualizations["missing_values"] = fig_missing.to_json()
        except Exception as e:
            insights.append(f"Could not generate missing values visualization: {e}")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched_cols or {},
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred during general insights: {e}"]
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides general insights as a fallback.
    """
    print(f"--- ⚠️ Required Columns Not Found for {analysis_name} ---")
    print(f"Missing: {missing_cols}")
    print("Falling back to General Insights.")
    
    # Generate general insights, passing the missing column info for inclusion in the report
    general_insights_data = show_general_insights(
        df, 
        analysis_name, # Pass the original analysis name for the warning message
        missing_cols=missing_cols,
        matched_cols=matched_cols
    )
    
    # Re-brand the general insights as a fallback response
    general_insights_data["analysis_type"] = analysis_name # Show the analysis that was *attempted*
    general_insights_data["status"] = "fallback" # Set status to fallback
    general_insights_data["message"] = f"Required columns were missing for '{analysis_name}'. Falling back to general insights."
    
    return general_insights_data

# ========== DATA LOADING ==========
def load_data(file_path, encoding='utf-8'):
    """Load data from CSV or Excel file with robust encoding support"""
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    print(f"File loaded successfully with {enc} encoding.")
                    return df
                except UnicodeDecodeError:
                    continue
            print("[ERROR] Failed to decode file with common encodings.")
            return None
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
            print("Excel file loaded successfully.")
            return df
        else:
            print("[ERROR] Unsupported file format. Please provide CSV or Excel file.")
            return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

# ========== REFACTORED ANALYSIS FUNCTIONS ==========

def a_b_testing_campaign_variant_analysis(df):
    analysis_name = "A/B Testing Campaign Variant Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_map = {
            'Variant': ['Variant', 'CampaignVariant', 'TreatmentGroup'],
            'Conversions': ['Conversions', 'NumConversions', 'Converted'],
            'TotalUsers': ['TotalUsers', 'Users', 'Impressions', 'Visits'],
            'Revenue': ['Revenue', 'TotalRevenue', 'Sales']
        }
        
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)
        
        critical_missing = [col for col in ['Variant', 'Conversions', 'TotalUsers'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
        df['TotalUsers'] = pd.to_numeric(df['TotalUsers'], errors='coerce')
        if 'Revenue' in df.columns:
            df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
        
        df = df.dropna(subset=['Variant', 'Conversions', 'TotalUsers'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['ConversionRate'] = df['Conversions'] / df['TotalUsers']
        if 'Revenue' in df.columns:
            df['AvgRevenuePerUser'] = df['Revenue'] / df['TotalUsers']
        else:
            df['AvgRevenuePerUser'] = np.nan

        variant_summary = df.groupby('Variant').agg(
            TotalUsers=('TotalUsers', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            MeanConversionRate=('ConversionRate', 'mean'),
            TotalRevenue=('Revenue', 'sum') if 'Revenue' in df.columns else ('TotalUsers', 'size')
        ).reset_index()
        
        variant_summary['OverallConversionRate'] = variant_summary['TotalConversions'] / variant_summary['TotalUsers']
        if 'Revenue' in df.columns:
            variant_summary['OverallRevenuePerUser'] = variant_summary['TotalRevenue'] / variant_summary['TotalUsers']
        else:
            variant_summary['OverallRevenuePerUser'] = np.nan

        insights.append(f"Analyzed {len(variant_summary)} variants.")

        # A/B Test
        stat, pval = 0.0, 0.0
        ab_test_results = "Not applicable for != 2 variants"
        
        if len(variant_summary) == 2:
            variant_A = variant_summary.loc[0]
            variant_B = variant_summary.loc[1]
            
            count = np.array([variant_A['TotalConversions'], variant_B['TotalConversions']])
            nobs = np.array([variant_A['TotalUsers'], variant_B['TotalUsers']])
            
            # Ensure nobs are > 0
            if nobs.sum() > 0 and count.sum() >= 0:
                stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
                ab_test_results = {"z_statistic": stat, "p_value": pval}
                
                insights.append(f"Z-test (Variant 0 vs 1): Z-stat={stat:.3f}, P-value={pval:.3f}")
                if pval < 0.05:
                    insights.append("Result: Statistically significant difference in conversion rates found.")
                else:
                    insights.append("Result: No statistically significant difference in conversion rates found.")
            else:
                insights.append("Could not run Z-test (no observations or invalid conversion data).")
                ab_test_results = "Could not run Z-test (no observations or invalid conversion data)."

        # Visualizations
        fig1 = px.bar(variant_summary, x='Variant', y='OverallConversionRate', title='Overall Conversion Rate by Variant')
        visualizations["Conversion_Rate_by_Variant_Bar"] = fig1.to_json()

        fig2_json = None
        if 'Revenue' in df.columns and not variant_summary['OverallRevenuePerUser'].isnull().all():
            fig2 = px.bar(variant_summary, x='Variant', y='OverallRevenuePerUser', title='Overall Revenue Per User by Variant')
            fig2_json = fig2.to_json()
        visualizations["Revenue_Per_User_by_Variant_Bar"] = fig2_json
        
        fig3 = px.histogram(df, x='ConversionRate', color='Variant', barmode='overlay', title='Distribution of Conversion Rates by Variant')
        visualizations["Conversion_Rates_Distribution_Histogram"] = fig3.to_json()

        metrics = {
            "variant_summary": variant_summary.to_dict(orient='records'),
            "ab_test_results": ab_test_results
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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def media_buying_and_ad_performance_analysis(df):
    analysis_name = "Media Buying and Ad Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_map = {
            'AdCampaignID': ['AdCampaignID', 'CampaignID', 'ID'],
            'Platform': ['Platform', 'AdPlatform', 'Channel'],
            'Spend': ['Spend', 'AdSpend', 'Cost'],
            'Impressions': ['Impressions', 'AdImpressions'],
            'Clicks': ['Clicks', 'AdClicks'],
            'Conversions': ['Conversions', 'AdConversions', 'Purchases'],
            'TargetAudience': ['TargetAudience', 'AudienceSegment']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Spend', 'Impressions', 'Clicks', 'Conversions'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Spend', 'Impressions', 'Clicks', 'Conversions']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Spend', 'Impressions', 'Clicks', 'Conversions'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CPM'] = (df['Spend'] / df['Impressions']) * 1000
        df['CPC'] = df['Spend'] / df['Clicks']
        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['CPA'] = df['Spend'] / df['Conversions']
        
        # Handle potential division by zero if sums are 0
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        group_col = 'AdCampaignID' if 'AdCampaignID' in df.columns else 'Platform'
        if group_col not in df.columns:
            group_col = 'AdCampaignID' # dummy for fallback
            df[group_col] = 'Default_Campaign'
            insights.append("Warning: No 'AdCampaignID' or 'Platform' found. Grouping all data as 'Default_Campaign'.")
            matched['AdCampaignID'] = None # Ensure it's marked as missing
        
        campaign_summary = df.groupby(group_col).agg(
            TotalSpend=('Spend', 'sum'),
            TotalImpressions=('Impressions', 'sum'),
            TotalClicks=('Clicks', 'sum'),
            TotalConversions=('Conversions', 'sum')
        ).reset_index()

        campaign_summary['OverallCPM'] = (campaign_summary['TotalSpend'] / campaign_summary['TotalImpressions']) * 1000
        campaign_summary['OverallCPC'] = campaign_summary['TotalSpend'] / campaign_summary['TotalClicks']
        campaign_summary['OverallCTR'] = (campaign_summary['TotalClicks'] / campaign_summary['TotalImpressions']) * 100
        campaign_summary['OverallConversionRate'] = (campaign_summary['TotalConversions'] / campaign_summary['TotalClicks']) * 100
        campaign_summary['OverallCPA'] = campaign_summary['TotalSpend'] / campaign_summary['TotalConversions']
        
        campaign_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"campaign_summary": campaign_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(campaign_summary)} campaigns/platforms.")
        
        # Visualizations
        fig1 = px.bar(campaign_summary, x=group_col, y='OverallCPA', title='Cost Per Acquisition (CPA) by Campaign')
        visualizations["CPA_by_Campaign_Bar"] = fig1.to_json()

        fig2 = px.bar(campaign_summary, x=group_col, y='OverallCTR', title='Click-Through Rate (CTR) by Campaign')
        visualizations["CTR_by_Campaign_Bar"] = fig2.to_json()

        fig3_json = None
        if 'Platform' in df.columns:
            platform_performance = df.groupby('Platform').agg(
                AvgCTR=('CTR', 'mean'),
                AvgCPA=('CPA', 'mean'),
                TotalSpend=('Spend', 'sum')
            ).reset_index()
            fig3 = px.bar(platform_performance, x='Platform', y='AvgCTR', title='Average CTR by Platform')
            fig3_json = fig3.to_json()
            metrics["platform_performance"] = platform_performance.to_dict(orient='records')
        visualizations["Avg_CTR_by_Platform_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def product_line_marketing_campaign_analysis(df):
    analysis_name = "Product Line Marketing Campaign Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'ProductLine': ['ProductLine', 'ProductCategory', 'ProductGroup'],
            'CampaignID': ['CampaignID', 'MarketingCampaignID', 'ID'],
            'Budget': ['Budget', 'CampaignBudget', 'Spend'],
            'SalesUnits': ['SalesUnits', 'UnitsSold', 'Quantity'],
            'Revenue': ['Revenue', 'TotalRevenue', 'Sales'],
            'ConversionRate': ['ConversionRate', 'ConversionPct']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['ProductLine', 'Budget', 'SalesUnits', 'Revenue'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Budget', 'SalesUnits', 'Revenue', 'ConversionRate']:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['ProductLine', 'Budget', 'SalesUnits', 'Revenue'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['ROI'] = ((df['Revenue'] - df['Budget']) / df['Budget']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        product_line_summary = df.groupby('ProductLine').agg(
            TotalBudget=('Budget', 'sum'),
            TotalSalesUnits=('SalesUnits', 'sum'),
            TotalRevenue=('Revenue', 'sum'),
            AvgROI=('ROI', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean') if 'ConversionRate' in df.columns else ('SalesUnits', 'size') # Placeholder
        ).reset_index()

        metrics = {"product_line_summary": product_line_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(product_line_summary)} product lines.")
        
        # Visualizations
        fig1 = px.bar(product_line_summary, x='ProductLine', y='TotalRevenue', title='Total Revenue by Product Line')
        visualizations["Total_Revenue_by_Product_Line_Bar"] = fig1.to_json()

        fig2 = px.bar(product_line_summary, x='ProductLine', y='AvgROI', title='Average ROI by Product Line')
        visualizations["Average_ROI_by_Product_Line_Bar"] = fig2.to_json()

        hover_col = 'CampaignID' if 'CampaignID' in df.columns else 'ProductLine'
        fig3 = px.scatter(df, x='Budget', y='Revenue', color='ProductLine', hover_name=hover_col,
                          title='Revenue vs. Budget by Product Line')
        visualizations["Revenue_vs_Budget_Scatter"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def website_engagement_and_ad_interaction_analysis(df):
    analysis_name = "Website Engagement and Ad Interaction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_map = {
            'UserID': ['UserID', 'VisitorID', 'CustomerID'],
            'PageViews': ['PageViews', 'PagesVisited', 'Views'],
            'TimeOnSiteSeconds': ['TimeOnSiteSeconds', 'SessionDuration', 'Duration'],
            'AdClicked': ['AdClicked', 'ClickedAd', 'AdInteraction'], # Binary (1/0 or True/False)
            'ConversionEvent': ['ConversionEvent', 'Converted', 'PurchaseEvent'] # Binary (1/0 or True/False)
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['PageViews', 'TimeOnSiteSeconds', 'AdClicked', 'ConversionEvent'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df_renamed
        
        df['PageViews'] = pd.to_numeric(df['PageViews'], errors='coerce')
        df['TimeOnSiteSeconds'] = pd.to_numeric(df['TimeOnSiteSeconds'], errors='coerce')
        
        df['AdClicked'] = df['AdClicked'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})

        df = df.dropna(subset=['PageViews', 'TimeOnSiteSeconds', 'AdClicked', 'ConversionEvent'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        avg_page_views = df['PageViews'].mean()
        avg_time_on_site_minutes = (df['TimeOnSiteSeconds'].mean() / 60)
        ad_click_rate = (df['AdClicked'].mean() * 100)
        overall_conversion_rate = (df['ConversionEvent'].mean() * 100)

        insights.append(f"Average Page Views per User: {avg_page_views:.1f}")
        insights.append(f"Average Time on Site: {avg_time_on_site_minutes:.1f} minutes")
        insights.append(f"Ad Click Rate: {ad_click_rate:.2f}%")
        insights.append(f"Overall Conversion Rate: {overall_conversion_rate:.2f}%")

        ad_interaction_summary = df.groupby('AdClicked').agg(
            AvgPageViews=('PageViews', 'mean'),
            AvgTimeOnSite=('TimeOnSiteSeconds', 'mean'),
            ConversionRate=('ConversionEvent', 'mean')
        ).reset_index()
        ad_interaction_summary['AvgTimeOnSite'] /= 60 # Convert to minutes
        ad_interaction_summary['ConversionRate'] *= 100

        metrics = {
            "Average Page Views": avg_page_views,
            "Average Time on Site (minutes)": avg_time_on_site_minutes,
            "Ad Click Rate": ad_click_rate,
            "Overall Conversion Rate": overall_conversion_rate,
            "Performance by Ad Interaction": ad_interaction_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.box(df, x='AdClicked', y='PageViews', title='Page Views by Ad Click Status')
        visualizations["Page_Views_by_Ad_Click_Box"] = fig1.to_json()

        fig2 = px.box(df, x='AdClicked', y='TimeOnSiteSeconds', title='Time on Site by Ad Click Status (Seconds)')
        visualizations["Time_on_Site_by_Ad_Click_Box"] = fig2.to_json()

        fig3 = px.bar(ad_interaction_summary, x='AdClicked', y='ConversionRate', title='Conversion Rate by Ad Click Status')
        visualizations["Conversion_Rate_by_Ad_Click_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def daily_sales_revenue_and_campaign_correlation_analysis(df):
    analysis_name = "Daily Sales Revenue and Campaign Correlation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'Date': ['Date', 'SaleDate', 'Day'],
            'DailyRevenue': ['DailyRevenue', 'Revenue', 'SalesAmount'],
            'MarketingSpend': ['MarketingSpend', 'CampaignSpend', 'AdSpend'],
            'CampaignType': ['CampaignType', 'Type', 'CampaignName']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Date', 'DailyRevenue', 'MarketingSpend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DailyRevenue'] = pd.to_numeric(df['DailyRevenue'], errors='coerce')
        df['MarketingSpend'] = pd.to_numeric(df['MarketingSpend'], errors='coerce')
        df = df.dropna(subset=['Date', 'DailyRevenue', 'MarketingSpend'])
        df = df.sort_values('Date')

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_revenue = df['DailyRevenue'].sum()
        total_spend = df['MarketingSpend'].sum()
        revenue_spend_correlation = df['DailyRevenue'].corr(df['MarketingSpend'])

        metrics = {
            "Total Revenue": total_revenue,
            "Total Marketing Spend": total_spend,
            "Revenue vs Spend Correlation": revenue_spend_correlation
        }
        
        insights.append(f"Total Revenue: ${total_revenue:,.2f}")
        insights.append(f"Total Marketing Spend: ${total_spend:,.2f}")
        insights.append(f"Correlation (Daily Revenue vs. Marketing Spend): {revenue_spend_correlation:.2f}")

        # Visualizations
        fig1 = px.line(df, x='Date', y='DailyRevenue', title='Daily Revenue Trend')
        visualizations["Daily_Revenue_Trend_Line"] = fig1.to_json()

        fig2 = px.scatter(df, x='MarketingSpend', y='DailyRevenue', color='CampaignType' if 'CampaignType' in df.columns else None,
                          title='Daily Revenue vs. Marketing Spend')
        visualizations["Daily_Revenue_vs_Marketing_Spend_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'CampaignType' in df.columns:
            campaign_type_impact = df.groupby('CampaignType').agg(
                AvgDailyRevenue=('DailyRevenue', 'mean'),
                AvgMarketingSpend=('MarketingSpend', 'mean')
            ).reset_index()
            fig3 = px.bar(campaign_type_impact, x='CampaignType', y='AvgDailyRevenue', title='Average Daily Revenue by Campaign Type')
            fig3_json = fig3.to_json()
            metrics["campaign_type_impact"] = campaign_type_impact.to_dict(orient='records')
        visualizations["Avg_Daily_Revenue_by_Campaign_Type_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def influencer_marketing_campaign_performance_analysis(df):
    analysis_name = "Influencer Marketing Campaign Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'InfluencerID': ['InfluencerID', 'Influencer', 'ID'],
            'CampaignName': ['CampaignName', 'Campaign', 'MarketingCampaign'],
            'FollowerCount': ['FollowerCount', 'Followers'],
            'EngagementRate': ['EngagementRate', 'AvgEngagement'],
            'PostsCount': ['PostsCount', 'NumPosts'],
            'Reach': ['Reach', 'TotalReach'],
            'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
            'Spend': ['Spend', 'PaymentToInfluencer', 'Cost']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['InfluencerID', 'FollowerCount', 'EngagementRate', 'Conversions', 'Spend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['FollowerCount', 'EngagementRate', 'PostsCount', 'Reach', 'Conversions', 'Spend']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['InfluencerID', 'FollowerCount', 'EngagementRate', 'Conversions', 'Spend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }
        
        # Note: This ROI calculation is a placeholder and may need adjustment based on business logic (e.g., avg. conversion value)
        avg_conversion_value = (df['Spend'].sum() / df['Conversions'].sum()) if df['Conversions'].sum() > 0 else 0
        if avg_conversion_value == 0:
            insights.append("Warning: Could not determine average conversion value. ROI calculation may be inaccurate.")
            avg_conversion_value = df['Spend'].mean() # Fallback

        df['CalculatedRevenue'] = df['Conversions'] * avg_conversion_value
        df['ROI'] = ((df['CalculatedRevenue'] - df['Spend']) / df['Spend']) * 100
        df['CostPerConversion'] = df['Spend'] / df['Conversions']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)


        influencer_summary = df.groupby('InfluencerID').agg(
            AvgFollowers=('FollowerCount', 'mean'),
            AvgEngagementRate=('EngagementRate', 'mean'),
            TotalConversions=('Conversions', 'sum'),
            TotalSpend=('Spend', 'sum'),
            AvgCostPerConversion=('CostPerConversion', 'mean'),
            AvgROI=('ROI', 'mean')
        ).reset_index()

        metrics = {"influencer_summary": influencer_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(influencer_summary)} influencers.")

        # Visualizations
        fig1 = px.scatter(df, x='EngagementRate', y='Conversions', size='FollowerCount', color='Spend',
                          hover_name='InfluencerID', title='Conversions vs. Engagement Rate (Sized by Followers, Colored by Spend)')
        visualizations["Conversions_vs_Engagement_Scatter"] = fig1.to_json()

        fig2 = px.bar(influencer_summary.sort_values('TotalConversions', ascending=False).head(10),
                      x='InfluencerID', y='TotalConversions', title='Top 10 Influencers by Total Conversions')
        visualizations["Top_10_Influencers_by_Conversions_Bar"] = fig2.to_json()

        fig3_json = None
        if 'CampaignName' in df.columns:
            campaign_performance = df.groupby('CampaignName').agg(
                TotalSpend=('Spend', 'sum'),
                TotalConversions=('Conversions', 'sum')
            ).reset_index()
            campaign_performance['CostPerConversion'] = campaign_performance['TotalSpend'] / campaign_performance['TotalConversions']
            campaign_performance.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            fig3 = px.bar(campaign_performance, x='CampaignName', y='CostPerConversion', title='Cost Per Conversion by Campaign')
            fig3_json = fig3.to_json()
            metrics["campaign_performance"] = campaign_performance.to_dict(orient='records')
        visualizations["Cost_Per_Conversion_by_Campaign_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def ad_copy_performance_analysis(df):
    analysis_name = "Ad Copy Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'AdCopyID': ['AdCopyID', 'CopyID', 'ID'],
            'AdCopyText': ['AdCopyText', 'CopyText', 'Headline'],
            'Clicks': ['Clicks', 'AdClicks'],
            'Impressions': ['Impressions', 'AdImpressions'],
            'Conversions': ['Conversions', 'AdConversions', 'Purchases'],
            'AdGroup': ['AdGroup', 'TargetGroup']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['AdCopyID', 'Clicks', 'Impressions', 'Conversions'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Clicks', 'Impressions', 'Conversions']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['AdCopyID', 'Clicks', 'Impressions', 'Conversions'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        ad_copy_summary = df.groupby('AdCopyID').agg(
            TotalImpressions=('Impressions', 'sum'),
            TotalClicks=('Clicks', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean')
        ).reset_index()

        metrics = {"ad_copy_summary": ad_copy_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(ad_copy_summary)} unique ad copies.")

        # Visualizations
        fig1 = px.bar(ad_copy_summary.sort_values('AvgCTR', ascending=False).head(10),
                      x='AdCopyID', y='AvgCTR', title='Top 10 Ad Copies by Average CTR')
        visualizations["Top_10_Ad_Copies_by_CTR_Bar"] = fig1.to_json()

        fig2 = px.bar(ad_copy_summary.sort_values('AvgConversionRate', ascending=False).head(10),
                      x='AdCopyID', y='AvgConversionRate', title='Top 10 Ad Copies by Average Conversion Rate')
        visualizations["Top_10_Ad_Copies_by_Conversion_Rate_Bar"] = fig2.to_json()

        fig3_json = None
        if 'AdGroup' in df.columns:
            ad_group_performance = df.groupby('AdGroup').agg(
                AvgCTR=('CTR', 'mean'),
                AvgConversionRate=('ConversionRate', 'mean')
            ).reset_index()
            fig3 = px.bar(ad_group_performance, x='AdGroup', y='AvgCTR', title='Average CTR by Ad Group')
            fig3_json = fig3.to_json()
            metrics["ad_group_performance"] = ad_group_performance.to_dict(orient='records')
        visualizations["Avg_CTR_by_Ad_Group_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def customer_referral_program_analysis(df):
    analysis_name = "Customer Referral Program Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'ReferrerID': ['ReferrerID', 'ReferralSourceID', 'UserID'],
            'ReferredCustomerID': ['ReferredCustomerID', 'NewCustomerID'],
            'ReferralDate': ['ReferralDate', 'Date'],
            'ReferralConversionStatus': ['ReferralConversionStatus', 'Status', 'Converted'],
            'ReferralRevenue': ['ReferralRevenue', 'Revenue', 'SalesAmount'],
            'RewardType': ['RewardType', 'IncentiveType']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['ReferrerID', 'ReferralDate', 'ReferralConversionStatus'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['ReferralDate'] = pd.to_datetime(df['ReferralDate'], errors='coerce')
        df['ReferralConversionStatus'] = df['ReferralConversionStatus'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        if 'ReferralRevenue' in df.columns:
            df['ReferralRevenue'] = pd.to_numeric(df['ReferralRevenue'], errors='coerce')
        
        df = df.dropna(subset=['ReferrerID', 'ReferralDate', 'ReferralConversionStatus'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_referrals = len(df)
        successful_referrals = int(df['ReferralConversionStatus'].sum())
        referral_conversion_rate = (successful_referrals / total_referrals) * 100 if total_referrals > 0 else 0
        total_referral_revenue = 0
        if 'ReferralRevenue' in df.columns:
            total_referral_revenue = df['ReferralRevenue'].sum()

        metrics = {
            "Total Referrals": total_referrals,
            "Successful Referrals": successful_referrals,
            "Referral Conversion Rate": referral_conversion_rate,
            "Total Revenue from Referrals": total_referral_revenue
        }
        
        insights.append(f"Total Referrals Initiated: {total_referrals}")
        insights.append(f"Successful Referrals: {successful_referrals} (Rate: {referral_conversion_rate:.2f}%)")
        insights.append(f"Total Revenue from Referrals: ${total_referral_revenue:,.2f}")

        group_col = 'ReferredCustomerID' if 'ReferredCustomerID' in df.columns else 'ReferrerID' # Use 'ReferrerID' as proxy count if 'ReferredCustomerID' missing
        
        referrer_performance = df.groupby('ReferrerID').agg(
            NumReferrals=(group_col, 'count'),
            SuccessfulReferrals=('ReferralConversionStatus', 'sum'),
            TotalRevenue=('ReferralRevenue', 'sum') if 'ReferralRevenue' in df.columns else ('ReferrerID', 'size')
        ).reset_index()
        referrer_performance['ConversionRate'] = (referrer_performance['SuccessfulReferrals'] / referrer_performance['NumReferrals']) * 100
        referrer_performance.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics["referrer_performance_summary"] = referrer_performance.to_dict(orient='records')

        # Visualizations
        fig1 = px.bar(referrer_performance.sort_values('SuccessfulReferrals', ascending=False).head(10),
                      x='ReferrerID', y='SuccessfulReferrals', title='Top 10 Referrers by Successful Referrals')
        visualizations["Top_10_Referrers_Bar"] = fig1.to_json()

        fig2 = px.pie(df, names='ReferralConversionStatus', title='Overall Referral Conversion Status')
        visualizations["Referral_Conversion_Status_Pie"] = fig2.to_json()

        fig3_json = None
        if 'RewardType' in df.columns:
            reward_type_performance = df.groupby('RewardType')['ReferralConversionStatus'].mean().reset_index()
            reward_type_performance['ConversionRate'] = reward_type_performance['ReferralConversionStatus'] * 100
            fig3 = px.bar(reward_type_performance, x='RewardType', y='ConversionRate', title='Referral Conversion Rate by Reward Type')
            fig3_json = fig3.to_json()
            metrics["reward_type_performance"] = reward_type_performance.to_dict(orient='records')
        visualizations["Referral_Conversion_Rate_by_Reward_Type_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def website_platform_engagement_metrics_analysis(df):
    analysis_name = "Website/Platform Engagement Metrics Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'UserID': ['UserID', 'VisitorID', 'CustomerID'],
            'Date': ['Date', 'VisitDate', 'ActivityDate'],
            'PageViews': ['PageViews', 'PagesVisited', 'Views'],
            'TimeOnSiteSeconds': ['TimeOnSiteSeconds', 'SessionDuration', 'Duration'],
            'BounceRate': ['BounceRate', 'Bounced'], 
            'Conversions': ['Conversions', 'GoalsCompleted', 'Purchases']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Date', 'PageViews', 'TimeOnSiteSeconds', 'BounceRate'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['PageViews'] = pd.to_numeric(df['PageViews'], errors='coerce')
        df['TimeOnSiteSeconds'] = pd.to_numeric(df['TimeOnSiteSeconds'], errors='coerce')
        df['BounceRate'] = pd.to_numeric(df['BounceRate'], errors='coerce')
        if 'Conversions' in df.columns:
            df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
        
        df = df.dropna(subset=['Date', 'PageViews', 'TimeOnSiteSeconds', 'BounceRate'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        avg_page_views = df['PageViews'].mean()
        avg_time_on_site_minutes = (df['TimeOnSiteSeconds'].mean() / 60)
        avg_bounce_rate = df['BounceRate'].mean()
        total_conversions = 0
        if 'Conversions' in df.columns:
            total_conversions = df['Conversions'].sum()

        metrics = {
            "Average Page Views": avg_page_views,
            "Average Time on Site (minutes)": avg_time_on_site_minutes,
            "Average Bounce Rate": avg_bounce_rate,
            "Total Conversions": total_conversions
        }

        insights.append(f"Average Page Views per Session: {avg_page_views:.1f}")
        insights.append(f"Average Time on Site per Session: {avg_time_on_site_minutes:.1f} minutes")
        insights.append(f"Average Bounce Rate: {avg_bounce_rate:.2f}%")
        insights.append(f"Total Conversions: {total_conversions}")

        daily_engagement = df.groupby('Date').agg(
            AvgPageViews=('PageViews', 'mean'),
            AvgTimeOnSiteMinutes=('TimeOnSiteSeconds', lambda x: x.mean() / 60),
            AvgBounceRate=('BounceRate', 'mean')
        ).reset_index()
        metrics["daily_engagement_summary"] = daily_engagement.to_dict(orient='records')

        # Visualizations
        fig1 = px.line(daily_engagement, x='Date', y='AvgPageViews', title='Daily Average Page Views Trend')
        visualizations["Daily_Avg_Page_Views_Trend_Line"] = fig1.to_json()

        fig2 = px.line(daily_engagement, x='Date', y='AvgTimeOnSiteMinutes', title='Daily Average Time on Site Trend')
        visualizations["Daily_Avg_Time_on_Site_Trend_Line"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='BounceRate', title='Distribution of Bounce Rates')
        visualizations["Bounce_Rates_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }

def customer_loyalty_program_engagement_analysis(df):
    analysis_name = "Customer Loyalty Program Engagement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_map = {
            'CustomerID': ['CustomerID', 'UserID', 'ID'],
            'EnrollmentDate': ['EnrollmentDate', 'DateEnrolled'],
            'PointsEarned': ['PointsEarned', 'LoyaltyPoints', 'Points'],
            'PointsRedeemed': ['PointsRedeemed', 'RedeemedPoints'],
            'PurchasesCount': ['PurchasesCount', 'NumPurchases'],
            'TotalSpend': ['TotalSpend', 'Spend', 'Revenue'],
            'Tier': ['Tier', 'LoyaltyTier', 'ProgramLevel']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['CustomerID', 'EnrollmentDate', 'PointsEarned', 'PointsRedeemed', 'PurchasesCount', 'TotalSpend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['EnrollmentDate'] = pd.to_datetime(df['EnrollmentDate'], errors='coerce')
        for col in ['PointsEarned', 'PointsRedeemed', 'PurchasesCount', 'TotalSpend']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['CustomerID', 'EnrollmentDate', 'PointsEarned', 'PointsRedeemed', 'PurchasesCount', 'TotalSpend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_customers_in_program = len(df)
        avg_points_earned = df['PointsEarned'].mean()
        avg_points_redeemed = df['PointsRedeemed'].mean()
        avg_purchases_per_customer = df['PurchasesCount'].mean()
        
        insights.append(f"Total Customers in Program: {total_customers_in_program}")
        insights.append(f"Average Points Earned: {avg_points_earned:.0f}")
        insights.append(f"Average Points Redeemed: {avg_points_redeemed:.0f}")
        insights.append(f"Average Purchases per Customer: {avg_purchases_per_customer:.1f}")

        metrics = {
            "Total Customers in Program": total_customers_in_program,
            "Average Points Earned": avg_points_earned,
            "Average Points Redeemed": avg_points_redeemed,
            "Average Purchases per Customer": avg_purchases_per_customer,
        }
        
        fig1_json, fig2_json = None, None
        
        if 'Tier' in df.columns:
            program_summary = df.groupby('Tier').agg(
                NumCustomers=('CustomerID', 'count'),
                AvgPointsEarned=('PointsEarned', 'mean'),
                AvgPointsRedeemed=('PointsRedeemed', 'mean'),
                AvgTotalSpend=('TotalSpend', 'mean')
            ).reset_index()
            metrics["Loyalty Program Performance by Tier"] = program_summary.to_dict(orient='records')
            
            fig1 = px.bar(program_summary, x='Tier', y='NumCustomers', title='Number of Customers by Loyalty Tier')
            fig1_json = fig1.to_json()
            
            fig2 = px.scatter(df, x='PointsEarned', y='TotalSpend', color='Tier',
                              hover_name='CustomerID', title='Total Spend vs. Points Earned')
            fig2_json = fig2.to_json()
        else:
            insights.append("Skipping Tier-based analysis: 'Tier' column not found.")
            fig2 = px.scatter(df, x='PointsEarned', y='TotalSpend',
                              hover_name='CustomerID', title='Total Spend vs. Points Earned')
            fig2_json = fig2.to_json()


        visualizations["Customers_by_Loyalty_Tier_Bar"] = fig1_json
        visualizations["Total_Spend_vs_Points_Earned_Scatter"] = fig2_json
        
        fig3 = px.histogram(df, x='PointsRedeemed', title='Distribution of Points Redeemed')
        visualizations["Points_Redeemed_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def discount_code_redemption_and_visit_analysis(df):
    analysis_name = "Discount Code Redemption and Visit Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'CodeID': ['CodeID', 'DiscountCode', 'ID'],
            'CustomerID': ['CustomerID', 'UserID'],
            'Redeemed': ['Redeemed', 'IsRedeemed', 'RedemptionStatus'],
            'VisitsBeforeRedemption': ['VisitsBeforeRedemption', 'PreRedemptionVisits'],
            'VisitsAfterRedemption': ['VisitsAfterRedemption', 'PostRedemptionVisits'],
            'PurchaseValue': ['PurchaseValue', 'Revenue', 'Sales'],
            'DiscountPercentage': ['DiscountPercentage', 'DiscountRate']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['CodeID', 'Redeemed', 'VisitsBeforeRedemption'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df_renamed

        df['Redeemed'] = df['Redeemed'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        for col in ['VisitsBeforeRedemption', 'VisitsAfterRedemption', 'PurchaseValue', 'DiscountPercentage']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['CodeID', 'Redeemed', 'VisitsBeforeRedemption'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_codes_issued = len(df)
        total_redemptions = int(df['Redeemed'].sum())
        redemption_rate = (total_redemptions / total_codes_issued) * 100 if total_codes_issued > 0 else 0
        avg_visits_before_redemption = df['VisitsBeforeRedemption'].mean()
        avg_purchase_value_redeemed = np.nan
        if 'PurchaseValue' in df.columns:
            avg_purchase_value_redeemed = df[df['Redeemed']]['PurchaseValue'].mean()

        metrics = {
            "Total Codes Issued": total_codes_issued,
            "Total Redemptions": total_redemptions,
            "Redemption Rate": redemption_rate,
            "Average Visits Before Redemption": avg_visits_before_redemption,
            "Average Purchase Value (Redeemed)": avg_purchase_value_redeemed
        }
        
        insights.append(f"Total Discount Codes Issued: {total_codes_issued}")
        insights.append(f"Total Redemptions: {total_redemptions} (Rate: {redemption_rate:.2f}%)")
        insights.append(f"Average Visits Before Redemption: {avg_visits_before_redemption:.1f}")
        if not pd.isna(avg_purchase_value_redeemed):
            insights.append(f"Average Purchase Value for Redeemed Codes: ${avg_purchase_value_redeemed:,.2f}")

        redemption_summary = df.groupby('Redeemed').agg(
            AvgVisitsBefore=('VisitsBeforeRedemption', 'mean'),
            AvgVisitsAfter=('VisitsAfterRedemption', 'mean') if 'VisitsAfterRedemption' in df.columns else ('VisitsBeforeRedemption', 'size'),
            AvgPurchaseValue=('PurchaseValue', 'mean') if 'PurchaseValue' in df.columns else ('VisitsBeforeRedemption', 'size')
        ).reset_index()
        metrics["Redemption Summary"] = redemption_summary.to_dict(orient='records')

        # Visualizations
        fig1 = px.pie(df, names='Redeemed', title='Discount Code Redemption Status')
        visualizations["Redemption_Status_Pie"] = fig1.to_json()

        fig2 = px.box(df, x='Redeemed', y='VisitsBeforeRedemption', title='Visits Before Redemption by Status')
        visualizations["Visits_Before_Redemption_Box"] = fig2.to_json()

        fig3_json = None
        if 'VisitsAfterRedemption' in df.columns:
            fig3 = px.box(df, x='Redeemed', y='VisitsAfterRedemption', title='Visits After Redemption by Status')
            fig3_json = fig3.to_json()
        visualizations["Visits_After_Redemption_Box"] = fig3_json
        
        fig4_json = None
        if 'DiscountPercentage' in df.columns:
            discount_impact = df.groupby('DiscountPercentage')['Redeemed'].mean().reset_index()
            discount_impact['RedemptionRate'] = discount_impact['Redeemed'] * 100
            fig4 = px.bar(discount_impact, x='DiscountPercentage', y='RedemptionRate', title='Redemption Rate by Discount Percentage')
            fig4_json = fig4.to_json()
            metrics["discount_impact"] = discount_impact.to_dict(orient='records')
        visualizations["Redemption_Rate_by_Discount_Percentage_Bar"] = fig4_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def seasonal_and_holiday_campaign_impact_analysis(df):
    analysis_name = "Seasonal and Holiday Campaign Impact Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'Date': ['Date', 'EventDate', 'Day'],
            'CampaignName': ['CampaignName', 'HolidayCampaign', 'SeasonalCampaign'],
            'Revenue': ['Revenue', 'Sales', 'TotalRevenue'],
            'Transactions': ['Transactions', 'Orders', 'NumTransactions'],
            'WebsiteTraffic': ['WebsiteTraffic', 'Visits', 'PageViews']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Date', 'Revenue', 'Transactions', 'WebsiteTraffic'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['Revenue', 'Transactions', 'WebsiteTraffic']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Date', 'Revenue', 'Transactions', 'WebsiteTraffic'])
        df = df.sort_values('Date')

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        # Identify campaign periods
        df['IsCampaignDay'] = df['CampaignName'].notna() & (df['CampaignName'] != '')
        
        avg_daily_revenue_campaign = df[df['IsCampaignDay']]['Revenue'].mean()
        avg_daily_revenue_non_campaign = df[~df['IsCampaignDay']]['Revenue'].mean()
        revenue_uplift = avg_daily_revenue_campaign - avg_daily_revenue_non_campaign

        insights.append(f"Average Daily Revenue (Campaign Days): ${avg_daily_revenue_campaign:,.2f}")
        insights.append(f"Average Daily Revenue (Non-Campaign Days): ${avg_daily_revenue_non_campaign:,.2f}")
        insights.append(f"Revenue Uplift during Campaigns: ${revenue_uplift:,.2f}")

        campaign_impact_summary = df.groupby('IsCampaignDay').agg(
            AvgRevenue=('Revenue', 'mean'),
            AvgTransactions=('Transactions', 'mean'),
            AvgWebsiteTraffic=('WebsiteTraffic', 'mean')
        ).reset_index()
        campaign_impact_summary['IsCampaignDay'] = campaign_impact_summary['IsCampaignDay'].map({True: 'Campaign Day', False: 'Non-Campaign Day'})

        metrics = {
            "Average Daily Revenue (Campaign)": avg_daily_revenue_campaign,
            "Average Daily Revenue (Non-Campaign)": avg_daily_revenue_non_campaign,
            "Average Daily Metrics by Campaign Status": campaign_impact_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.line(df, x='Date', y='Revenue', color='IsCampaignDay', title='Daily Revenue Trend by Campaign Status')
        visualizations["Daily_Revenue_Trend_by_Campaign_Status_Line"] = fig1.to_json()

        fig2 = px.bar(campaign_impact_summary, x='IsCampaignDay', y='AvgTransactions', title='Average Daily Transactions by Campaign Status')
        visualizations["Avg_Daily_Transactions_by_Campaign_Status_Bar"] = fig2.to_json()
        
        fig3_json = None
        if 'CampaignName' in df.columns:
            campaign_performance = df[df['IsCampaignDay']].groupby('CampaignName').agg(
                TotalRevenue=('Revenue', 'sum'),
                TotalTransactions=('Transactions', 'sum')
            ).reset_index()
            fig3 = px.bar(campaign_performance.sort_values('TotalRevenue', ascending=False).head(10),
                          x='CampaignName', y='TotalRevenue', title='Top 10 Campaigns by Total Revenue')
            fig3_json = fig3.to_json()
            metrics["campaign_performance"] = campaign_performance.to_dict(orient='records')
        visualizations["Top_10_Campaigns_by_Total_Revenue_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def video_marketing_engagement_analysis(df):
    analysis_name = "Video Marketing Engagement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'VideoID': ['VideoID', 'ID'],
            'VideoTitle': ['VideoTitle', 'Title'],
            'Views': ['Views', 'TotalViews', 'Playbacks'],
            'WatchTimeSeconds': ['WatchTimeSeconds', 'DurationWatched', 'AvgWatchTime'],
            'Likes': ['Likes', 'NumLikes'],
            'Shares': ['Shares', 'NumShares'],
            'Comments': ['Comments', 'NumComments'],
            'ConversionEvent': ['ConversionEvent', 'Converted', 'LeadEvent']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['VideoID', 'Views', 'WatchTimeSeconds', 'Likes', 'Shares', 'Comments'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Views', 'WatchTimeSeconds', 'Likes', 'Shares', 'Comments']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'ConversionEvent' in df.columns:
            df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        
        df = df.dropna(subset=['VideoID', 'Views', 'WatchTimeSeconds', 'Likes', 'Shares', 'Comments'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        avg_views = df['Views'].mean()
        avg_watch_time_minutes = (df['WatchTimeSeconds'].mean() / 60)
        avg_likes_per_view = (df['Likes'] / df['Views']).mean() * 100 if df['Views'].sum() > 0 else 0
        
        avg_conversion_rate = np.nan
        if 'ConversionEvent' in df.columns:
            avg_conversion_rate = df['ConversionEvent'].mean() * 100
            if not pd.isna(avg_conversion_rate):
                insights.append(f"Average Conversion Rate: {avg_conversion_rate:.2f}%")

        insights.append(f"Average Video Views: {avg_views:,.0f}")
        insights.append(f"Average Watch Time: {avg_watch_time_minutes:.1f} minutes")
        insights.append(f"Average Likes per View: {avg_likes_per_view:.2f}%")

        top_videos_by_views = df.sort_values('Views', ascending=False).head(10)
        
        metrics = {
            "Average Video Views": avg_views,
            "Average Watch Time (minutes)": avg_watch_time_minutes,
            "Average Likes per View": avg_likes_per_view,
            "Average Conversion Rate": avg_conversion_rate,
            "Top 10 Videos by Views": top_videos_by_views[['VideoTitle', 'Views', 'WatchTimeSeconds']].to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(top_videos_by_views, x='VideoTitle', y='Views', title='Top 10 Videos by Views')
        visualizations["Top_10_Videos_by_Views_Bar"] = fig1.to_json()

        fig2 = px.scatter(df, x='WatchTimeSeconds', y='Likes', size='Views', hover_name='VideoTitle',
                          title='Likes vs. Watch Time (Sized by Views)')
        visualizations["Likes_vs_Watch_Time_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'ConversionEvent' in df.columns and not df['ConversionEvent'].isnull().all():
            video_conversion = df.groupby('VideoTitle')['ConversionEvent'].mean().reset_index()
            video_conversion['ConversionRate'] = video_conversion['ConversionEvent'] * 100
            fig3 = px.bar(video_conversion.sort_values('ConversionRate', ascending=False).head(10),
                          x='VideoTitle', y='ConversionRate', title='Top 10 Videos by Conversion Rate')
            fig3_json = fig3.to_json()
            metrics["video_conversion_summary"] = video_conversion.to_dict(orient='records')
        visualizations["Top_10_Videos_by_Conversion_Rate_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def search_engine_marketing_sem_keyword_performance(df):
    analysis_name = "Search Engine Marketing (SEM) Keyword Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'Keyword': ['Keyword', 'SearchTerm'],
            'AdGroup': ['AdGroup', 'CampaignAdGroup'],
            'Impressions': ['Impressions', 'KeywordImpressions'],
            'Clicks': ['Clicks', 'KeywordClicks'],
            'Cost': ['Cost', 'Spend', 'KeywordCost'],
            'Conversions': ['Conversions', 'KeywordConversions', 'Sales']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Keyword', 'Impressions', 'Clicks', 'Cost', 'Conversions'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Impressions', 'Clicks', 'Cost', 'Conversions']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Keyword', 'Impressions', 'Clicks', 'Cost', 'Conversions'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['CPC'] = df['Cost'] / df['Clicks']
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['CPA'] = df['Cost'] / df['Conversions']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        keyword_summary = df.groupby('Keyword').agg(
            TotalImpressions=('Impressions', 'sum'),
            TotalClicks=('Clicks', 'sum'),
            TotalCost=('Cost', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgCPC=('CPC', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()

        metrics = {"keyword_summary": keyword_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(keyword_summary)} keywords.")

        # Visualizations
        fig1 = px.bar(keyword_summary.sort_values('TotalConversions', ascending=False).head(10),
                      x='Keyword', y='TotalConversions', title='Top 10 Keywords by Total Conversions')
        visualizations["Top_10_Keywords_by_Conversions_Bar"] = fig1.to_json()

        fig2 = px.scatter(keyword_summary, x='AvgCPC', y='AvgConversionRate', size='TotalImpressions',
                          hover_name='Keyword', title='Conversion Rate vs. CPC (Sized by Impressions)')
        visualizations["Conversion_Rate_vs_CPC_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'AdGroup' in df.columns:
            ad_group_performance = df.groupby('AdGroup').agg(
                TotalCost=('Cost', 'sum'),
                TotalConversions=('Conversions', 'sum'),
                AvgCPA=('CPA', 'mean')
            ).reset_index()
            fig3 = px.bar(ad_group_performance.sort_values('AvgCPA', ascending=True).head(10),
                          x='AdGroup', y='AvgCPA', title='Top 10 Ad Groups by Lowest Average CPA')
            fig3_json = fig3.to_json()
            metrics["ad_group_performance"] = ad_group_performance.to_dict(orient='records')
        visualizations["Top_10_Ad_Groups_by_Lowest_Avg_CPA_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def churn_prediction_and_targeted_campaign_analysis(df):
    analysis_name = "Churn Prediction and Targeted Campaign Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'CustomerID': ['CustomerID', 'ID', 'UserID'],
            'IsChurned': ['IsChurned', 'Churned', 'ChurnStatus'],
            'LastActivityDate': ['LastActivityDate', 'LastLogin', 'LastPurchaseDate'],
            'CampaignSegment': ['CampaignSegment', 'Segment', 'TargetGroup'],
            'CampaignResponse': ['CampaignResponse', 'Response', 'ConvertedToRetain'],
            'CustomerLifetimeValue': ['CustomerLifetimeValue', 'CLTV', 'LTV']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['CustomerID', 'IsChurned', 'CampaignSegment', 'CampaignResponse', 'CustomerLifetimeValue'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['IsChurned'] = df['IsChurned'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        df['CampaignResponse'] = df['CampaignResponse'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        df['CustomerLifetimeValue'] = pd.to_numeric(df['CustomerLifetimeValue'], errors='coerce')
        if 'LastActivityDate' in df.columns:
            df['LastActivityDate'] = pd.to_datetime(df['LastActivityDate'], errors='coerce')

        df = df.dropna(subset=['CustomerID', 'IsChurned', 'CampaignSegment', 'CampaignResponse', 'CustomerLifetimeValue'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        overall_churn_rate = df['IsChurned'].mean() * 100
        campaign_response_rate = df['CampaignResponse'].mean() * 100
        
        insights.append(f"Overall Churn Rate: {overall_churn_rate:.2f}%")
        insights.append(f"Overall Campaign Response Rate: {campaign_response_rate:.2f}%")

        churn_by_segment = df.groupby('CampaignSegment').agg(
            NumCustomers=('CustomerID', 'count'),
            ChurnRate=('IsChurned', 'mean'),
            ResponseRate=('CampaignResponse', 'mean'),
            AvgCLTV=('CustomerLifetimeValue', 'mean')
        ).reset_index()
        churn_by_segment['ChurnRate'] *= 100
        churn_by_segment['ResponseRate'] *= 100

        metrics = {
            "Overall Churn Rate": overall_churn_rate,
            "Overall Campaign Response Rate": campaign_response_rate,
            "Churn and Response Rates by Campaign Segment": churn_by_segment.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(churn_by_segment, x='CampaignSegment', y='ChurnRate', title='Churn Rate by Campaign Segment')
        visualizations["Churn_Rate_by_Segment_Bar"] = fig1.to_json()

        fig2 = px.bar(churn_by_segment, x='CampaignSegment', y='ResponseRate', title='Campaign Response Rate by Segment')
        visualizations["Campaign_Response_Rate_by_Segment_Bar"] = fig2.to_json()
        
        fig3 = px.box(df, x='IsChurned', y='CustomerLifetimeValue', title='Customer Lifetime Value by Churn Status')
        visualizations["CLTV_by_Churn_Status_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def newsletter_signup_attribution_analysis(df):
    analysis_name = "Newsletter Signup Attribution Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'SignupID': ['SignupID', 'ID'],
            'SignupDate': ['SignupDate', 'Date'],
            'Channel': ['Channel', 'Source', 'MarketingChannel'],
            'Device': ['Device', 'SignupDevice'],
            'ConvertedToCustomer': ['ConvertedToCustomer', 'IsCustomer', 'Converted'],
            'CustomerLifetimeValue': ['CustomerLifetimeValue', 'CLTV']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['SignupID', 'SignupDate', 'Channel', 'ConvertedToCustomer'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['SignupDate'] = pd.to_datetime(df['SignupDate'], errors='coerce')
        df['ConvertedToCustomer'] = df['ConvertedToCustomer'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        if 'CustomerLifetimeValue' in df.columns:
            df['CustomerLifetimeValue'] = pd.to_numeric(df['CustomerLifetimeValue'], errors='coerce')
        
        df = df.dropna(subset=['SignupID', 'SignupDate', 'Channel', 'ConvertedToCustomer'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_signups = len(df)
        customer_conversions = int(df['ConvertedToCustomer'].sum())
        conversion_rate = (customer_conversions / total_signups) * 100

        insights.append(f"Total Newsletter Signups: {total_signups}")
        insights.append(f"Signups Converted to Customer: {customer_conversions}")
        insights.append(f"Signup-to-Customer Conversion Rate: {conversion_rate:.2f}%")

        channel_performance = df.groupby('Channel').agg(
            NumSignups=('SignupID', 'count'),
            ConversionRate=('ConvertedToCustomer', 'mean'),
            AvgCLTV=('CustomerLifetimeValue', 'mean') if 'CustomerLifetimeValue' in df.columns else ('SignupID', 'size')
        ).reset_index()
        channel_performance['ConversionRate'] *= 100

        metrics = {
            "Total Signups": total_signups,
            "Signups Converted to Customer": customer_conversions,
            "Signup-to-Customer Conversion Rate": conversion_rate,
            "Conversion Performance by Signup Channel": channel_performance.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(channel_performance, x='Channel', y='NumSignups', title='Number of Signups by Channel')
        visualizations["Num_Signups_by_Channel_Bar"] = fig1.to_json()

        fig2 = px.bar(channel_performance, x='Channel', y='ConversionRate', title='Signup-to-Customer Conversion Rate by Channel')
        visualizations["Conversion_Rate_by_Channel_Bar"] = fig2.to_json()
        
        fig3_json = None
        if 'CustomerLifetimeValue' in df.columns and not df['CustomerLifetimeValue'].isnull().all():
            fig3 = px.box(df, x='Channel', y='CustomerLifetimeValue', title='Customer Lifetime Value by Signup Channel')
            fig3_json = fig3.to_json()
        visualizations["CLTV_by_Signup_Channel_Box"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def marketing_budget_allocation_and_spend_analysis(df):
    analysis_name = "Marketing Budget Allocation and Spend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'Date': ['Date', 'Month', 'ReportingPeriod'],
            'Channel': ['Channel', 'MarketingChannel', 'Platform'],
            'BudgetAllocated': ['BudgetAllocated', 'Budget', 'AllocatedSpend'],
            'ActualSpend': ['ActualSpend', 'Spend', 'Cost'],
            'RevenueAttributed': ['RevenueAttributed', 'Revenue', 'Sales']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Date', 'Channel', 'BudgetAllocated', 'ActualSpend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['BudgetAllocated', 'ActualSpend', 'RevenueAttributed']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Date', 'Channel', 'BudgetAllocated', 'ActualSpend'])
        df = df.sort_values('Date')

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_allocated_budget = df['BudgetAllocated'].sum()
        total_actual_spend = df['ActualSpend'].sum()
        total_revenue_attributed = 0
        if 'RevenueAttributed' in df.columns:
            total_revenue_attributed = df['RevenueAttributed'].sum()
        budget_utilization_rate = (total_actual_spend / total_allocated_budget) * 100 if total_allocated_budget > 0 else 0

        insights.append(f"Total Allocated Budget: ${total_allocated_budget:,.2f}")
        insights.append(f"Total Actual Spend: ${total_actual_spend:,.2f}")
        insights.append(f"Budget Utilization Rate: {budget_utilization_rate:.2f}%")
        insights.append(f"Total Revenue Attributed: ${total_revenue_attributed:,.2f}")

        channel_spend_summary = df.groupby('Channel').agg(
            TotalAllocated=('BudgetAllocated', 'sum'),
            TotalActualSpend=('ActualSpend', 'sum'),
            TotalRevenue=('RevenueAttributed', 'sum') if 'RevenueAttributed' in df.columns else ('BudgetAllocated', 'size')
        ).reset_index()
        
        channel_spend_summary['UtilizationRate'] = (channel_spend_summary['TotalActualSpend'] / channel_spend_summary['TotalAllocated']) * 100
        if 'RevenueAttributed' in df.columns:
            channel_spend_summary['ROI'] = ((channel_spend_summary['TotalRevenue'] - channel_spend_summary['TotalActualSpend']) / channel_spend_summary['TotalActualSpend']) * 100
        else:
            channel_spend_summary['ROI'] = np.nan
        
        channel_spend_summary.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        metrics = {
            "Total Allocated Budget": total_allocated_budget,
            "Total Actual Spend": total_actual_spend,
            "Budget Utilization Rate": budget_utilization_rate,
            "Total Revenue Attributed": total_revenue_attributed,
            "Budget and Performance by Channel": channel_spend_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(channel_spend_summary, x='Channel', y=['TotalAllocated', 'TotalActualSpend'],
                      barmode='group', title='Allocated Budget vs. Actual Spend by Channel')
        visualizations["Budget_vs_Spend_by_Channel_Bar"] = fig1.to_json()

        fig2 = px.pie(channel_spend_summary, names='Channel', values='TotalActualSpend', title='Distribution of Actual Spend by Channel')
        visualizations["Actual_Spend_Distribution_Pie"] = fig2.to_json()
        
        fig3_json = None
        if 'RevenueAttributed' in df.columns:
            fig3 = px.bar(channel_spend_summary, x='Channel', y='ROI', title='ROI by Channel')
            fig3_json = fig3.to_json()
        visualizations["ROI_by_Channel_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def social_media_competitive_and_sentiment_analysis(df):
    analysis_name = "Social Media Competitive and Sentiment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'Brand': ['Brand', 'Competitor', 'CompanyName'],
            'Date': ['Date', 'PostDate', 'AnalysisDate'],
            'Mentions': ['Mentions', 'TotalMentions', 'CountMentions'],
            'SentimentScore': ['SentimentScore', 'AvgSentiment', 'SentimentPolarity'],
            'EngagementRate': ['EngagementRate', 'AvgEngagement'],
            'FollowerGrowth': ['FollowerGrowth', 'NewFollowers']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['Brand', 'Date', 'Mentions', 'SentimentScore', 'EngagementRate'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Mentions', 'SentimentScore', 'EngagementRate', 'FollowerGrowth']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Brand', 'Date', 'Mentions', 'SentimentScore', 'EngagementRate'])
        df = df.sort_values(['Brand', 'Date'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        overall_avg_sentiment = df['SentimentScore'].mean()
        overall_avg_engagement = df['EngagementRate'].mean()
        
        insights.append(f"Overall Average Sentiment Score: {overall_avg_sentiment:.2f}")
        insights.append(f"Overall Average Engagement Rate: {overall_avg_engagement:.2f}%")

        brand_summary = df.groupby('Brand').agg(
            TotalMentions=('Mentions', 'sum'),
            AvgSentiment=('SentimentScore', 'mean'),
            AvgEngagementRate=('EngagementRate', 'mean'),
            TotalFollowerGrowth=('FollowerGrowth', 'sum') if 'FollowerGrowth' in df.columns else ('Mentions', 'size')
        ).reset_index()
        
        metrics = {
            "Overall Average Sentiment Score": overall_avg_sentiment,
            "Overall Average Engagement Rate": overall_avg_engagement,
            "Competitive Brand Performance Summary": brand_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.line(df, x='Date', y='SentimentScore', color='Brand', title='Sentiment Trend Over Time by Brand')
        visualizations["Sentiment_Trend_by_Brand_Line"] = fig1.to_json()

        fig2 = px.bar(brand_summary.sort_values('AvgEngagementRate', ascending=False),
                      x='Brand', y='AvgEngagementRate', title='Average Engagement Rate by Brand')
        visualizations["Avg_Engagement_Rate_by_Brand_Bar"] = fig2.to_json()
        
        fig3 = px.scatter(df, x='SentimentScore', y='EngagementRate', color='Brand', size='Mentions',
                          hover_name='Date', title='Engagement Rate vs. Sentiment Score by Brand (Sized by Mentions)')
        visualizations["Engagement_vs_Sentiment_Scatter"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def customer_service_sentiment_and_feedback_analysis(df):
    analysis_name = "Customer Service Sentiment and Feedback Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'TicketID': ['TicketID', 'CaseID', 'ID'],
            'Date': ['Date', 'TicketDate', 'FeedbackDate'],
            'Channel': ['Channel', 'ServiceChannel', 'ContactChannel'],
            'SentimentScore': ['SentimentScore', 'FeedbackSentiment', 'SentimentPolarity'],
            'ResolutionTimeHours': ['ResolutionTimeHours', 'TimeResolution', 'ResolveHours'],
            'CustomerSatisfactionRating': ['CustomerSatisfactionRating', 'CSAT', 'Rating'],
            'FeedbackCategory': ['FeedbackCategory', 'Category', 'IssueType']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['TicketID', 'Date', 'Channel', 'SentimentScore', 'CustomerSatisfactionRating'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        for col in ['SentimentScore', 'ResolutionTimeHours', 'CustomerSatisfactionRating']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['Date', 'Channel', 'SentimentScore', 'CustomerSatisfactionRating'])
        df = df.sort_values('Date')

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        overall_avg_sentiment = df['SentimentScore'].mean()
        overall_avg_csat = df['CustomerSatisfactionRating'].mean()
        avg_resolution_time_hours = np.nan
        if 'ResolutionTimeHours' in df.columns:
            avg_resolution_time_hours = df['ResolutionTimeHours'].mean()

        insights.append(f"Overall Average Sentiment Score: {overall_avg_sentiment:.2f}")
        insights.append(f"Overall Average Customer Satisfaction Rating: {overall_avg_csat:.2f}")
        if not pd.isna(avg_resolution_time_hours):
            insights.append(f"Average Resolution Time: {avg_resolution_time_hours:.1f} hours")

        channel_performance = df.groupby('Channel').agg(
            NumTickets=('TicketID', 'count'),
            AvgSentiment=('SentimentScore', 'mean'),
            AvgCSAT=('CustomerSatisfactionRating', 'mean'),
            AvgResolutionTime=('ResolutionTimeHours', 'mean') if 'ResolutionTimeHours' in df.columns else ('TicketID', 'size')
        ).reset_index()

        metrics = {
            "Overall Average Sentiment Score": overall_avg_sentiment,
            "Overall Average Customer Satisfaction Rating": overall_avg_csat,
            "Average Resolution Time (hours)": avg_resolution_time_hours,
            "Customer Service Performance by Channel": channel_performance.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.line(df, x='Date', y='SentimentScore', color='Channel', title='Sentiment Trend Over Time by Channel')
        visualizations["Sentiment_Trend_by_Channel_Line"] = fig1.to_json()

        fig2 = px.bar(channel_performance, x='Channel', y='AvgCSAT', title='Average Customer Satisfaction Rating by Channel')
        visualizations["Avg_CSAT_by_Channel_Bar"] = fig2.to_json()
        
        fig3_json = None
        if 'FeedbackCategory' in df.columns:
            category_counts = df['FeedbackCategory'].value_counts().reset_index()
            category_counts.columns = ['Category', 'Count']
            fig3 = px.bar(category_counts, x='Category', y='Count', title='Distribution of Feedback Categories')
            fig3_json = fig3.to_json()
            metrics["feedback_category_counts"] = category_counts.to_dict(orient='records')
        visualizations["Feedback_Categories_Distribution_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def rfm_based_customer_targeting_analysis(df):
    analysis_name = "RFM-Based Customer Targeting Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'CustomerID': ['CustomerID', 'ID', 'UserID'],
            'LastPurchaseDate': ['LastPurchaseDate', 'RecencyDate', 'DateOfLastPurchase'],
            'TotalPurchases': ['TotalPurchases', 'Frequency', 'NumOrders'],
            'TotalSpend': ['TotalSpend', 'Monetary', 'LifetimeValue'],
            'Segment': ['Segment', 'RFMSegment', 'CustomerSegment']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['CustomerID', 'LastPurchaseDate', 'TotalPurchases', 'TotalSpend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')
        df['TotalPurchases'] = pd.to_numeric(df['TotalPurchases'], errors='coerce')
        df['TotalSpend'] = pd.to_numeric(df['TotalSpend'], errors='coerce')
        df = df.dropna(subset=['CustomerID', 'LastPurchaseDate', 'TotalPurchases', 'TotalSpend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        max_date = df['LastPurchaseDate'].max()
        df['Recency'] = (max_date - df['LastPurchaseDate']).dt.days

        # Basic RFM quartiles
        df['R_Score'] = pd.qcut(df['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop')
        df['F_Score'] = pd.qcut(df['TotalPurchases'], q=4, labels=[1, 2, 3, 4], duplicates='drop')
        df['M_Score'] = pd.qcut(df['TotalSpend'], q=4, labels=[1, 2, 3, 4], duplicates='drop')

        if 'Segment' not in df.columns:
            df['Segment'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
            insights.append("Note: 'Segment' column not found, created a basic RFM score from R, F, M quartiles.")
        
        segment_distribution = df['Segment'].value_counts(normalize=True).reset_index()
        segment_distribution.columns = ['Segment', 'Percentage']
        segment_distribution['Percentage'] *= 100

        segment_performance = df.groupby('Segment').agg(
            AvgRecency=('Recency', 'mean'),
            AvgFrequency=('TotalPurchases', 'mean'),
            AvgMonetary=('TotalSpend', 'mean')
        ).reset_index()

        metrics = {
            "RFM Segment Distribution": segment_distribution.to_dict(orient='records'),
            "Average RFM Metrics by Segment": segment_performance.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.pie(segment_distribution, names='Segment', values='Percentage', title='Distribution of RFM Segments')
        visualizations["RFM_Segment_Distribution_Pie"] = fig1.to_json()

        fig2 = px.bar(segment_performance.sort_values('AvgMonetary', ascending=False).head(10),
                      x='Segment', y='AvgMonetary', title='Average Monetary Value by RFM Segment (Top 10)')
        visualizations["Avg_Monetary_Value_by_RFM_Segment_Bar"] = fig2.to_json()
        
        fig3 = px.scatter(df, x='Recency', y='TotalSpend', color='Segment', size='TotalPurchases',
                          hover_name='CustomerID', title='RFM: Spend vs. Recency (Sized by Purchases)')
        visualizations["RFM_Spend_vs_Recency_Scatter"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def webinar_performance_and_lead_generation_analysis(df):
    analysis_name = "Webinar Performance and Lead Generation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'WebinarID': ['WebinarID', 'ID'],
            'WebinarTitle': ['WebinarTitle', 'Title'],
            'Registrations': ['Registrations', 'NumRegistrations', 'Signups'],
            'Attendees': ['Attendees', 'NumAttendees', 'ActualAttendees'],
            'LeadsGenerated': ['LeadsGenerated', 'NewLeads', 'QualifiedLeads'],
            'CostPerWebinar': ['CostPerWebinar', 'Spend', 'Cost'],
            'Date': ['Date', 'WebinarDate']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['WebinarID', 'Registrations', 'Attendees', 'LeadsGenerated', 'CostPerWebinar'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Registrations', 'Attendees', 'LeadsGenerated', 'CostPerWebinar']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        df = df.dropna(subset=['WebinarID', 'Registrations', 'Attendees', 'LeadsGenerated', 'CostPerWebinar'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['AttendanceRate'] = (df['Attendees'] / df['Registrations']) * 100
        df['LeadConversionRate'] = (df['LeadsGenerated'] / df['Attendees']) * 100
        df['CPL'] = df['CostPerWebinar'] / df['LeadsGenerated']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        overall_attendance_rate = (df['Attendees'].sum() / df['Registrations'].sum()) * 100
        overall_lead_conversion_rate = (df['LeadsGenerated'].sum() / df['Attendees'].sum()) * 100
        overall_cpl = df['CostPerWebinar'].sum() / df['LeadsGenerated'].sum() if df['LeadsGenerated'].sum() > 0 else np.nan

        insights.append(f"Overall Attendance Rate: {overall_attendance_rate:.2f}%")
        insights.append(f"Overall Lead Conversion Rate (from Attendees): {overall_lead_conversion_rate:.2f}%")
        if not pd.isna(overall_cpl):
            insights.append(f"Overall Cost Per Lead (CPL): ${overall_cpl:,.2f}")

        webinar_summary = df.groupby('WebinarID').agg(
            TotalRegistrations=('Registrations', 'sum'),
            TotalAttendees=('Attendees', 'sum'),
            TotalLeads=('LeadsGenerated', 'sum'),
            AvgAttendanceRate=('AttendanceRate', 'mean'),
            AvgLeadConversionRate=('LeadConversionRate', 'mean'),
            AvgCPL=('CPL', 'mean')
        ).reset_index()

        metrics = {
            "Overall Attendance Rate": overall_attendance_rate,
            "Overall Lead Conversion Rate": overall_lead_conversion_rate,
            "Overall CPL": overall_cpl,
            "Webinar Performance Summary": webinar_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(webinar_summary.sort_values('TotalLeads', ascending=False).head(10),
                      x='WebinarID', y='TotalLeads', title='Top 10 Webinars by Leads Generated')
        visualizations["Top_10_Webinars_by_Leads_Bar"] = fig1.to_json()

        fig2 = px.scatter(webinar_summary, x='AvgAttendanceRate', y='AvgLeadConversionRate',
                          size='TotalRegistrations', hover_name='WebinarID',
                          title='Lead Conversion Rate vs. Attendance Rate (Sized by Registrations)')
        visualizations["Lead_Conversion_Rate_vs_Attendance_Rate_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'Date' in df.columns:
            daily_leads = df.groupby('Date')['LeadsGenerated'].sum().reset_index()
            fig3 = px.line(daily_leads, x='Date', y='LeadsGenerated', title='Daily Leads Generated Trend')
            fig3_json = fig3.to_json()
        visualizations["Daily_Leads_Generated_Trend_Line"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def event_marketing_effectiveness_analysis(df):
    analysis_name = "Event Marketing Effectiveness Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'EventID': ['EventID', 'ID', 'EventName'],
            'EventType': ['EventType', 'Type'],
            'Attendees': ['Attendees', 'NumAttendees', 'Registrants'],
            'LeadsGenerated': ['LeadsGenerated', 'NewLeads'],
            'SalesGenerated': ['SalesGenerated', 'Revenue', 'Sales'],
            'CostOfEvent': ['CostOfEvent', 'EventBudget', 'Spend'],
            'Date': ['Date', 'EventDate']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['EventID', 'Attendees', 'LeadsGenerated', 'SalesGenerated', 'CostOfEvent'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Attendees', 'LeadsGenerated', 'SalesGenerated', 'CostOfEvent']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        df = df.dropna(subset=['EventID', 'Attendees', 'LeadsGenerated', 'SalesGenerated', 'CostOfEvent'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['ROI'] = ((df['SalesGenerated'] - df['CostOfEvent']) / df['CostOfEvent']) * 100
        df['CPL'] = df['CostOfEvent'] / df['LeadsGenerated']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        overall_avg_attendees = df['Attendees'].mean()
        overall_total_sales = df['SalesGenerated'].sum()
        overall_total_cost = df['CostOfEvent'].sum()
        overall_roi = ((overall_total_sales - overall_total_cost) / overall_total_cost) * 100 if overall_total_cost > 0 else np.nan

        insights.append(f"Overall Average Attendees per Event: {overall_avg_attendees:,.0f}")
        insights.append(f"Overall Total Sales Generated: ${overall_total_sales:,.2f}")
        if not pd.isna(overall_roi):
            insights.append(f"Overall Event ROI: {overall_roi:.2f}%")

        event_summary = df.groupby('EventID').agg(
            TotalAttendees=('Attendees', 'sum'),
            TotalLeads=('LeadsGenerated', 'sum'),
            TotalSales=('SalesGenerated', 'sum'),
            TotalCost=('CostOfEvent', 'sum'),
            ROI=('ROI', 'mean'),
            CPL=('CPL', 'mean')
        ).reset_index()

        metrics = {
            "Overall Average Attendees": overall_avg_attendees,
            "Overall Total Sales Generated": overall_total_sales,
            "Overall Event ROI": overall_roi,
            "Event Performance Summary": event_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(event_summary.sort_values('TotalSales', ascending=False).head(10),
                      x='EventID', y='TotalSales', title='Top 10 Events by Total Sales Generated')
        visualizations["Top_10_Events_by_Sales_Bar"] = fig1.to_json()

        fig2 = px.scatter(event_summary, x='TotalCost', y='TotalSales', size='TotalAttendees', hover_name='EventID',
                          title='Sales vs. Cost (Sized by Attendees)')
        visualizations["Sales_vs_Cost_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'EventType' in df.columns:
            event_type_impact = df.groupby('EventType').agg(
                AvgROI=('ROI', 'mean'),
                AvgCPL=('CPL', 'mean')
            ).reset_index()
            fig3 = px.bar(event_type_impact, x='EventType', y='AvgROI', title='Average ROI by Event Type')
            fig3_json = fig3.to_json()
            metrics["event_type_impact"] = event_type_impact.to_dict(orient='records')
        visualizations["Avg_ROI_by_Event_Type_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def display_ad_banner_placement_performance_analysis(df):
    analysis_name = "Display Ad Banner Placement Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'PlacementID': ['PlacementID', 'ID', 'AdPlacement'],
            'BannerID': ['BannerID', 'AdID', 'CreativeID'],
            'WebsiteDomain': ['WebsiteDomain', 'PublisherSite'],
            'Impressions': ['Impressions', 'AdImpressions'],
            'Clicks': ['Clicks', 'AdClicks'],
            'Conversions': ['Conversions', 'AdConversions', 'Sales'],
            'Spend': ['Spend', 'Cost']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['PlacementID', 'Impressions', 'Clicks', 'Conversions', 'Spend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Impressions', 'Clicks', 'Conversions', 'Spend']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['PlacementID', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['CPA'] = df['Spend'] / df['Conversions']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        total_spend = df['Spend'].sum()
        overall_ctr = (df['Clicks'].sum() / df['Impressions'].sum()) * 100 if df['Impressions'].sum() > 0 else 0
        overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

        insights.append(f"Total Ad Spend: ${total_spend:,.2f}")
        insights.append(f"Overall CTR: {overall_ctr:.2f}%")
        if not pd.isna(overall_cpa):
            insights.append(f"Overall CPA: ${overall_cpa:,.2f}")

        placement_summary = df.groupby('PlacementID').agg(
            TotalImpressions=('Impressions', 'sum'),
            TotalClicks=('Clicks', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            TotalSpend=('Spend', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()

        metrics = {
            "Total Ad Spend": total_spend,
            "Overall CTR": overall_ctr,
            "Overall CPA": overall_cpa,
            "Ad Placement Performance Summary": placement_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(placement_summary.sort_values('TotalConversions', ascending=False).head(10),
                      x='PlacementID', y='TotalConversions', title='Top 10 Placements by Total Conversions')
        visualizations["Top_10_Placements_by_Conversions_Bar"] = fig1.to_json()

        fig2 = px.scatter(placement_summary, x='AvgCPA', y='AvgCTR', size='TotalSpend', hover_name='PlacementID',
                          title='CTR vs. CPA by Placement (Sized by Spend)')
        visualizations["CTR_vs_CPA_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'WebsiteDomain' in df.columns:
            domain_performance = df.groupby('WebsiteDomain').agg(
                AvgCTR=('CTR', 'mean'),
                AvgCPA=('CPA', 'mean')
            ).reset_index()
            fig3 = px.bar(domain_performance.sort_values('AvgCTR', ascending=False).head(10),
                          x='WebsiteDomain', y='AvgCTR', title='Top 10 Website Domains by Average CTR')
            fig3_json = fig3.to_json()
            metrics["domain_performance"] = domain_performance.to_dict(orient='records')
        visualizations["Top_10_Website_Domains_by_Avg_CTR_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def affiliate_marketing_performance_and_revenue_analysis(df):
    analysis_name = "Affiliate Marketing Performance and Revenue Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'AffiliateID': ['AffiliateID', 'ID', 'PartnerID'],
            'CampaignID': ['CampaignID', 'MarketingCampaignID'],
            'Clicks': ['Clicks', 'ReferralClicks'],
            'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
            'AffiliateCommission': ['AffiliateCommission', 'CommissionEarned', 'Payout'],
            'RevenueGenerated': ['RevenueGenerated', 'SalesRevenue', 'TotalSales'],
            'TrafficSource': ['TrafficSource', 'SourceMedium']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['AffiliateID', 'Clicks', 'Conversions', 'AffiliateCommission', 'RevenueGenerated'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Clicks', 'Conversions', 'AffiliateCommission', 'RevenueGenerated']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['AffiliateID', 'Clicks', 'Conversions', 'AffiliateCommission', 'RevenueGenerated'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['EPC'] = df['RevenueGenerated'] / df['Clicks'] # Earnings per click
        df['AffiliateROI'] = ((df['RevenueGenerated'] - df['AffiliateCommission']) / df['AffiliateCommission']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        total_commission_paid = df['AffiliateCommission'].sum()
        total_revenue_from_affiliates = df['RevenueGenerated'].sum()
        overall_affiliate_conversion_rate = (df['Conversions'].sum() / df['Clicks'].sum()) * 100 if df['Clicks'].sum() > 0 else 0

        insights.append(f"Total Commission Paid: ${total_commission_paid:,.2f}")
        insights.append(f"Total Revenue from Affiliates: ${total_revenue_from_affiliates:,.2f}")
        insights.append(f"Overall Affiliate Conversion Rate: {overall_affiliate_conversion_rate:.2f}%")

        affiliate_summary = df.groupby('AffiliateID').agg(
            TotalClicks=('Clicks', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            TotalCommission=('AffiliateCommission', 'sum'),
            TotalRevenue=('RevenueGenerated', 'sum'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgEPC=('EPC', 'mean'),
            AvgROI=('AffiliateROI', 'mean')
        ).reset_index()

        metrics = {
            "Total Commission Paid": total_commission_paid,
            "Total Revenue from Affiliates": total_revenue_from_affiliates,
            "Overall Affiliate Conversion Rate": overall_affiliate_conversion_rate,
            "Affiliate Performance Summary": affiliate_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(affiliate_summary.sort_values('TotalRevenue', ascending=False).head(10),
                      x='AffiliateID', y='TotalRevenue', title='Top 10 Affiliates by Revenue Generated')
        visualizations["Top_10_Affiliates_by_Revenue_Bar"] = fig1.to_json()

        fig2 = px.scatter(affiliate_summary, x='AvgEPC', y='AvgConversionRate', size='TotalClicks', hover_name='AffiliateID',
                          title='Affiliate Conversion Rate vs. EPC (Sized by Clicks)')
        visualizations["Affiliate_Conversion_Rate_vs_EPC_Scatter"] = fig2.to_json()
        
        fig3_json = None
        if 'TrafficSource' in df.columns:
            source_performance = df.groupby('TrafficSource').agg(
                TotalClicks=('Clicks', 'sum'),
                TotalConversions=('Conversions', 'sum'),
                TotalRevenue=('RevenueGenerated', 'sum')
            ).reset_index()
            fig3 = px.bar(source_performance, x='TrafficSource', y='TotalRevenue', title='Revenue by Traffic Source')
            fig3_json = fig3.to_json()
            metrics["source_performance"] = source_performance.to_dict(orient='records')
        visualizations["Revenue_by_Traffic_Source_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def clicked_link_position_and_device_analysis(df):
    analysis_name = "Clicked Link Position and Device Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'ClickID': ['ClickID', 'ID'],
            'LinkPosition': ['LinkPosition', 'AdPosition', 'Rank'],
            'DeviceType': ['DeviceType', 'Device', 'Platform'],
            'IsConversion': ['IsConversion', 'Converted', 'PurchaseEvent'],
            'CTR': ['CTR', 'ClickThroughRate'],
            'Impressions': ['Impressions', 'AdImpressions']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)
        
        # This analysis is complex. It needs either CTR or (Impressions + Clicks).
        # We'll assume each row is a click if 'Clicks' isn't present.
        
        critical_missing = [col for col in ['LinkPosition', 'DeviceType', 'IsConversion'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['LinkPosition'] = pd.to_numeric(df['LinkPosition'], errors='coerce')
        df['IsConversion'] = df['IsConversion'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
        
        analysis_df = None
        
        # If CTR is not a direct column, calculate it
        if 'CTR' not in df.columns:
            if 'Impressions' not in df.columns:
                 return create_fallback_response(analysis_name, ['CTR', 'Impressions'], matched, df)
            
            insights.append("Note: 'CTR' column not found. Calculating from 'Impressions' and assuming each row is one click.")
            df['Clicks'] = 1 # Each row is a click
            df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')

            grouped_data = df.groupby(['LinkPosition', 'DeviceType']).agg(
                TotalClicks=('Clicks', 'sum'),
                TotalImpressions=('Impressions', 'sum'),
                TotalConversions=('IsConversion', 'sum')
            ).reset_index()
            
            grouped_data['CTR'] = (grouped_data['TotalClicks'] / grouped_data['TotalImpressions']) * 100
            grouped_data['ConversionRate'] = (grouped_data['TotalConversions'] / grouped_data['TotalClicks']) * 100
            grouped_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            analysis_df = grouped_data
            
        elif 'CTR' in df.columns:
            insights.append("Note: Using provided 'CTR' column. Assuming 'Clicks' column exists or each row is a click for Conversion Rate.")
            df['CTR'] = pd.to_numeric(df['CTR'], errors='coerce')
            df['Clicks'] = 1 # Assuming each row is a click
            df['TotalConversions'] = df['IsConversion'].astype(int)
            df['ConversionRate'] = (df['TotalConversions'] / df['Clicks']) * 100 # This will be 100% or 0% per row
            analysis_df = df.copy()

        if analysis_df is None or analysis_df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning or calculation.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": insights
            }

        overall_ctr = analysis_df['CTR'].mean()
        overall_conversion_rate = analysis_df['ConversionRate'].mean()
        
        insights.append(f"Overall Average CTR: {overall_ctr:.2f}%")
        insights.append(f"Overall Average Conversion Rate: {overall_conversion_rate:.2f}%")

        metrics = {
            "Overall Average CTR": overall_ctr,
            "Overall Average Conversion Rate": overall_conversion_rate,
        }

        # Visualizations
        fig1 = px.bar(analysis_df.groupby('LinkPosition')['CTR'].mean().reset_index().sort_values('CTR', ascending=False).head(10), 
                      x='LinkPosition', y='CTR', title='Top 10 Link Positions by Avg. CTR')
        visualizations["Top_10_Link_Positions_by_CTR_Bar"] = fig1.to_json()

        device_performance = analysis_df.groupby('DeviceType').agg(
            AvgCTR=('CTR', 'mean'),
            ConversionRate=('ConversionRate', 'mean')
        ).reset_index()
        
        metrics["Performance by Device Type"] = device_performance.to_dict(orient='records')
        insights.append(f"Analyzed performance across {len(device_performance)} device types.")

        fig2 = px.bar(device_performance, x='DeviceType', y='AvgCTR', title='Average CTR by Device Type')
        visualizations["Avg_CTR_by_Device_Type_Bar"] = fig2.to_json()
        
        fig3 = px.bar(device_performance, x='DeviceType', y='ConversionRate', title='Conversion Rate by Device Type')
        visualizations["Conversion_Rate_by_Device_Type_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def dynamic_content_personalization_analysis(df):
    analysis_name = "Dynamic Content Personalization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'UserID': ['UserID', 'ID', 'CustomerID'],
            'PersonalizationSegment': ['PersonalizationSegment', 'Segment', 'DynamicContentGroup'],
            'ContentVariantShown': ['ContentVariantShown', 'ContentVariant', 'Variant'],
            'ClickThroughRate': ['ClickThroughRate', 'CTR'],
            'ConversionRate': ['ConversionRate', 'Converted', 'Conversion'],
            'Revenue': ['Revenue', 'Sales']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['PersonalizationSegment', 'ContentVariantShown', 'ClickThroughRate', 'ConversionRate'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['ClickThroughRate', 'ConversionRate', 'Revenue']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['PersonalizationSegment', 'ContentVariantShown', 'ClickThroughRate', 'ConversionRate'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        overall_ctr = df['ClickThroughRate'].mean()
        overall_conversion_rate = df['ConversionRate'].mean()
        overall_revenue = 0
        if 'Revenue' in df.columns:
            overall_revenue = df['Revenue'].sum()

        insights.append(f"Overall Average CTR: {overall_ctr:.2f}%")
        insights.append(f"Overall Average Conversion Rate: {overall_conversion_rate:.2f}%")
        if 'Revenue' in df.columns:
            insights.append(f"Overall Total Revenue: ${overall_revenue:,.2f}")

        segment_content_summary = df.groupby(['PersonalizationSegment', 'ContentVariantShown']).agg(
            AvgCTR=('ClickThroughRate', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            TotalRevenue=('Revenue', 'sum') if 'Revenue' in df.columns else ('ClickThroughRate', 'size')
        ).reset_index()

        metrics = {
            "Overall Average CTR": overall_ctr,
            "Overall Average Conversion Rate": overall_conversion_rate,
            "Overall Total Revenue": overall_revenue,
            "Performance by Segment and Content Variant": segment_content_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(segment_content_summary, x='PersonalizationSegment', y='AvgConversionRate', color='ContentVariantShown',
                      barmode='group', title='Average Conversion Rate by Segment and Content Variant')
        visualizations["Conversion_Rate_by_Segment_and_Variant_Bar"] = fig1.to_json()

        fig2 = px.bar(segment_content_summary, x='PersonalizationSegment', y='AvgCTR', color='ContentVariantShown',
                      barmode='group', title='Average CTR by Segment and Content Variant')
        visualizations["CTR_by_Segment_and_Variant_Bar"] = fig2.to_json()
        
        fig3_json = None
        if 'Revenue' in df.columns:
            fig3 = px.bar(segment_content_summary, x='PersonalizationSegment', y='TotalRevenue', color='ContentVariantShown',
                          barmode='group', title='Total Revenue by Segment and Content Variant')
            fig3_json = fig3.to_json()
        visualizations["Revenue_by_Segment_and_Variant_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def remarketing_campaign_performance_analysis(df):
    analysis_name = "Remarketing Campaign Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'UserID': ['UserID', 'ID', 'CustomerID'],
            'CampaignID': ['CampaignID', 'RemarketingCampaignID'],
            'Impressions': ['Impressions', 'AdImpressions'],
            'Clicks': ['Clicks', 'AdClicks'],
            'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
            'Spend': ['Spend', 'AdSpend', 'Cost'],
            'DaysSinceLastVisit': ['DaysSinceLastVisit', 'RecencyDays'],
            'PreviousInteractionType': ['PreviousInteractionType', 'LastInteraction', 'ActivityType']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['CampaignID', 'Impressions', 'Clicks', 'Conversions', 'Spend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Impressions', 'Clicks', 'Conversions', 'Spend', 'DaysSinceLastVisit']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['CampaignID', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['CPA'] = df['Spend'] / df['Conversions']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        overall_spend = df['Spend'].sum()
        overall_conversions = df['Conversions'].sum()
        overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

        insights.append(f"Total Remarketing Spend: ${overall_spend:,.2f}")
        insights.append(f"Total Remarketing Conversions: {overall_conversions}")
        if not pd.isna(overall_cpa):
            insights.append(f"Overall Remarketing CPA: ${overall_cpa:,.2f}")

        campaign_summary = df.groupby('CampaignID').agg(
            TotalSpend=('Spend', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()

        metrics = {
            "Total Remarketing Spend": overall_spend,
            "Total Remarketing Conversions": overall_conversions,
            "Overall Remarketing CPA": overall_cpa,
            "Remarketing Campaign Performance Summary": campaign_summary.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(campaign_summary.sort_values('TotalConversions', ascending=False).head(10),
                      x='CampaignID', y='TotalConversions', title='Top 10 Remarketing Campaigns by Conversions')
        visualizations["Top_10_Remarketing_Campaigns_Bar"] = fig1.to_json()

        fig2_json = None
        if 'PreviousInteractionType' in df.columns:
            interaction_type_performance = df.groupby('PreviousInteractionType').agg(
                AvgConversionRate=('ConversionRate', 'mean'),
                AvgCPA=('CPA', 'mean'),
                TotalSpend=('Spend', 'sum')
            ).reset_index()
            fig2 = px.bar(interaction_type_performance, x='PreviousInteractionType', y='AvgConversionRate',
                          title='Average Conversion Rate by Previous Interaction Type')
            fig2_json = fig2.to_json()
            metrics["interaction_type_performance"] = interaction_type_performance.to_dict(orient='records')
        visualizations["Avg_Conversion_Rate_by_Previous_Interaction_Type_Bar"] = fig2_json
        
        fig3_json = None
        if 'DaysSinceLastVisit' in df.columns:
            fig3 = px.scatter(df, x='DaysSinceLastVisit', y='Conversions', color='CampaignID',
                              hover_name='UserID' if 'UserID' in df.columns else None, 
                              title='Conversions vs. Days Since Last Visit')
            fig3_json = fig3.to_json()
        visualizations["Conversions_vs_Days_Since_Last_Visit_Scatter"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def ad_format_performance_and_cost_analysis(df):
    analysis_name = "Ad Format Performance and Cost Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'AdFormat': ['AdFormat', 'Format', 'CreativeType'],
            'AdChannel': ['AdChannel', 'Platform', 'Channel'],
            'Impressions': ['Impressions', 'AdImpressions'],
            'Clicks': ['Clicks', 'AdClicks'],
            'Conversions': ['Conversions', 'AdConversions', 'Sales'],
            'Spend': ['Spend', 'AdSpend', 'Cost']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['AdFormat', 'Impressions', 'Clicks', 'Conversions', 'Spend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        for col in ['Impressions', 'Clicks', 'Conversions', 'Spend']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['AdFormat', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        df['CTR'] = (df['Clicks'] / df['Impressions']) * 100
        df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100
        df['CPA'] = df['Spend'] / df['Conversions']
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        overall_spend = df['Spend'].sum()
        overall_ctr = (df['Clicks'].sum() / df['Impressions'].sum()) * 100 if df['Impressions'].sum() > 0 else 0
        overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

        insights.append(f"Total Ad Spend: ${overall_spend:,.2f}")
        insights.append(f"Overall CTR: {overall_ctr:.2f}%")
        if not pd.isna(overall_cpa):
            insights.append(f"Overall CPA: ${overall_cpa:,.2f}")

        format_performance = df.groupby('AdFormat').agg(
            TotalSpend=('Spend', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()

        metrics = {
            "Total Ad Spend": overall_spend,
            "Overall CTR": overall_ctr,
            "Overall CPA": overall_cpa,
            "Ad Format Performance Summary": format_performance.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.bar(format_performance.sort_values('TotalConversions', ascending=False),
                      x='AdFormat', y='TotalConversions', title='Total Conversions by Ad Format')
        visualizations["Total_Conversions_by_Ad_Format_Bar"] = fig1.to_json()

        fig2 = px.bar(format_performance.sort_values('AvgCPA', ascending=True),
                      x='AdFormat', y='AvgCPA', title='Average CPA by Ad Format (Lower is Better)')
        visualizations["Avg_CPA_by_Ad_Format_Bar"] = fig2.to_json()
        
        fig3_json = None
        if 'AdChannel' in df.columns:
            channel_format_performance = df.groupby(['AdChannel', 'AdFormat']).agg(
                TotalSpend=('Spend', 'sum'),
                TotalConversions=('Conversions', 'sum'),
                AvgCTR=('CTR', 'mean'),
                AvgConversionRate=('ConversionRate', 'mean')
            ).reset_index()
            fig3 = px.bar(channel_format_performance, x='AdChannel', y='AvgConversionRate', color='AdFormat',
                          barmode='group', title='Average Conversion Rate by Channel and Ad Format')
            fig3_json = fig3.to_json()
            metrics["channel_format_performance"] = channel_format_performance.to_dict(orient='records')
        visualizations["Avg_Conversion_Rate_by_Channel_and_Format_Bar"] = fig3_json

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


def geotargeting_accuracy_and_effectiveness_analysis(df):
    analysis_name = "Geotargeting Accuracy and Effectiveness Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected_map = {
            'AdImpressionID': ['AdImpressionID', 'ID', 'ImpressionID'],
            'TargetedRegion': ['TargetedRegion', 'TargetArea', 'GeoTarget'],
            'ActualRegion': ['ActualRegion', 'ActualLocation', 'UserLocation'],
            'ConversionEvent': ['ConversionEvent', 'Converted', 'Purchase', 'IsConverted'],
            'AdSpend': ['AdSpend', 'Spend', 'Cost']
        }
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), expected_map)

        critical_missing = [col for col in ['AdImpressionID', 'TargetedRegion', 'ActualRegion', 'ConversionEvent', 'AdSpend'] if matched[col] is None]
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = df_renamed

        df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({
            '1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False
        }).fillna(False)

        df['AdSpend'] = pd.to_numeric(df['AdSpend'], errors='coerce')
        df = df.dropna(subset=['TargetedRegion', 'ActualRegion', 'ConversionEvent', 'AdSpend'])

        if df.empty:
            return {
                "analysis_type": analysis_name, "status": "error", "error_message": "No sufficient data after cleaning.",
                "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        total_impressions = len(df)
        accurate_targets = int((df['TargetedRegion'] == df['ActualRegion']).sum())
        accuracy_rate = (accurate_targets / total_impressions) * 100

        insights.append(f"Total Ad Impressions: {total_impressions}")
        insights.append(f"Accurate Geotargets (TargetedRegion == ActualRegion): {accurate_targets}")
        insights.append(f"Geotargeting Accuracy Rate: {accuracy_rate:.2f}%")

        performance_by_accuracy = df.groupby(df['TargetedRegion'] == df['ActualRegion']).agg(
            NumImpressions=('AdImpressionID', 'count'),
            ConversionRate=('ConversionEvent', 'mean'),
            TotalSpend=('AdSpend', 'sum')
        ).reset_index()
        performance_by_accuracy.columns = ['IsAccurateTarget', 'NumImpressions', 'ConversionRate', 'TotalSpend']
        performance_by_accuracy['ConversionRate'] *= 100
        performance_by_accuracy['IsAccurateTarget'] = performance_by_accuracy['IsAccurateTarget'].map({True: 'Accurate', False: 'Inaccurate'})

        conversion_by_target_region = df.groupby('TargetedRegion').agg(
            TotalImpressions=('AdImpressionID', 'count'),
            TotalConversions=('ConversionEvent', lambda x: x.sum()),
            TotalSpend=('AdSpend', 'sum')
        ).reset_index()
        conversion_by_target_region['ConversionRate'] = (conversion_by_target_region['TotalConversions'] / conversion_by_target_region['TotalImpressions']) * 100
        conversion_by_target_region['CPA'] = (conversion_by_target_region['TotalSpend'] / conversion_by_target_region['TotalConversions']).replace([np.inf, -np.inf], np.nan)
        
        metrics = {
            "Total Impressions": total_impressions,
            "Accurate Geotargets": accurate_targets,
            "Geotargeting Accuracy Rate": accuracy_rate,
            "Performance by Geotargeting Accuracy": performance_by_accuracy.to_dict(orient='records'),
            "Conversion Performance by Targeted Region": conversion_by_target_region.to_dict(orient='records')
        }

        # Visualizations
        fig1 = px.pie(performance_by_accuracy, names='IsAccurateTarget', values='NumImpressions',
                      title='Geotargeting Accuracy Distribution (by Impressions)',
                      color_discrete_map={'Accurate': 'lightgreen', 'Inaccurate': 'salmon'})
        visualizations["Geotargeting_Accuracy_Pie"] = fig1.to_json()

        fig2 = px.bar(performance_by_accuracy, x='IsAccurateTarget', y='ConversionRate',
                      title='Conversion Rate by Geotargeting Accuracy',
                      text_auto='.2s', color='IsAccurateTarget',
                      color_discrete_map={'Accurate': 'lightgreen', 'Inaccurate': 'salmon'})
        fig2.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig2.update_layout(yaxis_title="Conversion Rate (%)")
        visualizations["Conversion_Rate_by_Accuracy_Bar"] = fig2.to_json()
        
        cross_tab_regions = pd.crosstab(df['TargetedRegion'], df['ActualRegion'], normalize='index')
        fig3 = px.imshow(cross_tab_regions, text_auto=True,
                          title='Targeted vs. Actual Region Impression Distribution (Normalized by Targeted Region)',
                          labels=dict(x="Actual Region", y="Targeted Region", color="Proportion"),
                          color_continuous_scale='Viridis')
        fig3.update_xaxes(side="top")
        visualizations["Targeted_vs_Actual_Region_Heatmap"] = fig3.to_json()

        fig4 = px.bar(conversion_by_target_region.sort_values(by='ConversionRate', ascending=False),
                      x='TargetedRegion', y='ConversionRate',
                      title='Conversion Rate by Targeted Region',
                      text_auto='.2s')
        fig4.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig4.update_layout(yaxis_title="Conversion Rate (%)")
        visualizations["Conversion_Rate_by_Targeted_Region_Bar"] = fig4.to_json()

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
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": visualizations, 
            "metrics": convert_to_native_types(metrics), "insights": insights
        }


# ========== REFACTORED STUB/SIMPLE FUNCTIONS ==========

def campaign_performance(df):
    analysis_name = "Campaign Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['conversions', 'cost']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df.dropna(subset=['conversions', 'cost'], inplace=True)
        
        total_conversions = df['conversions'].sum()
        total_cost = df['cost'].sum()
        cpa = np.nan
        if total_cost > 0 and total_conversions > 0:
            cpa = total_cost / total_conversions
            
        metrics = {
            "total_conversions": total_conversions,
            "total_cost": total_cost,
            "cost_per_acquisition": cpa
        }
        
        insights.append(f"Total Conversions: {total_conversions}")
        insights.append(f"Total Cost: ${total_cost:,.2f}")
        if not pd.isna(cpa):
            insights.append(f"Overall Cost Per Acquisition (CPA): ${cpa:.2f}")

        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def channel_analysis(df):
    analysis_name = "Channel Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['channel', 'conversions', 'cost']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df_renamed
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df.dropna(subset=['channel', 'conversions', 'cost'], inplace=True)

        channel_summary = df.groupby('channel').agg(
            total_conversions=('conversions', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        
        channel_summary['conversions_per_dollar'] = channel_summary['total_conversions'] / channel_summary['total_cost']
        channel_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"channel_summary": channel_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(channel_summary)} channels.")
        
        fig = px.bar(channel_summary, x='channel', y='total_conversions', color='total_cost',
                     title="Total Conversions and Cost by Channel")
        visualizations["conversions_by_channel"] = fig.to_json()

        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def customer_segmentation(df):
    analysis_name = "Customer Segmentation"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['customer_id', 'total_spend', 'customer_segment']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        missing_critical = [col for col in ['customer_id', 'total_spend'] if matched[col] is None]
        
        if missing_critical:
            return create_fallback_response(analysis_name, missing_critical, matched, df)
        
        df = df_renamed
        insights.append("Note: This is a descriptive analysis. For segmentation, advanced clustering (e.g., RFM, K-means) is recommended.")
        
        if 'customer_segment' in df.columns:
            segment_summary = df.groupby('customer_segment').agg(
                customer_count=('customer_id', 'nunique'),
                avg_spend=('total_spend', 'mean')
            ).reset_index()
            
            metrics = {"segment_summary": segment_summary.to_dict(orient='records')}
            insights.append(f"Analyzed {len(segment_summary)} pre-defined customer segments.")
            
            fig = px.pie(segment_summary, names='customer_segment', values='customer_count', title="Customer Count by Segment")
            visualizations["segment_distribution_pie"] = fig.to_json()
        else:
            insights.append("No 'customer_segment' column found. Showing total spend distribution.")
            df['total_spend'] = pd.to_numeric(df['total_spend'], errors='coerce')
            fig = px.histogram(df, x='total_spend', title="Distribution of Total Spend")
            visualizations["total_spend_distribution"] = fig.to_json()
            metrics = {"avg_spend": df['total_spend'].mean(), "median_spend": df['total_spend'].median()}

        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def funnel_analysis(df):
    analysis_name = "Funnel Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['impressions', 'clicks', 'leads', 'conversions']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        for col in expected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        total_impressions = df['impressions'].sum()
        total_clicks = df['clicks'].sum()
        total_leads = df['leads'].sum()
        total_conversions = df['conversions'].sum()

        ctr = (total_clicks / total_impressions) * 100 if total_impressions > 0 else 0
        leads_rate = (total_leads / total_clicks) * 100 if total_clicks > 0 else 0
        conv_rate = (total_conversions / total_leads) * 100 if total_leads > 0 else 0

        metrics = {
            "impressions": total_impressions,
            "clicks": total_clicks,
            "leads": total_leads,
            "conversions": total_conversions,
            "click_through_rate_pct": ctr,
            "lead_conversion_rate_pct": leads_rate,
            "sales_conversion_rate_pct": conv_rate
        }
        
        insights.append(f"Impressions: {total_impressions:,.0f}")
        insights.append(f"Clicks: {total_clicks:,.0f} (CTR: {ctr:.2f}%)")
        insights.append(f"Leads: {total_leads:,.0f} (Lead Rate: {leads_rate:.2f}%)")
        insights.append(f"Conversions: {total_conversions:,.0f} (Conversion Rate: {conv_rate:.2f}%)")

        funnel_data = pd.DataFrame(dict(
            number=[total_impressions, total_clicks, total_leads, total_conversions],
            stage=['Impressions', 'Clicks', 'Leads', 'Conversions']
        ))
        fig = px.funnel(funnel_data, x='number', y='stage', title="Marketing Funnel")
        visualizations["marketing_funnel"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def content_analysis(df):
    analysis_name = "Content Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['content_id', 'views', 'engagement_score']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['views'] = pd.to_numeric(df['views'], errors='coerce')
        df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce')
        df.dropna(subset=['content_id', 'views', 'engagement_score'], inplace=True)

        content_summary = df.groupby('content_id').agg(
            total_views=('views', 'sum'),
            avg_engagement=('engagement_score', 'mean')
        ).sort_values(by='total_views', ascending=False).reset_index()

        metrics = {"content_summary": content_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(content_summary)} pieces of content.")
        
        fig = px.bar(content_summary.head(10), x='content_id', y='total_views', color='avg_engagement',
                     title="Top 10 Content by Views and Engagement")
        visualizations["top_content_by_views"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def social_media(df):
    analysis_name = "Social Media Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['platform', 'impressions', 'engagement']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
        df['engagement'] = pd.to_numeric(df['engagement'], errors='coerce')
        df.dropna(subset=['platform', 'impressions', 'engagement'], inplace=True)

        social_media_summary = df.groupby('platform').agg(
            total_impressions=('impressions', 'sum'),
            total_engagement=('engagement', 'sum')
        ).reset_index()
        
        social_media_summary['engagement_rate'] = (social_media_summary['total_engagement'] / social_media_summary['total_impressions']) * 100
        social_media_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"social_media_summary": social_media_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(social_media_summary)} social media platforms.")
        
        fig = px.bar(social_media_summary, x='platform', y='engagement_rate', color='total_impressions',
                     title="Engagement Rate by Platform (Colored by Impressions)")
        visualizations["engagement_rate_by_platform"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def a_b_testing(df):
    analysis_name = "A/B Testing"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['variant', 'conversions', 'visitors']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce')
        df['visitors'] = pd.to_numeric(df['visitors'], errors='coerce')
        df.dropna(subset=['variant', 'conversions', 'visitors'], inplace=True)

        ab_summary = df.groupby('variant').agg(
            total_visitors=('visitors', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        
        ab_summary['conversion_rate'] = (ab_summary['total_conversions'] / ab_summary['total_visitors']) * 100
        ab_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"ab_summary": ab_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(ab_summary)} variants.")

        # Statistical Test
        if len(ab_summary) == 2:
            count = ab_summary['total_conversions'].values
            nobs = ab_summary['total_visitors'].values
            if nobs.sum() > 0 and count.sum() >= 0:
                stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
                metrics["z_test"] = {"z_statistic": stat, "p_value": pval}
                insights.append(f"Z-test (Variant A vs B): Z-stat={stat:.3f}, P-value={pval:.3f}")
                if pval < 0.05:
                    insights.append("Result: Statistically significant difference found.")
                else:
                    insights.append("Result: No statistically significant difference found.")
            else:
                 insights.append("Could not run Z-test (no observations or invalid conversion data).")
                 metrics["z_test"] = "Could not run Z-test (no observations or invalid conversion data)."
        
        fig = px.bar(ab_summary, x='variant', y='conversion_rate', title="Conversion Rate by Variant")
        visualizations["conversion_rate_by_variant"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def marketing_campaign_performance_and_roi_analysis(df):
    analysis_name = "Marketing Campaign Performance and ROI Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['campaign_id', 'revenue', 'cost']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df.dropna(subset=['campaign_id', 'revenue', 'cost'], inplace=True)

        campaign_roi = df.groupby('campaign_id').agg(
            total_revenue=('revenue', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        
        campaign_roi['ROI'] = ((campaign_roi['total_revenue'] - campaign_roi['total_cost']) / campaign_roi['total_cost']) * 100
        campaign_roi.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        metrics = {"campaign_roi_summary": campaign_roi.to_dict(orient='records')}
        insights.append(f"Analyzed {len(campaign_roi)} campaigns.")
        
        fig = px.bar(campaign_roi.sort_values(by='ROI', ascending=False), x='campaign_id', y='ROI',
                     title="Campaign ROI (%)")
        visualizations["campaign_roi_bar"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def customer_segmentation_and_campaign_response_analysis(df):
    analysis_name = "Customer Segmentation and Campaign Response Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['customer_segment', 'campaign_id', 'response']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['response'] = pd.to_numeric(df['response'], errors='coerce') # Assuming 1/0
        df.dropna(subset=['customer_segment', 'campaign_id', 'response'], inplace=True)

        response_by_segment = df.groupby(['customer_segment', 'campaign_id'])['response'].mean().mul(100).unstack().reset_index()
        
        metrics = {"response_rate_by_segment_campaign": response_by_segment.to_dict(orient='records')}
        insights.append(f"Analyzed response rates for {df['customer_segment'].nunique()} segments across {df['campaign_id'].nunique()} campaigns.")
        
        # Melt for plotting
        response_melted = response_by_segment.melt(id_vars='customer_segment', var_name='campaign_id', value_name='response_rate')
        
        fig = px.bar(response_melted, x='customer_segment', y='response_rate', color='campaign_id',
                     barmode='group', title="Campaign Response Rate (%) by Customer Segment")
        visualizations["response_rate_by_segment"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def email_marketing_campaign_effectiveness_analysis(df):
    analysis_name = "Email Marketing Campaign Effectiveness Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['email_campaign_id', 'sent', 'opens', 'clicks', 'conversions']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        for col in expected:
            if col != 'email_campaign_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        email_summary = df.groupby('email_campaign_id').agg(
            total_sent=('sent', 'sum'),
            total_opens=('opens', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        
        email_summary['open_rate'] = (email_summary['total_opens'] / email_summary['total_sent']) * 100
        email_summary['click_through_rate_open'] = (email_summary['total_clicks'] / email_summary['total_opens']) * 100
        email_summary['conversion_rate_click'] = (email_summary['total_conversions'] / email_summary['total_clicks']) * 100
        email_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"email_campaign_summary": email_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(email_summary)} email campaigns.")
        
        fig = px.bar(email_summary, x='email_campaign_id', y=['open_rate', 'click_through_rate_open', 'conversion_rate_click'],
                     title="Email Campaign Funnel Rates (%)", barmode='group')
        visualizations["email_funnel_rates"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def sms_marketing_campaign_performance_analysis(df):
    analysis_name = "SMS Marketing Campaign Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['sms_campaign_id', 'sent', 'delivered', 'clicks', 'conversions']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        for col in expected:
            if col != 'sms_campaign_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        sms_summary = df.groupby('sms_campaign_id').agg(
            total_sent=('sent', 'sum'),
            total_delivered=('delivered', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        
        sms_summary['delivery_rate'] = (sms_summary['total_delivered'] / sms_summary['total_sent']) * 100
        sms_summary['click_through_rate_delivered'] = (sms_summary['total_clicks'] / sms_summary['total_delivered']) * 100
        sms_summary['conversion_rate_click'] = (sms_summary['total_conversions'] / sms_summary['total_clicks']) * 100
        sms_summary.replace([np.inf, -np.inf], np.nan, inplace=True)

        metrics = {"sms_campaign_summary": sms_summary.to_dict(orient='records')}
        insights.append(f"Analyzed {len(sms_summary)} SMS campaigns.")
        
        fig = px.bar(sms_summary, x='sms_campaign_id', y=['delivery_rate', 'click_through_rate_delivered', 'conversion_rate_click'],
                     title="SMS Campaign Funnel Rates (%)", barmode='group')
        visualizations["sms_funnel_rates"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def multi_channel_campaign_attribution_analysis(df):
    analysis_name = "Multi-Channel Campaign Attribution Analysis"
    matched = {}
    
    try:
        expected = ['customer_id', 'channel', 'conversion_timestamp', 'event_timestamp']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        insights = [
            "This analysis typically requires a sequence of customer touchpoints and a chosen attribution model (e.g., last-click, linear, time decay).",
            "This function is a placeholder. Implement custom logic or use libraries like `ChannelAttribution` for a full analysis.",
            "A simple 'last-click' attribution would count conversions by the last 'channel' before the 'conversion_timestamp'."
        ]
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": {}, "metrics": {}, "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def daily_marketing_campaign_performance_tracking(df):
    analysis_name = "Daily Marketing Campaign Performance Tracking"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['date', 'campaign_id', 'conversions', 'cost']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce')
        df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        daily_performance = df.groupby(['date', 'campaign_id']).agg(
            daily_conversions=('conversions', 'sum'),
            daily_cost=('cost', 'sum')
        ).reset_index()
        
        metrics = {"daily_performance": daily_performance.to_dict(orient='records')}
        insights.append(f"Tracked performance across {daily_performance['date'].nunique()} days and {daily_performance['campaign_id'].nunique()} campaigns.")
        
        fig = px.line(daily_performance, x='date', y='daily_conversions', color='campaign_id',
                      title="Daily Conversions by Campaign")
        visualizations["daily_conversions_trend"] = fig.to_json()
        
        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def campaign_reach_frequency_and_lift_analysis(df):
    analysis_name = "Campaign Reach, Frequency, and Lift Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['campaign_id', 'user_id', 'impressions', 'conversion']
        df_renamed, missing, matched = check_and_rename_columns(df.copy(), {k: [k] for k in expected})
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df = df_renamed
        df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce')
        df['conversion'] = pd.to_numeric(df['conversion'], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        # Reach
        reach_by_campaign = df.groupby('campaign_id')['user_id'].nunique().reset_index(name='unique_users_reached')
        
        # Frequency
        frequency_by_campaign = df.groupby(['campaign_id', 'user_id'])['impressions'].sum().reset_index()
        avg_frequency = frequency_by_campaign.groupby('campaign_id')['impressions'].mean().reset_index(name='average_frequency')
        
        metrics = {
            "campaign_reach": reach_by_campaign.to_dict(orient='records'),
            "average_frequency": avg_frequency.to_dict(orient='records')
        }
        
        insights.append("Analyzed Reach (unique users) and Frequency (avg. impressions per user).")
        insights.append("Lift analysis requires a control group or pre/post data, which is not implemented in this basic function.")

        fig1 = px.bar(reach_by_campaign, x='campaign_id', y='unique_users_reached', title="Campaign Reach (Unique Users)")
        visualizations["campaign_reach_bar"] = fig1.to_json()
        
        fig2 = px.bar(avg_frequency, x='campaign_id', y='average_frequency', title="Average Frequency by Campaign")
        visualizations["campaign_frequency_bar"] = fig2.to_json()

        return {
            "analysis_type": analysis_name, "status": "success", "matched_columns": matched,
            "visualizations": visualizations, "metrics": convert_to_native_types(metrics), "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_name, "status": "error", "error_message": str(e),
            "matched_columns": matched, "visualizations": {}, "metrics": {}, "insights": []
        }

def customer_journey_and_touchpoint_analysis(df):
    """
    Maps and analyzes the customer journey, identifying key touchpoints and their impact.
    Requires granular data on customer interactions over time.
    """
    print("Performing customer journey and touchpoint analysis...")
    if 'customer_id' in df.columns and 'event_timestamp' in df.columns and 'touchpoint' in df.columns:
        # This is a complex analysis often involving sequence mining or path analysis.
        # Placeholder for actual journey mapping logic.
        print("This analysis requires identifying sequences of touchpoints for each customer.")
        print("Consider techniques like sequence mining or visualizing common customer paths.")
        # Example: Show the first few touchpoints for a customer
        # df.sort_values(by=['customer_id', 'event_timestamp']).groupby('customer_id')['touchpoint'].apply(list).head()
    else:
        print("Missing 'customer_id', 'event_timestamp', or 'touchpoint' for customer journey analysis.")
    return df

def coupon_redemption_and_usage_analysis(df):
    """
    Analyzes the redemption rates and usage patterns of coupons.
    """
    print("Performing coupon redemption and usage analysis...")
    if 'coupon_id' in df.columns and 'status' in df.columns and 'discount_amount' in df.columns:
        coupon_summary = df.groupby('coupon_id').agg(
            total_issued=('status', 'count'),
            total_redeemed=('status', lambda x: (x == 'redeemed').sum()),
            total_discount_given=('discount_amount', lambda x: x[df['status'] == 'redeemed'].sum())
        ).reset_index()
        coupon_summary['redemption_rate'] = (coupon_summary['total_redeemed'] / coupon_summary['total_issued']).replace([np.inf, -np.inf], np.nan)
        print(coupon_summary)
    else:
        print("Missing 'coupon_id', 'status', or 'discount_amount' for coupon analysis.")
    return df

def ab_test_creative_performance_analysis(df):
    """
    Analyzes the performance of different creative variants in A/B tests.
    """
    print("Performing A/B test creative performance analysis...")
    if 'test_id' in df.columns and 'creative_variant' in df.columns and 'clicks' in df.columns and 'impressions' in df.columns:
        creative_performance = df.groupby(['test_id', 'creative_variant']).agg(
            total_impressions=('impressions', 'sum'),
            total_clicks=('clicks', 'sum')
        ).reset_index()
        creative_performance['CTR'] = (creative_performance['total_clicks'] / creative_performance['total_impressions']).replace([np.inf, -np.inf], np.nan)
        print(creative_performance.sort_values(by='CTR', ascending=False))
    else:
        print("Missing 'test_id', 'creative_variant', 'clicks', or 'impressions' for creative performance analysis.")
    return df

def website_visitor_conversion_analysis(df):
    """
    Analyzes how website visitors convert, often looking at different traffic sources, landing pages, etc.
    """
    print("Performing website visitor conversion analysis...")
    if 'visitor_id' in df.columns and 'conversion_status' in df.columns and 'source' in df.columns:
        conversion_by_source = df.groupby('source')['conversion_status'].value_counts(normalize=True).unstack().fillna(0)
        print("Conversion Rate by Source:")
        print(conversion_by_source)
    else:
        print("Missing 'visitor_id', 'conversion_status', or 'source' for website conversion analysis.")
    return df

def digital_advertising_platform_performance_and_roas_analysis(df):
    """
    Analyzes the performance and Return on Ad Spend (ROAS) of digital advertising platforms.
    """
    print("Performing digital advertising platform performance and ROAS analysis...")
    if 'platform' in df.columns and 'ad_spend' in df.columns and 'revenue' in df.columns:
        platform_roas = df.groupby('platform').agg(
            total_ad_spend=('ad_spend', 'sum'),
            total_revenue=('revenue', 'sum')
        ).reset_index()
        platform_roas['ROAS'] = (platform_roas['total_revenue'] / platform_roas['total_ad_spend']).replace([np.inf, -np.inf], np.nan)
        print(platform_roas.sort_values(by='ROAS', ascending=False))
    else:
        print("Missing 'platform', 'ad_spend', or 'revenue' for digital advertising analysis.")
    return df

def customer_satisfaction_survey_analysis(df):
    """
    Analyzes customer satisfaction survey data (e.g., NPS, CSAT scores, sentiment).
    """
    print("Performing customer satisfaction survey analysis...")
    if 'survey_id' in df.columns and 'score' in df.columns and 'feedback' in df.columns:
        print(f"Average Satisfaction Score: {df['score'].mean():.2f}")
        # Further analysis could involve sentiment analysis on 'feedback'
        print("Consider performing sentiment analysis on the 'feedback' column.")
    else:
        print("Missing 'survey_id', 'score', or 'feedback' for customer satisfaction analysis.")
    return df

def ad_placement_and_engagement_analysis(df):
    """
    Analyzes the performance of different ad placements and the engagement they generate.
    """
    print("Performing ad placement and engagement analysis...")
    if 'ad_placement' in df.columns and 'impressions' in df.columns and 'clicks' in df.columns:
        placement_performance = df.groupby('ad_placement').agg(
            total_impressions=('impressions', 'sum'),
            total_clicks=('clicks', 'sum')
        ).reset_index()
        placement_performance['CTR'] = (placement_performance['total_clicks'] / placement_performance['total_impressions']).replace([np.inf, -np.inf], np.nan)
        print(placement_performance.sort_values(by='CTR', ascending=False))
    else:
        print("Missing 'ad_placement', 'impressions', or 'clicks' for ad placement analysis.")
    return df

def lead_generation_and_qualification_analysis(df):
    """
    Analyzes the effectiveness of lead generation efforts and the quality of generated leads.
    """
    print("Performing lead generation and qualification analysis...")
    if 'lead_source' in df.columns and 'lead_status' in df.columns and 'conversion_status' in df.columns:
        leads_by_source = df.groupby('lead_source')['lead_status'].value_counts().unstack().fillna(0)
        print("Leads by Source and Status:")
        print(leads_by_source)
        # Further analysis could look at conversion rates from qualified leads
        qualified_leads_conversion = df[df['lead_status'] == 'qualified'].groupby('lead_source')['conversion_status'].value_counts(normalize=True).unstack().fillna(0)
        print("\nConversion Rate of Qualified Leads by Source:")
        print(qualified_leads_conversion)
    else:
        print("Missing 'lead_source', 'lead_status', or 'conversion_status' for lead analysis.")
    return df

def cross_device_campaign_conversion_analysis(df):
    """
    Analyzes conversions that occur across different devices (e.g., user saw ad on mobile, converted on desktop).
    Requires unified user IDs across devices.
    """
    print("Performing cross-device campaign conversion analysis...")
    if 'user_id' in df.columns and 'device_type' in df.columns and 'conversion_status' in df.columns:
        # This requires identifying conversion paths across devices for the same user.
        print("This analysis is complex and requires tracking user journeys across multiple devices.")
        print("Consider identifying unique user IDs and their device interactions leading to conversion.")
        cross_device_conversions = df.groupby('user_id').agg(
            device_types=('device_type', lambda x: set(x)),
            converted=('conversion_status', lambda x: 'converted' in x.values)
        ).reset_index()
        cross_device_conversions['num_devices'] = cross_device_conversions['device_types'].apply(len)
        print(cross_device_conversions[cross_device_conversions['converted'] == True].groupby('num_devices').size())
    else:
        print("Missing 'user_id', 'device_type', or 'conversion_status' for cross-device analysis.")
    return df

def content_marketing_and_engagement_analysis(df):
    """
    Analyzes the performance and engagement of content marketing efforts.
    Similar to `content_analysis` but specifically for content marketing.
    """
    print("Performing content marketing and engagement analysis...")
    if 'content_type' in df.columns and 'views' in df.columns and 'shares' in df.columns and 'comments' in df.columns:
        content_marketing_summary = df.groupby('content_type').agg(
            total_views=('views', 'sum'),
            total_shares=('shares', 'sum'),
            total_comments=('comments', 'sum')
        ).reset_index()
        print(content_marketing_summary)
    else:
        print("Missing 'content_type', 'views', 'shares', or 'comments' for content marketing analysis.")
    return df

def customer_preference_and_personalization_analysis(df):
    """
    Analyzes customer preferences to inform personalization strategies.
    """
    print("Performing customer preference and personalization analysis...")
    if 'customer_id' in df.columns and 'product_category' in df.columns and 'interaction_type' in df.columns:
        # This would typically involve recommendation systems or preference matrices.
        print("This analysis often involves building customer profiles based on their interactions and preferences.")
        print("Consider techniques like collaborative filtering or content-based filtering for personalization insights.")
        customer_preferences = df.groupby('customer_id')['product_category'].value_counts().unstack().fillna(0)
        print(customer_preferences.head())
    else:
        print("Missing 'customer_id', 'product_category', or 'interaction_type' for customer preference analysis.")
    return df

def special_offer_and_discount_effectiveness_analysis(df):
    """
    Analyzes the effectiveness of special offers and discounts on sales and customer behavior.
    """
    print("Performing special offer and discount effectiveness analysis...")
    if 'offer_id' in df.columns and 'sales_amount' in df.columns and 'discount_applied' in df.columns:
        offer_effectiveness = df.groupby('offer_id').agg(
            total_sales_with_offer=('sales_amount', 'sum'),
            total_discounts_given=('discount_applied', 'sum'),
            num_transactions=('offer_id', 'count')
        ).reset_index()
        print(offer_effectiveness)
    else:
        print("Missing 'offer_id', 'sales_amount', or 'discount_applied' for special offer analysis.")
    return df

def geotargeted_campaign_performance_analysis(df):
    """
    Analyzes the performance of campaigns targeted at specific geographic locations.
    """
    print("Performing geotargeted campaign performance analysis...")
    if 'campaign_id' in df.columns and 'location' in df.columns and 'conversions' in df.columns and 'cost' in df.columns:
        geo_performance = df.groupby(['campaign_id', 'location']).agg(
            total_conversions=('conversions', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        geo_performance['CPA'] = (geo_performance['total_cost'] / geo_performance['total_conversions']).replace([np.inf, -np.inf], np.nan)
        print(geo_performance.sort_values(by='CPA'))
    else:
        print("Missing 'campaign_id', 'location', 'conversions', or 'cost' for geotargeted campaign analysis.")
    return df

def ab_testing_campaign_variant_analysis(df):
    """
    Analyzes the performance of specific campaign variants within an A/B test.
    This is essentially a more specific version of `ab_testing`.
    """
    print("Performing A/B testing campaign variant analysis...")
    # This function is very similar to the general `ab_testing` function.
    # It focuses on comparing different campaign variants specifically.
    if 'campaign_variant' in df.columns and 'conversions' in df.columns and 'visitors' in df.columns:
        variant_summary = df.groupby('campaign_variant').agg(
            total_visitors=('visitors', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        variant_summary['conversion_rate'] = (variant_summary['total_conversions'] / variant_summary['total_visitors']).replace([np.inf, -np.inf], np.nan)
        print(variant_summary)
        # Further statistical tests would be crucial here to determine significance.
    else:
        print("Missing 'campaign_variant', 'conversions', or 'visitors' for A/B testing variant analysis.")
    return df

def media_buying_and_ad_performance_analysis(df):
    """
    Analyzes the effectiveness of media buying strategies and overall ad performance.
    """
    print("Performing media buying and ad performance analysis...")
    if 'ad_platform' in df.columns and 'spend' in df.columns and 'impressions' in df.columns and 'clicks' in df.columns and 'conversions' in df.columns:
        media_performance = df.groupby('ad_platform').agg(
            total_spend=('spend', 'sum'),
            total_impressions=('impressions', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        media_performance['CPM'] = (media_performance['total_spend'] / media_performance['total_impressions'] * 1000).replace([np.inf, -np.inf], np.nan)
        media_performance['CPC'] = (media_performance['total_spend'] / media_performance['total_clicks']).replace([np.inf, -np.inf], np.nan)
        media_performance['CPA'] = (media_performance['total_spend'] / media_performance['total_conversions']).replace([np.inf, -np.inf], np.nan)
        media_performance['CTR'] = (media_performance['total_clicks'] / media_performance['total_impressions']).replace([np.inf, -np.inf], np.nan)
        print(media_performance)
    else:
        print("Missing key columns like 'ad_platform', 'spend', 'impressions', 'clicks', or 'conversions' for media buying analysis.")

    return df
def geotargeting_accuracy_and_effectiveness_analysis(df):
    print("\n--- Geotargeting Accuracy and Effectiveness Analysis ---")
    expected = {
        'AdImpressionID': ['AdImpressionID', 'ID', 'ImpressionID'],
        'TargetedRegion': ['TargetedRegion', 'TargetArea', 'GeoTarget'],
        'ActualRegion': ['ActualRegion', 'ActualLocation', 'UserLocation'],
        'ConversionEvent': ['ConversionEvent', 'Converted', 'Purchase', 'IsConverted'],
        'AdSpend': ['AdSpend', 'Spend', 'Cost']
    }
    df, missing, _ = check_and_rename_columns(df.copy(), expected) # Use .copy() to avoid SettingWithCopyWarning

    if missing:
        # Assuming show_missing_columns_warning is defined elsewhere
        # show_missing_columns_warning(missing, expected) 
        print(f"Missing required columns: {missing}")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    # Data Type Conversion and Cleaning
    # Ensure ConversionEvent is boolean
    df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({
        '1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False
    }).fillna(False) # Default to False if conversion status is unclear

    df['AdSpend'] = pd.to_numeric(df['AdSpend'], errors='coerce')
    df = df.dropna(subset=['TargetedRegion', 'ActualRegion', 'ConversionEvent', 'AdSpend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    # Core Calculations
    total_impressions = len(df)
    accurate_targets = int((df['TargetedRegion'] == df['ActualRegion']).sum())
    accuracy_rate = (accurate_targets / total_impressions) * 100

    print(f"\nTotal Ad Impressions: {total_impressions}")
    print(f"Accurate Geotargets (TargetedRegion == ActualRegion): {accurate_targets}")
    print(f"Geotargeting Accuracy Rate: {accuracy_rate:.2f}%")

    # Performance by Geotargeting Accuracy
    performance_by_accuracy = df.groupby(df['TargetedRegion'] == df['ActualRegion']).agg(
        NumImpressions=('AdImpressionID', 'count'),
        ConversionRate=('ConversionEvent', 'mean'), # Mean of True/False gives conversion rate
        TotalSpend=('AdSpend', 'sum')
    ).reset_index()
    performance_by_accuracy.columns = ['IsAccurateTarget', 'NumImpressions', 'ConversionRate', 'TotalSpend']
    performance_by_accuracy['ConversionRate'] *= 100 # Convert to percentage
    performance_by_accuracy['IsAccurateTarget'] = performance_by_accuracy['IsAccurateTarget'].map({True: 'Accurate', False: 'Inaccurate'})

    print("\nPerformance by Geotargeting Accuracy:")
    print(performance_by_accuracy.round(2))

    # Plotting
    fig1 = px.pie(performance_by_accuracy, names='IsAccurateTarget', values='NumImpressions', 
                  title='Geotargeting Accuracy Distribution (by Impressions)',
                  color_discrete_map={'Accurate': 'lightgreen', 'Inaccurate': 'salmon'})
    # fig1.show() # .show() will not work in this script environment, returning JSON
 
    fig2 = px.bar(performance_by_accuracy, x='IsAccurateTarget', y='ConversionRate', 
                  title='Conversion Rate by Geotargeting Accuracy',
                  text_auto='.2s', color='IsAccurateTarget',
                  color_discrete_map={'Accurate': 'lightgreen', 'Inaccurate': 'salmon'})
    fig2.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig2.update_layout(yaxis_title="Conversion Rate (%)")
    # fig2.show()
    
    # Targeted vs. Actual Region Analysis (Confusion Matrix like)
    cross_tab_regions = pd.crosstab(df['TargetedRegion'], df['ActualRegion'], normalize='index')
    print("\nTargeted vs Actual Region (Row-normalized):\n", cross_tab_regions.round(2))
    
    fig3 = px.imshow(cross_tab_regions, text_auto=True, 
                     title='Targeted vs. Actual Region Impression Distribution (Normalized by Targeted Region)',
                     labels=dict(x="Actual Region", y="Targeted Region", color="Proportion"),
                     color_continuous_scale='Viridis')
    fig3.update_xaxes(side="top")
    # fig3.show()

    # Further Effectiveness: Conversion Rate by Targeted Region
    conversion_by_target_region = df.groupby('TargetedRegion').agg(
        TotalImpressions=('AdImpressionID', 'count'),
        TotalConversions=('ConversionEvent', lambda x: x.sum()), # Sum of True values
        TotalSpend=('AdSpend', 'sum')
    ).reset_index()
    conversion_by_target_region['ConversionRate'] = (conversion_by_target_region['TotalConversions'] / conversion_by_target_region['TotalImpressions']) * 100
    conversion_by_target_region['CPA'] = (conversion_by_target_region['TotalSpend'] / conversion_by_target_region['TotalConversions']).replace([np.inf, -np.inf], np.nan)
    
    print("\nConversion Performance by Targeted Region:")
    print(conversion_by_target_region.round(2).sort_values(by='ConversionRate', ascending=False))

    fig4 = px.bar(conversion_by_target_region.sort_values(by='ConversionRate', ascending=False), 
                  x='TargetedRegion', y='ConversionRate', 
                  title='Conversion Rate by Targeted Region',
                  text_auto='.2s')
    fig4.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig4.update_layout(yaxis_title="Conversion Rate (%)")
    # fig4.show()

    return {
        "metrics": {
            "Total Impressions": total_impressions,
            "Accurate Geotargets": accurate_targets, # Corrected from accurate_impressions
            "Geotargeting Accuracy Rate": accuracy_rate,
            "Performance by Geotargeting Accuracy": performance_by_accuracy.to_dict(orient='records'),
            "Conversion Performance by Targeted Region": conversion_by_target_region.to_dict(orient='records')
        },
        "figures": {
            "Geotargeting_Accuracy_Pie": fig1.to_json(), # Convert figures to JSON for portability
            "Conversion_Rate_by_Accuracy_Bar": fig2.to_json(),
            "Targeted_vs_Actual_Region_Heatmap": fig3.to_json(),
            "Conversion_Rate_by_Targeted_Region_Bar": fig4.to_json()
        }
    }





def main_backend(file_path, encoding='utf-8', category=None, analysis=None, specific_analysis_name=None):
    """
    Main function to run manufacturing data analysis
    
    Parameters:
    - file_path: path to the data file (CSV or Excel)
    - encoding: file encoding (default: 'utf-8')
    - category: analysis category ('General' or 'Specific')
    - analysis: specific analysis name for general category
    - specific_analysis_name: specific analysis name for specific category
    
    Returns:
    - Dictionary with analysis results
    """
    
    # Load data
    df = load_data(file_path, encoding)
    if df is None:
        return {
            "analysis_type": "Data Loading",
            "status": "error", 
            "error_message": "Failed to load data file"
        }
    
    # Mapping of all analysis functions (simplified for this example)
    analysis_functions = {
        # General analyses
        "General Insights": show_general_insights,
        "Customer Segmentation": customer_segmentation,
        # Add other manufacturing analysis functions here...
    }
    
    # Determine which analysis to run
    if category == "General" and analysis in analysis_functions:
        result = analysis_functions[analysis](df)
    elif category == "Specific" and specific_analysis_name in analysis_functions:
        result = analysis_functions[specific_analysis_name](df)
    else:
        # Default to general insights
        result = show_general_insights(df)
    
    return result

# Keep the original main function for command-line usage
def main():
    """Main function for command-line usage"""
    print("🏭 Marketing Analytics Dashboard")

    # File path and encoding input
    file_path = input("Enter path to your marketing data file (e.g., data.csv or data.xlsx): ")
    encoding = input("Enter file encoding (e.g., utf-8, latin1, cp1252, default=utf-8): ")
    if not encoding:
        encoding = 'utf-8'

    df = load_data(file_path, encoding)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    
    # Diagnostic information
    print(f"\n📋 YOUR DATASET COLUMNS ({len(df.columns)} total):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}'")
   
    
    choice = input("Enter your choice (1 or 2): ")

# Example usage for API/backend
if __name__ == "__main__":
    # Example usage of the analysis functions
    file_path = "sample_manufacturing_data.csv"  # Replace with your actual file path
    
    # Run general insights
    result = main_backend(file_path)
    print("General Insights:", result.keys() if isinstance(result, dict) else "No result")
    
    # Run specific analysis
    result = main_backend(
        file_path, 
        category="Specific", 
        specific_analysis_name="regulatory_compliance_status_analysis"
    )
    print("Regulatory Compliance Analysis completed:", "status" in result if isinstance(result, dict) else "No result")

