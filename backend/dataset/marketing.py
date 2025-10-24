import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, chi2_contingency, f_oneway
from statsmodels.stats.proportion import proportions_ztest
# from fuzzywuzzy import process # Uncomment if you want to use fuzzy matching for column names

# Helper functions
def safe_numeric_conversion(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df.dropna(subset=[column_name])
    print(f"Warning: Column '{column_name}' not found for numeric conversion.")
    return df

def check_and_rename_columns(df, expected_cols_map):
    missing_cols = []
    renamed_df = df.copy()
    for standard_name, potential_names in expected_cols_map.items():
        found = False
        for p_name in potential_names:
            if p_name in renamed_df.columns:
                if p_name != standard_name:
                    renamed_df = renamed_df.rename(columns={p_name: standard_name})
                found = True
                break
        if not found and standard_name not in renamed_df.columns:
            missing_cols.append(standard_name)
    return renamed_df, missing_cols

def show_missing_columns_warning(missing_columns, matched_columns=None):
    print(f"\n--- WARNING: Required Columns Not Found ---")
    print(f"The following columns are needed but missing: {', '.join(missing_columns)}")
    if matched_columns:
        print("Expected column mappings attempted:")
        for key, value in matched_columns.items():
            if value is None:
                print(f"- '{key}' (e.g., '{key}' or a similar variation)")
    print("Analysis might be incomplete or aborted due to missing required data.")

# Marketing and Advertising Analysis Functions

def a_b_testing_campaign_variant_analysis(df):
    print("\n--- A/B Testing Campaign Variant Analysis ---")
    expected = {
        'Variant': ['Variant', 'CampaignVariant', 'TreatmentGroup'],
        'Conversions': ['Conversions', 'NumConversions', 'Converted'],
        'TotalUsers': ['TotalUsers', 'Users', 'Impressions', 'Visits'],
        'Revenue': ['Revenue', 'TotalRevenue', 'Sales']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['TotalUsers'] = pd.to_numeric(df['TotalUsers'], errors='coerce')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df = df.dropna(subset=['Variant', 'Conversions', 'TotalUsers'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['ConversionRate'] = df['Conversions'] / df['TotalUsers']
    df['AvgRevenuePerUser'] = df['Revenue'] / df['TotalUsers'] if 'Revenue' in df.columns else np.nan

    variant_summary = df.groupby('Variant').agg(
        TotalUsers=('TotalUsers', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        MeanConversionRate=('ConversionRate', 'mean'),
        TotalRevenue=('Revenue', 'sum') if 'Revenue' in df.columns else ('TotalUsers', 'size') # Placeholder if no revenue
    ).reset_index()
    
    variant_summary['OverallConversionRate'] = variant_summary['TotalConversions'] / variant_summary['TotalUsers']
    variant_summary['OverallRevenuePerUser'] = variant_summary['TotalRevenue'] / variant_summary['TotalUsers'] if 'Revenue' in df.columns else np.nan

    print("\nVariant Performance Summary:")
    print(variant_summary.round(4))

    # A/B Test for Conversion Rate (Z-test for proportions)
    if len(variant_summary) == 2:
        variant_A = variant_summary.loc[0]
        variant_B = variant_summary.loc[1]
        
        count = np.array([variant_A['TotalConversions'], variant_B['TotalConversions']])
        nobs = np.array([variant_A['TotalUsers'], variant_B['TotalUsers']])
        
        stat, pval = proportions_ztest(count, nobs, alternative='two-sided')
        print(f"\nZ-test for Conversion Rate (Variant 0 vs Variant 1):")
        print(f"  Z-statistic: {stat:.3f}")
        print(f"  P-value: {pval:.3f}")
        if pval < 0.05:
            print("  Result: Statistically significant difference in conversion rates.")
        else:
            print("  Result: No statistically significant difference in conversion rates.")
    elif len(variant_summary) > 2:
        print("\nNote: Z-test for proportions is typically for two variants. For multiple, consider Chi-squared or ANOVA on conversion rates if aggregated per variant is acceptable.")

    fig1 = px.bar(variant_summary, x='Variant', y='OverallConversionRate', title='Overall Conversion Rate by Variant')
    fig1.show()

    if 'Revenue' in df.columns:
        fig2 = px.bar(variant_summary, x='Variant', y='OverallRevenuePerUser', title='Overall Revenue Per User by Variant')
        fig2.show()
    
    if 'TotalUsers' in df.columns and 'Conversions' in df.columns:
        fig3 = px.histogram(df, x='ConversionRate', color='Variant', barmode='overlay', title='Distribution of Conversion Rates by Variant')
        fig3.show()

    return {
        "metrics": variant_summary.to_dict(orient='records'),
        "ab_test_results": {"z_statistic": stat, "p_value": pval} if len(variant_summary) == 2 else "Not applicable for >2 variants",
        "figures": {
            "Conversion_Rate_by_Variant_Bar": fig1,
            "Revenue_Per_User_by_Variant_Bar": fig2 if 'Revenue' in df.columns else None,
            "Conversion_Rates_Distribution_Histogram": fig3
        }
    }


def media_buying_and_ad_performance_analysis(df):
    print("\n--- Media Buying and Ad Performance Analysis ---")
    expected = {
        'AdCampaignID': ['AdCampaignID', 'CampaignID', 'ID'],
        'Platform': ['Platform', 'AdPlatform', 'Channel'],
        'Spend': ['Spend', 'AdSpend', 'Cost'],
        'Impressions': ['Impressions', 'AdImpressions'],
        'Clicks': ['Clicks', 'AdClicks'],
        'Conversions': ['Conversions', 'AdConversions', 'Purchases'],
        'TargetAudience': ['TargetAudience', 'AudienceSegment']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df = df.dropna(subset=['Spend', 'Impressions', 'Clicks', 'Conversions'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CPM'] = (df['Spend'] / df['Impressions']) * 1000 if df['Impressions'].sum() > 0 else 0
    df['CPC'] = df['Spend'] / df['Clicks'] if df['Clicks'].sum() > 0 else 0
    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['CPA'] = df['Spend'] / df['Conversions'] if df['Conversions'].sum() > 0 else 0

    campaign_summary = df.groupby('AdCampaignID').agg(
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

    print("\nAd Campaign Performance Summary:")
    print(campaign_summary.round(2))

    fig1 = px.bar(campaign_summary, x='AdCampaignID', y='OverallCPA', title='Cost Per Acquisition (CPA) by Campaign')
    fig1.show()

    fig2 = px.bar(campaign_summary, x='AdCampaignID', y='OverallCTR', title='Click-Through Rate (CTR) by Campaign')
    fig2.show()

    if 'Platform' in df.columns:
        platform_performance = df.groupby('Platform').agg(
            AvgCTR=('CTR', 'mean'),
            AvgCPA=('CPA', 'mean'),
            TotalSpend=('Spend', 'sum')
        ).reset_index()
        fig3 = px.bar(platform_performance, x='Platform', y='AvgCTR', title='Average CTR by Platform')
        fig3.show()

    return {
        "metrics": campaign_summary.to_dict(orient='records'),
        "figures": {
            "CPA_by_Campaign_Bar": fig1,
            "CTR_by_Campaign_Bar": fig2,
            "Avg_CTR_by_Platform_Bar": fig3 if 'Platform' in df.columns else None
        }
    }


def product_line_marketing_campaign_analysis(df):
    print("\n--- Product Line Marketing Campaign Analysis ---")
    expected = {
        'ProductLine': ['ProductLine', 'ProductCategory', 'ProductGroup'],
        'CampaignID': ['CampaignID', 'MarketingCampaignID', 'ID'],
        'Budget': ['Budget', 'CampaignBudget', 'Spend'],
        'SalesUnits': ['SalesUnits', 'UnitsSold', 'Quantity'],
        'Revenue': ['Revenue', 'TotalRevenue', 'Sales'],
        'ConversionRate': ['ConversionRate', 'ConversionPct']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    df['SalesUnits'] = pd.to_numeric(df['SalesUnits'], errors='coerce')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['ConversionRate'] = pd.to_numeric(df['ConversionRate'], errors='coerce')
    df = df.dropna(subset=['ProductLine', 'CampaignID', 'Budget', 'SalesUnits', 'Revenue'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['ROI'] = ((df['Revenue'] - df['Budget']) / df['Budget']) * 100 if df['Budget'].sum() > 0 else np.nan

    product_line_summary = df.groupby('ProductLine').agg(
        TotalBudget=('Budget', 'sum'),
        TotalSalesUnits=('SalesUnits', 'sum'),
        TotalRevenue=('Revenue', 'sum'),
        AvgROI=('ROI', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean')
    ).reset_index()

    print("\nProduct Line Performance Summary:")
    print(product_line_summary.round(2))

    fig1 = px.bar(product_line_summary, x='ProductLine', y='TotalRevenue', title='Total Revenue by Product Line')
    fig1.show()

    fig2 = px.bar(product_line_summary, x='ProductLine', y='AvgROI', title='Average ROI by Product Line')
    fig2.show()

    fig3 = px.scatter(df, x='Budget', y='Revenue', color='ProductLine', hover_name='CampaignID',
                     title='Revenue vs. Budget by Product Line')
    fig3.show()

    return {
        "metrics": product_line_summary.to_dict(orient='records'),
        "figures": {
            "Total_Revenue_by_Product_Line_Bar": fig1,
            "Average_ROI_by_Product_Line_Bar": fig2,
            "Revenue_vs_Budget_Scatter": fig3
        }
    }


def website_engagement_and_ad_interaction_analysis(df):
    print("\n--- Website Engagement and Ad Interaction Analysis ---")
    expected = {
        'UserID': ['UserID', 'VisitorID', 'CustomerID'],
        'PageViews': ['PageViews', 'PagesVisited', 'Views'],
        'TimeOnSiteSeconds': ['TimeOnSiteSeconds', 'SessionDuration', 'Duration'],
        'AdClicked': ['AdClicked', 'ClickedAd', 'AdInteraction'], # Binary (1/0 or True/False)
        'ConversionEvent': ['ConversionEvent', 'Converted', 'PurchaseEvent'] # Binary (1/0 or True/False)
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['PageViews'] = pd.to_numeric(df['PageViews'], errors='coerce')
    df['TimeOnSiteSeconds'] = pd.to_numeric(df['TimeOnSiteSeconds'], errors='coerce')
    
    df['AdClicked'] = df['AdClicked'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})

    df = df.dropna(subset=['PageViews', 'TimeOnSiteSeconds', 'AdClicked', 'ConversionEvent'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_page_views = df['PageViews'].mean()
    avg_time_on_site_minutes = df['TimeOnSiteSeconds'].mean() / 60
    ad_click_rate = df['AdClicked'].mean() * 100
    overall_conversion_rate = df['ConversionEvent'].mean() * 100

    print(f"Average Page Views per User: {avg_page_views:.1f}")
    print(f"Average Time on Site: {avg_time_on_site_minutes:.1f} minutes")
    print(f"Ad Click Rate: {ad_click_rate:.2f}%")
    print(f"Overall Conversion Rate: {overall_conversion_rate:.2f}%")

    ad_interaction_summary = df.groupby('AdClicked').agg(
        AvgPageViews=('PageViews', 'mean'),
        AvgTimeOnSite=('TimeOnSiteSeconds', 'mean'),
        ConversionRate=('ConversionEvent', 'mean')
    ).reset_index()
    ad_interaction_summary['AvgTimeOnSite'] /= 60 # Convert to minutes
    ad_interaction_summary['ConversionRate'] *= 100

    print("\nPerformance by Ad Interaction:")
    print(ad_interaction_summary.round(2))

    fig1 = px.box(df, x='AdClicked', y='PageViews', title='Page Views by Ad Click Status')
    fig1.show()

    fig2 = px.box(df, x='AdClicked', y='TimeOnSiteSeconds', title='Time on Site by Ad Click Status (Seconds)')
    fig2.show()

    fig3 = px.bar(ad_interaction_summary, x='AdClicked', y='ConversionRate', title='Conversion Rate by Ad Click Status')
    fig3.show()

    return {
        "metrics": {
            "Average Page Views": avg_page_views,
            "Average Time on Site (minutes)": avg_time_on_site_minutes,
            "Ad Click Rate": ad_click_rate,
            "Overall Conversion Rate": overall_conversion_rate,
            "Performance by Ad Interaction": ad_interaction_summary.to_dict(orient='records')
        },
        "figures": {
            "Page_Views_by_Ad_Click_Box": fig1,
            "Time_on_Site_by_Ad_Click_Box": fig2,
            "Conversion_Rate_by_Ad_Click_Bar": fig3
        }
    }


def daily_sales_revenue_and_campaign_correlation_analysis(df):
    print("\n--- Daily Sales Revenue and Campaign Correlation Analysis ---")
    expected = {
        'Date': ['Date', 'SaleDate', 'Day'],
        'DailyRevenue': ['DailyRevenue', 'Revenue', 'SalesAmount'],
        'MarketingSpend': ['MarketingSpend', 'CampaignSpend', 'AdSpend'],
        'CampaignType': ['CampaignType', 'Type', 'CampaignName']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DailyRevenue'] = pd.to_numeric(df['DailyRevenue'], errors='coerce')
    df['MarketingSpend'] = pd.to_numeric(df['MarketingSpend'], errors='coerce')
    df = df.dropna(subset=['Date', 'DailyRevenue', 'MarketingSpend'])
    df = df.sort_values('Date')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_revenue = df['DailyRevenue'].sum()
    total_spend = df['MarketingSpend'].sum()
    revenue_spend_correlation = df['DailyRevenue'].corr(df['MarketingSpend'])

    print(f"Total Revenue: ${total_revenue:,.2f}")
    print(f"Total Marketing Spend: ${total_spend:,.2f}")
    print(f"Correlation (Daily Revenue vs. Marketing Spend): {revenue_spend_correlation:.2f}")

    fig1 = px.line(df, x='Date', y='DailyRevenue', title='Daily Revenue Trend')
    fig1.show()

    fig2 = px.scatter(df, x='MarketingSpend', y='DailyRevenue', color='CampaignType' if 'CampaignType' in df.columns else None,
                     title='Daily Revenue vs. Marketing Spend')
    fig2.show()
    
    if 'CampaignType' in df.columns:
        campaign_type_impact = df.groupby('CampaignType').agg(
            AvgDailyRevenue=('DailyRevenue', 'mean'),
            AvgMarketingSpend=('MarketingSpend', 'mean')
        ).reset_index()
        fig3 = px.bar(campaign_type_impact, x='CampaignType', y='AvgDailyRevenue', title='Average Daily Revenue by Campaign Type')
        fig3.show()

    return {
        "metrics": {
            "Total Revenue": total_revenue,
            "Total Marketing Spend": total_spend,
            "Revenue vs Spend Correlation": revenue_spend_correlation
        },
        "figures": {
            "Daily_Revenue_Trend_Line": fig1,
            "Daily_Revenue_vs_Marketing_Spend_Scatter": fig2,
            "Avg_Daily_Revenue_by_Campaign_Type_Bar": fig3 if 'CampaignType' in df.columns else None
        }
    }


def influencer_marketing_campaign_performance_analysis(df):
    print("\n--- Influencer Marketing Campaign Performance Analysis ---")
    expected = {
        'InfluencerID': ['InfluencerID', 'Influencer', 'ID'],
        'CampaignName': ['CampaignName', 'Campaign', 'MarketingCampaign'],
        'FollowerCount': ['FollowerCount', 'Followers'],
        'EngagementRate': ['EngagementRate', 'AvgEngagement'],
        'PostsCount': ['PostsCount', 'NumPosts'],
        'Reach': ['Reach', 'TotalReach'],
        'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
        'Spend': ['Spend', 'PaymentToInfluencer', 'Cost']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['FollowerCount'] = pd.to_numeric(df['FollowerCount'], errors='coerce')
    df['EngagementRate'] = pd.to_numeric(df['EngagementRate'], errors='coerce')
    df['PostsCount'] = pd.to_numeric(df['PostsCount'], errors='coerce')
    df['Reach'] = pd.to_numeric(df['Reach'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
    df = df.dropna(subset=['InfluencerID', 'FollowerCount', 'EngagementRate', 'Conversions', 'Spend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['ROI'] = ((df['Conversions'] * df['Spend'].mean()) - df['Spend']) / df['Spend'] * 100 if df['Spend'].sum() > 0 and df['Conversions'].sum() > 0 else np.nan
    df['CostPerConversion'] = df['Spend'] / df['Conversions'] if df['Conversions'].sum() > 0 else np.nan

    influencer_summary = df.groupby('InfluencerID').agg(
        AvgFollowers=('FollowerCount', 'mean'),
        AvgEngagementRate=('EngagementRate', 'mean'),
        TotalConversions=('Conversions', 'sum'),
        TotalSpend=('Spend', 'sum'),
        AvgCostPerConversion=('CostPerConversion', 'mean')
    ).reset_index()

    print("\nInfluencer Performance Summary:")
    print(influencer_summary.round(2))

    fig1 = px.scatter(df, x='EngagementRate', y='Conversions', size='FollowerCount', color='Spend',
                     hover_name='InfluencerID', title='Conversions vs. Engagement Rate (Sized by Followers, Colored by Spend)')
    fig1.show()

    fig2 = px.bar(influencer_summary.sort_values('TotalConversions', ascending=False).head(10),
                  x='InfluencerID', y='TotalConversions', title='Top 10 Influencers by Total Conversions')
    fig2.show()

    if 'CampaignName' in df.columns:
        campaign_performance = df.groupby('CampaignName').agg(
            TotalSpend=('Spend', 'sum'),
            TotalConversions=('Conversions', 'sum')
        ).reset_index()
        campaign_performance['CostPerConversion'] = campaign_performance['TotalSpend'] / campaign_performance['TotalConversions']
        fig3 = px.bar(campaign_performance, x='CampaignName', y='CostPerConversion', title='Cost Per Conversion by Campaign')
        fig3.show()

    return {
        "metrics": influencer_summary.to_dict(orient='records'),
        "figures": {
            "Conversions_vs_Engagement_Scatter": fig1,
            "Top_10_Influencers_by_Conversions_Bar": fig2,
            "Cost_Per_Conversion_by_Campaign_Bar": fig3 if 'CampaignName' in df.columns else None
        }
    }


def ad_copy_performance_analysis(df):
    print("\n--- Ad Copy Performance Analysis ---")
    expected = {
        'AdCopyID': ['AdCopyID', 'CopyID', 'ID'],
        'AdCopyText': ['AdCopyText', 'CopyText', 'Headline'],
        'Clicks': ['Clicks', 'AdClicks'],
        'Impressions': ['Impressions', 'AdImpressions'],
        'Conversions': ['Conversions', 'AdConversions', 'Purchases'],
        'AdGroup': ['AdGroup', 'TargetGroup']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df = df.dropna(subset=['AdCopyID', 'Clicks', 'Impressions', 'Conversions'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0

    ad_copy_summary = df.groupby('AdCopyID').agg(
        TotalImpressions=('Impressions', 'sum'),
        TotalClicks=('Clicks', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        AvgCTR=('CTR', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean')
    ).reset_index()

    print("\nAd Copy Performance Summary:")
    print(ad_copy_summary.round(2))

    fig1 = px.bar(ad_copy_summary.sort_values('AvgCTR', ascending=False).head(10),
                  x='AdCopyID', y='AvgCTR', title='Top 10 Ad Copies by Average CTR')
    fig1.show()

    fig2 = px.bar(ad_copy_summary.sort_values('AvgConversionRate', ascending=False).head(10),
                  x='AdCopyID', y='AvgConversionRate', title='Top 10 Ad Copies by Average Conversion Rate')
    fig2.show()

    if 'AdGroup' in df.columns:
        ad_group_performance = df.groupby('AdGroup').agg(
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean')
        ).reset_index()
        fig3 = px.bar(ad_group_performance, x='AdGroup', y='AvgCTR', title='Average CTR by Ad Group')
        fig3.show()

    return {
        "metrics": ad_copy_summary.to_dict(orient='records'),
        "figures": {
            "Top_10_Ad_Copies_by_CTR_Bar": fig1,
            "Top_10_Ad_Copies_by_Conversion_Rate_Bar": fig2,
            "Avg_CTR_by_Ad_Group_Bar": fig3 if 'AdGroup' in df.columns else None
        }
    }


def customer_referral_program_analysis(df):
    print("\n--- Customer Referral Program Analysis ---")
    expected = {
        'ReferrerID': ['ReferrerID', 'ReferralSourceID', 'UserID'],
        'ReferredCustomerID': ['ReferredCustomerID', 'NewCustomerID'],
        'ReferralDate': ['ReferralDate', 'Date'],
        'ReferralConversionStatus': ['ReferralConversionStatus', 'Status', 'Converted'], # Binary (True/False)
        'ReferralRevenue': ['ReferralRevenue', 'Revenue', 'SalesAmount'],
        'RewardType': ['RewardType', 'IncentiveType']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['ReferralDate'] = pd.to_datetime(df['ReferralDate'], errors='coerce')
    df['ReferralConversionStatus'] = df['ReferralConversionStatus'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['ReferralRevenue'] = pd.to_numeric(df['ReferralRevenue'], errors='coerce')
    df = df.dropna(subset=['ReferrerID', 'ReferralDate', 'ReferralConversionStatus'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_referrals = len(df)
    successful_referrals = df['ReferralConversionStatus'].sum()
    referral_conversion_rate = (successful_referrals / total_referrals) * 100 if total_referrals > 0 else 0
    total_referral_revenue = df['ReferralRevenue'].sum() if 'ReferralRevenue' in df.columns else 0

    print(f"Total Referrals Initiated: {total_referrals}")
    print(f"Successful Referrals: {successful_referrals}")
    print(f"Referral Conversion Rate: {referral_conversion_rate:.2f}%")
    print(f"Total Revenue from Referrals: ${total_referral_revenue:,.2f}")

    referrer_performance = df.groupby('ReferrerID').agg(
        NumReferrals=('ReferredCustomerID', 'count'),
        SuccessfulReferrals=('ReferralConversionStatus', 'sum'),
        TotalRevenue=('ReferralRevenue', 'sum') if 'ReferralRevenue' in df.columns else ('ReferrerID', 'size') # Placeholder
    ).reset_index()
    referrer_performance['ConversionRate'] = (referrer_performance['SuccessfulReferrals'] / referrer_performance['NumReferrals']) * 100

    print("\nTop 10 Referrer Performance:")
    print(referrer_performance.sort_values('SuccessfulReferrals', ascending=False).head(10).round(2))

    fig1 = px.bar(referrer_performance.sort_values('SuccessfulReferrals', ascending=False).head(10),
                  x='ReferrerID', y='SuccessfulReferrals', title='Top 10 Referrers by Successful Referrals')
    fig1.show()

    fig2 = px.histogram(df, x='ReferralConversionStatus', title='Overall Referral Conversion Status')
    fig2.show()

    if 'RewardType' in df.columns:
        reward_type_performance = df.groupby('RewardType')['ReferralConversionStatus'].mean().reset_index()
        reward_type_performance['ConversionRate'] *= 100
        fig3 = px.bar(reward_type_performance, x='RewardType', y='ConversionRate', title='Referral Conversion Rate by Reward Type')
        fig3.show()

    return {
        "metrics": {
            "Total Referrals": total_referrals,
            "Successful Referrals": successful_referrals,
            "Referral Conversion Rate": referral_conversion_rate,
            "Total Revenue from Referrals": total_referral_revenue
        },
        "figures": {
            "Top_10_Referrers_Bar": fig1,
            "Referral_Conversion_Status_Histogram": fig2,
            "Referral_Conversion_Rate_by_Reward_Type_Bar": fig3 if 'RewardType' in df.columns else None
        }
    }


def website_platform_engagement_metrics_analysis(df):
    print("\n--- Website/Platform Engagement Metrics Analysis ---")
    expected = {
        'UserID': ['UserID', 'VisitorID', 'CustomerID'],
        'Date': ['Date', 'VisitDate', 'ActivityDate'],
        'PageViews': ['PageViews', 'PagesVisited', 'Views'],
        'TimeOnSiteSeconds': ['TimeOnSiteSeconds', 'SessionDuration', 'Duration'],
        'BounceRate': ['BounceRate', 'Bounced'], # Assumed as a percentage or 1/0 for bounced
        'Conversions': ['Conversions', 'GoalsCompleted', 'Purchases']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['PageViews'] = pd.to_numeric(df['PageViews'], errors='coerce')
    df['TimeOnSiteSeconds'] = pd.to_numeric(df['TimeOnSiteSeconds'], errors='coerce')
    df['BounceRate'] = pd.to_numeric(df['BounceRate'], errors='coerce') # If 0-100%, otherwise calculate if 1/0
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df = df.dropna(subset=['Date', 'PageViews', 'TimeOnSiteSeconds', 'BounceRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_page_views = df['PageViews'].mean()
    avg_time_on_site_minutes = df['TimeOnSiteSeconds'].mean() / 60
    avg_bounce_rate = df['BounceRate'].mean() # Assumed to be a percentage already or 0-1 value
    total_conversions = df['Conversions'].sum() if 'Conversions' in df.columns else 0

    print(f"Average Page Views per Session: {avg_page_views:.1f}")
    print(f"Average Time on Site per Session: {avg_time_on_site_minutes:.1f} minutes")
    print(f"Average Bounce Rate: {avg_bounce_rate:.2f}%")
    print(f"Total Conversions: {total_conversions}")

    daily_engagement = df.groupby('Date').agg(
        AvgPageViews=('PageViews', 'mean'),
        AvgTimeOnSiteMinutes=('TimeOnSiteSeconds', lambda x: x.mean() / 60),
        AvgBounceRate=('BounceRate', 'mean')
    ).reset_index()

    fig1 = px.line(daily_engagement, x='Date', y='AvgPageViews', title='Daily Average Page Views Trend')
    fig1.show()

    fig2 = px.line(daily_engagement, x='Date', y='AvgTimeOnSiteMinutes', title='Daily Average Time on Site Trend')
    fig2.show()
    
    fig3 = px.histogram(df, x='BounceRate', title='Distribution of Bounce Rates')
    fig3.show()

    return {
        "metrics": {
            "Average Page Views": avg_page_views,
            "Average Time on Site (minutes)": avg_time_on_site_minutes,
            "Average Bounce Rate": avg_bounce_rate,
            "Total Conversions": total_conversions
        },
        "figures": {
            "Daily_Avg_Page_Views_Trend_Line": fig1,
            "Daily_Avg_Time_on_Site_Trend_Line": fig2,
            "Bounce_Rates_Distribution_Histogram": fig3
        }
    }


def customer_loyalty_program_engagement_analysis(df):
    print("\n--- Customer Loyalty Program Engagement Analysis ---")
    expected = {
        'CustomerID': ['CustomerID', 'UserID', 'ID'],
        'EnrollmentDate': ['EnrollmentDate', 'DateEnrolled'],
        'PointsEarned': ['PointsEarned', 'LoyaltyPoints', 'Points'],
        'PointsRedeemed': ['PointsRedeemed', 'RedeemedPoints'],
        'PurchasesCount': ['PurchasesCount', 'NumPurchases'],
        'TotalSpend': ['TotalSpend', 'Spend', 'Revenue'],
        'Tier': ['Tier', 'LoyaltyTier', 'ProgramLevel']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['EnrollmentDate'] = pd.to_datetime(df['EnrollmentDate'], errors='coerce')
    df['PointsEarned'] = pd.to_numeric(df['PointsEarned'], errors='coerce')
    df['PointsRedeemed'] = pd.to_numeric(df['PointsRedeemed'], errors='coerce')
    df['PurchasesCount'] = pd.to_numeric(df['PurchasesCount'], errors='coerce')
    df['TotalSpend'] = pd.to_numeric(df['TotalSpend'], errors='coerce')
    df = df.dropna(subset=['CustomerID', 'EnrollmentDate', 'PointsEarned', 'PointsRedeemed', 'PurchasesCount', 'TotalSpend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_customers_in_program = len(df)
    avg_points_earned = df['PointsEarned'].mean()
    avg_points_redeemed = df['PointsRedeemed'].mean()
    avg_purchases_per_customer = df['PurchasesCount'].mean()
    
    print(f"Total Customers in Program: {total_customers_in_program}")
    print(f"Average Points Earned: {avg_points_earned:.0f}")
    print(f"Average Points Redeemed: {avg_points_redeemed:.0f}")
    print(f"Average Purchases per Customer: {avg_purchases_per_customer:.1f}")

    program_summary = df.groupby('Tier').agg(
        NumCustomers=('CustomerID', 'count'),
        AvgPointsEarned=('PointsEarned', 'mean'),
        AvgPointsRedeemed=('PointsRedeemed', 'mean'),
        AvgTotalSpend=('TotalSpend', 'mean')
    ).reset_index()

    print("\nLoyalty Program Performance by Tier:")
    print(program_summary.round(2))

    fig1 = px.bar(program_summary, x='Tier', y='NumCustomers', title='Number of Customers by Loyalty Tier')
    fig1.show()

    fig2 = px.scatter(df, x='PointsEarned', y='TotalSpend', color='Tier' if 'Tier' in df.columns else None,
                     hover_name='CustomerID', title='Total Spend vs. Points Earned')
    fig2.show()
    
    fig3 = px.histogram(df, x='PointsRedeemed', title='Distribution of Points Redeemed')
    fig3.show()

    return {
        "metrics": {
            "Total Customers in Program": total_customers_in_program,
            "Average Points Earned": avg_points_earned,
            "Average Points Redeemed": avg_points_redeemed,
            "Average Purchases per Customer": avg_purchases_per_customer,
            "Loyalty Program Performance by Tier": program_summary.to_dict(orient='records')
        },
        "figures": {
            "Customers_by_Loyalty_Tier_Bar": fig1,
            "Total_Spend_vs_Points_Earned_Scatter": fig2,
            "Points_Redeemed_Distribution_Histogram": fig3
        }
    }


def discount_code_redemption_and_visit_analysis(df):
    print("\n--- Discount Code Redemption and Visit Analysis ---")
    expected = {
        'CodeID': ['CodeID', 'DiscountCode', 'ID'],
        'CustomerID': ['CustomerID', 'UserID'],
        'Redeemed': ['Redeemed', 'IsRedeemed', 'RedemptionStatus'], # Binary (True/False)
        'VisitsBeforeRedemption': ['VisitsBeforeRedemption', 'PreRedemptionVisits'],
        'VisitsAfterRedemption': ['VisitsAfterRedemption', 'PostRedemptionVisits'],
        'PurchaseValue': ['PurchaseValue', 'Revenue', 'Sales'],
        'DiscountPercentage': ['DiscountPercentage', 'DiscountRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Redeemed'] = df['Redeemed'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['VisitsBeforeRedemption'] = pd.to_numeric(df['VisitsBeforeRedemption'], errors='coerce')
    df['VisitsAfterRedemption'] = pd.to_numeric(df['VisitsAfterRedemption'], errors='coerce')
    df['PurchaseValue'] = pd.to_numeric(df['PurchaseValue'], errors='coerce')
    df['DiscountPercentage'] = pd.to_numeric(df['DiscountPercentage'], errors='coerce')

    df = df.dropna(subset=['CodeID', 'Redeemed', 'VisitsBeforeRedemption'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_codes_issued = len(df)
    total_redemptions = df['Redeemed'].sum()
    redemption_rate = (total_redemptions / total_codes_issued) * 100 if total_codes_issued > 0 else 0
    avg_visits_before_redemption = df['VisitsBeforeRedemption'].mean()
    avg_purchase_value_redeemed = df[df['Redeemed']]['PurchaseValue'].mean() if 'PurchaseValue' in df.columns else np.nan

    print(f"Total Discount Codes Issued: {total_codes_issued}")
    print(f"Total Redemptions: {total_redemptions}")
    print(f"Redemption Rate: {redemption_rate:.2f}%")
    print(f"Average Visits Before Redemption: {avg_visits_before_redemption:.1f}")
    if not np.isnan(avg_purchase_value_redeemed):
        print(f"Average Purchase Value for Redeemed Codes: ${avg_purchase_value_redeemed:,.2f}")

    redemption_summary = df.groupby('Redeemed').agg(
        AvgVisitsBefore=('VisitsBeforeRedemption', 'mean'),
        AvgVisitsAfter=('VisitsAfterRedemption', 'mean') if 'VisitsAfterRedemption' in df.columns else ('VisitsBeforeRedemption', 'size'), # Placeholder
        AvgPurchaseValue=('PurchaseValue', 'mean') if 'PurchaseValue' in df.columns else ('VisitsBeforeRedemption', 'size') # Placeholder
    ).reset_index()

    print("\nVisits and Value by Redemption Status:")
    print(redemption_summary.round(2))

    fig1 = px.pie(df, names='Redeemed', title='Discount Code Redemption Status')
    fig1.show()

    fig2 = px.box(df, x='Redeemed', y='VisitsBeforeRedemption', title='Visits Before Redemption by Status')
    fig2.show()

    if 'VisitsAfterRedemption' in df.columns:
        fig3 = px.box(df, x='Redeemed', y='VisitsAfterRedemption', title='Visits After Redemption by Status')
        fig3.show()
    
    if 'DiscountPercentage' in df.columns:
        discount_impact = df.groupby('DiscountPercentage')['Redeemed'].mean().reset_index()
        discount_impact['RedemptionRate'] = discount_impact['Redeemed'] * 100
        fig4 = px.bar(discount_impact, x='DiscountPercentage', y='RedemptionRate', title='Redemption Rate by Discount Percentage')
        fig4.show()

    return {
        "metrics": {
            "Total Codes Issued": total_codes_issued,
            "Total Redemptions": total_redemptions,
            "Redemption Rate": redemption_rate,
            "Average Visits Before Redemption": avg_visits_before_redemption,
            "Average Purchase Value (Redeemed)": avg_purchase_value_redeemed,
            "Redemption Summary": redemption_summary.to_dict(orient='records')
        },
        "figures": {
            "Redemption_Status_Pie": fig1,
            "Visits_Before_Redemption_Box": fig2,
            "Visits_After_Redemption_Box": fig3 if 'VisitsAfterRedemption' in df.columns else None,
            "Redemption_Rate_by_Discount_Percentage_Bar": fig4 if 'DiscountPercentage' in df.columns else None
        }
    }


def seasonal_and_holiday_campaign_impact_analysis(df):
    print("\n--- Seasonal and Holiday Campaign Impact Analysis ---")
    expected = {
        'Date': ['Date', 'EventDate', 'Day'],
        'CampaignName': ['CampaignName', 'HolidayCampaign', 'SeasonalCampaign'],
        'Revenue': ['Revenue', 'Sales', 'TotalRevenue'],
        'Transactions': ['Transactions', 'Orders', 'NumTransactions'],
        'WebsiteTraffic': ['WebsiteTraffic', 'Visits', 'PageViews']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df['Transactions'] = pd.to_numeric(df['Transactions'], errors='coerce')
    df['WebsiteTraffic'] = pd.to_numeric(df['WebsiteTraffic'], errors='coerce')
    df = df.dropna(subset=['Date', 'Revenue', 'Transactions', 'WebsiteTraffic'])
    df = df.sort_values('Date')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    # Identify campaign periods (simple approach: days where CampaignName is present)
    df['IsCampaignDay'] = df['CampaignName'].notna()
    
    avg_daily_revenue_campaign = df[df['IsCampaignDay']]['Revenue'].mean()
    avg_daily_revenue_non_campaign = df[~df['IsCampaignDay']]['Revenue'].mean()
    
    print(f"Average Daily Revenue (Campaign Days): ${avg_daily_revenue_campaign:,.2f}")
    print(f"Average Daily Revenue (Non-Campaign Days): ${avg_daily_revenue_non_campaign:,.2f}")
    print(f"Revenue Uplift during Campaigns: ${(avg_daily_revenue_campaign - avg_daily_revenue_non_campaign):,.2f}")

    campaign_impact_summary = df.groupby('IsCampaignDay').agg(
        AvgRevenue=('Revenue', 'mean'),
        AvgTransactions=('Transactions', 'mean'),
        AvgWebsiteTraffic=('WebsiteTraffic', 'mean')
    ).reset_index()
    campaign_impact_summary['IsCampaignDay'] = campaign_impact_summary['IsCampaignDay'].map({True: 'Campaign Day', False: 'Non-Campaign Day'})

    print("\nAverage Daily Metrics by Campaign Status:")
    print(campaign_impact_summary.round(2))

    fig1 = px.line(df, x='Date', y='Revenue', color='IsCampaignDay', title='Daily Revenue Trend by Campaign Status')
    fig1.show()

    fig2 = px.bar(campaign_impact_summary, x='IsCampaignDay', y='AvgTransactions', title='Average Daily Transactions by Campaign Status')
    fig2.show()
    
    if 'CampaignName' in df.columns:
        campaign_performance = df.groupby('CampaignName').agg(
            TotalRevenue=('Revenue', 'sum'),
            TotalTransactions=('Transactions', 'sum')
        ).reset_index()
        fig3 = px.bar(campaign_performance.sort_values('TotalRevenue', ascending=False).head(10),
                      x='CampaignName', y='TotalRevenue', title='Top 10 Campaigns by Total Revenue')
        fig3.show()

    return {
        "metrics": {
            "Average Daily Revenue (Campaign)": avg_daily_revenue_campaign,
            "Average Daily Revenue (Non-Campaign)": avg_daily_revenue_non_campaign,
            "Average Daily Metrics by Campaign Status": campaign_impact_summary.to_dict(orient='records')
        },
        "figures": {
            "Daily_Revenue_Trend_by_Campaign_Status_Line": fig1,
            "Avg_Daily_Transactions_by_Campaign_Status_Bar": fig2,
            "Top_10_Campaigns_by_Total_Revenue_Bar": fig3 if 'CampaignName' in df.columns else None
        }
    }


def video_marketing_engagement_analysis(df):
    print("\n--- Video Marketing Engagement Analysis ---")
    expected = {
        'VideoID': ['VideoID', 'ID'],
        'VideoTitle': ['VideoTitle', 'Title'],
        'Views': ['Views', 'TotalViews', 'Playbacks'],
        'WatchTimeSeconds': ['WatchTimeSeconds', 'DurationWatched', 'AvgWatchTime'],
        'Likes': ['Likes', 'NumLikes'],
        'Shares': ['Shares', 'NumShares'],
        'Comments': ['Comments', 'NumComments'],
        'ConversionEvent': ['ConversionEvent', 'Converted', 'LeadEvent']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Views'] = pd.to_numeric(df['Views'], errors='coerce')
    df['WatchTimeSeconds'] = pd.to_numeric(df['WatchTimeSeconds'], errors='coerce')
    df['Likes'] = pd.to_numeric(df['Likes'], errors='coerce')
    df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce')
    df['Comments'] = pd.to_numeric(df['Comments'], errors='coerce')
    df['ConversionEvent'] = df['ConversionEvent'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})

    df = df.dropna(subset=['VideoID', 'Views', 'WatchTimeSeconds', 'Likes', 'Shares', 'Comments'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_views = df['Views'].mean()
    avg_watch_time_minutes = df['WatchTimeSeconds'].mean() / 60
    avg_likes_per_view = (df['Likes'] / df['Views']).mean() * 100 if df['Views'].sum() > 0 else 0
    avg_conversion_rate = df['ConversionEvent'].mean() * 100 if 'ConversionEvent' in df.columns else np.nan

    print(f"Average Video Views: {avg_views:,.0f}")
    print(f"Average Watch Time: {avg_watch_time_minutes:.1f} minutes")
    print(f"Average Likes per View: {avg_likes_per_view:.2f}%")
    if not np.isnan(avg_conversion_rate):
        print(f"Average Conversion Rate: {avg_conversion_rate:.2f}%")

    top_videos_by_views = df.sort_values('Views', ascending=False).head(10)
    print("\nTop 10 Videos by Views:")
    print(top_videos_by_views[['VideoTitle', 'Views', 'WatchTimeSeconds']].round(0))

    fig1 = px.bar(top_videos_by_views, x='VideoTitle', y='Views', title='Top 10 Videos by Views')
    fig1.show()

    fig2 = px.scatter(df, x='WatchTimeSeconds', y='Likes', size='Views', hover_name='VideoTitle',
                     title='Likes vs. Watch Time (Sized by Views)')
    fig2.show()
    
    if 'ConversionEvent' in df.columns:
        video_conversion = df.groupby('VideoTitle')['ConversionEvent'].mean().reset_index()
        video_conversion['ConversionRate'] *= 100
        fig3 = px.bar(video_conversion.sort_values('ConversionRate', ascending=False).head(10),
                      x='VideoTitle', y='ConversionRate', title='Top 10 Videos by Conversion Rate')
        fig3.show()

    return {
        "metrics": {
            "Average Video Views": avg_views,
            "Average Watch Time (minutes)": avg_watch_time_minutes,
            "Average Likes per View": avg_likes_per_view,
            "Average Conversion Rate": avg_conversion_rate,
            "Top 10 Videos by Views": top_videos_by_views[['VideoTitle', 'Views', 'WatchTimeSeconds']].to_dict(orient='records')
        },
        "figures": {
            "Top_10_Videos_by_Views_Bar": fig1,
            "Likes_vs_Watch_Time_Scatter": fig2,
            "Top_10_Videos_by_Conversion_Rate_Bar": fig3 if 'ConversionEvent' in df.columns else None
        }
    }


def search_engine_marketing_sem_keyword_performance(df):
    print("\n--- Search Engine Marketing (SEM) Keyword Performance Analysis ---")
    expected = {
        'Keyword': ['Keyword', 'SearchTerm'],
        'AdGroup': ['AdGroup', 'CampaignAdGroup'],
        'Impressions': ['Impressions', 'KeywordImpressions'],
        'Clicks': ['Clicks', 'KeywordClicks'],
        'Cost': ['Cost', 'Spend', 'KeywordCost'],
        'Conversions': ['Conversions', 'KeywordConversions', 'Sales']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Cost'] = pd.to_numeric(df['Cost'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df = df.dropna(subset=['Keyword', 'Impressions', 'Clicks', 'Cost', 'Conversions'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['CPC'] = df['Cost'] / df['Clicks'] if df['Clicks'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['CPA'] = df['Cost'] / df['Conversions'] if df['Conversions'].sum() > 0 else 0

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

    print("\nTop 10 Keywords by Conversions:")
    print(keyword_summary.sort_values('TotalConversions', ascending=False).head(10).round(2))

    fig1 = px.bar(keyword_summary.sort_values('TotalConversions', ascending=False).head(10),
                  x='Keyword', y='TotalConversions', title='Top 10 Keywords by Total Conversions')
    fig1.show()

    fig2 = px.scatter(keyword_summary, x='AvgCPC', y='AvgConversionRate', size='TotalImpressions',
                     hover_name='Keyword', title='Conversion Rate vs. CPC (Sized by Impressions)')
    fig2.show()
    
    if 'AdGroup' in df.columns:
        ad_group_performance = df.groupby('AdGroup').agg(
            TotalCost=('Cost', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()
        fig3 = px.bar(ad_group_performance.sort_values('AvgCPA', ascending=True).head(10),
                      x='AdGroup', y='AvgCPA', title='Top 10 Ad Groups by Lowest Average CPA')
        fig3.show()

    return {
        "metrics": keyword_summary.to_dict(orient='records'),
        "figures": {
            "Top_10_Keywords_by_Conversions_Bar": fig1,
            "Conversion_Rate_vs_CPC_Scatter": fig2,
            "Top_10_Ad_Groups_by_Lowest_Avg_CPA_Bar": fig3 if 'AdGroup' in df.columns else None
        }
    }


def churn_prediction_and_targeted_campaign_analysis(df):
    print("\n--- Churn Prediction and Targeted Campaign Analysis ---")
    expected = {
        'CustomerID': ['CustomerID', 'ID', 'UserID'],
        'IsChurned': ['IsChurned', 'Churned', 'ChurnStatus'], # Binary (True/False or 1/0)
        'LastActivityDate': ['LastActivityDate', 'LastLogin', 'LastPurchaseDate'],
        'CampaignSegment': ['CampaignSegment', 'Segment', 'TargetGroup'],
        'CampaignResponse': ['CampaignResponse', 'Response', 'ConvertedToRetain'], # Binary (True/False or 1/0)
        'CustomerLifetimeValue': ['CustomerLifetimeValue', 'CLTV', 'LTV']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['IsChurned'] = df['IsChurned'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['CampaignResponse'] = df['CampaignResponse'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['CustomerLifetimeValue'] = pd.to_numeric(df['CustomerLifetimeValue'], errors='coerce')
    df['LastActivityDate'] = pd.to_datetime(df['LastActivityDate'], errors='coerce')

    df = df.dropna(subset=['CustomerID', 'IsChurned', 'CampaignSegment', 'CampaignResponse', 'CustomerLifetimeValue'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    overall_churn_rate = df['IsChurned'].mean() * 100
    campaign_response_rate = df['CampaignResponse'].mean() * 100
    
    print(f"Overall Churn Rate: {overall_churn_rate:.2f}%")
    print(f"Overall Campaign Response Rate: {campaign_response_rate:.2f}%")

    churn_by_segment = df.groupby('CampaignSegment').agg(
        NumCustomers=('CustomerID', 'count'),
        ChurnRate=('IsChurned', 'mean'),
        ResponseRate=('CampaignResponse', 'mean'),
        AvgCLTV=('CustomerLifetimeValue', 'mean')
    ).reset_index()
    churn_by_segment['ChurnRate'] *= 100
    churn_by_segment['ResponseRate'] *= 100

    print("\nChurn and Response Rates by Campaign Segment:")
    print(churn_by_segment.round(2))

    fig1 = px.bar(churn_by_segment, x='CampaignSegment', y='ChurnRate', title='Churn Rate by Campaign Segment')
    fig1.show()

    fig2 = px.bar(churn_by_segment, x='CampaignSegment', y='ResponseRate', title='Campaign Response Rate by Segment')
    fig2.show()
    
    fig3 = px.box(df, x='IsChurned', y='CustomerLifetimeValue', title='Customer Lifetime Value by Churn Status')
    fig3.show()

    return {
        "metrics": {
            "Overall Churn Rate": overall_churn_rate,
            "Overall Campaign Response Rate": campaign_response_rate,
            "Churn and Response Rates by Campaign Segment": churn_by_segment.to_dict(orient='records')
        },
        "figures": {
            "Churn_Rate_by_Segment_Bar": fig1,
            "Campaign_Response_Rate_by_Segment_Bar": fig2,
            "CLTV_by_Churn_Status_Box": fig3
        }
    }


def newsletter_signup_attribution_analysis(df):
    print("\n--- Newsletter Signup Attribution Analysis ---")
    expected = {
        'SignupID': ['SignupID', 'ID'],
        'SignupDate': ['SignupDate', 'Date'],
        'Channel': ['Channel', 'Source', 'MarketingChannel'],
        'Device': ['Device', 'SignupDevice'],
        'ConvertedToCustomer': ['ConvertedToCustomer', 'IsCustomer', 'Converted'], # Binary (True/False)
        'CustomerLifetimeValue': ['CustomerLifetimeValue', 'CLTV']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['SignupDate'] = pd.to_datetime(df['SignupDate'], errors='coerce')
    df['ConvertedToCustomer'] = df['ConvertedToCustomer'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    df['CustomerLifetimeValue'] = pd.to_numeric(df['CustomerLifetimeValue'], errors='coerce')
    df = df.dropna(subset=['SignupID', 'SignupDate', 'Channel', 'ConvertedToCustomer'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_signups = len(df)
    customer_conversions = df['ConvertedToCustomer'].sum()
    conversion_rate = (customer_conversions / total_signups) * 100

    print(f"Total Newsletter Signups: {total_signups}")
    print(f"Signups Converted to Customer: {customer_conversions}")
    print(f"Signup-to-Customer Conversion Rate: {conversion_rate:.2f}%")

    channel_performance = df.groupby('Channel').agg(
        NumSignups=('SignupID', 'count'),
        ConversionRate=('ConvertedToCustomer', 'mean'),
        AvgCLTV=('CustomerLifetimeValue', 'mean') if 'CustomerLifetimeValue' in df.columns else ('SignupID', 'size') # Placeholder
    ).reset_index()
    channel_performance['ConversionRate'] *= 100

    print("\nConversion Performance by Signup Channel:")
    print(channel_performance.round(2))

    fig1 = px.bar(channel_performance, x='Channel', y='NumSignups', title='Number of Signups by Channel')
    fig1.show()

    fig2 = px.bar(channel_performance, x='Channel', y='ConversionRate', title='Signup-to-Customer Conversion Rate by Channel')
    fig2.show()
    
    if 'CustomerLifetimeValue' in df.columns:
        fig3 = px.box(df, x='Channel', y='CustomerLifetimeValue', title='Customer Lifetime Value by Signup Channel')
        fig3.show()

    return {
        "metrics": {
            "Total Signups": total_signups,
            "Signups Converted to Customer": customer_conversions,
            "Signup-to-Customer Conversion Rate": conversion_rate,
            "Conversion Performance by Signup Channel": channel_performance.to_dict(orient='records')
        },
        "figures": {
            "Num_Signups_by_Channel_Bar": fig1,
            "Conversion_Rate_by_Channel_Bar": fig2,
            "CLTV_by_Signup_Channel_Box": fig3 if 'CustomerLifetimeValue' in df.columns else None
        }
    }


def marketing_budget_allocation_and_spend_analysis(df):
    print("\n--- Marketing Budget Allocation and Spend Analysis ---")
    expected = {
        'Date': ['Date', 'Month', 'ReportingPeriod'],
        'Channel': ['Channel', 'MarketingChannel', 'Platform'],
        'BudgetAllocated': ['BudgetAllocated', 'Budget', 'AllocatedSpend'],
        'ActualSpend': ['ActualSpend', 'Spend', 'Cost'],
        'RevenueAttributed': ['RevenueAttributed', 'Revenue', 'Sales']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['BudgetAllocated'] = pd.to_numeric(df['BudgetAllocated'], errors='coerce')
    df['ActualSpend'] = pd.to_numeric(df['ActualSpend'], errors='coerce')
    df['RevenueAttributed'] = pd.to_numeric(df['RevenueAttributed'], errors='coerce')
    df = df.dropna(subset=['Date', 'Channel', 'BudgetAllocated', 'ActualSpend'])
    df = df.sort_values('Date')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_allocated_budget = df['BudgetAllocated'].sum()
    total_actual_spend = df['ActualSpend'].sum()
    total_revenue_attributed = df['RevenueAttributed'].sum() if 'RevenueAttributed' in df.columns else 0
    budget_utilization_rate = (total_actual_spend / total_allocated_budget) * 100 if total_allocated_budget > 0 else 0

    print(f"Total Allocated Budget: ${total_allocated_budget:,.2f}")
    print(f"Total Actual Spend: ${total_actual_spend:,.2f}")
    print(f"Budget Utilization Rate: {budget_utilization_rate:.2f}%")
    print(f"Total Revenue Attributed: ${total_revenue_attributed:,.2f}")

    channel_spend_summary = df.groupby('Channel').agg(
        TotalAllocated=('BudgetAllocated', 'sum'),
        TotalActualSpend=('ActualSpend', 'sum'),
        TotalRevenue=('RevenueAttributed', 'sum') if 'RevenueAttributed' in df.columns else ('BudgetAllocated', 'size') # Placeholder
    ).reset_index()
    
    channel_spend_summary['UtilizationRate'] = (channel_spend_summary['TotalActualSpend'] / channel_spend_summary['TotalAllocated']) * 100
    if 'RevenueAttributed' in df.columns:
        channel_spend_summary['ROI'] = ((channel_spend_summary['TotalRevenue'] - channel_spend_summary['TotalActualSpend']) / channel_spend_summary['TotalActualSpend']) * 100
    else:
        channel_spend_summary['ROI'] = np.nan

    print("\nBudget and Performance by Channel:")
    print(channel_spend_summary.round(2))

    fig1 = px.bar(channel_spend_summary, x='Channel', y=['TotalAllocated', 'TotalActualSpend'],
                  barmode='group', title='Allocated Budget vs. Actual Spend by Channel')
    fig1.show()

    fig2 = px.pie(channel_spend_summary, names='Channel', values='TotalActualSpend', title='Distribution of Actual Spend by Channel')
    fig2.show()
    
    if 'RevenueAttributed' in df.columns:
        fig3 = px.bar(channel_spend_summary, x='Channel', y='ROI', title='ROI by Channel')
        fig3.show()

    return {
        "metrics": {
            "Total Allocated Budget": total_allocated_budget,
            "Total Actual Spend": total_actual_spend,
            "Budget Utilization Rate": budget_utilization_rate,
            "Total Revenue Attributed": total_revenue_attributed,
            "Budget and Performance by Channel": channel_spend_summary.to_dict(orient='records')
        },
        "figures": {
            "Budget_vs_Spend_by_Channel_Bar": fig1,
            "Actual_Spend_Distribution_Pie": fig2,
            "ROI_by_Channel_Bar": fig3 if 'RevenueAttributed' in df.columns else None
        }
    }


def social_media_competitive_and_sentiment_analysis(df):
    print("\n--- Social Media Competitive and Sentiment Analysis ---")
    expected = {
        'Brand': ['Brand', 'Competitor', 'CompanyName'],
        'Date': ['Date', 'PostDate', 'AnalysisDate'],
        'Mentions': ['Mentions', 'TotalMentions', 'CountMentions'],
        'SentimentScore': ['SentimentScore', 'AvgSentiment', 'SentimentPolarity'],
        'EngagementRate': ['EngagementRate', 'AvgEngagement'],
        'FollowerGrowth': ['FollowerGrowth', 'NewFollowers']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Mentions'] = pd.to_numeric(df['Mentions'], errors='coerce')
    df['SentimentScore'] = pd.to_numeric(df['SentimentScore'], errors='coerce')
    df['EngagementRate'] = pd.to_numeric(df['EngagementRate'], errors='coerce')
    df['FollowerGrowth'] = pd.to_numeric(df['FollowerGrowth'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Brand', 'Date', 'Mentions', 'SentimentScore', 'EngagementRate'])
    df = df.sort_values(['Brand', 'Date'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    overall_avg_sentiment = df['SentimentScore'].mean()
    overall_avg_engagement = df['EngagementRate'].mean()
    
    print(f"Overall Average Sentiment Score: {overall_avg_sentiment:.2f}")
    print(f"Overall Average Engagement Rate: {overall_avg_engagement:.2f}%")

    brand_summary = df.groupby('Brand').agg(
        TotalMentions=('Mentions', 'sum'),
        AvgSentiment=('SentimentScore', 'mean'),
        AvgEngagementRate=('EngagementRate', 'mean'),
        TotalFollowerGrowth=('FollowerGrowth', 'sum') if 'FollowerGrowth' in df.columns else ('Mentions', 'size') # Placeholder
    ).reset_index()
    
    print("\nCompetitive Brand Performance Summary:")
    print(brand_summary.round(2))

    fig1 = px.line(df, x='Date', y='SentimentScore', color='Brand', title='Sentiment Trend Over Time by Brand')
    fig1.show()

    fig2 = px.bar(brand_summary.sort_values('AvgEngagementRate', ascending=False),
                  x='Brand', y='AvgEngagementRate', title='Average Engagement Rate by Brand')
    fig2.show()
    
    fig3 = px.scatter(df, x='SentimentScore', y='EngagementRate', color='Brand', size='Mentions',
                     hover_name='Date', title='Engagement Rate vs. Sentiment Score by Brand (Sized by Mentions)')
    fig3.show()

    return {
        "metrics": {
            "Overall Average Sentiment Score": overall_avg_sentiment,
            "Overall Average Engagement Rate": overall_avg_engagement,
            "Competitive Brand Performance Summary": brand_summary.to_dict(orient='records')
        },
        "figures": {
            "Sentiment_Trend_by_Brand_Line": fig1,
            "Avg_Engagement_Rate_by_Brand_Bar": fig2,
            "Engagement_vs_Sentiment_Scatter": fig3
        }
    }


def customer_service_sentiment_and_feedback_analysis(df):
    print("\n--- Customer Service Sentiment and Feedback Analysis ---")
    expected = {
        'TicketID': ['TicketID', 'CaseID', 'ID'],
        'Date': ['Date', 'TicketDate', 'FeedbackDate'],
        'Channel': ['Channel', 'ServiceChannel', 'ContactChannel'],
        'SentimentScore': ['SentimentScore', 'FeedbackSentiment', 'SentimentPolarity'],
        'ResolutionTimeHours': ['ResolutionTimeHours', 'TimeResolution', 'ResolveHours'],
        'CustomerSatisfactionRating': ['CustomerSatisfactionRating', 'CSAT', 'Rating'],
        'FeedbackCategory': ['FeedbackCategory', 'Category', 'IssueType']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['SentimentScore'] = pd.to_numeric(df['SentimentScore'], errors='coerce')
    df['ResolutionTimeHours'] = pd.to_numeric(df['ResolutionTimeHours'], errors='coerce')
    df['CustomerSatisfactionRating'] = pd.to_numeric(df['CustomerSatisfactionRating'], errors='coerce')
    df = df.dropna(subset=['Date', 'Channel', 'SentimentScore', 'CustomerSatisfactionRating'])
    df = df.sort_values('Date')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    overall_avg_sentiment = df['SentimentScore'].mean()
    overall_avg_csat = df['CustomerSatisfactionRating'].mean()
    avg_resolution_time_hours = df['ResolutionTimeHours'].mean() if 'ResolutionTimeHours' in df.columns else np.nan

    print(f"Overall Average Sentiment Score: {overall_avg_sentiment:.2f}")
    print(f"Overall Average Customer Satisfaction Rating: {overall_avg_csat:.2f}")
    if not np.isnan(avg_resolution_time_hours):
        print(f"Average Resolution Time: {avg_resolution_time_hours:.1f} hours")

    channel_performance = df.groupby('Channel').agg(
        NumTickets=('TicketID', 'count'),
        AvgSentiment=('SentimentScore', 'mean'),
        AvgCSAT=('CustomerSatisfactionRating', 'mean'),
        AvgResolutionTime=('ResolutionTimeHours', 'mean') if 'ResolutionTimeHours' in df.columns else ('TicketID', 'size')
    ).reset_index()

    print("\nCustomer Service Performance by Channel:")
    print(channel_performance.round(2))

    fig1 = px.line(df, x='Date', y='SentimentScore', color='Channel', title='Sentiment Trend Over Time by Channel')
    fig1.show()

    fig2 = px.bar(channel_performance, x='Channel', y='AvgCSAT', title='Average Customer Satisfaction Rating by Channel')
    fig2.show()
    
    if 'FeedbackCategory' in df.columns:
        category_counts = df['FeedbackCategory'].value_counts().reset_index()
        category_counts.columns = ['Category', 'Count']
        fig3 = px.bar(category_counts, x='Category', y='Count', title='Distribution of Feedback Categories')
        fig3.show()

    return {
        "metrics": {
            "Overall Average Sentiment Score": overall_avg_sentiment,
            "Overall Average Customer Satisfaction Rating": overall_avg_csat,
            "Average Resolution Time (hours)": avg_resolution_time_hours,
            "Customer Service Performance by Channel": channel_performance.to_dict(orient='records')
        },
        "figures": {
            "Sentiment_Trend_by_Channel_Line": fig1,
            "Avg_CSAT_by_Channel_Bar": fig2,
            "Feedback_Categories_Distribution_Bar": fig3 if 'FeedbackCategory' in df.columns else None
        }
    }


def rfm_based_customer_targeting_analysis(df):
    print("\n--- RFM-Based Customer Targeting Analysis ---")
    expected = {
        'CustomerID': ['CustomerID', 'ID', 'UserID'],
        'LastPurchaseDate': ['LastPurchaseDate', 'RecencyDate', 'DateOfLastPurchase'],
        'TotalPurchases': ['TotalPurchases', 'Frequency', 'NumOrders'],
        'TotalSpend': ['TotalSpend', 'Monetary', 'LifetimeValue'],
        'Segment': ['Segment', 'RFMSegment', 'CustomerSegment']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['LastPurchaseDate'] = pd.to_datetime(df['LastPurchaseDate'], errors='coerce')
    df['TotalPurchases'] = pd.to_numeric(df['TotalPurchases'], errors='coerce')
    df['TotalSpend'] = pd.to_numeric(df['TotalSpend'], errors='coerce')
    df = df.dropna(subset=['CustomerID', 'LastPurchaseDate', 'TotalPurchases', 'TotalSpend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    # Calculate Recency (days since last purchase) relative to a most recent date in data
    max_date = df['LastPurchaseDate'].max()
    df['Recency'] = (max_date - df['LastPurchaseDate']).dt.days

    # Basic RFM quartiles (for demonstration, a full RFM usually involves custom scoring)
    df['R_Score'] = pd.qcut(df['Recency'], q=4, labels=[4, 3, 2, 1], duplicates='drop') # Lower recency is better
    df['F_Score'] = pd.qcut(df['TotalPurchases'], q=4, labels=[1, 2, 3, 4], duplicates='drop') # Higher frequency is better
    df['M_Score'] = pd.qcut(df['TotalSpend'], q=4, labels=[1, 2, 3, 4], duplicates='drop') # Higher monetary is better

    if 'Segment' not in df.columns:
        df['Segment'] = df['R_Score'].astype(str) + df['F_Score'].astype(str) + df['M_Score'].astype(str)
        print("Note: 'Segment' column not found, created a basic RFM score from R, F, M quartiles.")
    
    segment_distribution = df['Segment'].value_counts(normalize=True).reset_index()
    segment_distribution.columns = ['Segment', 'Percentage']
    segment_distribution['Percentage'] *= 100

    print("\nRFM Segment Distribution:")
    print(segment_distribution.round(2))

    fig1 = px.pie(segment_distribution, names='Segment', values='Percentage', title='Distribution of RFM Segments')
    fig1.show()

    segment_performance = df.groupby('Segment').agg(
        AvgRecency=('Recency', 'mean'),
        AvgFrequency=('TotalPurchases', 'mean'),
        AvgMonetary=('TotalSpend', 'mean')
    ).reset_index()

    print("\nAverage RFM Metrics by Segment:")
    print(segment_performance.round(2))

    fig2 = px.bar(segment_performance.sort_values('AvgMonetary', ascending=False).head(10),
                  x='Segment', y='AvgMonetary', title='Average Monetary Value by RFM Segment (Top 10)')
    fig2.show()
    
    fig3 = px.scatter(df, x='Recency', y='TotalSpend', color='Segment', size='TotalPurchases',
                     hover_name='CustomerID', title='RFM: Spend vs. Recency (Sized by Purchases)')
    fig3.show()

    return {
        "metrics": {
            "RFM Segment Distribution": segment_distribution.to_dict(orient='records'),
            "Average RFM Metrics by Segment": segment_performance.to_dict(orient='records')
        },
        "figures": {
            "RFM_Segment_Distribution_Pie": fig1,
            "Avg_Monetary_Value_by_RFM_Segment_Bar": fig2,
            "RFM_Spend_vs_Recency_Scatter": fig3
        }
    }


def webinar_performance_and_lead_generation_analysis(df):
    print("\n--- Webinar Performance and Lead Generation Analysis ---")
    expected = {
        'WebinarID': ['WebinarID', 'ID'],
        'WebinarTitle': ['WebinarTitle', 'Title'],
        'Registrations': ['Registrations', 'NumRegistrations', 'Signups'],
        'Attendees': ['Attendees', 'NumAttendees', 'ActualAttendees'],
        'LeadsGenerated': ['LeadsGenerated', 'NewLeads', 'QualifiedLeads'],
        'CostPerWebinar': ['CostPerWebinar', 'Spend', 'Cost'],
        'Date': ['Date', 'WebinarDate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Registrations'] = pd.to_numeric(df['Registrations'], errors='coerce')
    df['Attendees'] = pd.to_numeric(df['Attendees'], errors='coerce')
    df['LeadsGenerated'] = pd.to_numeric(df['LeadsGenerated'], errors='coerce')
    df['CostPerWebinar'] = pd.to_numeric(df['CostPerWebinar'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['WebinarID', 'Registrations', 'Attendees', 'LeadsGenerated', 'CostPerWebinar'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['AttendanceRate'] = (df['Attendees'] / df['Registrations']) * 100 if df['Registrations'].sum() > 0 else 0
    df['LeadConversionRate'] = (df['LeadsGenerated'] / df['Attendees']) * 100 if df['Attendees'].sum() > 0 else 0
    df['CPL'] = df['CostPerWebinar'] / df['LeadsGenerated'] if df['LeadsGenerated'].sum() > 0 else np.nan

    overall_attendance_rate = (df['Attendees'].sum() / df['Registrations'].sum()) * 100
    overall_lead_conversion_rate = (df['LeadsGenerated'].sum() / df['Attendees'].sum()) * 100
    overall_cpl = df['CostPerWebinar'].sum() / df['LeadsGenerated'].sum() if df['LeadsGenerated'].sum() > 0 else np.nan

    print(f"Overall Attendance Rate: {overall_attendance_rate:.2f}%")
    print(f"Overall Lead Conversion Rate: {overall_lead_conversion_rate:.2f}%")
    if not np.isnan(overall_cpl):
        print(f"Overall Cost Per Lead (CPL): ${overall_cpl:,.2f}")

    webinar_summary = df.groupby('WebinarID').agg(
        TotalRegistrations=('Registrations', 'sum'),
        TotalAttendees=('Attendees', 'sum'),
        TotalLeads=('LeadsGenerated', 'sum'),
        AvgAttendanceRate=('AttendanceRate', 'mean'),
        AvgLeadConversionRate=('LeadConversionRate', 'mean'),
        AvgCPL=('CPL', 'mean')
    ).reset_index()

    print("\nWebinar Performance Summary:")
    print(webinar_summary.sort_values('TotalLeads', ascending=False).head(10).round(2))

    fig1 = px.bar(webinar_summary.sort_values('TotalLeads', ascending=False).head(10),
                  x='WebinarID', y='TotalLeads', title='Top 10 Webinars by Leads Generated')
    fig1.show()

    fig2 = px.scatter(webinar_summary, x='AvgAttendanceRate', y='AvgLeadConversionRate',
                     size='TotalRegistrations', hover_name='WebinarID',
                     title='Lead Conversion Rate vs. Attendance Rate (Sized by Registrations)')
    fig2.show()
    
    if 'Date' in df.columns:
        daily_leads = df.groupby('Date')['LeadsGenerated'].sum().reset_index()
        fig3 = px.line(daily_leads, x='Date', y='LeadsGenerated', title='Daily Leads Generated Trend')
        fig3.show()

    return {
        "metrics": {
            "Overall Attendance Rate": overall_attendance_rate,
            "Overall Lead Conversion Rate": overall_lead_conversion_rate,
            "Overall CPL": overall_cpl,
            "Webinar Performance Summary": webinar_summary.to_dict(orient='records')
        },
        "figures": {
            "Top_10_Webinars_by_Leads_Bar": fig1,
            "Lead_Conversion_Rate_vs_Attendance_Rate_Scatter": fig2,
            "Daily_Leads_Generated_Trend_Line": fig3 if 'Date' in df.columns else None
        }
    }


def event_marketing_effectiveness_analysis(df):
    print("\n--- Event Marketing Effectiveness Analysis ---")
    expected = {
        'EventID': ['EventID', 'ID', 'EventName'],
        'EventType': ['EventType', 'Type'],
        'Attendees': ['Attendees', 'NumAttendees', 'Registrants'],
        'LeadsGenerated': ['LeadsGenerated', 'NewLeads'],
        'SalesGenerated': ['SalesGenerated', 'Revenue', 'Sales'],
        'CostOfEvent': ['CostOfEvent', 'EventBudget', 'Spend'],
        'Date': ['Date', 'EventDate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Attendees'] = pd.to_numeric(df['Attendees'], errors='coerce')
    df['LeadsGenerated'] = pd.to_numeric(df['LeadsGenerated'], errors='coerce')
    df['SalesGenerated'] = pd.to_numeric(df['SalesGenerated'], errors='coerce')
    df['CostOfEvent'] = pd.to_numeric(df['CostOfEvent'], errors='coerce')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['EventID', 'Attendees', 'LeadsGenerated', 'SalesGenerated', 'CostOfEvent'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['ROI'] = ((df['SalesGenerated'] - df['CostOfEvent']) / df['CostOfEvent']) * 100 if df['CostOfEvent'].sum() > 0 else np.nan
    df['CPL'] = df['CostOfEvent'] / df['LeadsGenerated'] if df['LeadsGenerated'].sum() > 0 else np.nan

    overall_avg_attendees = df['Attendees'].mean()
    overall_total_sales = df['SalesGenerated'].sum()
    overall_total_cost = df['CostOfEvent'].sum()
    overall_roi = ((overall_total_sales - overall_total_cost) / overall_total_cost) * 100 if overall_total_cost > 0 else np.nan

    print(f"Overall Average Attendees per Event: {overall_avg_attendees:,.0f}")
    print(f"Overall Total Sales Generated: ${overall_total_sales:,.2f}")
    print(f"Overall Event ROI: {overall_roi:.2f}%" if not np.isnan(overall_roi) else "Overall Event ROI: N/A")

    event_summary = df.groupby('EventID').agg(
        TotalAttendees=('Attendees', 'sum'),
        TotalLeads=('LeadsGenerated', 'sum'),
        TotalSales=('SalesGenerated', 'sum'),
        TotalCost=('CostOfEvent', 'sum'),
        ROI=('ROI', 'mean'),
        CPL=('CPL', 'mean')
    ).reset_index()

    print("\nEvent Performance Summary (Top 10 by Sales):")
    print(event_summary.sort_values('TotalSales', ascending=False).head(10).round(2))

    fig1 = px.bar(event_summary.sort_values('TotalSales', ascending=False).head(10),
                  x='EventID', y='TotalSales', title='Top 10 Events by Total Sales Generated')
    fig1.show()

    fig2 = px.scatter(event_summary, x='TotalCost', y='TotalSales', size='TotalAttendees', hover_name='EventID',
                     title='Sales vs. Cost (Sized by Attendees)')
    fig2.show()
    
    if 'EventType' in df.columns:
        event_type_impact = df.groupby('EventType').agg(
            AvgROI=('ROI', 'mean'),
            AvgCPL=('CPL', 'mean')
        ).reset_index()
        fig3 = px.bar(event_type_impact, x='EventType', y='AvgROI', title='Average ROI by Event Type')
        fig3.show()

    return {
        "metrics": {
            "Overall Average Attendees": overall_avg_attendees,
            "Overall Total Sales Generated": overall_total_sales,
            "Overall Event ROI": overall_roi,
            "Event Performance Summary": event_summary.to_dict(orient='records')
        },
        "figures": {
            "Top_10_Events_by_Sales_Bar": fig1,
            "Sales_vs_Cost_Scatter": fig2,
            "Avg_ROI_by_Event_Type_Bar": fig3 if 'EventType' in df.columns else None
        }
    }


def display_ad_banner_placement_performance_analysis(df):
    print("\n--- Display Ad Banner Placement Performance Analysis ---")
    expected = {
        'PlacementID': ['PlacementID', 'ID', 'AdPlacement'],
        'BannerID': ['BannerID', 'AdID', 'CreativeID'],
        'WebsiteDomain': ['WebsiteDomain', 'PublisherSite'],
        'Impressions': ['Impressions', 'AdImpressions'],
        'Clicks': ['Clicks', 'AdClicks'],
        'Conversions': ['Conversions', 'AdConversions', 'Sales'],
        'Spend': ['Spend', 'Cost']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
    df = df.dropna(subset=['PlacementID', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['CPA'] = df['Spend'] / df['Conversions'] if df['Conversions'].sum() > 0 else np.nan

    total_spend = df['Spend'].sum()
    overall_ctr = (df['Clicks'].sum() / df['Impressions'].sum()) * 100 if df['Impressions'].sum() > 0 else 0
    overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

    print(f"Total Ad Spend: ${total_spend:,.2f}")
    print(f"Overall CTR: {overall_ctr:.2f}%")
    if not np.isnan(overall_cpa):
        print(f"Overall CPA: ${overall_cpa:,.2f}")

    placement_summary = df.groupby('PlacementID').agg(
        TotalImpressions=('Impressions', 'sum'),
        TotalClicks=('Clicks', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        TotalSpend=('Spend', 'sum'),
        AvgCTR=('CTR', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean'),
        AvgCPA=('CPA', 'mean')
    ).reset_index()

    print("\nAd Placement Performance Summary (Top 10 by Conversions):")
    print(placement_summary.sort_values('TotalConversions', ascending=False).head(10).round(2))

    fig1 = px.bar(placement_summary.sort_values('TotalConversions', ascending=False).head(10),
                  x='PlacementID', y='TotalConversions', title='Top 10 Placements by Total Conversions')
    fig1.show()

    fig2 = px.scatter(placement_summary, x='AvgCPA', y='AvgCTR', size='TotalSpend', hover_name='PlacementID',
                     title='CTR vs. CPA by Placement (Sized by Spend)')
    fig2.show()
    
    if 'WebsiteDomain' in df.columns:
        domain_performance = df.groupby('WebsiteDomain').agg(
            AvgCTR=('CTR', 'mean'),
            AvgCPA=('CPA', 'mean')
        ).reset_index()
        fig3 = px.bar(domain_performance.sort_values('AvgCTR', ascending=False).head(10),
                      x='WebsiteDomain', y='AvgCTR', title='Top 10 Website Domains by Average CTR')
        fig3.show()

    return {
        "metrics": {
            "Total Ad Spend": total_spend,
            "Overall CTR": overall_ctr,
            "Overall CPA": overall_cpa,
            "Ad Placement Performance Summary": placement_summary.to_dict(orient='records')
        },
        "figures": {
            "Top_10_Placements_by_Conversions_Bar": fig1,
            "CTR_vs_CPA_Scatter": fig2,
            "Top_10_Website_Domains_by_Avg_CTR_Bar": fig3 if 'WebsiteDomain' in df.columns else None
        }
    }


def affiliate_marketing_performance_and_revenue_analysis(df):
    print("\n--- Affiliate Marketing Performance and Revenue Analysis ---")
    expected = {
        'AffiliateID': ['AffiliateID', 'ID', 'PartnerID'],
        'CampaignID': ['CampaignID', 'MarketingCampaignID'],
        'Clicks': ['Clicks', 'ReferralClicks'],
        'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
        'AffiliateCommission': ['AffiliateCommission', 'CommissionEarned', 'Payout'],
        'RevenueGenerated': ['RevenueGenerated', 'SalesRevenue', 'TotalSales'],
        'TrafficSource': ['TrafficSource', 'SourceMedium']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['AffiliateCommission'] = pd.to_numeric(df['AffiliateCommission'], errors='coerce')
    df['RevenueGenerated'] = pd.to_numeric(df['RevenueGenerated'], errors='coerce')
    df = df.dropna(subset=['AffiliateID', 'Clicks', 'Conversions', 'AffiliateCommission', 'RevenueGenerated'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['EPC'] = df['RevenueGenerated'] / df['Clicks'] if df['Clicks'].sum() > 0 else np.nan # Earnings per click
    df['AffiliateROI'] = ((df['RevenueGenerated'] - df['AffiliateCommission']) / df['AffiliateCommission']) * 100 if df['AffiliateCommission'].sum() > 0 else np.nan

    total_commission_paid = df['AffiliateCommission'].sum()
    total_revenue_from_affiliates = df['RevenueGenerated'].sum()
    overall_affiliate_conversion_rate = (df['Conversions'].sum() / df['Clicks'].sum()) * 100 if df['Clicks'].sum() > 0 else 0

    print(f"Total Commission Paid: ${total_commission_paid:,.2f}")
    print(f"Total Revenue from Affiliates: ${total_revenue_from_affiliates:,.2f}")
    print(f"Overall Affiliate Conversion Rate: {overall_affiliate_conversion_rate:.2f}%")

    affiliate_summary = df.groupby('AffiliateID').agg(
        TotalClicks=('Clicks', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        TotalCommission=('AffiliateCommission', 'sum'),
        TotalRevenue=('RevenueGenerated', 'sum'),
        AvgConversionRate=('ConversionRate', 'mean'),
        AvgEPC=('EPC', 'mean'),
        AvgROI=('AffiliateROI', 'mean')
    ).reset_index()

    print("\nAffiliate Performance Summary (Top 10 by Revenue Generated):")
    print(affiliate_summary.sort_values('TotalRevenue', ascending=False).head(10).round(2))

    fig1 = px.bar(affiliate_summary.sort_values('TotalRevenue', ascending=False).head(10),
                  x='AffiliateID', y='TotalRevenue', title='Top 10 Affiliates by Revenue Generated')
    fig1.show()

    fig2 = px.scatter(affiliate_summary, x='AvgEPC', y='AvgConversionRate', size='TotalClicks', hover_name='AffiliateID',
                     title='Affiliate Conversion Rate vs. EPC (Sized by Clicks)')
    fig2.show()
    
    if 'TrafficSource' in df.columns:
        source_performance = df.groupby('TrafficSource').agg(
            TotalClicks=('Clicks', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            TotalRevenue=('RevenueGenerated', 'sum')
        ).reset_index()
        fig3 = px.bar(source_performance, x='TrafficSource', y='TotalRevenue', title='Revenue by Traffic Source')
        fig3.show()

    return {
        "metrics": {
            "Total Commission Paid": total_commission_paid,
            "Total Revenue from Affiliates": total_revenue_from_affiliates,
            "Overall Affiliate Conversion Rate": overall_affiliate_conversion_rate,
            "Affiliate Performance Summary": affiliate_summary.to_dict(orient='records')
        },
        "figures": {
            "Top_10_Affiliates_by_Revenue_Bar": fig1,
            "Affiliate_Conversion_Rate_vs_EPC_Scatter": fig2,
            "Revenue_by_Traffic_Source_Bar": fig3 if 'TrafficSource' in df.columns else None
        }
    }


def clicked_link_position_and_device_analysis(df):
    print("\n--- Clicked Link Position and Device Analysis ---")
    expected = {
        'ClickID': ['ClickID', 'ID'],
        'LinkPosition': ['LinkPosition', 'AdPosition', 'Rank'],
        'DeviceType': ['DeviceType', 'Device', 'Platform'],
        'IsConversion': ['IsConversion', 'Converted', 'PurchaseEvent'], # Binary (True/False)
        'CTR': ['CTR', 'ClickThroughRate'], # Per click, if available, otherwise will recalculate
        'Impressions': ['Impressions', 'AdImpressions'] # Needed to calculate CTR if not provided
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['LinkPosition'] = pd.to_numeric(df['LinkPosition'], errors='coerce')
    df['IsConversion'] = df['IsConversion'].astype(str).str.lower().map({'1': True, '0': False, 'true': True, 'false': False, 'yes': True, 'no': False})
    
    # If CTR is not a direct column, calculate it (requires Impressions in the df)
    if 'CTR' not in df.columns and 'Impressions' in df.columns:
        # Assuming each row is a click and Impressions is for that specific click context
        # This might be tricky if data is aggregated. A more robust approach would be to calculate CTR per (LinkPosition, DeviceType) group
        df['Clicks'] = 1 # Each row is a click
        grouped_data = df.groupby(['LinkPosition', 'DeviceType']).agg(
            TotalClicks=('Clicks', 'sum'),
            TotalImpressions=('Impressions', 'sum'),
            TotalConversions=('IsConversion', 'sum')
        ).reset_index()
        grouped_data['CTR'] = (grouped_data['TotalClicks'] / grouped_data['TotalImpressions']) * 100
        grouped_data['ConversionRate'] = (grouped_data['TotalConversions'] / grouped_data['TotalClicks']) * 100
        
        analysis_df = grouped_data
        print("Note: CTR/ConversionRate calculated from aggregated clicks/impressions/conversions.")
    elif 'CTR' in df.columns:
        df['CTR'] = pd.to_numeric(df['CTR'], errors='coerce')
        df = df.dropna(subset=['LinkPosition', 'DeviceType', 'IsConversion', 'CTR'])
        analysis_df = df.copy()
        analysis_df['Clicks'] = 1 # Assuming each row is a click for simpler group by if needed
        analysis_df['TotalConversions'] = analysis_df['IsConversion'].astype(int) # Convert boolean to int for sum
        print("Note: Using provided 'CTR' column directly.")
    else:
        print("Error: Neither 'CTR' nor 'Impressions' column found to calculate CTR. Analysis limited.")
        return {"message": "Missing CTR or Impressions column."}


    if analysis_df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    overall_ctr = analysis_df['CTR'].mean() if 'CTR' in analysis_df.columns else np.nan
    overall_conversion_rate = analysis_df['ConversionRate'].mean() if 'ConversionRate' in analysis_df.columns else np.nan
    
    print(f"Overall Average CTR: {overall_ctr:.2f}%" if not np.isnan(overall_ctr) else "Overall Average CTR: N/A")
    print(f"Overall Average Conversion Rate: {overall_conversion_rate:.2f}%" if not np.isnan(overall_conversion_rate) else "Overall Average Conversion Rate: N/A")

    if 'CTR' in analysis_df.columns:
        fig1 = px.bar(analysis_df.sort_values('CTR', ascending=False).head(10), x='LinkPosition', y='CTR',
                      title='Top 10 Link Positions by CTR')
        fig1.show()

    if 'DeviceType' in analysis_df.columns:
        device_performance = analysis_df.groupby('DeviceType').agg(
            AvgCTR=('CTR', 'mean') if 'CTR' in analysis_df.columns else ('TotalClicks', lambda x: x.sum() / analysis_df['TotalImpressions'].sum() * 100),
            ConversionRate=('ConversionRate', 'mean') if 'ConversionRate' in analysis_df.columns else ('TotalConversions', lambda x: x.sum() / analysis_df['TotalClicks'].sum() * 100)
        ).reset_index()
        device_performance.columns = ['DeviceType', 'AvgCTR', 'ConversionRate'] # Standardize columns after aggregation
        
        print("\nPerformance by Device Type:")
        print(device_performance.round(2))

        fig2 = px.bar(device_performance, x='DeviceType', y='AvgCTR', title='Average CTR by Device Type')
        fig2.show()
        
        fig3 = px.bar(device_performance, x='DeviceType', y='ConversionRate', title='Conversion Rate by Device Type')
        fig3.show()

    return {
        "metrics": {
            "Overall Average CTR": overall_ctr,
            "Overall Average Conversion Rate": overall_conversion_rate,
            "Performance by Device Type": device_performance.to_dict(orient='records') if 'DeviceType' in analysis_df.columns else "N/A"
        },
        "figures": {
            "Top_10_Link_Positions_by_CTR_Bar": fig1 if 'CTR' in analysis_df.columns else None,
            "Avg_CTR_by_Device_Type_Bar": fig2 if 'DeviceType' in analysis_df.columns else None,
            "Conversion_Rate_by_Device_Type_Bar": fig3 if 'DeviceType' in analysis_df.columns else None
        }
    }


def dynamic_content_personalization_analysis(df):
    print("\n--- Dynamic Content Personalization Analysis ---")
    expected = {
        'UserID': ['UserID', 'ID', 'CustomerID'],
        'PersonalizationSegment': ['PersonalizationSegment', 'Segment', 'DynamicContentGroup'],
        'ContentVariantShown': ['ContentVariantShown', 'ContentVariant', 'Variant'],
        'ClickThroughRate': ['ClickThroughRate', 'CTR'],
        'ConversionRate': ['ConversionRate', 'Converted'],
        'Revenue': ['Revenue', 'Sales']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['ClickThroughRate'] = pd.to_numeric(df['ClickThroughRate'], errors='coerce')
    df['ConversionRate'] = pd.to_numeric(df['ConversionRate'], errors='coerce')
    df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')
    df = df.dropna(subset=['UserID', 'PersonalizationSegment', 'ContentVariantShown', 'ClickThroughRate', 'ConversionRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    overall_ctr = df['ClickThroughRate'].mean()
    overall_conversion_rate = df['ConversionRate'].mean()
    overall_revenue = df['Revenue'].sum() if 'Revenue' in df.columns else 0

    print(f"Overall Average CTR: {overall_ctr:.2f}%")
    print(f"Overall Average Conversion Rate: {overall_conversion_rate:.2f}%")
    print(f"Overall Total Revenue: ${overall_revenue:,.2f}")

    segment_content_summary = df.groupby(['PersonalizationSegment', 'ContentVariantShown']).agg(
        AvgCTR=('ClickThroughRate', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean'),
        TotalRevenue=('Revenue', 'sum') if 'Revenue' in df.columns else ('ClickThroughRate', 'size')
    ).reset_index()

    print("\nPerformance by Personalization Segment and Content Variant:")
    print(segment_content_summary.round(2))

    fig1 = px.bar(segment_content_summary, x='PersonalizationSegment', y='AvgConversionRate', color='ContentVariantShown',
                  barmode='group', title='Average Conversion Rate by Segment and Content Variant')
    fig1.show()

    fig2 = px.bar(segment_content_summary, x='PersonalizationSegment', y='AvgCTR', color='ContentVariantShown',
                  barmode='group', title='Average CTR by Segment and Content Variant')
    fig2.show()
    
    if 'Revenue' in df.columns:
        fig3 = px.bar(segment_content_summary, x='PersonalizationSegment', y='TotalRevenue', color='ContentVariantShown',
                      barmode='group', title='Total Revenue by Segment and Content Variant')
        fig3.show()

    return {
        "metrics": {
            "Overall Average CTR": overall_ctr,
            "Overall Average Conversion Rate": overall_conversion_rate,
            "Overall Total Revenue": overall_revenue,
            "Performance by Segment and Content Variant": segment_content_summary.to_dict(orient='records')
        },
        "figures": {
            "Conversion_Rate_by_Segment_and_Variant_Bar": fig1,
            "CTR_by_Segment_and_Variant_Bar": fig2,
            "Revenue_by_Segment_and_Variant_Bar": fig3 if 'Revenue' in df.columns else None
        }
    }


def remarketing_campaign_performance_analysis(df):
    print("\n--- Remarketing Campaign Performance Analysis ---")
    expected = {
        'UserID': ['UserID', 'ID', 'CustomerID'],
        'CampaignID': ['CampaignID', 'RemarketingCampaignID'],
        'Impressions': ['Impressions', 'AdImpressions'],
        'Clicks': ['Clicks', 'AdClicks'],
        'Conversions': ['Conversions', 'SalesConversions', 'Purchases'],
        'Spend': ['Spend', 'AdSpend', 'Cost'],
        'DaysSinceLastVisit': ['DaysSinceLastVisit', 'RecencyDays'],
        'PreviousInteractionType': ['PreviousInteractionType', 'LastInteraction', 'ActivityType'] # e.g., viewed product, abandoned cart
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
    df['DaysSinceLastVisit'] = pd.to_numeric(df['DaysSinceLastVisit'], errors='coerce')
    df = df.dropna(subset=['UserID', 'CampaignID', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['CPA'] = df['Spend'] / df['Conversions'] if df['Conversions'].sum() > 0 else np.nan

    overall_spend = df['Spend'].sum()
    overall_conversions = df['Conversions'].sum()
    overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

    print(f"Total Remarketing Spend: ${overall_spend:,.2f}")
    print(f"Total Remarketing Conversions: {overall_conversions}")
    if not np.isnan(overall_cpa):
        print(f"Overall Remarketing CPA: ${overall_cpa:,.2f}")

    campaign_summary = df.groupby('CampaignID').agg(
        TotalSpend=('Spend', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        AvgCTR=('CTR', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean'),
        AvgCPA=('CPA', 'mean')
    ).reset_index()

    print("\nRemarketing Campaign Performance Summary:")
    print(campaign_summary.round(2))

    fig1 = px.bar(campaign_summary.sort_values('TotalConversions', ascending=False).head(10),
                  x='CampaignID', y='TotalConversions', title='Top 10 Remarketing Campaigns by Conversions')
    fig1.show()

    if 'PreviousInteractionType' in df.columns:
        interaction_type_performance = df.groupby('PreviousInteractionType').agg(
            AvgConversionRate=('ConversionRate', 'mean'),
            AvgCPA=('CPA', 'mean'),
            TotalSpend=('Spend', 'sum')
        ).reset_index()
        fig2 = px.bar(interaction_type_performance, x='PreviousInteractionType', y='AvgConversionRate',
                      title='Average Conversion Rate by Previous Interaction Type')
        fig2.show()
    
    if 'DaysSinceLastVisit' in df.columns:
        fig3 = px.scatter(df, x='DaysSinceLastVisit', y='Conversions', color='CampaignID',
                         hover_name='UserID', title='Conversions vs. Days Since Last Visit')
        fig3.show()

    return {
        "metrics": {
            "Total Remarketing Spend": overall_spend,
            "Total Remarketing Conversions": overall_conversions,
            "Overall Remarketing CPA": overall_cpa,
            "Remarketing Campaign Performance Summary": campaign_summary.to_dict(orient='records')
        },
        "figures": {
            "Top_10_Remarketing_Campaigns_Bar": fig1,
            "Avg_Conversion_Rate_by_Previous_Interaction_Type_Bar": fig2 if 'PreviousInteractionType' in df.columns else None,
            "Conversions_vs_Days_Since_Last_Visit_Scatter": fig3 if 'DaysSinceLastVisit' in df.columns else None
        }
    }


def ad_format_performance_and_cost_analysis(df):
    print("\n--- Ad Format Performance and Cost Analysis ---")
    expected = {
        'AdFormat': ['AdFormat', 'Format', 'CreativeType'],
        'AdChannel': ['AdChannel', 'Platform', 'Channel'],
        'Impressions': ['Impressions', 'AdImpressions'],
        'Clicks': ['Clicks', 'AdClicks'],
        'Conversions': ['Conversions', 'AdConversions', 'Sales'],
        'Spend': ['Spend', 'AdSpend', 'Cost']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['Impressions'] = pd.to_numeric(df['Impressions'], errors='coerce')
    df['Clicks'] = pd.to_numeric(df['Clicks'], errors='coerce')
    df['Conversions'] = pd.to_numeric(df['Conversions'], errors='coerce')
    df['Spend'] = pd.to_numeric(df['Spend'], errors='coerce')
    df = df.dropna(subset=['AdFormat', 'Impressions', 'Clicks', 'Conversions', 'Spend'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['CTR'] = (df['Clicks'] / df['Impressions']) * 100 if df['Impressions'].sum() > 0 else 0
    df['ConversionRate'] = (df['Conversions'] / df['Clicks']) * 100 if df['Clicks'].sum() > 0 else 0
    df['CPA'] = df['Spend'] / df['Conversions'] if df['Conversions'].sum() > 0 else np.nan

    overall_spend = df['Spend'].sum()
    overall_ctr = (df['Clicks'].sum() / df['Impressions'].sum()) * 100 if df['Impressions'].sum() > 0 else 0
    overall_cpa = df['Spend'].sum() / df['Conversions'].sum() if df['Conversions'].sum() > 0 else np.nan

    print(f"Total Ad Spend: ${overall_spend:,.2f}")
    print(f"Overall CTR: {overall_ctr:.2f}%")
    if not np.isnan(overall_cpa):
        print(f"Overall CPA: ${overall_cpa:,.2f}")

    format_performance = df.groupby('AdFormat').agg(
        TotalSpend=('Spend', 'sum'),
        TotalConversions=('Conversions', 'sum'),
        AvgCTR=('CTR', 'mean'),
        AvgConversionRate=('ConversionRate', 'mean'),
        AvgCPA=('CPA', 'mean')
    ).reset_index()

    print("\nAd Format Performance Summary:")
    print(format_performance.round(2))

    fig1 = px.bar(format_performance.sort_values('TotalConversions', ascending=False),
                  x='AdFormat', y='TotalConversions', title='Total Conversions by Ad Format')
    fig1.show()

    fig2 = px.bar(format_performance.sort_values('AvgCPA', ascending=True),
                  x='AdFormat', y='AvgCPA', title='Average CPA by Ad Format (Lower is Better)')
    fig2.show()
    
    if 'AdChannel' in df.columns:
        channel_format_performance = df.groupby(['AdChannel', 'AdFormat']).agg(
            TotalSpend=('Spend', 'sum'),
            TotalConversions=('Conversions', 'sum'),
            AvgCTR=('CTR', 'mean'),
            AvgConversionRate=('ConversionRate', 'mean')
        ).reset_index()
        fig3 = px.bar(channel_format_performance, x='AdChannel', y='AvgConversionRate', color='AdFormat',
                      barmode='group', title='Average Conversion Rate by Channel and Ad Format')
        fig3.show()

    return {
        "metrics": {
            "Total Ad Spend": overall_spend,
            "Overall CTR": overall_ctr,
            "Overall CPA": overall_cpa,
            "Ad Format Performance Summary": format_performance.to_dict(orient='records')
        },
        "figures": {
            "Total_Conversions_by_Ad_Format_Bar": fig1,
            "Avg_CPA_by_Ad_Format_Bar": fig2,
            "Avg_Conversion_Rate_by_Channel_and_Format_Bar": fig3 if 'AdChannel' in df.columns else None
        }
    }
import pandas as pd
import numpy as np

def campaign_performance(df):
    """
    Performs a general analysis of campaign performance.
    This could include metrics like total spend, impressions, clicks, conversions, ROI.
    """
    print("Performing campaign performance analysis...")
    # Example: Calculate total conversions and cost
    if 'conversions' in df.columns and 'cost' in df.columns:
        total_conversions = df['conversions'].sum()
        total_cost = df['cost'].sum()
        print(f"Total Conversions: {total_conversions}")
        print(f"Total Cost: {total_cost}")
        if total_cost > 0:
            cpa = total_cost / total_conversions if total_conversions > 0 else np.inf
            print(f"Cost Per Acquisition (CPA): {cpa:.2f}")
    else:
        print("Missing 'conversions' or 'cost' columns for detailed analysis.")
    # Add more detailed analysis logic here (e.g., ROI, CTR, CVR)
    return df

def channel_analysis(df):
    """
    Analyzes performance across different marketing channels (e.g., social, email, paid search).
    """
    print("Performing channel analysis...")
    if 'channel' in df.columns and 'conversions' in df.columns and 'cost' in df.columns:
        channel_summary = df.groupby('channel').agg(
            total_conversions=('conversions', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        channel_summary['ROI'] = (channel_summary['total_conversions'] / channel_summary['total_cost']).replace([np.inf, -np.inf], np.nan)
        print(channel_summary)
    else:
        print("Missing 'channel', 'conversions', or 'cost' columns for detailed analysis.")
    return df

def customer_segmentation(df):
    """
    Segments customers based on various criteria (e.g., demographics, behavior, purchase history).
    """
    print("Performing customer segmentation...")
    # Example: Simple segmentation based on a 'customer_type' column
    if 'customer_id' in df.columns and 'total_spend' in df.columns:
        # This is a placeholder for actual segmentation logic (e.g., K-means, RFM)
        print("Consider using clustering algorithms (e.g., K-means) or RFM analysis here.")
        # For a very basic example, if you have a 'customer_segment' column already:
        if 'customer_segment' in df.columns:
            print(df.groupby('customer_segment')['customer_id'].count())
        else:
            print("No 'customer_segment' column found for pre-defined segmentation. Implement custom segmentation logic.")
    else:
        print("Missing 'customer_id' or 'total_spend' columns for detailed analysis.")
    return df

def funnel_analysis(df):
    """
    Analyzes customer progression through a marketing or sales funnel.
    Requires data on different stages (e.g., impressions, clicks, leads, conversions).
    """
    print("Performing funnel analysis...")
    # Example: Assume columns like 'impressions', 'clicks', 'leads', 'conversions'
    funnel_stages = ['impressions', 'clicks', 'leads', 'conversions']
    for i in range(len(funnel_stages)):
        if funnel_stages[i] not in df.columns:
            print(f"Missing column: {funnel_stages[i]} for funnel analysis.")
            return df
    
    total_impressions = df['impressions'].sum()
    total_clicks = df['clicks'].sum()
    total_leads = df['leads'].sum()
    total_conversions = df['conversions'].sum()

    print(f"Impressions: {total_impressions}")
    print(f"Clicks: {total_clicks} (Click-through Rate: {total_clicks / total_impressions:.2%} if total_impressions > 0 else 'N/A')")
    print(f"Leads: {total_leads} (Lead Conversion Rate: {total_leads / total_clicks:.2%} if total_clicks > 0 else 'N/A')")
    print(f"Conversions: {total_conversions} (Sales Conversion Rate: {total_conversions / total_leads:.2%} if total_leads > 0 else 'N/A')")
    return df

def content_analysis(df):
    """
    Analyzes the performance of marketing content (e.g., blog posts, videos, ads).
    Metrics could include views, engagement, shares, time on page.
    """
    print("Performing content analysis...")
    if 'content_id' in df.columns and 'views' in df.columns and 'engagement_score' in df.columns:
        content_summary = df.groupby('content_id').agg(
            total_views=('views', 'sum'),
            avg_engagement=('engagement_score', 'mean')
        ).sort_values(by='total_views', ascending=False)
        print(content_summary.head())
    else:
        print("Missing 'content_id', 'views', or 'engagement_score' columns for detailed analysis.")
    return df

def social_media(df):
    """
    Analyzes social media campaign performance.
    Metrics: likes, shares, comments, reach, impressions, engagement rate.
    """
    print("Performing social media analysis...")
    if 'platform' in df.columns and 'impressions' in df.columns and 'engagement' in df.columns:
        social_media_summary = df.groupby('platform').agg(
            total_impressions=('impressions', 'sum'),
            total_engagement=('engagement', 'sum')
        ).reset_index()
        social_media_summary['engagement_rate'] = (social_media_summary['total_engagement'] / social_media_summary['total_impressions']).replace([np.inf, -np.inf], np.nan)
        print(social_media_summary)
    else:
        print("Missing 'platform', 'impressions', or 'engagement' columns for detailed analysis.")
    return df

def ab_testing(df):
    """
    Performs a general A/B testing analysis. This might involve comparing metrics
    between two or more variants.
    """
    print("Performing A/B testing analysis...")
    if 'variant' in df.columns and 'conversions' in df.columns and 'visitors' in df.columns:
        ab_summary = df.groupby('variant').agg(
            total_visitors=('visitors', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        ab_summary['conversion_rate'] = (ab_summary['total_conversions'] / ab_summary['total_visitors']).replace([np.inf, -np.inf], np.nan)
        print(ab_summary)
        # Further statistical tests (e.g., t-test, chi-squared) would go here
    else:
        print("Missing 'variant', 'conversions', or 'visitors' columns for detailed analysis.")
    return df

def marketing_campaign_performance_and_roi_analysis(df):
    """
    Analyzes the performance and Return on Investment (ROI) of marketing campaigns.
    """
    print("Performing marketing campaign performance and ROI analysis...")
    if 'campaign_id' in df.columns and 'revenue' in df.columns and 'cost' in df.columns:
        campaign_roi = df.groupby('campaign_id').agg(
            total_revenue=('revenue', 'sum'),
            total_cost=('cost', 'sum')
        ).reset_index()
        campaign_roi['ROI'] = ((campaign_roi['total_revenue'] - campaign_roi['total_cost']) / campaign_roi['total_cost']).replace([np.inf, -np.inf], np.nan)
        print(campaign_roi.sort_values(by='ROI', ascending=False))
    else:
        print("Missing 'campaign_id', 'revenue', or 'cost' columns for detailed analysis.")
    return df

def customer_segmentation_and_campaign_response_analysis(df):
    """
    Segments customers and analyzes how different segments respond to campaigns.
    """
    print("Performing customer segmentation and campaign response analysis...")
    if 'customer_segment' in df.columns and 'campaign_id' in df.columns and 'response' in df.columns:
        response_by_segment = df.groupby(['customer_segment', 'campaign_id'])['response'].mean().unstack()
        print(response_by_segment)
    else:
        print("Missing 'customer_segment', 'campaign_id', or 'response' columns for detailed analysis.")
    return df

def email_marketing_campaign_effectiveness_analysis(df):
    """
    Analyzes key metrics for email marketing campaigns: open rates, click-through rates, conversions.
    """
    print("Performing email marketing campaign effectiveness analysis...")
    if 'email_campaign_id' in df.columns and 'sent' in df.columns and 'opens' in df.columns and 'clicks' in df.columns and 'conversions' in df.columns:
        email_summary = df.groupby('email_campaign_id').agg(
            total_sent=('sent', 'sum'),
            total_opens=('opens', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        email_summary['open_rate'] = (email_summary['total_opens'] / email_summary['total_sent']).replace([np.inf, -np.inf], np.nan)
        email_summary['click_through_rate'] = (email_summary['total_clicks'] / email_summary['total_opens']).replace([np.inf, -np.inf], np.nan)
        email_summary['conversion_rate'] = (email_summary['total_conversions'] / email_summary['total_clicks']).replace([np.inf, -np.inf], np.nan)
        print(email_summary)
    else:
        print("Missing necessary columns for email marketing analysis (e.g., 'sent', 'opens', 'clicks', 'conversions').")
    return df

def sms_marketing_campaign_performance_analysis(df):
    """
    Analyzes key metrics for SMS marketing campaigns: delivery rates, open rates (if applicable), click-through rates, conversions.
    """
    print("Performing SMS marketing campaign performance analysis...")
    if 'sms_campaign_id' in df.columns and 'sent' in df.columns and 'delivered' in df.columns and 'clicks' in df.columns and 'conversions' in df.columns:
        sms_summary = df.groupby('sms_campaign_id').agg(
            total_sent=('sent', 'sum'),
            total_delivered=('delivered', 'sum'),
            total_clicks=('clicks', 'sum'),
            total_conversions=('conversions', 'sum')
        ).reset_index()
        sms_summary['delivery_rate'] = (sms_summary['total_delivered'] / sms_summary['total_sent']).replace([np.inf, -np.inf], np.nan)
        sms_summary['click_through_rate'] = (sms_summary['total_clicks'] / sms_summary['total_delivered']).replace([np.inf, -np.inf], np.nan)
        sms_summary['conversion_rate'] = (sms_summary['total_conversions'] / sms_summary['total_clicks']).replace([np.inf, -np.inf], np.nan)
        print(sms_summary)
    else:
        print("Missing necessary columns for SMS marketing analysis (e.g., 'sent', 'delivered', 'clicks', 'conversions').")
    return df

def multi_channel_campaign_attribution_analysis(df):
    """
    Analyzes the contribution of different marketing channels to conversions,
    often using attribution models (e.g., first-touch, last-touch, linear).
    """
    print("Performing multi-channel campaign attribution analysis...")
    if 'customer_id' in df.columns and 'channel' in df.columns and 'conversion_timestamp' in df.columns and 'event_timestamp' in df.columns:
        # This requires more complex logic for attribution modeling.
        # Placeholder for where attribution logic would go.
        print("This analysis typically requires a sequence of customer touchpoints and a chosen attribution model (e.g., last-click, linear, time decay).")
        print("Consider using libraries like `ChannelAttribution` or implementing custom logic.")
    else:
        print("Missing 'customer_id', 'channel', 'conversion_timestamp', or 'event_timestamp' for multi-channel attribution.")
    return df

def daily_marketing_campaign_performance_tracking(df):
    """
    Tracks marketing campaign performance on a daily basis.
    """
    print("Performing daily marketing campaign performance tracking...")
    if 'date' in df.columns and 'campaign_id' in df.columns and 'conversions' in df.columns and 'cost' in df.columns:
        daily_performance = df.groupby(['date', 'campaign_id']).agg(
            daily_conversions=('conversions', 'sum'),
            daily_cost=('cost', 'sum')
        ).reset_index()
        print(daily_performance.sort_values(by='date'))
    else:
        print("Missing 'date', 'campaign_id', 'conversions', or 'cost' for daily tracking.")
    return df

def campaign_reach_frequency_and_lift_analysis(df):
    """
    Analyzes campaign reach (unique users), frequency (average exposures), and lift (incremental impact).
    """
    print("Performing campaign reach, frequency, and lift analysis...")
    if 'campaign_id' in df.columns and 'user_id' in df.columns and 'impressions' in df.columns and 'conversion' in df.columns:
        # Reach: unique users exposed
        reach_by_campaign = df.groupby('campaign_id')['user_id'].nunique().reset_index(name='unique_users_reached')
        print("Campaign Reach:")
        print(reach_by_campaign)

        # Frequency: average impressions per user
        frequency_by_campaign = df.groupby(['campaign_id', 'user_id'])['impressions'].sum().reset_index()
        avg_frequency = frequency_by_campaign.groupby('campaign_id')['impressions'].mean().reset_index(name='average_frequency')
        print("\nCampaign Average Frequency:")
        print(avg_frequency)

        # Lift Analysis (requires control group data or pre/post analysis)
        print("\nLift analysis typically requires a control group or pre/post campaign data to compare conversion rates or other metrics.")
    else:
        print("Missing 'campaign_id', 'user_id', 'impressions', or 'conversion' for reach/frequency/lift analysis.")
    return df

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
    df, missing = check_and_rename_columns(df.copy(), expected) # Use .copy() to avoid SettingWithCopyWarning

    if missing:
        show_missing_columns_warning(missing, expected)
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
    accurate_targets = (df['TargetedRegion'] == df['ActualRegion']).sum()
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
    fig1.show()

    fig2 = px.bar(performance_by_accuracy, x='IsAccurateTarget', y='ConversionRate', 
                  title='Conversion Rate by Geotargeting Accuracy',
                  text_auto='.2s', color='IsAccurateTarget',
                  color_discrete_map={'Accurate': 'lightgreen', 'Inaccurate': 'salmon'})
    fig2.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
    fig2.update_layout(yaxis_title="Conversion Rate (%)")
    fig2.show()
    
    # Targeted vs. Actual Region Analysis (Confusion Matrix like)
    cross_tab_regions = pd.crosstab(df['TargetedRegion'], df['ActualRegion'], normalize='index')
    print("\nTargeted vs Actual Region (Row-normalized):\n", cross_tab_regions.round(2))
    
    fig3 = px.imshow(cross_tab_regions, text_auto=True, 
                     title='Targeted vs. Actual Region Impression Distribution (Normalized by Targeted Region)',
                     labels=dict(x="Actual Region", y="Targeted Region", color="Proportion"),
                     color_continuous_scale='Viridis')
    fig3.update_xaxes(side="top")
    fig3.show()

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
    fig4.show()

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

# Data Loading Function
def load_data():
    """
    Loads a sample DataFrame for geotargeting analysis.
    """
    data = {
        'AdImpressionID': range(1, 21),
        'TargetedRegion': ['Chennai', 'Mumbai', 'Chennai', 'Delhi', 'Bangalore', 
                           'Mumbai', 'Chennai', 'Delhi', 'Chennai', 'Bangalore',
                           'Chennai', 'Mumbai', 'Delhi', 'Chennai', 'Bangalore',
                           'Mumbai', 'Chennai', 'Delhi', 'Chennai', 'Bangalore'],
        'ActualRegion': ['Chennai', 'Mumbai', 'Bangalore', 'Delhi', 'Bangalore',
                         'Pune', 'Chennai', 'Delhi', 'Hyderabad', 'Bangalore',
                         'Chennai', 'Mumbai', 'Delhi', 'Kolkata', 'Chennai',
                         'Mumbai', 'Chennai', 'Delhi', 'Chennai', 'Bangalore'],
        'ConversionEvent': [1, 1, 0, 1, 1, 
                            0, 1, 1, 0, 1, 
                            1, 1, 1, 0, 0,
                            1, 1, 1, 1, 1],
        'AdSpend': [10, 12, 8, 15, 11, 
                    9, 10, 14, 7, 12,
                    10, 13, 16, 8, 9,
                    11, 10, 15, 10, 13]
    }
    df = pd.DataFrame(data)

    # Add some alternative column names to test the check_and_rename_columns function
    df.rename(columns={'AdImpressionID': 'ID', 'TargetedRegion': 'GeoTarget', 'ActualRegion': 'UserLocation', 'ConversionEvent': 'IsConverted', 'AdSpend': 'Cost'}, inplace=True)
    
    print("Sample Data Loaded:")
    print(df.head())
    return df
def show_general_insights(df,title="General Insights"):
    """
    Show general insights for a given dataframe including shape, dtypes, missing values, and basic statistics.
    """
    import pandas as pd

    insights = {}

    print("--- General Insights ---")

    # Shape
    insights['Shape'] = df.shape
    print(f"Shape: {df.shape}")

    # Data types
    dtype_counts = df.dtypes.value_counts()
    insights['Data Types Count'] = dtype_counts.to_dict()
    print("Data Types Count:")
    print(dtype_counts)

    # Missing values
    missing_values_count = df.isnull().sum()
    total_missing = missing_values_count.sum()
    insights['Total Missing Values'] = total_missing
    print(f"Total Missing Values: {total_missing}")
    print("Missing Values by Column:")
    print(missing_values_count[missing_values_count > 0])

    # Basic statistics for numerical columns
    if not df.select_dtypes(include=["number"]).empty:
        desc_stats = df.describe().T
        insights['Numerical Columns Description'] = desc_stats
        print("\nDescriptive Statistics for Numerical Columns:")
        print(desc_stats)
    else:
        print("No numerical columns to describe.")

    # Unique value counts for categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nUnique value counts for categorical columns:")
        unique_counts = df[categorical_cols].nunique()
        insights['Categorical Unique Counts'] = unique_counts.to_dict()
        print(unique_counts)
    else:
        print("No categorical columns found.")

    return insights


def main():
    print("📈 Marketing Analytics Dashboard")
    file_path = input("Enter path to your marketing data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252): ")
    if not encoding:
        encoding = 'utf-8'
    
    df = load_data(file_path, encoding=encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    
    analysis_options = [
        "campaign_performance",
        "channel_analysis",
        "customer_segmentation", 
        "funnel_analysis",
        "content_analysis",
        "social_media",
        "a/b_testing",
        "marketing_campaign_performance_and_roi_analysis",
        "customer_segmentation_and_campaign_response_analysis",
        "email_marketing_campaign_effectiveness_analysis",
        "sms_marketing_campaign_performance_analysis",
        "multi-channel_campaign_attribution_analysis",
        "daily_marketing_campaign_performance_tracking",
        "campaign_reach_frequency_and_lift_analysis",
        "customer_journey_and_touchpoint_analysis",
        "coupon_redemption_and_usage_analysis",
        "a/b_test_creative_performance_analysis",
        "website_visitor_conversion_analysis",
        "digital_advertising_platform_performance_and_roas_analysis",
        "customer_satisfaction_survey_analysis",
        "ad_placement_and_engagement_analysis",
        "lead_generation_and_qualification_analysis",
        "cross-device_campaign_conversion_analysis",
        "content_marketing_and_engagement_analysis",
        "customer_preference_and_personalization_analysis",
        "special_offer_and_discount_effectiveness_analysis",
        "geotargeted_campaign_performance_analysis",
        "a/b_testing_campaign_variant_analysis",
        "media_buying_and_ad_performance_analysis",
        "product_line_marketing_campaign_analysis",
        "website_engagement_and_ad_interaction_analysis",
        "daily_sales_revenue_and_campaign_correlation_analysis",
        "influencer_marketing_campaign_performance_analysis",
        "ad_copy_performance_analysis",
        "customer_referral_program_analysis",
        "website/platform_engagement_metrics_analysis",
        "customer_loyalty_program_engagement_analysis",
        "discount_code_redemption_and_visit_analysis",
        "seasonal_and_holiday_campaign_impact_analysis",
        "video_marketing_engagement_analysis",
        "search_engine_marketing_(sem)_keyword_performance",
        "churn_prediction_and_targeted_campaign_analysis",
        "geotargeting_accuracy_and_effectiveness_analysis",
        "newsletter_signup_attribution_analysis",
        "marketing_budget_allocation_and_spend_analysis",
        "social_media_competitive_and_sentiment_analysis",
        "customer_service_sentiment_and_feedback_analysis",
        "rfm-based_customer_targeting_analysis",
        "webinar_performance_and_lead_generation_analysis",
        "event_marketing_effectiveness_analysis",
        "display_ad_banner_placement_performance_analysis",
        "affiliate_marketing_performance_and_revenue_analysis",
        "clicked_link_position_and_device_analysis",
        "dynamic_content_personalization_analysis",
        "remarketing_campaign_performance_analysis",
        "ad_format_performance_and_cost_analysis",
        "General Insights"
    ]

    print("\nSelect Analysis to Perform:")
    for i, option in enumerate(analysis_options):
        print(f"{i}: {option}")
    
    choice = input("Enter the option number: ")
    try:
        choice = int(choice)
    except ValueError:
        print("Invalid input. Showing General Insights.")
        choice = len(analysis_options) - 1

    selected = analysis_options[choice] if 0 <= choice < len(analysis_options) else "General Insights"

    # Execute analysis based on selection
    if selected == "campaign_performance":
        campaign_performance(df)
    elif selected == "channel_analysis":
        channel_analysis(df)
    elif selected == "customer_segmentation":
        customer_segmentation(df)
    elif selected == "funnel_analysis":
        funnel_analysis(df)
    elif selected == "content_analysis":
        content_analysis(df)
    elif selected == "social_media":
        social_media(df)
    elif selected == "a/b_testing":
        ab_testing(df)
    elif selected == "marketing_campaign_performance_and_roi_analysis":
        marketing_campaign_performance_and_roi_analysis(df)
    elif selected == "customer_segmentation_and_campaign_response_analysis":
        customer_segmentation_and_campaign_response_analysis(df)
    elif selected == "email_marketing_campaign_effectiveness_analysis":
        email_marketing_campaign_effectiveness_analysis(df)
    elif selected == "sms_marketing_campaign_performance_analysis":
        sms_marketing_campaign_performance_analysis(df)
    elif selected == "multi-channel_campaign_attribution_analysis":
        multi_channel_campaign_attribution_analysis(df)
    elif selected == "daily_marketing_campaign_performance_tracking":
        daily_marketing_campaign_performance_tracking(df)
    elif selected == "campaign_reach_frequency_and_lift_analysis":
        campaign_reach_frequency_and_lift_analysis(df)
    elif selected == "customer_journey_and_touchpoint_analysis":
        customer_journey_and_touchpoint_analysis(df)
    elif selected == "coupon_redemption_and_usage_analysis":
        coupon_redemption_and_usage_analysis(df)
    elif selected == "a/b_test_creative_performance_analysis":
        ab_test_creative_performance_analysis(df)
    elif selected == "website_visitor_conversion_analysis":
        website_visitor_conversion_analysis(df)
    elif selected == "digital_advertising_platform_performance_and_roas_analysis":
        digital_advertising_platform_performance_and_roas_analysis(df)
    elif selected == "customer_satisfaction_survey_analysis":
        customer_satisfaction_survey_analysis(df)
    elif selected == "ad_placement_and_engagement_analysis":
        ad_placement_and_engagement_analysis(df)
    elif selected == "lead_generation_and_qualification_analysis":
        lead_generation_and_qualification_analysis(df)
    elif selected == "cross-device_campaign_conversion_analysis":
        cross_device_campaign_conversion_analysis(df)
    elif selected == "content_marketing_and_engagement_analysis":
        content_marketing_and_engagement_analysis(df)
    elif selected == "customer_preference_and_personalization_analysis":
        customer_preference_and_personalization_analysis(df)
    elif selected == "special_offer_and_discount_effectiveness_analysis":
        special_offer_and_discount_effectiveness_analysis(df)
    elif selected == "geotargeted_campaign_performance_analysis":
        geotargeted_campaign_performance_analysis(df)
    elif selected == "a/b_testing_campaign_variant_analysis":
        ab_testing_campaign_variant_analysis(df)
    elif selected == "media_buying_and_ad_performance_analysis":
        media_buying_and_ad_performance_analysis(df)
    elif selected == "product_line_marketing_campaign_analysis":
        product_line_marketing_campaign_analysis(df)
    elif selected == "website_engagement_and_ad_interaction_analysis":
        website_engagement_and_ad_interaction_analysis(df)
    elif selected == "daily_sales_revenue_and_campaign_correlation_analysis":
        daily_sales_revenue_and_campaign_correlation_analysis(df)
    elif selected == "influencer_marketing_campaign_performance_analysis":
        influencer_marketing_campaign_performance_analysis(df)
    elif selected == "ad_copy_performance_analysis":
        ad_copy_performance_analysis(df)
    elif selected == "customer_referral_program_analysis":
        customer_referral_program_analysis(df)
    elif selected == "website/platform_engagement_metrics_analysis":
        website_platform_engagement_metrics_analysis(df)
    elif selected == "customer_loyalty_program_engagement_analysis":
        customer_loyalty_program_engagement_analysis(df)
    elif selected == "discount_code_redemption_and_visit_analysis":
        discount_code_redemption_and_visit_analysis(df)
    elif selected == "seasonal_and_holiday_campaign_impact_analysis":
        seasonal_and_holiday_campaign_impact_analysis(df)
    elif selected == "video_marketing_engagement_analysis":
        video_marketing_engagement_analysis(df)
    elif selected == "search_engine_marketing_(sem)_keyword_performance":
        search_engine_marketing_sem_keyword_performance(df)
    elif selected == "churn_prediction_and_targeted_campaign_analysis":
        churn_prediction_and_targeted_campaign_analysis(df)
    elif selected == "geotargeting_accuracy_and_effectiveness_analysis":
        geotargeting_accuracy_and_effectiveness_analysis(df)
    elif selected == "newsletter_signup_attribution_analysis":
        newsletter_signup_attribution_analysis(df)
    elif selected == "marketing_budget_allocation_and_spend_analysis":
        marketing_budget_allocation_and_spend_analysis(df)
    elif selected == "social_media_competitive_and_sentiment_analysis":
        social_media_competitive_and_sentiment_analysis(df)
    elif selected == "customer_service_sentiment_and_feedback_analysis":
        customer_service_sentiment_and_feedback_analysis(df)
    elif selected == "rfm-based_customer_targeting_analysis":
        rfm_based_customer_targeting_analysis(df)
    elif selected == "webinar_performance_and_lead_generation_analysis":
        webinar_performance_and_lead_generation_analysis(df)
    elif selected == "event_marketing_effectiveness_analysis":
        event_marketing_effectiveness_analysis(df)
    elif selected == "display_ad_banner_placement_performance_analysis":
        display_ad_banner_placement_performance_analysis(df)
    elif selected == "affiliate_marketing_performance_and_revenue_analysis":
        affiliate_marketing_performance_and_revenue_analysis(df)
    elif selected == "clicked_link_position_and_device_analysis":
        clicked_link_position_and_device_analysis(df)
    elif selected == "dynamic_content_personalization_analysis":
        dynamic_content_personalization_analysis(df)
    elif selected == "remarketing_campaign_performance_analysis":
        remarketing_campaign_performance_analysis(df)
    elif selected == "ad_format_performance_and_cost_analysis":
        ad_format_performance_and_cost_analysis(df)
    else:
        print(f"Analysis option '{selected}' not recognized or not implemented.")
        show_general_insights(df)

if __name__ == "__main__":
    main()

   