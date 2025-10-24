import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, linregress
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
            print("Failed to decode file. Try another encoding.")
            return None
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            print("Unsupported file format")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def show_general_insights(df, title="General Insights"):
    print(f"--- {title} ---")
    
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Total Properties: {total_records}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

    if len(numeric_cols) > 0:
        print("\nNumeric Features Analysis:")
        for col in numeric_cols:
            print(f"\n--- Distribution of {col} ---")
            print(df[col].describe())
            fig1 = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig1.show()
            fig2 = px.box(df, y=col, title=f"Box Plot of {col}")
            fig2.show()
        
        if len(numeric_cols) >= 2:
            print("\n--- Feature Correlations ---")
            corr = df[numeric_cols].corr()
            print(corr)
            fig3 = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
            fig3.show()
    else:
        print("No numeric columns found for analysis.")
    
    if len(categorical_cols) > 0:
        print("\nCategorical Features Analysis:")
        for col in categorical_cols:
            print(f"\n--- Distribution of {col} ---")
            value_counts = df[col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']
            print(value_counts)
            fig4 = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Distribution of {col}")
            fig4.show()
    else:
        print("No categorical columns found for analysis.")

# Analysis Functions

def property_analysis(df):
    print("\n--- Property Analysis ---")
    expected = ['property_id', 'address', 'property_type', 'bedrooms', 
               'bathrooms', 'sqft', 'year_built', 'price', 'location']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist() # Simplified fuzzy matching to exact
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns}) # Simplified renaming

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df = df.dropna(subset=['price', 'sqft'])
    
    total_properties = len(df)
    avg_price = df['price'].mean()
    avg_sqft = df['sqft'].mean()
    avg_price_per_sqft = avg_price / avg_sqft if avg_sqft > 0 else 0
    
    print(f"Total Properties: {total_properties}")
    print(f"Average Price: ${avg_price:,.0f}")
    print(f"Average Sqft: {avg_sqft:,.0f}")
    print(f"Average Price/Sqft: ${avg_price_per_sqft:,.2f}")
    
    if 'property_type' in df.columns:
        fig1 = px.pie(df, names='property_type', title="Property Type Distribution")
        fig1.show()
    
    if 'price' in df.columns:
        fig2 = px.histogram(df, x='price', title="Price Distribution")
        fig2.show()
    
    if 'price' in df.columns and 'sqft' in df.columns:
        fig3 = px.scatter(df, x='sqft', y='price', color='property_type',
                         title="Price vs Square Footage", trendline="ols")
        fig3.show()

def location_analysis(df):
    print("\n--- Location Analysis ---")
    expected = ['property_id', 'address', 'latitude', 'longitude', 
               'neighborhood', 'zipcode', 'price']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})

    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['latitude', 'longitude', 'price'])
    
    unique_neighborhoods = df['neighborhood'].nunique() if 'neighborhood' in df.columns else 0
    unique_zipcodes = df['zipcode'].nunique() if 'zipcode' in df.columns else 0
    
    print(f"Total Properties: {len(df)}")
    print(f"Unique Neighborhoods: {unique_neighborhoods}")
    print(f"Unique Zipcodes: {unique_zipcodes}")
    
    if 'latitude' in df.columns and 'longitude' in df.columns and 'price' in df.columns:
        fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude',
                                color='price', size='price',
                                hover_name='address',
                                zoom=10, height=600,
                                title="Property Price Map")
        fig1.update_layout(mapbox_style="open-street-map")
        fig1.show()
    
    if 'neighborhood' in df.columns and 'price' in df.columns:
        neighborhood_prices = df.groupby('neighborhood')['price'].mean().sort_values(ascending=False).reset_index()
        fig2 = px.bar(neighborhood_prices.head(20), x='neighborhood', y='price',
                     title="Average Price by Neighborhood (Top 20)")
        fig2.show()

def price_trends_analysis(df):
    print("\n--- Price Trends Analysis ---")
    expected = ['property_id', 'date', 'price', 'property_type', 'sqft']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    
    if 'date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df = df.dropna(subset=['date', 'price'])
    
    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return

    df['month'] = df['date'].dt.to_period('M').astype(str)
    monthly_trends = df.groupby('month')['price'].agg(['mean', 'count']).reset_index()
    
    avg_price_change = (monthly_trends['mean'].pct_change() * 100).mean()
    total_sales = monthly_trends['count'].sum()
    
    print(f"Average Monthly Price Change: {avg_price_change:.1f}%")
    print(f"Total Sales in Period: {total_sales}")
    
    fig1 = px.line(monthly_trends, x='month', y='mean', title="Average Price Over Time")
    fig1.show()
    
    if 'property_type' in df.columns:
        type_trends = df.groupby(['month', 'property_type'])['price'].mean().reset_index()
        fig2 = px.line(type_trends, x='month', y='price', color='property_type',
                       title="Price Trends by Property Type")
        fig2.show()

def rental_analysis(df):
    print("\n--- Rental Analysis ---")
    expected = ['property_id', 'address', 'property_type', 'bedrooms', 
               'bathrooms', 'sqft', 'monthly_rent', 'location', 'occupancy_status']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})

    df['monthly_rent'] = pd.to_numeric(df['monthly_rent'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df = df.dropna(subset=['monthly_rent'])
    
    total_rentals = len(df)
    avg_rent = df['monthly_rent'].mean()
    occupancy_rate = (df['occupancy_status'] == 'Occupied').mean() * 100 if 'occupancy_status' in df.columns else 0
    avg_rent_per_sqft = avg_rent / df['sqft'].mean() if 'sqft' in df.columns and df['sqft'].mean() > 0 else 0
    
    print(f"Total Rentals: {total_rentals}")
    print(f"Average Monthly Rent: ${avg_rent:,.0f}")
    print(f"Occupancy Rate: {occupancy_rate:.1f}%")
    print(f"Average Rent/Sqft: ${avg_rent_per_sqft:,.2f}")
    
    fig1 = px.histogram(df, x='monthly_rent', title="Monthly Rent Distribution")
    fig1.show()
    
    if 'bedrooms' in df.columns:
        fig2 = px.box(df, x='bedrooms', y='monthly_rent', title="Rent by Number of Bedrooms")
        fig2.show()

def investment_analysis(df):
    print("\n--- Investment Analysis ---")
    expected = ['property_id', 'purchase_price', 'purchase_date', 
               'current_value', 'rental_income', 'expenses', 'location']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})

    df['purchase_price'] = pd.to_numeric(df['purchase_price'], errors='coerce')
    df['current_value'] = pd.to_numeric(df['current_value'], errors='coerce')
    df['rental_income'] = pd.to_numeric(df['rental_income'], errors='coerce')
    df['expenses'] = pd.to_numeric(df['expenses'], errors='coerce')
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')

    df = df.dropna(subset=['purchase_price', 'current_value'])
    
    total_investment = df['purchase_price'].sum()
    current_value_total = df['current_value'].sum()
    total_roi = ((current_value_total - total_investment) / total_investment) * 100 if total_investment > 0 else 0
    annual_rental_income = df['rental_income'].sum() * 12 if 'rental_income' in df.columns else 0
    
    print(f"Total Investment: ${total_investment:,.0f}")
    print(f"Current Value: ${current_value_total:,.0f}")
    print(f"Total ROI: {total_roi:.1f}%")
    print(f"Annual Rental Income: ${annual_rental_income:,.0f}")
    
    if 'purchase_price' in df.columns and 'current_value' in df.columns:
        df['roi'] = ((df['current_value'] - df['purchase_price']) / df['purchase_price']) * 100
        fig1 = px.histogram(df, x='roi', title="Return on Investment Distribution")
        fig1.show()
    
    if 'purchase_date' in df.columns and 'purchase_price' in df.columns and 'current_value' in df.columns:
        df['year'] = df['purchase_date'].dt.year
        yearly_performance = df.groupby('year').agg(
            purchase_price=('purchase_price', 'sum'),
            current_value=('current_value', 'sum')
        ).reset_index()
        yearly_performance['roi'] = ((yearly_performance['current_value'] - yearly_performance['purchase_price']) / 
                                     yearly_performance['purchase_price']) * 100
        
        fig2 = px.line(yearly_performance, x='year', y='roi', title="Investment Performance by Purchase Year")
        fig2.show()

def market_comparison_analysis(df):
    print("\n--- Market Comparison Analysis ---")
    expected = ['property_id', 'price', 'sqft', 'bedrooms', 
               'bathrooms', 'property_type', 'location']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')
    df = df.dropna(subset=['price', 'sqft'])
    
    if 'price' in df.columns and 'sqft' in df.columns:
        df['price_per_sqft'] = df['price'] / df['sqft']
    
    # Placeholder for user choice as this is not an interactive Streamlit app
    compare_by = 'property_type' # Default for non-interactive mode. Can be 'bedrooms', 'bathrooms', 'location'
    if compare_by not in df.columns:
        print(f"Warning: '{compare_by}' column not found. Defaulting comparison to 'property_type' if available.")
        if 'property_type' in df.columns:
            compare_by = 'property_type'
        elif 'bedrooms' in df.columns:
            compare_by = 'bedrooms'
        else:
            print("No suitable column for comparison found. Skipping comparison charts.")
            return

    # Filter data (no multiselect in non-interactive mode)
    # Assuming full dataset is used for analysis
    
    if 'price' in df.columns and 'price_per_sqft' in df.columns and compare_by in df.columns:
        fig1 = px.box(df, x=compare_by, y='price',
                     title=f"Price Distribution by {compare_by}")
        fig1.show()
        
        fig2 = px.box(df, x=compare_by, y='price_per_sqft',
                     title=f"Price per Sqft by {compare_by}")
        fig2.show()

def affordability_analysis(df):
    print("\n--- Affordability Analysis ---")
    expected = ['property_id', 'price', 'bedrooms', 'bathrooms', 
               'sqft', 'property_type', 'location']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['sqft'] = pd.to_numeric(df['sqft'], errors='coerce')
    df = df.dropna(subset=['price'])

    # Default values for affordability calculation (no user input in non-interactive mode)
    annual_income = 100000
    down_payment_pct = 20
    interest_rate = 5.0
    loan_term = 30
    dti_ratio = 36
    
    property_tax_rate = 1.2
    insurance_rate = 0.5
    hoa_fees = 0
    
    down_payment = down_payment_pct / 100
    monthly_income = annual_income / 12
    max_monthly_payment = monthly_income * (dti_ratio / 100)
    
    monthly_interest_rate = (interest_rate / 100) / 12
    loan_term_months = loan_term * 12
    
    def calculate_max_price(monthly_payment):
        if monthly_interest_rate == 0: # Handle 0 interest rate to avoid division by zero
            loan_amount = monthly_payment * loan_term_months
        else:
            loan_amount = (monthly_payment * (1 - (1 + monthly_interest_rate) ** -loan_term_months)) / monthly_interest_rate
        return loan_amount / (1 - down_payment)
        
    max_affordable_price = calculate_max_price(max_monthly_payment)
    
    affordable_properties = df[df['price'] <= max_affordable_price]
    num_affordable = len(affordable_properties)
    pct_affordable = (num_affordable / len(df)) * 100 if len(df) > 0 else 0
    
    print("\nAffordability Summary:")
    print(f"With an annual income of ${annual_income:,.0f} and a {down_payment_pct}% down payment:")
    print(f"Maximum affordable home price: ${max_affordable_price:,.0f}")
    print(f"Affordable properties in market: {num_affordable} ({pct_affordable:.1f}% of total)")
    
    if num_affordable > 0:
        print("\nTop 10 Affordable Properties:")
        print(affordable_properties.head(10))
    else:
        print("No properties in this dataset are affordable with the current criteria.")

def boston_housing_price_prediction_analysis(df):
    print("\n--- Boston Housing Price Prediction Analysis ---")
    expected = ['crim', 'zn', 'indus', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'b', 'lstat', 'medv']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist() # Simplified matching
    missing = [col for col in expected if col not in df.columns] # Re-check columns after renaming
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    median_value = df['medv'].median() * 1000
    avg_rooms = df['rm'].mean()
    lstat_corr = df['lstat'].corr(df['medv'])

    print(f"Median Home Value: ${median_value:,.0f}")
    print(f"Average Rooms per Dwelling: {avg_rooms:.2f}")
    print(f"% Lower Status/Value Correlation: {lstat_corr:.2f}")

    fig1 = px.scatter(df, x='rm', y='medv', color='crim',
                      title="Median Value vs. Average Number of Rooms",
                      labels={'rm': 'Average Rooms', 'medv': 'Median Value ($1000s)', 'crim': 'Crime Rate'})
    fig1.show()

    fig2 = px.scatter(df, x='lstat', y='medv', trendline='ols',
                      title="Median Value vs. % Lower Status of Population",
                      labels={'lstat': '% Lower Status Population', 'medv': 'Median Value ($1000s)'})
    fig2.show()

def real_estate_listing_description_and_time_on_market_analysis(df):
    print("\n--- Real Estate Listing Description and Time-on-Market Analysis ---")
    expected = ['full_description', 'deal_days']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    df['deal_days'] = pd.to_numeric(df['deal_days'], errors='coerce')
    df.dropna(subset=['full_description', 'deal_days'], inplace=True)
    df['desc_length'] = df['full_description'].str.len()

    avg_days_on_market = df['deal_days'].mean()
    avg_desc_length = df['desc_length'].mean()
    corr = df['desc_length'].corr(df['deal_days'])

    print(f"Average Days on Market: {avg_days_on_market:.1f}")
    print(f"Average Description Length: {avg_desc_length:.0f}")
    print(f"Length/Days Correlation: {corr:.2f}")

    fig1 = px.histogram(df, x='deal_days', nbins=50, title="Distribution of Days on Market")
    fig1.show()

    fig2 = px.scatter(df, x='desc_length', y='deal_days', trendline='ols',
                      title="Days on Market vs. Description Length")
    fig2.show()

def property_valuation_zestimate_and_feature_analysis(df):
    print("\n--- Property Valuation (Zestimate) and Feature Analysis ---")
    expected = ['zestimate', 'bedroom_number', 'bathroom_number', 'price_per_unit', 'living_space', 'property_type']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['zestimate', 'bedroom_number', 'bathroom_number', 'price_per_unit', 'living_space']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_zestimate = df['zestimate'].mean()
    avg_price_per_unit = df['price_per_unit'].mean()
    
    print(f"Average Zestimate: ${avg_zestimate:,.0f}")
    print(f"Average Price per Unit: ${avg_price_per_unit:,.0f}")

    fig1 = px.scatter(df, x='living_space', y='zestimate', color='property_type',
                      title="Zestimate vs. Living Space by Property Type")
    fig1.show()

    fig2 = px.box(df, x='bedroom_number', y='zestimate', title="Zestimate Distribution by Number of Bedrooms")
    fig2.show()

def real_estate_transaction_analysis_by_area_and_furnishing(df):
    print("\n--- Real Estate Transaction Analysis by Area and Furnishing ---")
    expected = ['area', 'furnishing', 'transaction']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    df['area'] = pd.to_numeric(df['area'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['transaction', 'furnishing']).agg(
        count=('area', 'count'),
        avg_area=('area', 'mean')
    ).reset_index()
    
    print("\nTransaction Summary:")
    print(summary)
    
    fig1 = px.sunburst(df, path=['transaction', 'furnishing'], title="Hierarchical View of Transactions")
    fig1.show()
    
    fig2 = px.box(df, x='furnishing', y='area', color='transaction', title="Area Distribution by Furnishing and Transaction Type")
    fig2.show()

def real_estate_price_prediction_based_on_location_and_features(df):
    print("\n--- Real Estate Price Prediction (Location & Features) ---")
    expected = ['house_age', 'distance_to_the_nearest_mrt_station', 'number_of_convenience_stores', 'latitude', 'longitude', 'house_price_of_unit_area']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_price = df['house_price_of_unit_area'].mean()
    
    print(f"Average House Price of Unit Area: {avg_price:.2f}")

    if 'latitude' in df.columns and 'longitude' in df.columns and 'house_price_of_unit_area' in df.columns:
        fig1 = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='house_price_of_unit_area', size='house_price_of_unit_area',
                                 hover_name='house_price_of_unit_area', zoom=10, height=600, title="Geospatial Price Distribution")
        fig1.update_layout(mapbox_style="open-street-map")
        fig1.show()
    
    fig2 = px.scatter(df, x='distance_to_the_nearest_mrt_station', y='house_price_of_unit_area',
                      color='number_of_convenience_stores', title="Price vs. Distance to MRT")
    fig2.show()

def property_sales_and_assessment_ratio_analysis(df):
    print("\n--- Property Sales and Assessment Ratio Analysis ---")
    expected = ['town', 'assessed_value', 'sale_amount', 'sales_ratio', 'property_type', 'residential_type']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['assessed_value', 'sale_amount', 'sales_ratio']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_ratio = df['sales_ratio'].mean()
    print(f"Average Sales Ratio (Sale Amount / Assessed Value): {avg_ratio:.3f}")

    ratio_by_town = df.groupby('town')['sales_ratio'].mean().nlargest(20).reset_index()
    fig1 = px.bar(ratio_by_town, x='town', y='sales_ratio', title="Top 20 Towns by Average Sales Ratio")
    fig1.show()
    
    fig2 = px.scatter(df, x='assessed_value', y='sale_amount', color='property_type',
                     title="Sale Amount vs. Assessed Value", log_x=True, log_y=True)
    fig2.show()

def neighborhood_property_characteristics_analysis(df):
    print("\n--- Neighborhood Property Characteristics Analysis ---")
    expected = ['beds', 'size', 'baths', 'neighborhood']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['beds', 'size', 'baths']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    summary = df.groupby('neighborhood').agg(
        avg_beds=('beds', 'mean'),
        avg_baths=('baths', 'mean'),
        avg_size=('size', 'mean'),
        property_count=('size', 'count')
    ).reset_index()
    
    print("\nNeighborhood Summary:")
    print(summary.round(2))
    
    fig = px.scatter(summary, x='avg_size', y='property_count', color='neighborhood',
                     size='avg_beds', title="Property Count vs. Average Size by Neighborhood")
    fig.show()

def house_price_prediction_based_on_property_features(df):
    print("\n--- House Price Prediction based on Property Features ---")
    expected = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'parking', 'furnishingstatus']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    median_price = df['price'].median()
    avg_area = df['area'].mean()
    price_area_corr = df['area'].corr(df['price'])
    
    print(f"Median Price: ${median_price:,.0f}")
    print(f"Average Area (sqft): {avg_area:,.0f}")
    print(f"Price/Area Correlation: {price_area_corr:.2f}")
    
    fig1 = px.scatter(df, x='area', y='price', color='furnishingstatus',
                     title="Price vs. Area by Furnishing Status")
    fig1.show()
    
    fig2 = px.box(df, x='bedrooms', y='price', color='stories', title="Price by Bedrooms and Stories")
    fig2.show()

def property_sales_and_appraisal_data_analysis(df):
    print("\n--- Property Sales and Appraisal Data Analysis ---")
    expected = ['saledate', 'totalappraisedvalue', 'totalfinishedarea', 'livingunits', 'xrprimaryneighborhoodid']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    for col in ['totalappraisedvalue', 'totalfinishedarea', 'livingunits']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_appraisal = df['totalappraisedvalue'].mean()
    print(f"Average Appraised Value: ${avg_appraisal:,.0f}")

    appraisal_over_time = df.groupby(df['saledate'].dt.to_period('Y').astype(str))['totalappraisedvalue'].mean().reset_index()
    fig1 = px.bar(appraisal_over_time, x='saledate', y='totalappraisedvalue', title="Average Appraised Value Over Years")
    fig1.show()
    
    fig2 = px.scatter(df, x='totalfinishedarea', y='totalappraisedvalue', color='xrprimaryneighborhoodid',
                     title="Appraised Value vs. Finished Area by Neighborhood")
    fig2.show()

def real_estate_pricing_and_feature_analysis(df):
    print("\n--- Real Estate Pricing and Feature Analysis ---")
    expected = ['rate', 'carpet_area', 'floor', 'bedroom', 'bathroom', 'parking', 'ownership']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['rate', 'carpet_area', 'bedroom', 'bathroom']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_rate = df['rate'].mean()
    print(f"Average Rate: ${avg_rate:,.0f}")
    
    fig1 = px.scatter(df, x='carpet_area', y='rate', color='ownership',
                     title="Rate vs. Carpet Area by Ownership Type")
    fig1.show()
    
    rate_by_bedroom = df.groupby('bedroom')['rate'].mean().reset_index()
    fig2 = px.bar(rate_by_bedroom, x='bedroom', y='rate', title="Average Rate by Number of Bedrooms")
    fig2.show()

def property_listing_time_on_market_analysis(df):
    print("\n--- Property Listing Time-on-Market Analysis ---")
    expected = ['bedrooms', 'bathrooms', 'square_feet', 'days_on_market', 'property_type']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['bedrooms', 'bathrooms', 'square_feet', 'days_on_market']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_dom = df['days_on_market'].mean()
    print(f"Average Days on Market: {avg_dom:.1f}")
    
    dom_by_type = df.groupby('property_type')['days_on_market'].mean().reset_index()
    fig1 = px.bar(dom_by_type, x='property_type', y='days_on_market', title="Average Days on Market by Property Type")
    fig1.show()
    
    fig2 = px.scatter(df, x='square_feet', y='days_on_market', title="Days on Market vs. Square Feet")
    fig2.show()

def real_estate_sales_price_analysis(df):
    print("\n--- Real Estate Sales Price Analysis ---")
    expected = ['list_price', 'sale_price', 'bedrooms', 'bathrooms', 'square_footage', 'year_built']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['sale_to_list_ratio'] = df['sale_price'] / df['list_price']
    avg_ratio = df['sale_to_list_ratio'].mean()
    print(f"Average Sale-to-List Price Ratio: {avg_ratio:.3f}")
    
    fig1 = px.scatter(df, x='list_price', y='sale_price', title="Sale Price vs. List Price")
    fig1.add_shape(type='line', x0=df['list_price'].min(), y0=df['list_price'].min(), 
                   x1=df['list_price'].max(), y1=df['list_price'].max(),
                   line=dict(color='red', dash='dash'))
    fig1.show()

    fig2 = px.scatter(df, x='square_footage', y='sale_price', color='sale_to_list_ratio',
                     title="Sale Price vs. Square Footage (Colored by Sale-to-List Ratio)")
    fig2.show()

def neighborhood_property_sales_trend_analysis(df):
    print("\n--- Neighborhood Property Sales Trend Analysis ---")
    expected = ['neighborhood', 'area_sqft', 'bedrooms', 'sale_date', 'sale_price']
    matched = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in ['neighborhood', 'sale_date', 'sale_price'] if col not in matched]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    df.dropna(inplace=True)
    
    df['sale_year'] = df['sale_date'].dt.year
    price_over_time = df.groupby(['sale_year', 'neighborhood'])['sale_price'].mean().reset_index()
    
    fig = px.line(price_over_time, x='sale_year', y='sale_price', color='neighborhood',
                  title="Average Sale Price Over Time by Neighborhood")
    fig.show()

def property_listing_price_vs_final_sale_price_analysis(df):
    print("\n--- Property Listing Price vs. Final Sale Price Analysis ---")
    expected = ['list_price', 'final_price', 'beds', 'baths', 'living_area']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['negotiation_diff'] = df['list_price'] - df['final_price']
    df['negotiation_perc'] = (df['negotiation_diff'] / df['list_price']) * 100
    
    avg_negotiation_perc = df['negotiation_perc'].mean()
    print(f"Average Negotiation Discount: {avg_negotiation_perc:.2f}%")
    
    fig1 = px.histogram(df, x='negotiation_perc', title="Distribution of Negotiation Percentage")
    fig1.show()
    
    fig2 = px.scatter(df, x='living_area', y='negotiation_perc',
                     title="Negotiation % vs. Living Area")
    fig2.show()

def housing_market_analysis_by_postal_code(df):
    print("\n--- Housing Market Analysis by Postal Code ---")
    expected = ['postal_code', 'asking_price', 'closed_price', 'square_feet', 'date_listed']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    for col in ['asking_price', 'closed_price', 'square_feet']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['price_per_sqft'] = df['closed_price'] / df['square_feet']
    
    summary = df.groupby('postal_code').agg(
        avg_price=('closed_price', 'mean'),
        avg_price_sqft=('price_per_sqft', 'mean'),
        num_listings=('postal_code', 'count')
    ).reset_index()
    
    print("\nMarket Summary by Postal Code:")
    print(summary.round(2))
    
    # In a non-Streamlit context, user selection for ZIP codes is not dynamic.
    # We'll just show for a few dominant ones or a random sample if many.
    if len(summary['postal_code'].unique()) > 5:
        sample_zips = summary['postal_code'].unique()[:5]
        filtered_summary = summary[summary['postal_code'].isin(sample_zips)]
    else:
        filtered_summary = summary.copy()

    fig = px.scatter(filtered_summary, x='avg_price', y='avg_price_sqft', size='num_listings',
                     color='postal_code', title="Avg. Price vs. Avg. Price/SqFt by Postal Code (Sample)")
    fig.show()

def residential_property_feature_and_price_analysis(df):
    print("\n--- Residential Property Feature and Price Analysis ---")
    expected = ['saleprice', 'bedcount', 'bathcount', 'floorarea', 'landarea', 'yearbuilt']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    fig1 = px.scatter(df, x='floorarea', y='saleprice', color='bedcount',
                     title="Sale Price vs. Floor Area (Colored by Bedroom Count)")
    fig1.show()
    
    if 'landarea' in df.columns and 'yearbuilt' in df.columns:
        fig2 = px.scatter_3d(df, x='floorarea', y='landarea', z='saleprice', color='yearbuilt',
                             title="3D View: Price by Floor Area, Land Area, and Year Built")
        fig2.show()

def county_level_real_estate_market_analysis(df):
    print("\n--- County-Level Real Estate Market Analysis ---")
    expected = ['county', 'price_usd', 'beds', 'baths', 'sqft', 'saledate']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    for col in ['price_usd', 'beds', 'baths', 'sqft']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('county').agg(
        median_price=('price_usd', 'median'),
        avg_sqft=('sqft', 'mean'),
        num_sales=('price_usd', 'count')
    ).reset_index()
    
    print("\nMarket Summary by County:")
    print(summary.round(0))
    
    fig = px.bar(summary, x='county', y='median_price', color='avg_sqft',
                     title="Median Price by County (Colored by Average SqFt)")
    fig.show()

def real_estate_sales_data_analysis_by_realtor(df):
    print("\n--- Real Estate Sales Data Analysis by Realtor ---")
    expected = ['realtor', 'saledate', 'saleprice', 'listing_status']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v:k for k,v in {k:k for k in expected}.items() if v in df.columns})
    df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('realtor').agg(
        total_sales_value=('saleprice', 'sum'),
        num_sales=('saleprice', 'count'),
        avg_sale_price=('saleprice', 'mean')
    ).nlargest(15, 'total_sales_value').reset_index()
    
    print("\nTop 15 Realtors by Sales Volume:")
    print(summary.round(0))
    
    fig = px.bar(summary, x='realtor', y='total_sales_value', color='avg_sale_price',
                     title="Top Realtors by Total Sales Value (Colored by Average Sale Price)")
    fig.show()

def web_scraped_real_estate_listing_analysis(df):
    print("\n--- Web-Scraped Real Estate Listing Analysis ---")
    expected = ['beds', 'baths', 'area', 'lotsize', 'yearbuilt', 'agentname']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['beds', 'baths', 'area', 'lotsize', 'yearbuilt']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    top_agents = df['agentname'].value_counts().nlargest(15).reset_index()
    top_agents.columns = ['AgentName', 'ListingsCount'] # Rename columns for clarity in print/plot
    print("\nTop 15 Agents by Number of Listings:")
    print(top_agents)
    
    fig1 = px.bar(top_agents, x='AgentName', y='ListingsCount', title="Top 15 Agents by Number of Listings")
    fig1.show()
    
    fig2 = px.histogram(df, x='yearbuilt', title="Distribution of Property Construction Year")
    fig2.show()

def real_estate_market_analysis_by_agency_and_zip_code(df):
    print("\n--- Real Estate Market Analysis by Agency and ZIP Code ---")
    expected = ['zip_code', 'listing_price', 'sale_amount', 'agentcompany']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['listing_price', 'sale_amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby(['agentcompany', 'zip_code'])['sale_amount'].agg(['sum', 'count']).reset_index()
    
    fig = px.treemap(summary, path=[px.Constant("All Agencies"), 'agentcompany', 'zip_code'],
                     values='sum', color='count',
                     title="Sales Volume by Agency and ZIP Code (Color by Number of Sales)")
    fig.show()

def property_listing_and_sales_data_correlation(df):
    print("\n--- Property Listing and Sales Data Correlation ---")
    expected = ['listprice', 'soldprice', 'bedrooms', 'bathrooms', 'livingsqft', 'landsqft', 'yearbuilt']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    print("\n--- Correlation Matrix of Property Features ---")
    corr_matrix = df[expected].corr()
    print(corr_matrix)
    fig = px.imshow(corr_matrix, text_auto=True, title="Feature Correlation Heatmap")
    fig.show()

def neighborhood_based_property_price_analysis(df):
    print("\n--- Neighborhood-Based Property Price Analysis ---")
    expected = ['neighborhood', 'listprice', 'saleprice', 'beds', 'baths', 'livingarea']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['listprice', 'saleprice', 'beds', 'baths', 'livingarea']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('neighborhood')['saleprice'].agg(['mean', 'median', 'count']).reset_index()
    
    print("\nNeighborhood Summary (Sales Price):")
    print(summary.sort_values('median', ascending=False).round(2))
    
    fig = px.bar(summary.sort_values('median', ascending=False), 
                     x='neighborhood', y='median', color='count',
                     title="Median Sale Price by Neighborhood (Colored by Number of Sales)")
    fig.show()

def real_estate_market_dynamics_analysis(df):
    print("\n--- Real Estate Market Dynamics Analysis ---")
    expected = ['county', 'zip', 'list_price', 'sold_price', 'daysonmarket']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in ['list_price', 'sold_price', 'daysonmarket']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['sale_to_list_ratio'] = df['sold_price'] / df['list_price']
    
    summary = df.groupby('county').agg(
        avg_dom=('daysonmarket', 'mean'),
        avg_ratio=('sale_to_list_ratio', 'mean'),
        median_price=('sold_price', 'median')
    ).reset_index()
    
    print("\nMarket Dynamics Summary by County:")
    print(summary.round(2))
    
    fig = px.scatter(summary, x='avg_dom', y='avg_ratio', size='median_price',
                     color='county', title="Market Dynamics by County (Avg DOM vs. Avg Sale/List Ratio, Sized by Median Price)")
    fig.show()

def property_sales_price_vs_list_price_analysis(df):
    print("\n--- Property Sales Price vs. List Price Analysis ---")
    expected = ['pricesold', 'pricelist', 'beds', 'baths']
    df, missing = check_and_rename_columns(df, {k:[k] for k in expected})[0].columns.tolist()
    missing = [col for col in expected if col not in df.columns]
    
    if missing:
        show_missing_columns_warning(missing)
        show_general_insights(df, "General Analysis")
        return
    
    df = df.rename(columns={v: k for k, v in {k: k for k in expected}.items() if v in df.columns})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['ratio'] = df['pricesold'] / df['pricelist']
    
    print(f"Average Sale-to-List Ratio: {df['ratio'].mean():.3f}")
    
    fig = px.histogram(df, x='ratio', title="Distribution of Sale-to-List Price Ratios")
    fig.show()

def zip_code_level_housing_market_trend_analysis(df):
    print("\n--- ZIP Code-Level Housing Market Trend Analysis ---")
    expected = ['zip', 'beds', 'baths', 'livingspace', 'saledate', 'saleprice']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy() # Use the potentially renamed DataFrame

    if any(col not in df.columns for col in ['zip', 'saledate', 'saleprice']):
        missing_critical = [col for col in ['zip', 'saledate', 'saleprice'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
    df['zip'] = df['zip'].astype(str)
    df.dropna(subset=['saledate', 'saleprice', 'zip'], inplace=True)
    
    df['sale_year_month'] = df['saledate'].dt.to_period('M').astype(str)
    summary = df.groupby(['sale_year_month', 'zip'])['saleprice'].median().reset_index()
    
    # Non-interactive selection: just show for the top N ZIPs by sales volume
    top_zips = df['zip'].value_counts().nlargest(3).index.tolist()
    filtered_df = summary[summary['zip'].isin(top_zips)]

    print(f"\nMedian Sale Price Trend for Top {len(top_zips)} ZIP Codes:")
    
    fig = px.line(filtered_df, x='sale_year_month', y='saleprice', color='zip',
                     title="Median Sale Price Trend by ZIP Code")
    fig.show()

def county_level_housing_price_and_feature_analysis(df):
    print("\n--- County-Level Housing Price and Feature Analysis ---")
    expected = ['county', 'zip', 'listprice', 'closedprice', 'bedrooms', 'bathrooms', 'sqft']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['county', 'closedprice', 'sqft', 'bedrooms']):
        missing_critical = [col for col in ['county', 'closedprice', 'sqft', 'bedrooms'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    for col in ['listprice', 'closedprice', 'bedrooms', 'bathrooms', 'sqft']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    summary = df.groupby('county').agg(
        median_price=('closedprice', 'median'),
        avg_sqft=('sqft', 'mean'),
        avg_beds=('bedrooms', 'mean')
    ).reset_index()
    
    print("\nCounty-Level Housing Summary:")
    print(summary.round(2))
    
    fig = px.scatter(summary, x='avg_sqft', y='median_price', size='avg_beds',
                     color='county', title="Median Price vs. Avg. SqFt by County (Sized by Avg. Bedrooms)")
    fig.show()

def real_estate_listing_duration_and_price_analysis(df):
    print("\n--- Real Estate Listing Duration and Price Analysis ---")
    expected = ['zipcode', 'askingprice', 'finalsaleprice', 'dayslisted']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['askingprice', 'finalsaleprice', 'dayslisted']):
        missing_critical = [col for col in ['askingprice', 'finalsaleprice', 'dayslisted'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    for col in ['askingprice', 'finalsaleprice', 'dayslisted']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['price_diff_perc'] = (df['askingprice'] - df['finalsaleprice']) / df['askingprice'] * 100
    
    print(f"\nAverage Price Reduction Percentage: {df['price_diff_perc'].mean():.2f}%")
    print(f"Average Days Listed: {df['dayslisted'].mean():.1f}")
    
    fig = px.scatter(df, x='dayslisted', y='price_diff_perc',
                     title="Price Reduction % vs. Days Listed")
    fig.show()

def agency_performance_in_real_estate_sales(df):
    print("\n--- Agency Performance in Real Estate Sales ---")
    expected = ['agency', 'pricelist', 'pricesold']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['agency', 'pricelist', 'pricesold']):
        missing_critical = [col for col in ['agency', 'pricelist', 'pricesold'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    for col in ['pricelist', 'pricesold']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['sale_to_list_ratio'] = df['pricesold'] / df['pricelist']
    
    summary = df.groupby('agency').agg(
        total_volume=('pricesold', 'sum'),
        num_sales=('pricesold', 'count'),
        avg_ratio=('sale_to_list_ratio', 'mean')
    ).nlargest(15, 'total_volume').reset_index()
    
    print("\nTop 15 Agencies by Sales Volume:")
    print(summary.round(2))
    
    fig = px.bar(summary, x='agency', y='total_volume', color='avg_ratio',
                     title="Top 15 Agencies by Sales Volume (Colored by Average Sale-to-List Ratio)")
    fig.show()

def real_estate_transaction_and_status_analysis(df):
    print("\n--- Real Estate Transaction and Status Analysis ---")
    expected = ['addr', 'zip', 'listprice', 'saleprice', 'mls_status', 'closingdate']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['mls_status']):
        missing_critical = [col for col in ['mls_status'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    status_counts = df['mls_status'].value_counts().reset_index()
    status_counts.columns = ['MLS_Status', 'Count'] # Renaming for clarity in print/plot
    print("\nDistribution of MLS Statuses:")
    print(status_counts)

    fig = px.pie(status_counts, names='MLS_Status', values='Count', title="Distribution of MLS Statuses")
    fig.show()

def property_sales_data_analysis_by_location(df):
    print("\n--- Property Sales Data Analysis by Location ---")
    expected = ['fulladdress', 'zipcode', 'listingprice', 'soldprice', 'squarefeet']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['zipcode', 'soldprice', 'squarefeet']):
        missing_critical = [col for col in ['zipcode', 'soldprice', 'squarefeet'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Sales Analysis")
        return
    
    for col in ['listingprice', 'soldprice', 'squarefeet']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    df['price_per_sqft'] = df['soldprice'] / df['squarefeet']
    
    summary = df.groupby('zipcode')['price_per_sqft'].median().reset_index()
    
    print("\nMedian Price per Square Foot by ZIP Code:")
    print(summary.sort_values('price_per_sqft', ascending=False).round(2))
    
    fig = px.bar(summary.sort_values('price_per_sqft', ascending=False).head(20), # Showing top 20 for brevity
                     x='zipcode', y='price_per_sqft', title="Median Price per Square Foot by ZIP Code (Top 20)")
    fig.show()

def neighborhood_specific_real_estate_market_analysis(df):
    print("\n--- Neighborhood-Specific Real Estate Market Analysis ---")
    expected = ['neighborhood', 'list_price', 'sale_price', 'beds', 'baths', 'livingarea']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['neighborhood', 'sale_price', 'livingarea', 'beds']):
        missing_critical = [col for col in ['neighborhood', 'sale_price', 'livingarea', 'beds'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    for col in ['list_price', 'sale_price', 'beds', 'baths', 'livingarea']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    # Non-interactive selection: just pick the first neighborhood or a prominent one
    if not df['neighborhood'].empty:
        neighborhood = df['neighborhood'].iloc[0]
        print(f"\nAnalyzing market for: {neighborhood}")
        df_hood = df[df['neighborhood'] == neighborhood]
    else:
        print("No neighborhoods available for analysis.")
        return

    print(f"\nMarket Snapshot for: {neighborhood}")
    print(f"Median Sale Price: ${df_hood['sale_price'].median():,.0f}")
    print(f"Average Living Area: {df_hood['livingarea'].mean():,.0f} sqft")
    print(f"Number of Sales: {len(df_hood)}")
    
    fig = px.scatter(df_hood, x='livingarea', y='sale_price', color='beds',
                     title=f"Sale Price vs. Living Area in {neighborhood} (Colored by Bedrooms)")
    fig.show()

def real_estate_market_time_series_analysis(df):
    print("\n--- Real Estate Market Time-Series Analysis ---")
    expected = ['county', 'zipcode', 'listprice', 'saleprice', 'saledate']
    matched_cols, missing_cols = check_and_rename_columns(df, {k:[k] for k in expected})
    df = matched_cols.copy()

    if any(col not in df.columns for col in ['saledate', 'saleprice']):
        missing_critical = [col for col in ['saledate', 'saleprice'] if col not in df.columns]
        show_missing_columns_warning(missing_critical)
        show_general_insights(df, "General Analysis")
        return
    
    df['saledate'] = pd.to_datetime(df['saledate'], errors='coerce')
    df['saleprice'] = pd.to_numeric(df['saleprice'], errors='coerce')
    df.dropna(subset=['saledate', 'saleprice'], inplace=True)
    
    df['sale_month'] = df['saledate'].dt.to_period('M').astype(str)
    
    summary = df.groupby('sale_month').agg(
        median_price=('saleprice', 'median'),
        num_sales=('saleprice', 'count')
    ).reset_index()
    
    print("\nMarket Summary Over Time:")
    print(summary.round(2))
    
    fig1 = px.line(summary, x='sale_month', y='median_price', title="Median Sale Price Over Time")
    fig1.show()
    
    fig2 = px.bar(summary, x='sale_month', y='num_sales', title="Number of Sales Over Time")
    fig2.show()

# Main execution logic for the console application
def main():
    print("🏠 Real Estate Analytics Console Application")
    
    file_path = input("Enter path to your real estate data file (e.g., data.csv or data.xlsx): ")
    encoding = input("Enter file encoding (e.g., utf-8, latin1, cp1252), or press Enter for 'utf-8': ")
    if not encoding:
        encoding = 'utf-8'
    
    df = load_data(file_path, encoding=encoding)
    
    if df is None:
        print("Could not load data. Please check the file path and encoding. Exiting.")
        return

    print("\nData loaded successfully!")
    print(f"First 5 rows of the loaded data:\n{df.head()}")

    analysis_options_map = {
        "property_analysis": property_analysis,
        "location_analysis": location_analysis,
        "price_trends_analysis": price_trends_analysis,
        "rental_analysis": rental_analysis,
        "investment_analysis": investment_analysis,
        "market_comparison": market_comparison_analysis,
        "affordability_analysis": affordability_analysis,
        "boston_housing_price_prediction_analysis": boston_housing_price_prediction_analysis,
        "real_estate_listing_description_and_time-on-market_analysis": real_estate_listing_description_and_time_on_market_analysis,
        "property_valuation_(zestimate)_and_feature_analysis": property_valuation_zestimate_and_feature_analysis,
        "real_estate_transaction_analysis_by_area_and_furnishing": real_estate_transaction_analysis_by_area_and_furnishing,
        "real_estate_price_prediction_based_on_location_and_features": real_estate_price_prediction_based_on_location_and_features,
        "property_sales_and_assessment_ratio_analysis": property_sales_and_assessment_ratio_analysis,
        "neighborhood_property_characteristics_analysis": neighborhood_property_characteristics_analysis,
        "house_price_prediction_based_on_property_features": house_price_prediction_based_on_property_features,
        "property_sales_and_appraisal_data_analysis": property_sales_and_appraisal_data_analysis,
        "real_estate_pricing_and_feature_analysis": real_estate_pricing_and_feature_analysis,
        "property_listing_time-on-market_analysis": property_listing_time_on_market_analysis,
        "real_estate_sales_price_analysis": real_estate_sales_price_analysis,
        "neighborhood_property_sales_trend_analysis": neighborhood_property_sales_trend_analysis,
        "property_listing_price_vs._final_sale_price_analysis": property_listing_price_vs_final_sale_price_analysis,
        "housing_market_analysis_by_postal_code": housing_market_analysis_by_postal_code,
        "residential_property_feature_and_price_analysis": residential_property_feature_and_price_analysis,
        "county-level_real_estate_market_analysis": county_level_real_estate_market_analysis,
        "real_estate_sales_data_analysis_by_realtor": real_estate_sales_data_analysis_by_realtor,
        "web-scraped_real_estate_listing_analysis": web_scraped_real_estate_listing_analysis,
        "real_estate_market_analysis_by_agency_and_zip_code": real_estate_market_analysis_by_agency_and_zip_code,
        "property_listing_and_sales_data_correlation": property_listing_and_sales_data_correlation,
        "neighborhood-based_property_price_analysis": neighborhood_based_property_price_analysis,
        "real_estate_market_dynamics_analysis": real_estate_market_dynamics_analysis,
        "property_sales_price_vs._list_price_analysis": property_sales_price_vs_list_price_analysis,
        "zip_code-level_housing_market_trend_analysis": zip_code_level_housing_market_trend_analysis,
        "county-level_housing_price_and_feature_analysis": county_level_housing_price_and_feature_analysis,
        "real_estate_listing_duration_and_price_analysis": real_estate_listing_duration_and_price_analysis,
        "agency_performance_in_real_estate_sales": agency_performance_in_real_estate_sales,
        "real_estate_transaction_and_status_analysis": real_estate_transaction_and_status_analysis,
        "property_sales_data_analysis_by_location": property_sales_data_analysis_by_location,
        "neighborhood-specific_real_estate_market_analysis": neighborhood_specific_real_estate_market_analysis,
        "real_estate_market_time-series_analysis": real_estate_market_time_series_analysis,
    }

    analysis_options_list = list(analysis_options_map.keys())
    analysis_options_list.sort() # Sort alphabetically for easier selection

    while True:
        print("\n--- Available Analyses ---")
        for i, option_key in enumerate(analysis_options_list):
            print(f"{i + 1}. {option_key.replace('_', ' ').title()}")
        print(f"{len(analysis_options_list) + 1}. General Insights (Overview of your data)")
        print("0. Exit")

        choice_str = input("Enter the number corresponding to your desired analysis: ")
        
        if choice_str == '0':
            print("Exiting application. Goodbye!")
            break

        try:
            choice_idx = int(choice_str) - 1
            if 0 <= choice_idx < len(analysis_options_list):
                selected_analysis_name = analysis_options_list[choice_idx]
                selected_function = analysis_options_map[selected_analysis_name]
                print(f"\n--- Running: {selected_analysis_name.replace('_', ' ').title()} ---")
                
                # Call the selected function with a copy of the DataFrame
                try:
                    selected_function(df.copy())
                except Exception as e:
                    print(f"An error occurred during the '{selected_analysis_name}' analysis: {e}")
                    import traceback
                    traceback.print_exc() # Print full traceback for debugging
            elif choice_idx == len(analysis_options_list): # General Insights
                show_general_insights(df.copy())
            else:
                print("Invalid selection. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for debugging

if __name__ == "__main__":
    main()