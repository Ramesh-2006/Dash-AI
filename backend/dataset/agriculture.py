import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process # Assuming fuzzywuzzy is installed and used by fuzzy_match_column

import warnings
warnings.filterwarnings('ignore')

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
    print("\n--- ⚠️ Required Columns Not Found ---")
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
        print("Available numeric features:")
        for i, col in enumerate(numeric_cols):
            print(f"{i}: {col}")
        try:
            selected_num_col_idx = int(input("Select numeric feature to analyze (enter index): "))
            selected_num_col = numeric_cols[selected_num_col_idx]

            fig1 = px.histogram(df, x=selected_num_col,
                                    title=f"Distribution of {selected_num_col}")
            fig1.show()

            fig2 = px.box(df, y=selected_num_col,
                                  title=f"Box Plot of {selected_num_col}")
            fig2.show()
        except (ValueError, IndexError):
            print("[WARNING] Invalid selection or no numeric column selected.")
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
        try:
            selected_cat_col_idx = int(input("Select categorical feature to analyze (enter index): "))
            selected_cat_col = categorical_cols[selected_cat_col_idx]

            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']

            fig4 = px.bar(value_counts.head(10), x='Value', y='Count',
                                  title=f"Distribution of {selected_cat_col}")
            fig4.show()
        except (ValueError, IndexError):
            print("[WARNING] Invalid selection or no categorical column selected.")
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
def crop_analysis(df):
    print("\n--- General Crop Analysis ---")
    # A broad list of expected columns for this category
    expected = ['crop', 'crop_type', 'yield_tons_per_hectare', 'yeilds', 'production', 'area', 'grain2020', 'rice_production', 'wheat_production_1000_tons']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find any standard crop-related columns like 'crop', 'yield', or 'production'.")
        show_general_insights(df, "General Analysis")
        return

    # --- Yield Analysis ---
    yield_col_name = matched.get('yield_tons_per_hectare') or matched.get('yeilds')
    crop_col_name = matched.get('crop') or matched.get('crop_type')
    if yield_col_name and crop_col_name:
        print("\n--- Yield by Crop Type ---")
        df[yield_col_name] = pd.to_numeric(df[yield_col_name], errors='coerce')
        avg_yield = df[yield_col_name].mean()
        top_crop = df.groupby(crop_col_name)[yield_col_name].mean().idxmax()
        print(f"Average Yield: {avg_yield:,.2f}")
        print(f"Highest Average Yield Crop: {top_crop}")

        yield_by_crop = df.groupby(crop_col_name)[yield_col_name].mean().sort_values(ascending=False)
        fig = px.bar(yield_by_crop, title="Average Yield by Crop", labels={'value': 'Average Yield', 'index': 'Crop'})
        fig.show()

    # --- Production & Area Analysis ---
    prod_col_name = matched.get('production')
    area_col_name = matched.get('area')
    if prod_col_name and area_col_name:
        print("\n--- Production and Area Insights ---")
        df[prod_col_name] = pd.to_numeric(df[prod_col_name], errors='coerce')
        df[area_col_name] = pd.to_numeric(df[area_col_name], errors='coerce')
        df['yield_calculated'] = df[prod_col_name] / df[area_col_name]

        fig = px.scatter(df, x=area_col_name, y=prod_col_name,
                         size='yield_calculated', color=crop_col_name,
                         title="Production vs. Area Planted")
        fig.show()

def soil_analysis(df):
    print("\n--- General Soil Analysis ---")
    expected = ['soil_type', 'ph', 'nitrogen', 'potassium', 'phosphorous', 'om_perc', 'soc', 'bulk_density', 'coneindex']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find any standard soil-related columns like 'soil_type', 'ph', or 'nitrogen'.")
        show_general_insights(df, "General Insights")
        return

    # --- Chemical Property Analysis ---
    chem_cols = ['ph', 'nitrogen', 'potassium', 'phosphorous', 'om_perc', 'soc']
    found_chem_cols = [matched[c] for c in chem_cols if matched.get(c)]

    if found_chem_cols:
        print("\n--- General Soil Chemistry Analysis ---")
        df_chem = df[found_chem_cols].copy()
        for col in df_chem.columns:
            df_chem[col] = pd.to_numeric(df_chem[col], errors='coerce')

        print(df_chem.describe().to_string()) # Use to_string() for full DataFrame printing

        corr_matrix = df_chem.corr(numeric_only=True)
        fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap of Soil Nutrients")
        fig.show()

    # --- Physical Property Analysis ---
    if matched.get('bulk_density') and matched.get('coneindex'):
        print("\n--- Soil Physical Properties Analysis ---")
        df['bulk_density'] = pd.to_numeric(df[matched['bulk_density']], errors='coerce')
        df['coneindex'] = pd.to_numeric(df[matched['coneindex']], errors='coerce')

        fig = px.scatter(df, x=matched['bulk_density'], y=matched['coneindex'],
                         title="Soil Compaction (Cone Index) vs. Bulk Density")
        fig.show()

def weather_impact_analysis(df):
    print("\n--- Weather Impact Analysis ---")
    expected = ['rainfall_mm', 'temperature_celsius', 'humidity', 'weather_condition', 'yield_tons_per_hectare', 'production']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find weather columns like 'rainfall_mm' or 'temperature_celsius'.")
        show_general_insights(df, "General Analysis")
        return

    target_col = matched.get('yield_tons_per_hectare') or matched.get('production')
    rain_col = matched.get('rainfall_mm')
    temp_col = matched.get('temperature_celsius')

    if not target_col:
        print("Could not find a target variable like 'yield' or 'production' to analyze against.")
        return

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')

    if rain_col:
        print("\n--- Impact of Rainfall ---")
        df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
        rain_corr = df[rain_col].corr(df[target_col])
        print(f"Rainfall / {target_col.replace('_', ' ').title()} Correlation: {rain_corr:.2f}")
        fig1 = px.scatter(df, x=rain_col, y=target_col, trendline='ols', title=f"Impact of Rainfall on {target_col.replace('_', ' ').title()}")
        fig1.show()

    if temp_col:
        print("\n--- Impact of Temperature ---")
        df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
        temp_corr = df[temp_col].corr(df[target_col])
        print(f"Temperature / {target_col.replace('_', ' ').title()} Correlation: {temp_corr:.2f}")
        fig2 = px.scatter(df, x=temp_col, y=target_col, trendline='ols', title=f"Impact of Temperature on {target_col.replace('_', ' ').title()}")
        fig2.show()

def irrigation_analysis(df):
    print("\n--- General Irrigation Analysis ---")
    expected = ['irrigation_used', 'irrigation_method', 'irrigation_amount', 'water_table_depth', 'yield_tons_per_hectare']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find any irrigation-related columns.")
        show_general_insights(df, "General Analysis")
        return

    if matched.get('irrigation_used') and matched.get('yield_tons_per_hectare'):
        print("\n--- Impact of Irrigation on Yield ---")
        df[matched['yield_tons_per_hectare']] = pd.to_numeric(df[matched['yield_tons_per_hectare']], errors='coerce')
        fig = px.box(df, x=matched['irrigation_used'], y=matched['yield_tons_per_hectare'],
                     title="Crop Yield for Irrigated vs. Non-Irrigated Plots")
        fig.show()

    if matched.get('irrigation_method') and matched.get('irrigation_amount'):
        print("\n--- Water Usage by Irrigation Method ---")
        df[matched['irrigation_amount']] = pd.to_numeric(df[matched['irrigation_amount']], errors='coerce')

        water_by_method = df.groupby(matched['irrigation_method']).agg(
            total_irrigation_amount=(matched['irrigation_amount'], 'sum')
        ).reset_index()

        fig = px.pie(water_by_method,
                     names=matched['irrigation_method'],
                     values='total_irrigation_amount',
                     title="Total Water Amount by Irrigation Method")
        fig.show()

def pest_disease_analysis(df):
    print("\n--- Pest & Disease Analysis ---")
    expected = ['estimated_insects_count', 'pesticide_use_category', 'crop_damage', 'larval_mortality_perc', 'mite_count_after_shake']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find any pest or disease-related columns.")
        show_general_insights(df, "General Analysis")
        return

    if matched.get('estimated_insects_count') and matched.get('crop_damage'):
        print("\n--- Insect Count vs. Crop Damage ---")
        df[matched['estimated_insects_count']] = pd.to_numeric(df[matched['estimated_insects_count']], errors='coerce')
        df[matched['crop_damage']] = df[matched['crop_damage']].astype(str)
        fig = px.box(df, x=matched['crop_damage'], y=matched['estimated_insects_count'],
                     title="Insect Count by Crop Damage Category (0=Alive, 1=Damaged, 2=Pesticide Damage)")
        fig.show()

    if matched.get('pesticide_use_category') and matched.get('estimated_insects_count'):
        print("\n--- Impact of Pesticide Use ---")
        df[matched['pesticide_use_category']] = df[matched['pesticide_use_category']].astype(str)
        fig = px.violin(df, x=matched['pesticide_use_category'], y=matched['estimated_insects_count'],
                        title="Insect Count by Pesticide Use (1=Never, 2=Previous, 3=Current)")
        fig.show()

def economic_analysis(df):
    print("\n--- General Economic Analysis ---")
    expected = ['cost_of_cultivation_hectare_c2', 'cost_of_production_quintal_c2', 'yield_quintal_hectare', 'price', 'revenue', 'profit']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find any economic columns like 'cost', 'price', or 'profit'.")
        show_general_insights(df, "General Analysis")
        return

    cost_col = matched.get('cost_of_cultivation_hectare_c2')
    yield_col = matched.get('yield_quintal_hectare')
    cost_prod_col = matched.get('cost_of_production_quintal_c2')

    if cost_col and yield_col and cost_prod_col:
        print("\n--- Crop Profitability Analysis ---")
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        df[yield_col] = pd.to_numeric(df[yield_col], errors='coerce')
        df[cost_prod_col] = pd.to_numeric(df[cost_prod_col], errors='coerce')

        # We can estimate revenue and profit from these columns
        df['estimated_revenue'] = df[yield_col] * df[cost_prod_col] * 1.2 # Assume 20% markup for price
        df['estimated_profit'] = df['estimated_revenue'] - df[cost_col]

        avg_cost = df[cost_col].mean()
        avg_revenue = df['estimated_revenue'].mean()
        avg_profit = df['estimated_profit'].mean()

        print(f"Average Cost per Hectare: ${avg_cost:,.0f}")
        print(f"Average Revenue per Hectare: ${avg_revenue:,.0f}")
        print(f"Average Profit per Hectare: ${avg_profit:,.0f}")

        fig = px.scatter(df, x=cost_col, y='estimated_profit', title="Profit vs. Cost of Cultivation")
        fig.show()

def sustainability_analysis(df):
    print("\n--- General Sustainability Analysis ---")
    expected = ['land_use', 'grazingland', 'cropland', 'tillage', 'no_till', 'om_perc', 'soc', 'total_n', 'nitrogen_elem']
    matched = fuzzy_match_column(df, expected)

    found_cols = {k:v for k,v in matched.items() if v}
    if not found_cols:
        print("Could not find sustainability-related columns like 'land_use', 'tillage', or 'om_perc'.")
        show_general_insights(df, "General Analysis")
        return

    if matched.get('grazingland') and matched.get('cropland'):
        print("\n--- Land Use Distribution ---")
        grazing_total = pd.to_numeric(df[matched['grazingland']], errors='coerce').sum()
        cropland_total = pd.to_numeric(df[matched['cropland']], errors='coerce').sum()
        pie_df = pd.DataFrame({'Land Type': ['Grazing Land', 'Crop Land'], 'Area': [grazing_total, cropland_total]})
        fig1 = px.pie(pie_df, names='Land Type', values='Area', title="Grazing vs. Cropland Area")
        fig1.show()

    if matched.get('tillage') and matched.get('soc'): # SOC = Soil Organic Carbon
        print("\n--- Impact of Tillage on Soil Organic Carbon ---")
        df[matched['soc']] = pd.to_numeric(df[matched['soc']], errors='coerce')
        soc_by_tillage = df.groupby(matched['tillage'])[matched['soc']].mean().reset_index()
        fig2 = px.bar(soc_by_tillage, x=matched['tillage'], y=matched['soc'],
                      title="Average Soil Organic Carbon by Tillage Practice")
        fig2.show()

def crop_yield_prediction_analysis(df):
    print("\n--- Crop Yield Prediction Analysis ---")
    expected = ['rainfall_mm', 'temperature_celsius', 'fertilizer_used', 'crop', 'yield_tons_per_hectare']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    # Convert data types
    for col in ['rainfall_mm', 'temperature_celsius', 'yield_tons_per_hectare']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    avg_yield = df['yield_tons_per_hectare'].mean()
    rainfall_corr = df['rainfall_mm'].corr(df['yield_tons_per_hectare'])
    temp_corr = df['temperature_celsius'].corr(df['yield_tons_per_hectare'])

    print(f"Average Yield (tons/ha): {avg_yield:.2f}")
    print(f"Rainfall/Yield Correlation: {rainfall_corr:.2f}")
    print(f"Temp/Yield Correlation: {temp_corr:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='rainfall_mm', y='yield_tons_per_hectare', color='crop',
                      title="Rainfall vs. Crop Yield",
                      labels={'rainfall_mm': 'Rainfall (mm)', 'yield_tons_per_hectare': 'Yield (tons/hectare)'})
    fig1.show()

    fig2 = px.scatter(df, x='temperature_celsius', y='yield_tons_per_hectare', color='crop',
                      title="Temperature vs. Crop Yield",
                      labels={'temperature_celsius': 'Temperature (°C)', 'yield_tons_per_hectare': 'Yield (tons/hectare)'})
    fig2.show()

def state_level_agricultural_production_trend_analysis(df):
    print("\n--- State-Level Agricultural Production Trend Analysis ---")
    expected = ['state_name', 'crop_year', 'crop', 'area', 'production']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    # Convert data types
    for col in ['crop_year', 'area', 'production']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    total_production = df['production'].sum()
    peak_year = df.groupby('crop_year')['production'].sum().idxmax()
    top_state = df.groupby('state_name')['production'].sum().idxmax()

    print(f"Total Production (all years): {total_production:,.0f}")
    print(f"Peak Production Year: {peak_year}")
    print(f"Top Producing State: {top_state}")

    # Visualizations
    production_over_time = df.groupby('crop_year')['production'].sum().reset_index()
    fig1 = px.line(production_over_time, x='crop_year', y='production',
                    title="Total Agricultural Production Over Time",
                    labels={'crop_year': 'Year', 'production': 'Total Production'})
    fig1.show()

    production_by_state = df.groupby('state_name')['production'].sum().sort_values(ascending=False).head(10).reset_index()
    fig2 = px.bar(production_by_state, x='state_name', y='production',
                    title="Top 10 Producing States",
                    labels={'state_name': 'State', 'production': 'Total Production'})
    fig2.show()

def comprehensive_county_level_agricultural_output_analysis(df):
    print("\n--- Comprehensive County-Level Agricultural Output Analysis ---")
    expected = ['crop_production', 'livestock_production', 'aquaculture', 'maize', 'rice', 'beans', 'indigenous_cattle', 'exotic_chicken_layers']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    # Convert to numeric
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Metrics
    total_crop_prod = df['crop_production'].sum()
    total_livestock_prod = df['livestock_production'].sum()
    total_aquaculture = df['aquaculture'].sum()

    print(f"Total Crop Production: {total_crop_prod:,.0f}")
    print(f"Total Livestock Production: {total_livestock_prod:,.0f}")
    print(f"Total Aquaculture: {total_aquaculture:,.0f}")

    # Visualizations
    output_summary = df[['crop_production', 'livestock_production', 'aquaculture']].sum().reset_index()
    output_summary.columns = ['Category', 'Total Production']
    fig1 = px.pie(output_summary, names='Category', values='Total Production',
                    title="Share of Agricultural Output by Category", hole=0.4)
    fig1.show()

    top_crops = df[['maize', 'rice', 'beans']].sum(numeric_only=True).sort_values(ascending=False).head(5)
    fig2 = px.bar(top_crops, x=top_crops.index, y=top_crops.values,
                    title="Top Crop Outputs", labels={'x': 'Crop', 'y': 'Total Production'})
    fig2.show()

def environmental_factor_analysis_for_rice_production(df):
    print("\n--- Environmental Factor Analysis for Rice Production ---")
    expected = ['annual_rain', 'nitrogen', 'potash', 'phosphate', 'loamy_alfisol', 'rice_production']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    avg_rice_prod = df['rice_production'].mean()
    rain_corr = df['annual_rain'].corr(df['rice_production'])
    nitrogen_corr = df['nitrogen'].corr(df['rice_production'])

    print(f"Average Rice Production: {avg_rice_prod:,.2f}")
    print(f"Rain/Production Correlation: {rain_corr:.2f}")
    print(f"Nitrogen/Production Correlation: {nitrogen_corr:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='annual_rain', y='rice_production',
                      title="Annual Rain vs. Rice Production", trendline='ols')
    fig1.show()

    # For this visualization, we assume loamy_alfisol is a binary/percentage indicator of that soil type
    fig2 = px.scatter(df, x='nitrogen', y='rice_production', color='loamy_alfisol',
                      title="Nitrogen Level vs. Rice Production by Loamy Alfisol Presence",
                      labels={'nitrogen': 'Nitrogen Level', 'rice_production': 'Rice Production'})
    fig2.show()

def district_level_rice_yield_and_soil_type_correlation_analysis(df):
    print("\n--- District-Level Rice Yield and Soil Type Analysis ---")
    expected = ['dist_name', 'rice_yield', 'annual_rain', 'nitrogen', 'vertisols']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    for col in ['rice_yield', 'annual_rain', 'nitrogen', 'vertisols']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    avg_yield = df['rice_yield'].mean()
    top_district = df.groupby('dist_name')['rice_yield'].mean().idxmax()
    avg_yield_vertisols = df[df['vertisols'] > 0.5]['rice_yield'].mean() # Assuming vertisols is a percentage

    print(f"Average Rice Yield (Kg/ha): {avg_yield:,.2f}")
    print(f"Top Yielding District: {top_district}")
    print(f"Avg. Yield in Vertisols: {avg_yield_vertisols:,.2f}")

    # Visualizations
    top_districts = df.groupby('dist_name')['rice_yield'].mean().sort_values(ascending=False).head(10)
    fig1 = px.bar(top_districts, title="Top 10 Districts by Average Rice Yield")
    fig1.show()

    fig2 = px.box(df, x='vertisols', y='rice_yield', title="Rice Yield Distribution by Vertisols Soil Presence")
    fig2.show()

def crop_type_recommendation_analysis(df):
    print("\n--- Crop Type Recommendation Analysis ---")
    expected = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    for col in ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    num_crops = df['label'].nunique()
    avg_temp = df['temperature'].mean()
    avg_rainfall = df['rainfall'].mean()

    print(f"Number of Crop Types: {num_crops}")
    print(f"Average Temperature (°C): {avg_temp:.1f}")
    print(f"Average Rainfall (mm): {avg_rainfall:.1f}")

    # Visualizations
    print("Average Environmental Conditions per Crop")
    crop_conditions = df.groupby('label')[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']].mean().reset_index()
    fig1 = px.parallel_coordinates(crop_conditions, color="label",
                                     title="Environmental Profiles for Different Crops",
                                     labels={'n': 'Nitrogen', 'p': 'Phosphorus', 'k': 'Potassium', 'label': 'Crop'})
    fig1.show()

    fig2 = px.box(df, x='label', y='rainfall', title="Rainfall Requirements by Crop")
    fig2.show()

def national_land_use_change_analysis(df):
    print("\n--- National Land Use Change Analysis ---")
    expected = ['country', 'year', 'land_use', 'grazingland', 'cropland']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    for col in ['year', 'grazingland', 'cropland']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    latest_year = df['year'].max()
    latest_cropland = df[df['year'] == latest_year]['cropland'].sum()
    latest_grazingland = df[df['year'] == latest_year]['grazingland'].sum()

    print(f"Most Recent Year: {latest_year}")
    print(f"Latest Cropland Area: {latest_cropland:,.0f}")
    print(f"Latest Grazingland Area: {latest_grazingland:,.0f}")

    # Visualizations
    df_melted = df.melt(id_vars=['country', 'year'], value_vars=['grazingland', 'cropland'],
                        var_name='land_type', value_name='area')
    fig1 = px.line(df_melted, x='year', y='area', color='land_type',
                    title="National Land Use Trends Over Time",
                    labels={'year': 'Year', 'area': 'Area (e.g., in sq km)', 'land_type': 'Land Use Type'})
    fig1.show()

    # User choice for country-specific data
    if len(df['country'].unique()) > 1:
        show_country_data = input("Show data by country? (yes/no): ").lower()
        if show_country_data == 'yes':
            country_select = input(f"Select a country ({', '.join(df['country'].unique())}): ")
            if country_select in df['country'].unique():
                country_data = df_melted[df_melted['country'] == country_select]
                fig2 = px.area(country_data, x='year', y='area', color='land_type',
                                 title=f"Land Use Trend for {country_select}")
                fig2.show()
            else:
                print("Invalid country selected.")

def agricultural_census_crop_production_analysis(df):
    print("\n--- Agricultural Census Crop Production Analysis ---")
    expected = ['census_year', 'county', 'type_of_crop', 'value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    df['census_year'] = pd.to_numeric(df['census_year'], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    total_value = df['value'].sum()
    latest_year = df['census_year'].max()
    top_crop = df.groupby('type_of_crop')['value'].sum().idxmax()

    print(f"Total Production Value: ${total_value:,.0f}")
    print(f"Latest Census Year: {latest_year}")
    print(f"Top Crop by Value: {top_crop}")

    # Visualizations
    value_by_year = df.groupby('census_year')['value'].sum().reset_index()
    fig1 = px.bar(value_by_year, x='census_year', y='value', title="Total Production Value by Census Year")
    fig1.show()

    value_by_crop = df.groupby('type_of_crop')['value'].sum().sort_values(ascending=False).head(10)
    fig2 = px.pie(value_by_crop, names=value_by_crop.index, values=value_by_crop.values,
                    title="Top 10 Crops by Production Value", hole=0.4)
    fig2.show()

def insect_population_estimation_analysis(df):
    print("\n--- Insect Population Estimation Analysis ---")
    expected = ['estimated_insects_count', 'crop_type', 'soil_type', 'pesticide_use_category', 'season']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    df['estimated_insects_count'] = pd.to_numeric(df['estimated_insects_count'], errors='coerce')
    df['pesticide_use_category'] = pd.to_numeric(df['pesticide_use_category'], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    avg_insect_count = df['estimated_insects_count'].mean()
    highest_infestation_crop = df.groupby('crop_type')['estimated_insects_count'].mean().idxmax()
    highest_infestation_season = df.groupby('season')['estimated_insects_count'].mean().idxmax()

    print(f"Avg. Estimated Insect Count: {avg_insect_count:,.0f}")
    print(f"Most Infested Crop: {highest_infestation_crop}")
    print(f"Most Infested Season: {highest_infestation_season}")

    # Visualizations
    fig1 = px.box(df, x='crop_type', y='estimated_insects_count', color='season',
                    title="Insect Counts by Crop Type and Season")
    fig1.show()

    fig2 = px.violin(df, x='pesticide_use_category', y='estimated_insects_count',
                     title="Insect Counts by Pesticide Use Category",
                     labels={'pesticide_use_category': 'Pesticide Use (1=Never, 2=Prev. Used, 3=Currently Using)'})
    fig2.show()

def crop_damage_prediction_from_pest_infestation(df):
    print("\n--- Crop Damage Prediction from Pest Infestation ---")
    expected = ['estimated_insects_count', 'crop_type', 'pesticide_use_category', 'number_doses_week', 'crop_damage']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    for col in ['estimated_insects_count', 'number_doses_week', 'crop_damage']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)
    df['crop_damage_label'] = df['crop_damage'].map({0: 'Alive', 1: 'Damaged (Other)', 2: 'Damaged (Pesticide)'})

    # Metrics
    damage_rate = (df['crop_damage'] > 0).mean() * 100
    insect_damage_corr = df['estimated_insects_count'].corr(df['crop_damage'])

    print(f"Overall Damage Rate: {damage_rate:.2f}%")
    print(f"Insect Count/Damage Correlation: {insect_damage_corr:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='estimated_insects_count', y='number_doses_week', color='crop_damage_label',
                      title="Insect Count vs. Pesticide Doses by Damage Type",
                      labels={'estimated_insects_count': 'Estimated Insect Count', 'number_doses_week': 'Doses per Week'})
    fig1.show()

    damage_dist = df['crop_damage_label'].value_counts().reset_index()
    fig2 = px.pie(damage_dist, names='index', values='crop_damage_label', title="Distribution of Crop Damage Types")
    fig2.show()

def fertilizer_recommendation_system_analysis(df):
    print("\n--- Fertilizer Recommendation System Analysis ---")
    expected = ['temperature', 'humidity', 'nitrogen', 'potassium', 'phosphorous', 'crop_type', 'fertilizer_name']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['temperature', 'humidity', 'nitrogen', 'potassium', 'phosphorous']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    num_fertilizers = df['fertilizer_name'].nunique()
    top_fertilizer = df['fertilizer_name'].mode()[0]
    top_crop = df['crop_type'].mode()[0]

    print(f"Unique Fertilizer Types: {num_fertilizers}")
    print(f"Most Common Fertilizer: {top_fertilizer}")
    print(f"Most Common Crop: {top_crop}")

    # Visualizations
    print("Nutrient Requirements by Crop Type")
    crop_conditions = df.groupby('crop_type')[['nitrogen', 'potassium', 'phosphorous', 'temperature', 'humidity', 'ph', 'rainfall']].mean().reset_index() # ph, rainfall might be missing here if not in dataset
    fig1 = px.parallel_coordinates(crop_conditions, color="crop_type",
                                      title="Environmental Profiles for Different Crops",
                                      labels={'nitrogen': 'Nitrogen', 'phosphorous': 'Phosphorus', 'potassium': 'Potassium', 'crop_type': 'Crop'})
    fig1.show()

    fertilizer_counts = df['fertilizer_name'].value_counts().reset_index()
    fig2 = px.bar(fertilizer_counts, x='index', y='fertilizer_name',
                      title="Frequency of Recommended Fertilizers",
                      labels={'index': 'Fertilizer', 'fertilizer_name': 'Count'})
    fig2.show()


def agricultural_yield_and_price_fluctuation_analysis(df):
    print("\n--- Agricultural Yield and Price Fluctuation Analysis ---")
    expected = ['year', 'crops', 'yeilds', 'price', 'rainfall', 'temperature']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['year', 'yeilds', 'price', 'rainfall', 'temperature']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=expected, inplace=True)

    # Metrics
    avg_yield = df['yeilds'].mean()
    avg_price = df['price'].mean()
    yield_price_corr = df['yeilds'].corr(df['price'])

    print(f"Average Yield: {avg_yield:,.2f}")
    print(f"Average Price: ${avg_price:,.2f}")
    print(f"Yield/Price Correlation: {yield_price_corr:.2f}")

    # Visualizations
    df_yearly = df.groupby('year')[['yeilds', 'price']].mean().reset_index()
    fig1 = px.line(df_yearly, x='year', y=['yeilds', 'price'],
                    title="Average Yield and Price Over Time", facet_row="variable",
                    labels={'value': 'Value', 'year': 'Year'})
    fig1.update_yaxes(matches=None)
    fig1.show()

    fig2 = px.scatter(df, x='rainfall', y='yeilds', color='crops',
                      title="Rainfall vs. Yields by Crop",
                      labels={'rainfall': 'Rainfall', 'yeilds': 'Yields'})
    fig2.show()


def long_term_tillage_and_crop_rotation_impact_analysis(df):
    print("\n--- Long-Term Tillage and Crop Rotation Impact Analysis ---")
    # Choosing representative columns for grain over years
    expected = ['tillage', 'endingcroppingsystem', 'grain2018', 'grain2019', 'grain2020']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['grain2018', 'grain2019', 'grain2020']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_yield_2020 = df['grain2020'].mean()
    df['yield_gain'] = df['grain2020'] - df['grain2018']
    avg_yield_gain = df['yield_gain'].mean()

    print(f"Avg. Grain Yield in 2020: {avg_yield_2020:.2f}")
    print(f"Avg. Yield Gain (2018-2020): {avg_yield_gain:.2f}")

    # Visualizations
    fig1 = px.box(df, x='tillage', y='grain2020',
                    title="Impact of Tillage System on 2020 Grain Yield")
    fig1.show()

    fig2 = px.box(df, x='endingcroppingsystem', y='yield_gain',
                    title="Yield Gain (2018-2020) by Ending Cropping System",
                    labels={'endingcroppingsystem': 'Cropping System', 'yield_gain': 'Yield Gain'})
    fig2.show()


def soil_compaction_and_organic_matter_analysis(df):
    print("\n--- Soil Compaction and Organic Matter Analysis ---")
    expected = ['depth_upper', 'coneindex', 'tillage', 'om_perc', 'bd'] # BD = Bulk Density
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['depth_upper', 'coneindex', 'om_perc', 'bd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_cone_index = df['coneindex'].mean() # Cone index measures compaction
    avg_om = df['om_perc'].mean()

    print(f"Average Cone Index (Compaction): {avg_cone_index:.2f}")
    print(f"Average Organic Matter: {avg_om:.2f}%")

    # Visualizations
    fig1 = px.scatter(df, x='om_perc', y='bd', color='tillage',
                        title="Organic Matter vs. Bulk Density by Tillage Type",
                        labels={'om_perc': 'Organic Matter %', 'bd': 'Bulk Density'})
    fig1.show()

    fig2 = px.line(df.sort_values('depth_upper'), x='depth_upper', y='coneindex', color='tillage',
                    title="Soil Compaction (Cone Index) by Depth and Tillage")
    fig2.show()


def monthly_temperature_variation_analysis(df):
    print("\n--- Monthly Temperature Variation Analysis ---")
    expected = ['year', 'may_tmin_c', 'may_tmean_c', 'may_tmax_c']
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

    # Metrics
    avg_mean_temp = df['may_tmean_c'].mean()
    max_temp = df['may_tmax_c'].max()
    min_temp = df['may_tmin_c'].min()

    print(f"Avg. May Mean Temp: {avg_mean_temp:.1f}°C")
    print(f"Highest Max Temp Recorded: {max_temp:.1f}°C")
    print(f"Lowest Min Temp Recorded: {min_temp:.1f}°C")

    # Visualizations
    df_melted = df.melt(id_vars='year', value_vars=['may_tmin_c', 'may_tmean_c', 'may_tmax_c'],
                        var_name='temp_type', value_name='temperature')
    fig1 = px.line(df_melted, x='year', y='temperature', color='temp_type',
                    title="May Temperature Trends Over Years")
    fig1.show()

def daily_precipitation_data_analysis(df):
    print("\n--- Daily Precipitation Data Analysis ---")
    expected = ['date', 'doy', 'month', 'year', 'mm']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['mm'] = pd.to_numeric(df['mm'], errors='coerce')
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_precip = df['mm'].sum()
    wettest_day = df.loc[df['mm'].idxmax()]['date'].strftime('%Y-%m-%d')
    wettest_day_mm = df['mm'].max()

    print(f"Total Precipitation Recorded: {total_precip:,.1f} mm")
    print(f"Wettest Day: {wettest_day}")
    print(f"Precipitation on Wettest Day: {wettest_day_mm:.1f} mm")

    # Visualizations
    yearly_precip = df.groupby('year')['mm'].sum().reset_index()
    fig1 = px.bar(yearly_precip, x='year', y='mm', title="Total Annual Precipitation")
    fig1.show()

    monthly_precip = df.groupby(df['date'].dt.month)['mm'].mean().reset_index()
    fig2 = px.bar(monthly_precip, x='date', y='mm', title="Average Daily Precipitation by Month")
    fig2.update_xaxes(dtick="M1", tickformat="%b")
    fig2.show()


def soil_carbon_and_nitrogen_dynamics_analysis(df):
    print("\n--- Soil Carbon and Nitrogen Dynamics Analysis ---")
    expected = ['tillage', 'depth', 'totaln_perc', 'totalc_perc', 'organicc_perc', 'ph']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['totaln_perc', 'totalc_perc', 'organicc_perc', 'ph']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_c_n_ratio = (df['totalc_perc'] / df['totaln_perc']).mean()
    avg_ph = df['ph'].mean()

    print(f"Average C:N Ratio: {avg_c_n_ratio:.2f}")
    print(f"Average Soil pH: {avg_ph:.2f}")

    # Visualizations
    fig1 = px.box(df, x='tillage', y=['totalc_perc', 'totaln_perc'],
                    title="Total Carbon and Nitrogen Percentage by Tillage System")
    fig1.show()

    fig2 = px.scatter(df, x='ph', y='organicc_perc', color='tillage',
                        title="Organic Carbon vs. pH by Tillage System",
                        labels={'ph': 'Soil pH', 'organicc_perc': 'Organic Carbon %'})
    fig2.show()


def soil_chemistry_and_nutrient_level_change_analysis(df):
    print("\n--- Soil Chemistry and Nutrient Level Change Analysis ---")
    expected = ['rotn', 'tillage', 'nrate', 'delta_ph', 'delta_kcmol', 'delta_cacmol']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['nrate', 'delta_ph', 'delta_kcmol', 'delta_cacmol']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_ph_change = df['delta_ph'].mean()
    avg_k_change = df['delta_kcmol'].mean()
    avg_ca_change = df['delta_cacmol'].mean()

    print(f"Average pH Change: {avg_ph_change:.3f}")
    print(f"Avg. Potassium Change (cmol/kg): {avg_k_change:.3f}")
    print(f"Avg. Calcium Change (cmol/kg): {avg_ca_change:.3f}")

    # Visualizations
    fig1 = px.box(df, x='tillage', y='delta_ph', color='rotn',
                    title="Change in Soil pH by Tillage and Rotation")
    fig1.show()

    fig2 = px.scatter(df, x='nrate', y='delta_ph', color='tillage',
                        title="Impact of Nitrogen Rate on pH Change",
                        labels={'nrate': 'Nitrogen Rate', 'delta_ph': 'Change in pH'})
    fig2.show()

def plant_dissection_and_larval_mortality_analysis(df):
    print("\n--- Plant Dissection and Larval Mortality Analysis ---")
    expected = ['line', 'total_#plants', 'stunting_score_mean', 'total_dead_mean', 'total_live_mean', 'larval_mortality_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col != 'line':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_mortality = df['larval_mortality_perc'].mean()
    most_resistant_line = df.loc[df['larval_mortality_perc'].idxmax()]['line']
    most_stunted_line = df.loc[df['stunting_score_mean'].idxmax()]['line']

    print(f"Average Larval Mortality: {avg_mortality:.2f}%")
    print(f"Most Resistant Line (by Mortality): {most_resistant_line}")
    print(f"Most Stunted Line: {most_stunted_line}")

    # Visualizations
    mortality_by_line = df.groupby('line')['larval_mortality_perc'].mean().sort_values(ascending=False).reset_index()
    fig1 = px.bar(mortality_by_line, x='line', y='larval_mortality_perc',
                      title="Average Larval Mortality by Plant Line")
    fig1.show()

    fig2 = px.scatter(df, x='stunting_score_mean', y='larval_mortality_perc', hover_name='line',
                      title="Stunting Score vs. Larval Mortality",
                      labels={'stunting_score_mean': 'Mean Stunting Score', 'larval_mortality_perc': 'Larval Mortality %'})
    fig2.show()

def detailed_pest_resistance_scoring_analysis(df):
    print("\n--- Detailed Pest Resistance Scoring Analysis ---")
    expected = ['line', 'plant#', 'stunting_score', 'resistance_score_leaves', 'total_live', 'total_dead', 'total_larvae']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col != 'line':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_total_larvae = df['total_larvae'].mean()
    avg_resistance_score = df['resistance_score_leaves'].mean()

    print(f"Average Larvae Count per Plant: {avg_total_larvae:.1f}")
    print(f"Average Leaf Resistance Score: {avg_resistance_score:.2f}")

    # Visualizations
    fig1 = px.density_heatmap(df, x="total_live", y="total_dead",
                              title="Heatmap of Live vs. Dead Larvae Counts")
    fig1.show()

    fig2 = px.box(df, x='line', y='total_larvae',
                      title="Total Larvae Distribution by Plant Line")
    fig2.show()

def coffee_plantation_area_analysis(df):
    print("\n--- Coffee Plantation Area Analysis (Arabica vs. Robusta) ---")
    expected = ['year', 'arabica_in_hectares', 'robusta_in_hectares', 'total_in_hectares']
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

    # Metrics
    latest_year = df['year'].max()
    latest_total = df.loc[df['year'] == latest_year, 'total_in_hectares'].values[0]
    latest_arabica_perc = (df.loc[df['year'] == latest_year, 'arabica_in_hectares'] / latest_total * 100).values[0]

    print(f"Most Recent Year: {latest_year}")
    print(f"Total Area (ha) in Latest Year: {latest_total:,.0f}")
    print(f"Arabica Share in Latest Year: {latest_arabica_perc:.1f}%")

    # Visualizations
    df_melted = df.melt(id_vars='year', value_vars=['arabica_in_hectares', 'robusta_in_hectares'],
                        var_name='coffee_type', value_name='area_ha')
    fig1 = px.area(df_melted, x='year', y='area_ha', color='coffee_type',
                    title="Coffee Plantation Area by Type Over Time")
    fig1.show()

    df['arabica_share'] = df['arabica_in_hectares'] / df['total_in_hectares'] * 100
    fig2 = px.line(df, x='year', y='arabica_share', title="Percentage Share of Arabica Over Time")
    fig2.show()


def comprehensive_district_level_crop_production_and_yield_analysis(df):
    print("\n--- Comprehensive District-Level Crop Production and Yield Analysis ---")
    expected = ['dist_name', 'rice_production_1000_tons', 'rice_yield_kg_per_ha', 'wheat_production_1000_tons', 'wheat_yield_kg_per_ha', 'maize_production_1000_tons']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if 'dist_name' not in col:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_rice_prod = df['rice_production_1000_tons'].sum()
    avg_wheat_yield = df['wheat_yield_kg_per_ha'].mean()
    top_maize_dist = df.groupby('dist_name')['maize_production_1000_tons'].sum().idxmax()

    print(f"Total Rice Production (k-tons): {total_rice_prod:,.0f}")
    print(f"Avg. Wheat Yield (kg/ha): {avg_wheat_yield:,.1f}")
    print(f"Top Maize District: {top_maize_dist}")

    # Visualizations
    top_districts_rice = df.groupby('dist_name')['rice_production_1000_tons'].sum().nlargest(10).reset_index()
    fig1 = px.bar(top_districts_rice, x='dist_name', y='rice_production_1000_tons',
                      title="Top 10 Districts by Rice Production")
    fig1.show()

    fig2 = px.scatter(df, x='rice_yield_kg_per_ha', y='wheat_yield_kg_per_ha',
                      hover_name='dist_name', title="Rice Yield vs. Wheat Yield by District")
    fig2.show()

def nutrient_retention_factor_analysis_in_foods(df):
    print("\n--- Nutrient Retention Factor Analysis in Foods ---")
    expected = ['fdgrp_cd', 'nutrdesc', 'retn_factor', 'date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['retn_factor'] = pd.to_numeric(df['retn_factor'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_retention = df['retn_factor'].mean()
    most_retained_nutrient = df.loc[df['retn_factor'].idxmax()]['nutrdesc']
    least_retained_nutrient = df.loc[df['retn_factor'].idxmin()]['nutrdesc']

    print(f"Average Retention Factor: {avg_retention:.2f}")
    print(f"Most Retained Nutrient: {most_retained_nutrient}")
    print(f"Least Retained Nutrient: {least_retained_nutrient}")

    # Visualizations
    top_nutrients = df.groupby('nutrdesc')['retn_factor'].mean().nlargest(15).reset_index()
    fig1 = px.bar(top_nutrients, x='nutrdesc', y='retn_factor',
                      title="Top 15 Nutrients by Average Retention Factor")
    fig1.show()

    fig2 = px.box(df, x='fdgrp_cd', y='retn_factor', title="Nutrient Retention by Food Group Code")
    fig2.show()


def food_cooking_yield_and_nutrient_change_analysis(df):
    print("\n--- Food Cooking Yield and Nutrient Change Analysis ---")
    expected = ['food_group_code', 'preparation_method1', 'cooking_yield_perc', 'moisture_gain_loss_perc', 'fat_gain_loss_perc']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['cooking_yield_perc', 'moisture_gain_loss_perc', 'fat_gain_loss_perc']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_yield = df['cooking_yield_perc'].mean()
    avg_moisture_change = df['moisture_gain_loss_perc'].mean()
    avg_fat_change = df['fat_gain_loss_perc'].mean()

    print(f"Average Cooking Yield: {avg_yield:.1f}%")
    print(f"Avg. Moisture Change: {avg_moisture_change:.1f}%")
    print(f"Avg. Fat Change: {avg_fat_change:.1f}%")

    # Visualizations
    yield_by_prep = df.groupby('preparation_method1')['cooking_yield_perc'].mean().sort_values().reset_index()
    fig1 = px.bar(yield_by_prep, x='cooking_yield_perc', y='preparation_method1',
                      orientation='h', title="Average Cooking Yield by Preparation Method")
    fig1.show()

    fig2 = px.scatter(df, x='moisture_gain_loss_perc', y='fat_gain_loss_perc', color='preparation_method1',
                      title="Moisture vs. Fat Change During Cooking")
    fig2.show()

def field_operation_and_tillage_log_analysis(df):
    print("\n--- Field Operation and Tillage Log Analysis ---")
    expected = ['siteid', 'plotid', 'date', 'cashcrop', 'tillage_type', 'depth']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    common_tillage = df['tillage_type'].mode()[0]
    common_crop = df['cashcrop'].mode()[0]
    avg_depth = df['depth'].mean()

    print(f"Most Common Tillage: {common_tillage}")
    print(f"Most Common Crop: {common_crop}")
    print(f"Average Tillage Depth: {avg_depth:.2f}")

    # Visualizations
    tillage_counts = df['tillage_type'].value_counts().reset_index()
    fig1 = px.pie(tillage_counts, names='index', values='tillage_type', title="Distribution of Tillage Types")
    fig1.show()

    fig2 = px.box(df, x='tillage_type', y='depth', title="Tillage Depth by Type")
    fig2.show()

def fertilizer_and_manure_application_analysis(df):
    print("\n--- Fertilizer and Manure Application Analysis ---")
    expected = ['cashcrop', 'operation_type', 'fertilizer_form', 'fertilizer_rate', 'manure_source', 'manure_rate', 'nitrogen_elem']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['fertilizer_rate', 'manure_rate', 'nitrogen_elem']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_n_rate = df['nitrogen_elem'].mean()
    avg_fert_rate = df['fertilizer_rate'].mean()
    avg_manure_rate = df['manure_rate'].mean()

    print(f"Average Nitrogen Rate: {avg_n_rate:.2f}")
    print(f"Average Fertilizer Rate: {avg_fert_rate:.2f}")
    print(f"Average Manure Rate: {avg_manure_rate:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='fertilizer_rate', y='nitrogen_elem', color='cashcrop',
                        title="Elemental Nitrogen vs. Total Fertilizer Rate")
    fig1.show()

    rate_by_crop = df.groupby('cashcrop')[['fertilizer_rate', 'manure_rate']].mean().reset_index()
    fig2 = px.bar(rate_by_crop, x='cashcrop', y=['fertilizer_rate', 'manure_rate'],
                      title="Average Application Rates by Crop", barmode='group')
    fig2.show()

def no_till_farming_practice_impact_analysis(df):
    print("\n--- No-Till Farming Practice Impact Analysis ---")
    # This dataset is very simple, likely needs to be joined with another dataset (e.g., yield)
    # to be useful. Analysis will be basic.
    expected = ['siteid', 'plotid', 'year_crop', 'crop', 'notill']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['notill'] = df['notill'].astype(str)

    # Metrics
    no_till_perc = (df['notill'].str.lower().isin(['yes', '1', 'true'])).mean() * 100

    print(f"Percentage of No-Till Plots: {no_till_perc:.1f}%")

    # Visualizations
    notill_counts = df['notill'].value_counts().reset_index()
    fig1 = px.pie(notill_counts, names='index', values='notill', title="Proportion of No-Till vs. Conventional Till Plots")
    fig1.show()

    crop_counts = df.groupby(['crop', 'notill']).size().reset_index(name='count')
    fig2 = px.bar(crop_counts, x='crop', y='count', color='notill',
                      title="Tillage Practice by Crop Type", barmode='group')
    fig2.show()


def cash_crop_planting_schedule_analysis(df):
    print("\n--- Cash Crop Planting Schedule Analysis ---")
    expected = ['plotid', 'year_calendar', 'date', 'cashcrop']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)
    df['day_of_year'] = df['date'].dt.dayofyear

    # Metrics
    avg_planting_doy = df['day_of_year'].mean()

    print(f"Average Planting Day of Year: {avg_planting_doy:.0f}")

    # Visualizations
    fig1 = px.histogram(df, x='day_of_year', color='cashcrop', marginal='box',
                        title="Distribution of Planting Dates by Crop")
    fig1.show()

    yearly_planting = df.groupby(['year_calendar', 'cashcrop'])['day_of_year'].mean().reset_index()
    fig2 = px.line(yearly_planting, x='year_calendar', y='day_of_year', color='cashcrop',
                    title="Trend of Average Planting Day Over Years")
    fig2.show()

def plant_hybrid_and_seeding_rate_performance_analysis(df):
    print("\n--- Plant Hybrid and Seeding Rate Performance Analysis ---")
    expected = ['cashcrop', 'plant_hybrid', 'plant_maturity', 'plant_rate', 'plant_rate_units']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['plant_maturity', 'plant_rate']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_seeding_rate = df['plant_rate'].mean()
    most_common_hybrid = df['plant_hybrid'].mode()[0]

    print(f"Average Seeding Rate: {avg_seeding_rate:,.0f}")
    print(f"Most Common Hybrid: {most_common_hybrid}")

    # Visualizations
    fig1 = px.box(df, x='cashcrop', y='plant_rate', color='plant_hybrid',
                      title="Seeding Rate by Crop and Hybrid")
    fig1.show()

    fig2 = px.scatter(df, x='plant_maturity', y='plant_rate', color='cashcrop',
                      title="Seeding Rate vs. Plant Maturity",
                      labels={'plant_maturity': 'Plant Maturity (days)', 'plant_rate': 'Seeding Rate'})
    fig2.show()

def irrigation_scheduling_and_water_usage_analysis(df):
    print("\n--- Irrigation Scheduling and Water Usage Analysis ---")
    expected = ['irrigation_method', 'year_calendar', 'date_irrigation_start', 'date_irrigation_end', 'irrigation_amount']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date_irrigation_start'] = pd.to_datetime(df['date_irrigation_start'], errors='coerce')
    df['irrigation_amount'] = pd.to_numeric(df['irrigation_amount'], errors='coerce')
    df['year_calendar'] = pd.to_numeric(df['year_calendar'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_water_used = df['irrigation_amount'].sum()
    avg_water_per_event = df['irrigation_amount'].mean()
    most_common_method = df['irrigation_method'].mode()[0]

    print(f"Total Water Used: {total_water_used:,.2f}")
    print(f"Avg. Water per Event: {avg_water_per_event:.2f}")
    print(f"Most Common Method: {most_common_method}")

    # Visualizations
    water_by_year = df.groupby('year_calendar')['irrigation_amount'].sum().reset_index()
    fig1 = px.bar(water_by_year, x='year_calendar', y='irrigation_amount',
                      title="Total Irrigation Water Used Per Year")
    fig1.show()

    water_by_method = df.groupby('irrigation_method')['irrigation_amount'].sum().reset_index()
    fig2 = px.pie(water_by_method, names='irrigation_method', values='irrigation_amount',
                      title="Total Water Usage by Irrigation Method")
    fig2.show()

def water_drainage_control_structure_analysis(df):
    print("\n--- Water Drainage Control Structure Analysis ---")
    expected = ['plot_id', 'year_calendar', 'date', 'outlet_depth', 'outlet_height']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['outlet_depth', 'outlet_height']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_depth = df['outlet_depth'].mean()
    avg_height = df['outlet_height'].mean()

    print(f"Average Outlet Depth: {avg_depth:.2f}")
    print(f"Average Outlet Height: {avg_height:.2f}")

    # Visualizations
    df_daily = df.groupby(df['date'].dt.date)[['outlet_depth', 'outlet_height']].mean().reset_index()
    fig1 = px.line(df_daily, x='date', y=['outlet_depth', 'outlet_height'],
                    title="Drainage Outlet Settings Over Time")
    fig1.show()

    fig2 = px.scatter(df, x='outlet_depth', y='outlet_height',
                      title="Outlet Depth vs. Height Settings")
    fig2.show()

def agro_meteorological_data_and_evapotranspiration_analysis(df):
    print("\n--- Agro-Meteorological Data and Evapotranspiration Analysis ---")
    expected = ['date', 'precipitation', 'air_temp_avg', 'solar_radiation', 'wind_speed', 'et']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['precipitation', 'air_temp_avg', 'solar_radiation', 'wind_speed', 'et']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_et = df['et'].mean() # Evapotranspiration
    avg_temp = df['air_temp_avg'].mean()
    total_precip = df['precipitation'].sum()

    print(f"Average ET: {avg_et:.2f}")
    print(f"Average Air Temp: {avg_temp:.2f}°C")
    print(f"Total Precipitation: {total_precip:.2f}")

    # Visualizations
    df_daily_avg = df.groupby(df['date'].dt.date)[['et', 'precipitation', 'air_temp_avg']].mean().reset_index()
    fig1 = px.line(df_daily_avg, x='date', y=['et', 'precipitation'],
                    title="Evapotranspiration and Precipitation Over Time")
    fig1.show()

    fig2 = px.scatter(df, x='air_temp_avg', y='et', color='solar_radiation',
                      title="Evapotranspiration vs. Air Temperature",
                      labels={'air_temp_avg': 'Average Air Temp (°C)', 'et': 'Evapotranspiration'})
    fig2.show()

def crop_cultivation_cost_and_profitability_analysis(df):
    print("\n--- Crop Cultivation Cost and Profitability Analysis ---")
    expected = ['crop', 'state', 'cost_of_cultivation_hectare_c2', 'cost_of_production_quintal_c2', 'yield_quintal_hectare']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col not in ['crop', 'state']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Assume price is cost of production * some markup, or analyze cost efficiency
    df['revenue_per_hectare'] = df['cost_of_production_quintal_c2'] * df['yield_quintal_hectare']
    df['profit_per_hectare'] = df['revenue_per_hectare'] - df['cost_of_cultivation_hectare_c2']

    # Metrics
    most_profitable_crop = df.groupby('crop')['profit_per_hectare'].mean().idxmax()
    avg_profit = df['profit_per_hectare'].mean()
    highest_cost_crop = df.groupby('crop')['cost_of_cultivation_hectare_c2'].mean().idxmax()

    print(f"Most Profitable Crop: {most_profitable_crop}")
    print(f"Average Profit per Hectare: ${avg_profit:,.2f}")
    print(f"Highest Cost Crop: {highest_cost_crop}")

    # Visualizations
    profit_by_crop = df.groupby('crop')[['cost_of_cultivation_hectare_c2', 'revenue_per_hectare', 'profit_per_hectare']].mean().reset_index()
    fig1 = px.bar(profit_by_crop.sort_values('profit_per_hectare', ascending=False),
                  x='crop', y=['cost_of_cultivation_hectare_c2', 'profit_per_hectare'],
                  title="Cost and Profit per Hectare by Crop", barmode='stack')
    fig1.show()

    fig2 = px.scatter(df, x='cost_of_cultivation_hectare_c2', y='yield_quintal_hectare',
                      color='crop', title="Cost of Cultivation vs. Yield")
    fig2.show()

def detailed_soil_physicochemical_property_analysis(df):
    print("\n--- Detailed Soil Physicochemical Property Analysis ---")
    expected = ['depth', 'percent_sand', 'percent_silt', 'percent_clay', 'bulk_density', 'ph_water', 'soc', 'total_n']
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

    # Metrics
    avg_ph = df['ph_water'].mean()
    avg_bulk_density = df['bulk_density'].mean()
    avg_soc = df['soc'].mean() # Soil Organic Carbon

    print(f"Average Soil pH: {avg_ph:.2f}")
    print(f"Average Bulk Density: {avg_bulk_density:.2f} g/cm³")
    print(f"Average Soil Organic Carbon: {avg_soc:.2f}%")

    # Visualizations
    print("Correlation Between Soil Properties")
    corr_matrix = df[['percent_sand', 'bulk_density', 'ph_water', 'soc', 'total_n']].corr(numeric_only=True)
    fig1 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                      title="Correlation Heatmap of Soil Properties")
    fig1.show()

    fig2 = px.scatter(df, x='soc', y='total_n', size='bulk_density',
                      title="Soil Organic Carbon vs. Total Nitrogen",
                      labels={'soc': 'Soil Organic Carbon (%)', 'total_n': 'Total Nitrogen (%)'})
    fig2.show()


def soil_moisture_temperature_and_electrical_conductivity_analysis(df):
    print("\n--- Soil Moisture, Temperature, and EC Analysis ---")
    expected = ['date', 'depth', 'soil_moisture', 'soil_temperature', 'soil_ec']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['depth', 'soil_moisture', 'soil_temperature', 'soil_ec']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_moisture = df['soil_moisture'].mean()
    avg_temp = df['soil_temperature'].mean()
    avg_ec = df['soil_ec'].mean()

    print(f"Average Soil Moisture: {avg_moisture:.3f} m³/m³")
    print(f"Average Soil Temperature: {avg_temp:.2f}°C")
    print(f"Average Soil EC: {avg_ec:.2f} dS/m")

    # Visualizations
    df_daily_avg = df.groupby(df['date'].dt.date)[['soil_moisture', 'soil_temperature']].mean().reset_index()
    fig1 = px.line(df_daily_avg, x='date', y=['soil_moisture', 'soil_temperature'],
                    title="Daily Average Soil Moisture and Temperature Over Time", facet_row='variable')
    fig1.update_yaxes(matches=None)
    fig1.show()

    df_depth_profile = df.groupby('depth')[['soil_moisture', 'soil_ec']].mean().reset_index()
    fig2 = px.line(df_depth_profile, x='depth', y=['soil_moisture', 'soil_ec'],
                    title="Average Soil Moisture and EC by Depth", facet_row='variable')
    fig2.update_yaxes(matches=None)
    fig2.show()


def crop_biomass_and_nutrient_content_analysis(df):
    print("\n--- Crop Biomass and Nutrient Content Analysis ---")
    expected = ['crop', 'crop_yield', 'vegetative_biomass', 'grain_biomass', 'vegetative_total_n', 'grain_total_n']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['crop_yield', 'vegetative_biomass', 'grain_biomass', 'vegetative_total_n', 'grain_total_n']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_grain_biomass = df['grain_biomass'].sum()
    avg_grain_n = df['grain_total_n'].mean()
    most_biomass_crop = df.groupby('crop')['grain_biomass'].sum().idxmax()

    print(f"Total Grain Biomass: {total_grain_biomass:,.2f}")
    print(f"Average Grain Total Nitrogen: {avg_grain_n:.2f}%")
    print(f"Crop with Highest Grain Biomass: {most_biomass_crop}")

    # Visualizations
    fig1 = px.bar(df.groupby('crop')['grain_biomass'].sum().reset_index(),
                  x='crop', y='grain_biomass', title="Total Grain Biomass by Crop Type")
    fig1.show()

    fig2 = px.scatter(df, x='vegetative_total_n', y='grain_total_n', color='crop',
                      title="Vegetative N vs. Grain N Content by Crop",
                      labels={'vegetative_total_n': 'Vegetative Total Nitrogen (%)', 'grain_total_n': 'Grain Total Nitrogen (%)'})
    fig2.show()

def crop_growth_stage_monitoring_analysis(df):
    print("\n--- Crop Growth Stage Monitoring Analysis ---")
    expected = ['plot_id', 'crop', 'date', 'growth_stage', 'biomass_g_per_m2', 'height_cm']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['biomass_g_per_m2', 'height_cm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    num_plots = df['plot_id'].nunique()
    common_growth_stage = df['growth_stage'].mode()[0]
    avg_biomass = df['biomass_g_per_m2'].mean()

    print(f"Number of Plots Monitored: {num_plots}")
    print(f"Most Common Growth Stage: {common_growth_stage}")
    print(f"Average Biomass: {avg_biomass:.2f} g/m²")

    # Visualizations
    fig1 = px.line(df.sort_values('date'), x='date', y='biomass_g_per_m2', color='crop',
                   title="Biomass Growth Over Time by Crop Type")
    fig1.show()

    fig2 = px.box(df, x='growth_stage', y='height_cm', color='crop',
                  title="Crop Height Distribution by Growth Stage and Crop")
    fig2.show()

def water_table_depth_fluctuation_analysis(df):
    print("\n--- Water Table Depth Fluctuation Analysis ---")
    expected = ['date', 'site_id', 'water_table_depth_cm', 'rainfall_mm', 'irrigation_amount_mm']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['water_table_depth_cm', 'rainfall_mm', 'irrigation_amount_mm']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_depth = df['water_table_depth_cm'].mean()
    max_depth = df['water_table_depth_cm'].max()
    min_depth = df['water_table_depth_cm'].min()

    print(f"Average Water Table Depth: {avg_depth:.2f} cm")
    print(f"Maximum Water Table Depth: {max_depth:.2f} cm")
    print(f"Minimum Water Table Depth: {min_depth:.2f} cm")

    # Visualizations
    fig1 = px.line(df.sort_values('date'), x='date', y='water_table_depth_cm', color='site_id',
                   title="Water Table Depth Fluctuation Over Time by Site")
    fig1.show()

    fig2 = px.scatter(df, x='rainfall_mm', y='water_table_depth_cm', color='site_id',
                      title="Rainfall vs. Water Table Depth",
                      labels={'rainfall_mm': 'Rainfall (mm)', 'water_table_depth_cm': 'Water Table Depth (cm)'})
    fig2.show()

def agricultural_water_quality_monitoring_analysis(df):
    print("\n--- Agricultural Water Quality Monitoring Analysis ---")
    expected = ['sample_date', 'source_type', 'ph', 'ec_us_per_cm', 'nitrate_mg_per_l', 'phosphate_mg_per_l']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    for col in ['ph', 'ec_us_per_cm', 'nitrate_mg_per_l', 'phosphate_mg_per_l']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_ph = df['ph'].mean()
    avg_ec = df['ec_us_per_cm'].mean()
    avg_nitrate = df['nitrate_mg_per_l'].mean()

    print(f"Average pH: {avg_ph:.2f}")
    print(f"Average EC: {avg_ec:.2f} µS/cm")
    print(f"Average Nitrate: {avg_nitrate:.2f} mg/L")

    # Visualizations
    fig1 = px.box(df, x='source_type', y=['ph', 'ec_us_per_cm'],
                  title="Water pH and EC by Source Type", facet_row='variable')
    fig1.update_yaxes(matches=None)
    fig1.show()

    fig2 = px.scatter(df, x='nitrate_mg_per_l', y='phosphate_mg_per_l', color='source_type',
                      title="Nitrate vs. Phosphate Levels by Source Type")
    fig2.show()

def drain_flow_and_nutrient_load_analysis(df):
    print("\n--- Drain Flow and Nutrient Load Analysis ---")
    expected = ['date', 'drain_flow_l_per_sec', 'nitrate_load_kg', 'phosphate_load_kg', 'site_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['drain_flow_l_per_sec', 'nitrate_load_kg', 'phosphate_load_kg']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    total_drain_flow = df['drain_flow_l_per_sec'].sum()
    total_nitrate_load = df['nitrate_load_kg'].sum()
    total_phosphate_load = df['phosphate_load_kg'].sum()

    print(f"Total Drain Flow: {total_drain_flow:,.2f} L/sec")
    print(f"Total Nitrate Load: {total_nitrate_load:,.2f} kg")
    print(f"Total Phosphate Load: {total_phosphate_load:,.2f} kg")

    # Visualizations
    fig1 = px.line(df.sort_values('date'), x='date', y='drain_flow_l_per_sec', color='site_id',
                   title="Drain Flow Over Time by Site")
    fig1.show()

    fig2 = px.scatter(df, x='nitrate_load_kg', y='phosphate_load_kg', color='site_id',
                      title="Nitrate Load vs. Phosphate Load in Drains")
    fig2.show()

def miticide_efficacy_analysis_for_varroa_destructor(df):
    print("\n--- Miticide Efficacy Analysis for Varroa Destructor ---")
    expected = ['treatment_group', 'colony_id', 'pre_treatment_mite_count', 'post_treatment_mite_count', 'efficacy_percent']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['pre_treatment_mite_count', 'post_treatment_mite_count', 'efficacy_percent']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_efficacy = df['efficacy_percent'].mean()
    highest_efficacy_treatment = df.groupby('treatment_group')['efficacy_percent'].mean().idxmax()

    print(f"Average Miticide Efficacy: {avg_efficacy:.2f}%")
    print(f"Best Performing Treatment: {highest_efficacy_treatment}")

    # Visualizations
    fig1 = px.box(df, x='treatment_group', y='efficacy_percent',
                  title="Miticide Efficacy Distribution by Treatment Group")
    fig1.show()

    fig2 = px.scatter(df, x='pre_treatment_mite_count', y='efficacy_percent', color='treatment_group',
                      title="Pre-Treatment Mite Count vs. Efficacy by Treatment",
                      labels={'pre_treatment_mite_count': 'Pre-Treatment Mite Count', 'efficacy_percent': 'Efficacy (%)'})
    fig2.show()

def groundwater_quality_and_suitability_analysis_for_irrigation(df):
    print("\n--- Groundwater Quality & Suitability for Irrigation Analysis ---")
    expected = ['well_id', 'sample_date', 'ph', 'ec_us_per_cm', 'sodium_mg_per_l', 'chloride_mg_per_l', 'suitability_category']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    for col in ['ph', 'ec_us_per_cm', 'sodium_mg_per_l', 'chloride_mg_per_l']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_ph = df['ph'].mean()
    avg_ec = df['ec_us_per_cm'].mean()
    most_common_suitability = df['suitability_category'].mode()[0]

    print(f"Average pH: {avg_ph:.2f}")
    print(f"Average EC: {avg_ec:.2f} µS/cm")
    print(f"Most Common Suitability: {most_common_suitability}")

    # Visualizations
    fig1 = px.box(df, x='suitability_category', y=['ph', 'ec_us_per_cm'],
                  title="Water Quality Parameters by Suitability Category", facet_row='variable')
    fig1.update_yaxes(matches=None)
    fig1.show()

    fig2 = px.scatter(df, x='sodium_mg_per_l', y='chloride_mg_per_l', color='suitability_category',
                      title="Sodium vs. Chloride Levels by Suitability",
                      labels={'sodium_mg_per_l': 'Sodium (mg/L)', 'chloride_mg_per_l': 'Chloride (mg/L)'})
    fig2.show()

def soil_acidity_and_cation_exchange_capacity_analysis(df):
    print("\n--- Soil Acidity and Cation Exchange Capacity Analysis ---")
    expected = ['site_id', 'depth', 'ph_h2o', 'cec_cmol_per_kg', 'exchangeable_calcium', 'exchangeable_magnesium']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['ph_h2o', 'cec_cmol_per_kg', 'exchangeable_calcium', 'exchangeable_magnesium']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_ph = df['ph_h2o'].mean()
    avg_cec = df['cec_cmol_per_kg'].mean()
    avg_calcium = df['exchangeable_calcium'].mean()

    print(f"Average pH (H2O): {avg_ph:.2f}")
    print(f"Average CEC: {avg_cec:.2f} cmol/kg")
    print(f"Average Exchangeable Calcium: {avg_calcium:.2f} cmol/kg")

    # Visualizations
    fig1 = px.scatter(df, x='ph_h2o', y='cec_cmol_per_kg', color='depth',
                      title="Soil pH vs. CEC by Depth",
                      labels={'ph_h2o': 'pH (H2O)', 'cec_cmol_per_kg': 'CEC (cmol/kg)'})
    fig1.show()

    fig2 = px.box(df, x='depth', y=['exchangeable_calcium', 'exchangeable_magnesium'],
                  title="Exchangeable Calcium and Magnesium by Depth", facet_row='variable')
    fig2.update_yaxes(matches=None)
    fig2.show()

def varroa_mite_population_assessment_in_beehives(df):
    print("\n--- Varroa Mite Population Assessment in Beehives ---")
    expected = ['colony_id', 'sample_date', 'mite_count_after_shake', 'mite_fall_24hr', 'mite_infestation_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
    for col in ['mite_count_after_shake', 'mite_fall_24hr', 'mite_infestation_level']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_mite_shake = df['mite_count_after_shake'].mean()
    avg_mite_fall = df['mite_fall_24hr'].mean()
    highest_infestation_colony = df.loc[df['mite_infestation_level'].idxmax()]['colony_id']

    print(f"Average Mite Count (Sugar Shake): {avg_mite_shake:.1f}")
    print(f"Average 24hr Mite Fall: {avg_mite_fall:.1f}")
    print(f"Colony with Highest Infestation: {highest_infestation_colony}")

    # Visualizations
    fig1 = px.line(df.sort_values('sample_date'), x='sample_date', y='mite_infestation_level', color='colony_id',
                   title="Mite Infestation Level Over Time by Colony")
    fig1.show()

    fig2 = px.box(df, x='mite_infestation_level', y='mite_count_after_shake',
                  title="Mite Count (Shake) Distribution by Infestation Level")
    fig2.show()


# ========== MAIN APP / EXECUTION LOGIC ==========
def main():
    """Main function to run the Agricultural Analytics script."""
    print("🌾 Agricultural Analytics Script")

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

    # This dictionary maps the analysis names to the actual Python functions
    specific_agri_function_mapping = {
        "crop_analysis": crop_analysis,
        "soil_analysis": soil_analysis,
        "weather_impact_analysis": weather_impact_analysis,
        "irrigation_analysis": irrigation_analysis,
        "pest_disease_analysis": pest_disease_analysis,
        "economic_analysis": economic_analysis,
        "sustainability_analysis": sustainability_analysis,
        "crop_yield_prediction_analysis": crop_yield_prediction_analysis,
        "state_level_agricultural_production_trend_analysis": state_level_agricultural_production_trend_analysis,
        "comprehensive_county_level_agricultural_output_analysis": comprehensive_county_level_agricultural_output_analysis,
        "environmental_factor_analysis_for_rice_production": environmental_factor_analysis_for_rice_production,
        "district_level_rice_yield_and_soil_type_correlation_analysis": district_level_rice_yield_and_soil_type_correlation_analysis,
        "crop_type_recommendation_analysis": crop_type_recommendation_analysis,
        "national_land_use_change_analysis": national_land_use_change_analysis,
        "agricultural_census_crop_production_analysis": agricultural_census_crop_production_analysis,
        "insect_population_estimation_analysis": insect_population_estimation_analysis,
        "crop_damage_prediction_from_pest_infestation": crop_damage_prediction_from_pest_infestation,
        "fertilizer_recommendation_system_analysis": fertilizer_recommendation_system_analysis,
        "agricultural_yield_and_price_fluctuation_analysis": agricultural_yield_and_price_fluctuation_analysis,
        "long_term_tillage_and_crop_rotation_impact_analysis": long_term_tillage_and_crop_rotation_impact_analysis,
        "soil_compaction_and_organic_matter_analysis": soil_compaction_and_organic_matter_analysis,
        "monthly_temperature_variation_analysis": monthly_temperature_variation_analysis,
        "daily_precipitation_data_analysis": daily_precipitation_data_analysis,
        "soil_carbon_and_nitrogen_dynamics_analysis": soil_carbon_and_nitrogen_dynamics_analysis,
        "soil_chemistry_and_nutrient_level_change_analysis": soil_chemistry_and_nutrient_level_change_analysis,
        "plant_dissection_and_larval_mortality_analysis": plant_dissection_and_larval_mortality_analysis,
        "detailed_pest_resistance_scoring_analysis": detailed_pest_resistance_scoring_analysis,
        "coffee_plantation_area_analysis": coffee_plantation_area_analysis,
        "comprehensive_district_level_crop_production_and_yield_analysis": comprehensive_district_level_crop_production_and_yield_analysis,
        "nutrient_retention_factor_analysis_in_foods": nutrient_retention_factor_analysis_in_foods,
        "food_cooking_yield_and_nutrient_change_analysis": food_cooking_yield_and_nutrient_change_analysis,
        "field_operation_and_tillage_log_analysis": field_operation_and_tillage_log_analysis,
        "fertilizer_and_manure_application_analysis": fertilizer_and_manure_application_analysis,
        "no_till_farming_practice_impact_analysis": no_till_farming_practice_impact_analysis,
        "cash_crop_planting_schedule_analysis": cash_crop_planting_schedule_analysis,
        "plant_hybrid_and_seeding_rate_performance_analysis": plant_hybrid_and_seeding_rate_performance_analysis,
        "irrigation_scheduling_and_water_usage_analysis": irrigation_scheduling_and_water_usage_analysis,
        "water_drainage_control_structure_analysis": water_drainage_control_structure_analysis,
        "agro_meteorological_data_and_evapotranspiration_analysis": agro_meteorological_data_and_evapotranspiration_analysis,
        "crop_cultivation_cost_and_profitability_analysis": crop_cultivation_cost_and_profitability_analysis,
        "detailed_soil_physicochemical_property_analysis": detailed_soil_physicochemical_property_analysis,
        "soil_moisture_temperature_and_electrical_conductivity_analysis": soil_moisture_temperature_and_electrical_conductivity_analysis,
        "crop_biomass_and_nutrient_content_analysis": crop_biomass_and_nutrient_content_analysis,
        "crop_growth_stage_monitoring_analysis": crop_growth_stage_monitoring_analysis,
        "water_table_depth_fluctuation_analysis": water_table_depth_fluctuation_analysis,
        "agricultural_water_quality_monitoring_analysis": agricultural_water_quality_monitoring_analysis,
        "drain_flow_and_nutrient_load_analysis": drain_flow_and_nutrient_load_analysis,
        "miticide_efficacy_analysis_for_varroa_destructor": miticide_efficacy_analysis_for_varroa_destructor,
        "groundwater_quality_and_suitability_analysis_for_irrigation": groundwater_quality_and_suitability_analysis_for_irrigation,
        "soil_acidity_and_cation_exchange_capacity_analysis": soil_acidity_and_cation_exchange_capacity_analysis,
        "varroa_mite_population_assessment_in_beehives": varroa_mite_population_assessment_in_beehives,
    }

    # --- Analysis Selection ---
    print("\nSelect an Agricultural Analysis to Perform:")
    all_analysis_names = list(specific_agri_function_mapping.keys())
    for i, name in enumerate(all_analysis_names):
        print(f"{i+1}: {name.replace('_', ' ').title()}") # Nicer display name
    print(f"{len(all_analysis_names)+1}: General Insights (Data Overview)")

    choice_str = input(f"Enter the number of your choice (1-{len(all_analysis_names)+1}): ")
    try:
        choice_idx = int(choice_str) - 1
        if 0 <= choice_idx < len(all_analysis_names):
            selected_analysis_key = all_analysis_names[choice_idx]
            selected_function = specific_agri_function_mapping.get(selected_analysis_key)
            if selected_function:
                try:
                    selected_function(df)
                except Exception as e:
                    print(f"\n[ERROR] An error occurred while running the analysis '{selected_analysis_key.replace('_', ' ').title()}':")
                    print(f"Error details: {e}")
            else:
                print(f"\n[ERROR] Function for '{selected_analysis_key.replace('_', ' ').title()}' not found. This should not happen.")
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