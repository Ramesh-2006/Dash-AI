import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import warnings
import json # Added for describe() output

warnings.filterwarnings('ignore')

# ========== NUMPY TO PYTHON TYPE CONVERSION UTILITY ==========

def convert_numpy_types(data):
    """
    Recursively converts numpy types in a data structure (dict, list, value)
    to native Python types for JSON serialization.
    """
    if isinstance(data, (np.integer, np.int_)):
        return int(data)
    if isinstance(data, (np.floating, np.float_)):
        return float(data)
    if isinstance(data, (np.bool_, np.bool)):
        return bool(data)
    if isinstance(data, np.ndarray):
        return [convert_numpy_types(x) for x in data.tolist()]
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [convert_numpy_types(x) for x in data]
    if isinstance(data, pd.Timestamp):
        return data.isoformat()
    if pd.isna(data):
        return None
    return data

# ========== UTILITY FUNCTIONS ==========

def get_key_metrics(df):
    """Returns key metrics about the dataset as a dictionary"""
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    return {
        "total_records": total_records,
        "total_features": len(df.columns),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols)
    }

def convert_to_native_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.isna(obj):
        return None
    return obj

def get_missing_columns_message(missing_cols, matched_cols=None):
    """Returns a list of insight strings for missing columns."""
    insights = ["⚠️ Required Columns Not Found",
                "The following columns are needed for this analysis but weren't found in your data:"]
    for col in missing_cols:
        match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""
        insights.append(f" - {col}{match_info}")
    return insights

def general_insights_analysis(df, title="General Insights Analysis"):
    """
    Show general data visualizations.
    This is the non-interactive version of show_general_insights.
    It analyzes the *first* numeric and categorical columns found.
    """
    analysis_type = "general_insights_analysis"
    try:
        visualizations = {}
        metrics = {}
        insights = [f"--- {title} ---"]

        # Key Metrics
        key_metrics = get_key_metrics(df)
        metrics.update(key_metrics)
        insights.append(f"Total Records: {key_metrics['total_records']}")
        insights.append(f"Total Features: {key_metrics['total_features']}")

        # Numeric columns analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 0:
            selected_num_col = numeric_cols[0]
            insights.append(f"\nNumeric Features Analysis (showing first column: {selected_num_col})")

            # Histogram
            fig1 = px.histogram(df, x=selected_num_col,
                                title=f"Distribution of {selected_num_col}")
            visualizations["numeric_histogram"] = fig1.to_json()

            # Box Plot
            fig2 = px.box(df, y=selected_num_col,
                                title=f"Box Plot of {selected_num_col}")
            visualizations["numeric_boxplot"] = fig2.to_json()
        else:
            insights.append("[INFO] No numeric columns found for analysis.")

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            insights.append("\nFeature Correlations:")
            corr = df[numeric_cols].corr(numeric_only=True)
            fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                                   title="Correlation Between Numeric Features")
            visualizations["correlation_heatmap"] = fig3.to_json()
            metrics["correlation_matrix"] = json.loads(corr.to_json(orient="split"))

        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            selected_cat_col = categorical_cols[0]
            insights.append(f"\nCategorical Features Analysis (showing first column: {selected_cat_col})")

            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']

            fig4 = px.bar(value_counts.head(20), x='Value', y='Count',
                                title=f"Top 20 Distribution of {selected_cat_col}")
            visualizations["categorical_barchart"] = fig4.to_json()
        else:
            insights.append("[INFO] No categorical columns found for analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "General insights generated successfully.",
            "matched_columns": {}, # Not applicable
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred in general_insights_analysis: {str(e)}",
            "matched_columns": {},
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


# ========== DATA LOADING ==========
def load_data(file_path, encoding='utf-8'):
    """
    Loads data from a CSV or Excel file.
    Returns a DataFrame or None if loading fails.
    """
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            # If all encodings fail, return None (or raise error)
            return None
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}") # This print is for server-side logging
        return None

def fuzzy_match_column(df, target_columns):
    """
    Matches target column names to available columns in the DataFrame
    using fuzzy string matching.
    """
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        # Case-insensitive exact match
        lower_available = {col.lower(): col for col in available}
        if target.lower() in lower_available:
             matched[target] = lower_available[target.lower()]
             continue
            
        match, score = process.extractOne(target, available)
        matched[target] = match if score >= 70 else None
    return matched

# ========== ANALYSIS FUNCTIONS ==========

def crop_analysis(df):
    analysis_type = "crop_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- General Crop Analysis ---"]
        
        expected = ['crop', 'crop_type', 'yield_tons_per_hectare', 'yeilds', 'production', 'area', 'grain2020', 'rice_production', 'wheat_production_1000_tons']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find any standard crop-related columns like 'crop', 'yield', or 'production'.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # --- Yield Analysis ---
        yield_col_name = matched.get('yield_tons_per_hectare') or matched.get('yeilds')
        crop_col_name = matched.get('crop') or matched.get('crop_type')
        if yield_col_name and crop_col_name:
            insights.append("--- Yield by Crop Type ---")
            df[yield_col_name] = pd.to_numeric(df[yield_col_name], errors='coerce')
            avg_yield = df[yield_col_name].mean()
            top_crop = df.groupby(crop_col_name)[yield_col_name].mean().idxmax()
            
            metrics["average_yield"] = avg_yield
            metrics["highest_yield_crop"] = str(top_crop) # Ensure native type
            
            insights.append(f"Average Yield: {avg_yield:,.2f}")
            insights.append(f"Highest Average Yield Crop: {top_crop}")

            yield_by_crop = df.groupby(crop_col_name)[yield_col_name].mean().sort_values(ascending=False)
            fig = px.bar(yield_by_crop, title="Average Yield by Crop", labels={'value': 'Average Yield', 'index': 'Crop'})
            visualizations["yield_by_crop"] = fig.to_json()

        # --- Production & Area Analysis ---
        prod_col_name = matched.get('production')
        area_col_name = matched.get('area')
        if prod_col_name and area_col_name and crop_col_name:
            insights.append("--- Production and Area Insights ---")
            df[prod_col_name] = pd.to_numeric(df[prod_col_name], errors='coerce')
            df[area_col_name] = pd.to_numeric(df[area_col_name], errors='coerce')
            df['yield_calculated'] = df[prod_col_name] / df[area_col_name]
            
            insights.append("Calculated yield (production/area) where possible.")
            metrics["average_calculated_yield"] = df['yield_calculated'].mean()

            fig = px.scatter(df, x=area_col_name, y=prod_col_name,
                             size='yield_calculated', color=crop_col_name,
                             title="Production vs. Area Planted (Sized by Calculated Yield)")
            visualizations["production_vs_area"] = fig.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def soil_analysis(df):
    analysis_type = "soil_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- General Soil Analysis ---"]

        expected = ['soil_type', 'ph', 'nitrogen', 'potassium', 'phosphorous', 'om_perc', 'soc', 'bulk_density', 'coneindex']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find any standard soil-related columns like 'soil_type', 'ph', or 'nitrogen'.")
            fallback_data = general_insights_analysis(df, "General Insights")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        # --- Chemical Property Analysis ---
        chem_cols = ['ph', 'nitrogen', 'potassium', 'phosphorous', 'om_perc', 'soc']
        found_chem_cols = [matched[c] for c in chem_cols if matched.get(c)]

        if found_chem_cols:
            insights.append("--- General Soil Chemistry Analysis ---")
            df_chem = df[found_chem_cols].copy()
            for col in df_chem.columns:
                df_chem[col] = pd.to_numeric(df_chem[col], errors='coerce')

            # Get summary statistics
            chem_summary = df_chem.describe()
            metrics["chemistry_summary_table"] = json.loads(chem_summary.to_json(orient="split"))
            insights.append("Generated summary statistics for soil chemical properties.")
            
            # Correlation matrix
            corr_matrix = df_chem.corr(numeric_only=True)
            fig = px.imshow(corr_matrix, text_auto=True, title="Correlation Heatmap of Soil Nutrients")
            visualizations["soil_nutrient_correlation"] = fig.to_json()
            metrics["chemistry_correlation_matrix"] = json.loads(corr_matrix.to_json(orient="split"))

        # --- Physical Property Analysis ---
        if matched.get('bulk_density') and matched.get('coneindex'):
            insights.append("--- Soil Physical Properties Analysis ---")
            df['bulk_density'] = pd.to_numeric(df[matched['bulk_density']], errors='coerce')
            df['coneindex'] = pd.to_numeric(df[matched['coneindex']], errors='coerce')
            
            metrics["average_bulk_density"] = df['bulk_density'].mean()
            metrics["average_cone_index"] = df['coneindex'].mean()

            fig = px.scatter(df, x=matched['bulk_density'], y=matched['coneindex'],
                             trendline='ols', title="Soil Compaction (Cone Index) vs. Bulk Density")
            visualizations["compaction_vs_density"] = fig.to_json()
            
            corr_val = df['bulk_density'].corr(df['coneindex'])
            insights.append(f"Correlation between Bulk Density and Cone Index: {corr_val:.2f}")
            metrics["density_compaction_correlation"] = corr_val

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def weather_impact_analysis(df):
    analysis_type = "weather_impact_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Weather Impact Analysis ---"]
        
        expected = ['rainfall_mm', 'temperature_celsius', 'humidity', 'weather_condition', 'yield_tons_per_hectare', 'production']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols or (not matched.get('rainfall_mm') and not matched.get('temperature_celsius')):
            insights.append("Could not find key weather columns like 'rainfall_mm' or 'temperature_celsius'.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        target_col = matched.get('yield_tons_per_hectare') or matched.get('production')
        rain_col = matched.get('rainfall_mm')
        temp_col = matched.get('temperature_celsius')

        if not target_col:
            insights.append("Could not find a target variable like 'yield' or 'production' to analyze against.")
            # We can still show weather data distribution
            if rain_col:
                df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
                metrics["average_rainfall"] = df[rain_col].mean()
                fig_rain_dist = px.histogram(df, x=rain_col, title="Rainfall Distribution")
                visualizations["rainfall_distribution"] = fig_rain_dist.to_json()
            if temp_col:
                df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
                metrics["average_temperature"] = df[temp_col].mean()
                fig_temp_dist = px.histogram(df, x=temp_col, title="Temperature Distribution")
                visualizations["temperature_distribution"] = fig_temp_dist.to_json()
            
            return {
                "analysis_type": analysis_type,
                "status": "success", # Partial success
                "message": "Weather impact analysis performed (target not found, showing distributions).",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": convert_numpy_types(metrics),
                "insights": insights
            }

        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        target_label = target_col.replace('_', ' ').title()

        if rain_col:
            insights.append(f"--- Impact of Rainfall on {target_label} ---")
            df[rain_col] = pd.to_numeric(df[rain_col], errors='coerce')
            rain_corr = df[rain_col].corr(df[target_col])
            metrics["rainfall_correlation"] = rain_corr
            insights.append(f"Rainfall / {target_label} Correlation: {rain_corr:.2f}")
            fig1 = px.scatter(df, x=rain_col, y=target_col, trendline='ols', title=f"Impact of Rainfall on {target_label}")
            visualizations["rainfall_impact"] = fig1.to_json()

        if temp_col:
            insights.append(f"--- Impact of Temperature on {target_label} ---")
            df[temp_col] = pd.to_numeric(df[temp_col], errors='coerce')
            temp_corr = df[temp_col].corr(df[target_col])
            metrics["temperature_correlation"] = temp_corr
            insights.append(f"Temperature / {target_label} Correlation: {temp_corr:.2f}")
            fig2 = px.scatter(df, x=temp_col, y=target_col, trendline='ols', title=f"Impact of Temperature on {target_label}")
            visualizations["temperature_impact"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def irrigation_analysis(df):
    analysis_type = "irrigation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- General Irrigation Analysis ---"]
        
        expected = ['irrigation_used', 'irrigation_method', 'irrigation_amount', 'water_table_depth', 'yield_tons_per_hectare']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find any irrigation-related columns.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        if matched.get('irrigation_used') and matched.get('yield_tons_per_hectare'):
            insights.append("--- Impact of Irrigation on Yield ---")
            df[matched['yield_tons_per_hectare']] = pd.to_numeric(df[matched['yield_tons_per_hectare']], errors='coerce')
            
            # Calculate metrics
            yield_irrigated = df[df[matched['irrigation_used']] == True][matched['yield_tons_per_hectare']].mean()
            yield_not_irrigated = df[df[matched['irrigation_used']] == False][matched['yield_tons_per_hectare']].mean()
            metrics["average_yield_irrigated"] = yield_irrigated
            metrics["average_yield_not_irrigated"] = yield_not_irrigated
            insights.append(f"Average Yield (Irrigated): {yield_irrigated:.2f}")
            insights.append(f"Average Yield (Not Irrigated): {yield_not_irrigated:.2f}")

            fig = px.box(df, x=matched['irrigation_used'], y=matched['yield_tons_per_hectare'],
                         title="Crop Yield for Irrigated vs. Non-Irrigated Plots")
            visualizations["irrigation_impact_on_yield"] = fig.to_json()

        if matched.get('irrigation_method') and matched.get('irrigation_amount'):
            insights.append("--- Water Usage by Irrigation Method ---")
            df[matched['irrigation_amount']] = pd.to_numeric(df[matched['irrigation_amount']], errors='coerce')

            water_by_method = df.groupby(matched['irrigation_method']).agg(
                total_irrigation_amount=(matched['irrigation_amount'], 'sum')
            ).reset_index()
            
            metrics["water_usage_by_method"] = json.loads(water_by_method.to_json(orient="split"))

            fig = px.pie(water_by_method,
                         names=matched['irrigation_method'],
                         values='total_irrigation_amount',
                         title="Total Water Amount by Irrigation Method")
            visualizations["water_usage_by_method"] = fig.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def pest_disease_analysis(df):
    analysis_type = "pest_disease_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Pest & Disease Analysis ---"]
        
        expected = ['estimated_insects_count', 'pesticide_use_category', 'crop_damage', 'larval_mortality_perc', 'mite_count_after_shake']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find any pest or disease-related columns.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        if matched.get('estimated_insects_count') and matched.get('crop_damage'):
            insights.append("--- Insect Count vs. Crop Damage ---")
            df[matched['estimated_insects_count']] = pd.to_numeric(df[matched['estimated_insects_count']], errors='coerce')
            df[matched['crop_damage']] = df[matched['crop_damage']].astype(str)
            
            damage_groups = df.groupby(matched['crop_damage'])[matched['estimated_insects_count']].mean()
            metrics["average_insects_by_damage"] = json.loads(damage_groups.to_json(orient="split"))
            insights.append("Calculated average insect count by damage category.")
            
            fig = px.box(df, x=matched['crop_damage'], y=matched['estimated_insects_count'],
                         title="Insect Count by Crop Damage Category (0=Alive, 1=Damaged, 2=Pesticide Damage)")
            visualizations["insects_vs_damage"] = fig.to_json()

        if matched.get('pesticide_use_category') and matched.get('estimated_insects_count'):
            insights.append("--- Impact of Pesticide Use ---")
            df[matched['pesticide_use_category']] = df[matched['pesticide_use_category']].astype(str)
            
            pesticide_groups = df.groupby(matched['pesticide_use_category'])[matched['estimated_insects_count']].mean()
            metrics["average_insects_by_pesticide_use"] = json.loads(pesticide_groups.to_json(orient="split"))
            
            fig = px.violin(df, x=matched['pesticide_use_category'], y=matched['estimated_insects_count'],
                            box=True,
                            title="Insect Count by Pesticide Use (1=Never, 2=Previous, 3=Current)")
            visualizations["insects_vs_pesticide_use"] = fig.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def economic_analysis(df):
    analysis_type = "economic_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- General Economic Analysis ---"]
        
        expected = ['cost_of_cultivation_hectare_c2', 'cost_of_production_quintal_c2', 'yield_quintal_hectare', 'price', 'revenue', 'profit', 'crop']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols or not (matched.get('cost_of_cultivation_hectare_c2') and matched.get('yield_quintal_hectare') and matched.get('cost_of_production_quintal_c2')):
            insights.append("Could not find key economic columns like 'cost_of_cultivation_hectare_c2', 'yield_quintal_hectare', and 'cost_of_production_quintal_c2'.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        cost_col = matched.get('cost_of_cultivation_hectare_c2')
        yield_col = matched.get('yield_quintal_hectare')
        cost_prod_col = matched.get('cost_of_production_quintal_c2')
        crop_col = matched.get('crop') # Optional, for grouping

        if cost_col and yield_col and cost_prod_col:
            insights.append("--- Crop Profitability Analysis ---")
            df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
            df[yield_col] = pd.to_numeric(df[yield_col], errors='coerce')
            df[cost_prod_col] = pd.to_numeric(df[cost_prod_col], errors='coerce')

            # Estimate revenue and profit
            df['estimated_revenue'] = df[yield_col] * df[cost_prod_col] * 1.2 # Assume 20% markup for price
            df['estimated_profit'] = df['estimated_revenue'] - df[cost_col]
            insights.append("Estimated revenue and profit (assuming 20% markup on production cost for price).")

            avg_cost = df[cost_col].mean()
            avg_revenue = df['estimated_revenue'].mean()
            avg_profit = df['estimated_profit'].mean()

            metrics["average_cost_per_hectare"] = avg_cost
            metrics["average_revenue_per_hectare"] = avg_revenue
            metrics["average_profit_per_hectare"] = avg_profit
            
            insights.append(f"Average Cost per Hectare: ${avg_cost:,.0f}")
            insights.append(f"Average Revenue per Hectare: ${avg_revenue:,.0f}")
            insights.append(f"Average Profit per Hectare: ${avg_profit:,.0f}")
            
            scatter_color = crop_col if crop_col else None
            fig = px.scatter(df, x=cost_col, y='estimated_profit', color=scatter_color,
                             title="Profit vs. Cost of Cultivation")
            visualizations["profit_vs_cost"] = fig.to_json()
            
            if crop_col:
                profit_by_crop = df.groupby(crop_col)['estimated_profit'].mean().sort_values(ascending=False)
                metrics["profit_by_crop"] = json.loads(profit_by_crop.to_json(orient="split"))
                fig_crop = px.bar(profit_by_crop, title="Average Estimated Profit by Crop")
                visualizations["profit_by_crop"] = fig_crop.to_json()


        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def sustainability_analysis(df):
    analysis_type = "sustainability_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- General Sustainability Analysis ---"]
        
        expected = ['land_use', 'grazingland', 'cropland', 'tillage', 'no_till', 'om_perc', 'soc', 'total_n', 'nitrogen_elem']
        matched = fuzzy_match_column(df, expected)

        found_cols = {k:v for k,v in matched.items() if v}
        if not found_cols:
            insights.append("Could not find sustainability-related columns like 'land_use', 'tillage', or 'om_perc'.")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns not found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        if matched.get('grazingland') and matched.get('cropland'):
            insights.append("--- Land Use Distribution ---")
            grazing_total = pd.to_numeric(df[matched['grazingland']], errors='coerce').sum()
            cropland_total = pd.to_numeric(df[matched['cropland']], errors='coerce').sum()
            
            metrics["total_grazing_land"] = grazing_total
            metrics["total_crop_land"] = cropland_total
            
            pie_df = pd.DataFrame({'Land Type': ['Grazing Land', 'Crop Land'], 'Area': [grazing_total, cropland_total]})
            fig1 = px.pie(pie_df, names='Land Type', values='Area', title="Grazing vs. Cropland Area")
            visualizations["land_use_distribution"] = fig1.to_json()

        if matched.get('tillage') and matched.get('soc'): # SOC = Soil Organic Carbon
            insights.append("--- Impact of Tillage on Soil Organic Carbon ---")
            df[matched['soc']] = pd.to_numeric(df[matched['soc']], errors='coerce')
            soc_by_tillage = df.groupby(matched['tillage'])[matched['soc']].mean().reset_index()
            
            metrics["soc_by_tillage"] = json.loads(soc_by_tillage.to_json(orient="split"))
            
            fig2 = px.bar(soc_by_tillage, x=matched['tillage'], y=matched['soc'],
                          title="Average Soil Organic Carbon by Tillage Practice")
            visualizations["tillage_impact_on_soc"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def crop_yield_prediction_analysis(df):
    analysis_type = "crop_yield_prediction_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Yield Prediction Analysis ---"]

        expected = ['rainfall_mm', 'temperature_celsius', 'fertilizer_used', 'crop', 'yield_tons_per_hectare']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['rainfall_mm', 'temperature_celsius', 'yield_tons_per_hectare']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        avg_yield = df['yield_tons_per_hectare'].mean()
        rainfall_corr = df['rainfall_mm'].corr(df['yield_tons_per_hectare'])
        temp_corr = df['temperature_celsius'].corr(df['yield_tons_per_hectare'])

        metrics["average_yield_tons_ha"] = avg_yield
        metrics["rainfall_yield_correlation"] = rainfall_corr
        metrics["temp_yield_correlation"] = temp_corr

        insights.append(f"Average Yield (tons/ha): {avg_yield:.2f}")
        insights.append(f"Rainfall/Yield Correlation: {rainfall_corr:.2f}")
        insights.append(f"Temp/Yield Correlation: {temp_corr:.2f}")

        fig1 = px.scatter(df, x='rainfall_mm', y='yield_tons_per_hectare', color='crop',
                          title="Rainfall vs. Crop Yield",
                          labels={'rainfall_mm': 'Rainfall (mm)', 'yield_tons_per_hectare': 'Yield (tons/hectare)'})
        visualizations["rainfall_vs_yield"] = fig1.to_json()

        fig2 = px.scatter(df, x='temperature_celsius', y='yield_tons_per_hectare', color='crop',
                          title="Temperature vs. Crop Yield",
                          labels={'temperature_celsius': 'Temperature (°C)', 'yield_tons_per_hectare': 'Yield (tons/hectare)'})
        visualizations["temperature_vs_yield"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def state_level_agricultural_production_trend_analysis(df):
    analysis_type = "state_level_agricultural_production_trend_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- State-Level Agricultural Production Trend Analysis ---"]

        expected = ['state_name', 'crop_year', 'crop', 'area', 'production']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['crop_year', 'area', 'production']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        total_production = df['production'].sum()
        peak_year = df.groupby('crop_year')['production'].sum().idxmax()
        top_state = df.groupby('state_name')['production'].sum().idxmax()

        metrics["total_production"] = total_production
        metrics["peak_production_year"] = peak_year
        metrics["top_producing_state"] = str(top_state)

        insights.append(f"Total Production (all years): {total_production:,.0f}")
        insights.append(f"Peak Production Year: {peak_year}")
        insights.append(f"Top Producing State: {top_state}")

        production_over_time = df.groupby('crop_year')['production'].sum().reset_index()
        fig1 = px.line(production_over_time, x='crop_year', y='production',
                       title="Total Agricultural Production Over Time",
                       labels={'crop_year': 'Year', 'production': 'Total Production'})
        visualizations["production_over_time"] = fig1.to_json()

        production_by_state = df.groupby('state_name')['production'].sum().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.bar(production_by_state, x='state_name', y='production',
                       title="Top 10 Producing States",
                       labels={'state_name': 'State', 'production': 'Total Production'})
        visualizations["top_10_states"] = fig2.to_json()
        
        metrics["production_by_state_top10"] = json.loads(production_by_state.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def comprehensive_county_level_agricultural_output_analysis(df):
    analysis_type = "comprehensive_county_level_agricultural_output_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Comprehensive County-Level Agricultural Output Analysis ---"]

        expected = ['crop_production', 'livestock_production', 'aquaculture', 'maize', 'rice', 'beans', 'indigenous_cattle', 'exotic_chicken_layers', 'county']
        matched = fuzzy_match_column(df, expected)
        # This analysis is broad, so we don't strictly need all columns
        missing = [col for col in expected if matched[col] is None]

        if not any(matched.values()):
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "No relevant columns found, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})

        # Convert all expected columns that were found
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # --- Sector Analysis ---
        sector_cols = ['crop_production', 'livestock_production', 'aquaculture']
        found_sectors = [col for col in sector_cols if col in df.columns]
        
        if len(found_sectors) > 0:
            insights.append("--- Output by Sector ---")
            output_summary = df[found_sectors].sum().reset_index()
            output_summary.columns = ['Category', 'Total Production']
            metrics["output_by_sector"] = json.loads(output_summary.to_json(orient="split"))

            for _, row in output_summary.iterrows():
                insights.append(f"Total {row['Category'].replace('_', ' ').title()}: {row['Total Production']:,.0f}")
                metrics[f"total_{row['Category']}"] = row['Total Production']

            fig1 = px.pie(output_summary, names='Category', values='Total Production',
                          title="Share of Agricultural Output by Category", hole=0.4)
            visualizations["sector_output_pie"] = fig1.to_json()

        # --- Specific Product Analysis ---
        crop_cols = ['maize', 'rice', 'beans']
        found_crops = [col for col in crop_cols if col in df.columns]
        
        if len(found_crops) > 0:
            insights.append("--- Top Crop Outputs ---")
            top_crops = df[found_crops].sum(numeric_only=True).sort_values(ascending=False)
            metrics["top_crop_outputs"] = json.loads(top_crops.to_json(orient="split"))

            fig2 = px.bar(top_crops, x=top_crops.index, y=top_crops.values,
                          title="Top Crop Outputs", labels={'x': 'Crop', 'y': 'Total Production'})
            visualizations["top_crop_outputs_bar"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def environmental_factor_analysis_for_rice_production(df):
    analysis_type = "environmental_factor_analysis_for_rice_production"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Environmental Factor Analysis for Rice Production ---"]

        expected = ['annual_rain', 'nitrogen', 'potash', 'phosphate', 'loamy_alfisol', 'rice_production']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in expected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        avg_rice_prod = df['rice_production'].mean()
        rain_corr = df['annual_rain'].corr(df['rice_production'])
        nitrogen_corr = df['nitrogen'].corr(df['rice_production'])

        metrics["average_rice_production"] = avg_rice_prod
        metrics["rain_production_correlation"] = rain_corr
        metrics["nitrogen_production_correlation"] = nitrogen_corr

        insights.append(f"Average Rice Production: {avg_rice_prod:,.2f}")
        insights.append(f"Rain/Production Correlation: {rain_corr:.2f}")
        insights.append(f"Nitrogen/Production Correlation: {nitrogen_corr:.2f}")

        fig1 = px.scatter(df, x='annual_rain', y='rice_production',
                          title="Annual Rain vs. Rice Production", trendline='ols')
        visualizations["rain_vs_production"] = fig1.to_json()

        # Assuming loamy_alfisol is a binary/percentage indicator
        fig2 = px.scatter(df, x='nitrogen', y='rice_production', color='loamy_alfisol',
                          title="Nitrogen Level vs. Rice Production by Loamy Alfisol Presence",
                          labels={'nitrogen': 'Nitrogen Level', 'rice_production': 'Rice Production'})
        visualizations["nitrogen_vs_production_by_soil"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def district_level_rice_yield_and_soil_type_correlation_analysis(df):
    analysis_type = "district_level_rice_yield_and_soil_type_correlation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- District-Level Rice Yield and Soil Type Analysis ---"]

        expected = ['dist_name', 'rice_yield', 'annual_rain', 'nitrogen', 'vertisols']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['rice_yield', 'annual_rain', 'nitrogen', 'vertisols']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        avg_yield = df['rice_yield'].mean()
        top_district = df.groupby('dist_name')['rice_yield'].mean().idxmax()
        # Assuming vertisols is a percentage/fraction
        avg_yield_vertisols = df[df['vertisols'] > 0.5]['rice_yield'].mean() 

        metrics["average_rice_yield_kg_ha"] = avg_yield
        metrics["top_yielding_district"] = str(top_district)
        metrics["avg_yield_in_vertisols"] = avg_yield_vertisols

        insights.append(f"Average Rice Yield (Kg/ha): {avg_yield:,.2f}")
        insights.append(f"Top Yielding District: {top_district}")
        insights.append(f"Avg. Yield in Vertisols (where > 50%): {avg_yield_vertisols:,.2f}")

        top_districts = df.groupby('dist_name')['rice_yield'].mean().sort_values(ascending=False).head(10)
        fig1 = px.bar(top_districts, title="Top 10 Districts by Average Rice Yield")
        visualizations["top_10_districts_yield"] = fig1.to_json()
        metrics["top_10_districts_yield"] = json.loads(top_districts.to_json(orient="split"))

        # Create a binary category for better box plotting
        df['vertisols_presence'] = pd.qcut(df['vertisols'], q=2, labels=['Low Vertisols', 'High Vertisols'], duplicates='drop')
        fig2 = px.box(df, x='vertisols_presence', y='rice_yield', title="Rice Yield Distribution by Vertisols Soil Presence")
        visualizations["yield_by_vertisols"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def crop_type_recommendation_analysis(df):
    analysis_type = "crop_type_recommendation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Type Recommendation Analysis ---"]

        expected = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        num_crops = df['label'].nunique()
        avg_temp = df['temperature'].mean()
        avg_rainfall = df['rainfall'].mean()

        metrics["number_of_crop_types"] = num_crops
        metrics["average_temperature"] = avg_temp
        metrics["average_rainfall"] = avg_rainfall

        insights.append(f"Number of Crop Types: {num_crops}")
        insights.append(f"Average Temperature (°C): {avg_temp:.1f}")
        insights.append(f"Average Rainfall (mm): {avg_rainfall:.1f}")

        insights.append("Average Environmental Conditions per Crop")
        crop_conditions = df.groupby('label')[['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']].mean().reset_index()
        metrics["crop_environmental_profiles"] = json.loads(crop_conditions.to_json(orient="split"))
        
        fig1 = px.parallel_coordinates(crop_conditions, color="label",
                                       title="Environmental Profiles for Different Crops",
                                       labels={'n': 'Nitrogen', 'p': 'Phosphorus', 'k': 'Potassium', 'label': 'Crop'})
        visualizations["crop_profiles_parallel_coords"] = fig1.to_json()

        fig2 = px.box(df, x='label', y='rainfall', title="Rainfall Requirements by Crop")
        visualizations["rainfall_by_crop"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def national_land_use_change_analysis(df):
    analysis_type = "national_land_use_change_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- National Land Use Change Analysis ---"]

        expected = ['country', 'year', 'land_use', 'grazingland', 'cropland']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['year', 'grazingland', 'cropland']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['year', 'grazingland', 'cropland', 'country'], inplace=True) # Ensure key cols are present

        latest_year = df['year'].max()
        latest_df = df[df['year'] == latest_year]
        latest_cropland = latest_df['cropland'].sum()
        latest_grazingland = latest_df['grazingland'].sum()

        metrics["latest_year"] = latest_year
        metrics["latest_cropland_area"] = latest_cropland
        metrics["latest_grazingland_area"] = latest_grazingland

        insights.append(f"Most Recent Year: {latest_year}")
        insights.append(f"Latest Cropland Area: {latest_cropland:,.0f}")
        insights.append(f"Latest Grazingland Area: {latest_grazingland:,.0f}")

        # Aggregate data by year for national trend
        df_yearly = df.groupby('year')[['grazingland', 'cropland']].sum().reset_index()
        df_melted = df_yearly.melt(id_vars=['year'], value_vars=['grazingland', 'cropland'],
                                   var_name='land_type', value_name='area')
                                   
        fig1 = px.line(df_melted, x='year', y='area', color='land_type',
                       title="National Land Use Trends Over Time",
                       labels={'year': 'Year', 'area': 'Area (e.g., in sq km)', 'land_type': 'Land Use Type'})
        visualizations["national_land_use_trend"] = fig1.to_json()
        
        # Add insight about country-specific data
        countries = df['country'].unique()
        if len(countries) > 1:
            insights.append(f"Data covers {len(countries)} countries. Country-specific analysis is possible.")
            metrics["countries_present"] = countries.tolist()
            
            # Show top 5 countries by latest cropland
            top_5_cropland = latest_df.groupby('country')['cropland'].sum().nlargest(5).reset_index()
            fig_top5 = px.bar(top_5_cropland, x='country', y='cropland', title="Top 5 Countries by Cropland (Latest Year)")
            visualizations["top_5_countries_cropland"] = fig_top5.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def agricultural_census_crop_production_analysis(df):
    analysis_type = "agricultural_census_crop_production_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Agricultural Census Crop Production Analysis ---"]

        expected = ['census_year', 'county', 'type_of_crop', 'value']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df['census_year'] = pd.to_numeric(df['census_year'], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        total_value = df['value'].sum()
        latest_year = df['census_year'].max()
        top_crop = df.groupby('type_of_crop')['value'].sum().idxmax()

        metrics["total_production_value"] = total_value
        metrics["latest_census_year"] = latest_year
        metrics["top_crop_by_value"] = str(top_crop)

        insights.append(f"Total Production Value: ${total_value:,.0f}")
        insights.append(f"Latest Census Year: {latest_year}")
        insights.append(f"Top Crop by Value: {top_crop}")

        value_by_year = df.groupby('census_year')['value'].sum().reset_index()
        fig1 = px.bar(value_by_year, x='census_year', y='value', title="Total Production Value by Census Year")
        visualizations["value_by_year"] = fig1.to_json()

        value_by_crop = df.groupby('type_of_crop')['value'].sum().sort_values(ascending=False).head(10)
        fig2 = px.pie(value_by_crop, names=value_by_crop.index, values=value_by_crop.values,
                      title="Top 10 Crops by Production Value", hole=0.4)
        visualizations["top_10_crops_by_value"] = fig2.to_json()
        metrics["top_10_crops_by_value"] = json.loads(value_by_crop.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def insect_population_estimation_analysis(df):
    analysis_type = "insect_population_estimation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Insect Population Estimation Analysis ---"]

        expected = ['estimated_insects_count', 'crop_type', 'soil_type', 'pesticide_use_category', 'season']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        df['estimated_insects_count'] = pd.to_numeric(df['estimated_insects_count'], errors='coerce')
        df['pesticide_use_category'] = pd.to_numeric(df['pesticide_use_category'], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        avg_insect_count = df['estimated_insects_count'].mean()
        highest_infestation_crop = df.groupby('crop_type')['estimated_insects_count'].mean().idxmax()
        highest_infestation_season = df.groupby('season')['estimated_insects_count'].mean().idxmax()

        metrics["avg_estimated_insect_count"] = avg_insect_count
        metrics["most_infested_crop"] = str(highest_infestation_crop)
        metrics["most_infested_season"] = str(highest_infestation_season)

        insights.append(f"Avg. Estimated Insect Count: {avg_insect_count:,.0f}")
        insights.append(f"Most Infested Crop: {highest_infestation_crop}")
        insights.append(f"Most Infested Season: {highest_infestation_season}")

        fig1 = px.box(df, x='crop_type', y='estimated_insects_count', color='season',
                      title="Insect Counts by Crop Type and Season")
        visualizations["insects_by_crop_and_season"] = fig1.to_json()

        fig2 = px.violin(df, x='pesticide_use_category', y='estimated_insects_count',
                         box=True,
                         title="Insect Counts by Pesticide Use Category",
                         labels={'pesticide_use_category': 'Pesticide Use (1=Never, 2=Prev. Used, 3=Currently Using)'})
        visualizations["insects_by_pesticide_use"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def crop_damage_prediction_from_pest_infestation(df):
    analysis_type = "crop_damage_prediction_from_pest_infestation"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Damage Prediction from Pest Infestation ---"]

        expected = ['estimated_insects_count', 'crop_type', 'pesticide_use_category', 'number_doses_week', 'crop_damage']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in ['estimated_insects_count', 'number_doses_week', 'crop_damage']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)
        df['crop_damage_label'] = df['crop_damage'].map({0: 'Alive', 1: 'Damaged (Other)', 2: 'Damaged (Pesticide)'})

        damage_rate = (df['crop_damage'] > 0).mean() * 100
        insect_damage_corr = df['estimated_insects_count'].corr(df['crop_damage'])

        metrics["overall_damage_rate_perc"] = damage_rate
        metrics["insect_count_damage_correlation"] = insect_damage_corr

        insights.append(f"Overall Damage Rate: {damage_rate:.2f}%")
        insights.append(f"Insect Count/Damage Correlation: {insect_damage_corr:.2f}")

        fig1 = px.scatter(df, x='estimated_insects_count', y='number_doses_week', color='crop_damage_label',
                          title="Insect Count vs. Pesticide Doses by Damage Type",
                          labels={'estimated_insects_count': 'Estimated Insect Count', 'number_doses_week': 'Doses per Week'})
        visualizations["insects_vs_doses_by_damage"] = fig1.to_json()

        damage_dist = df['crop_damage_label'].value_counts().reset_index()
        damage_dist.columns = ['Damage Type', 'Count'] # Fix for new pandas
        fig2 = px.pie(damage_dist, names='Damage Type', values='Count', title="Distribution of Crop Damage Types")
        visualizations["damage_type_distribution"] = fig2.to_json()
        metrics["damage_type_distribution"] = json.loads(damage_dist.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def fertilizer_recommendation_system_analysis(df):
    analysis_type = "fertilizer_recommendation_system_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Fertilizer Recommendation System Analysis ---"]

        expected = ['temperature', 'humidity', 'nitrogen', 'potassium', 'phosphorous', 'crop_type', 'fertilizer_name']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        base_cols = ['temperature', 'humidity', 'nitrogen', 'potassium', 'phosphorous']
        for col in base_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        num_fertilizers = df['fertilizer_name'].nunique()
        top_fertilizer = df['fertilizer_name'].mode()[0]
        top_crop = df['crop_type'].mode()[0]

        metrics["unique_fertilizer_types"] = num_fertilizers
        metrics["most_common_fertilizer"] = str(top_fertilizer)
        metrics["most_common_crop"] = str(top_crop)

        insights.append(f"Unique Fertilizer Types: {num_fertilizers}")
        insights.append(f"Most Common Fertilizer: {top_fertilizer}")
        insights.append(f"Most Common Crop: {top_crop}")

        insights.append("Nutrient Requirements by Crop Type")
        # Find other env columns that might exist
        other_env_cols = [col for col in ['ph', 'rainfall'] if col in df.columns]
        group_cols = base_cols + other_env_cols
        crop_conditions = df.groupby('crop_type')[group_cols].mean().reset_index()
        metrics["crop_nutrient_profiles"] = json.loads(crop_conditions.to_json(orient="split"))
        
        fig1 = px.parallel_coordinates(crop_conditions, color="crop_type",
                                       title="Environmental Profiles for Different Crops",
                                       labels={'nitrogen': 'Nitrogen', 'phosphorous': 'Phosphorus', 'potassium': 'Potassium', 'crop_type': 'Crop'})
        visualizations["crop_profiles_parallel_coords"] = fig1.to_json()

        fertilizer_counts = df['fertilizer_name'].value_counts().reset_index()
        fertilizer_counts.columns = ['Fertilizer', 'Count'] # Fix for new pandas
        fig2 = px.bar(fertilizer_counts, x='Fertilizer', y='Count',
                      title="Frequency of Recommended Fertilizers")
        visualizations["fertilizer_frequency"] = fig2.to_json()
        metrics["fertilizer_frequency"] = json.loads(fertilizer_counts.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def agricultural_yield_and_price_fluctuation_analysis(df):
    analysis_type = "agricultural_yield_and_price_fluctuation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Agricultural Yield and Price Fluctuation Analysis ---"]

        expected = ['year', 'crops', 'yeilds', 'price', 'rainfall', 'temperature']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['year', 'yeilds', 'price', 'rainfall', 'temperature']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=expected, inplace=True)

        avg_yield = df['yeilds'].mean()
        avg_price = df['price'].mean()
        yield_price_corr = df['yeilds'].corr(df['price'])

        metrics["average_yield"] = avg_yield
        metrics["average_price"] = avg_price
        metrics["yield_price_correlation"] = yield_price_corr

        insights.append(f"Average Yield: {avg_yield:,.2f}")
        insights.append(f"Average Price: ${avg_price:,.2f}")
        insights.append(f"Yield/Price Correlation: {yield_price_corr:.2f} (Note: negative correlation is common)")

        df_yearly = df.groupby('year')[['yeilds', 'price']].mean().reset_index()
        fig1 = px.line(df_yearly, x='year', y=['yeilds', 'price'],
                       title="Average Yield and Price Over Time", facet_row="variable",
                       labels={'value': 'Value', 'year': 'Year'})
        fig1.update_yaxes(matches=None) # Allow y-axes to scale independently
        visualizations["yield_price_over_time"] = fig1.to_json()

        fig2 = px.scatter(df, x='rainfall', y='yeilds', color='crops',
                          title="Rainfall vs. Yields by Crop",
                          labels={'rainfall': 'Rainfall', 'yeilds': 'Yields'})
        visualizations["rainfall_vs_yield"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def long_term_tillage_and_crop_rotation_impact_analysis(df):
    analysis_type = "long_term_tillage_and_crop_rotation_impact_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Long-Term Tillage and Crop Rotation Impact Analysis ---"]

        expected = ['tillage', 'endingcroppingsystem', 'grain2018', 'grain2019', 'grain2020']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['grain2018', 'grain2019', 'grain2020']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_yield_2020 = df['grain2020'].mean()
        df['yield_gain_18_20'] = df['grain2020'] - df['grain2018']
        avg_yield_gain = df['yield_gain_18_20'].mean()

        metrics["avg_grain_yield_2020"] = avg_yield_2020
        metrics["avg_yield_gain_2018_2020"] = avg_yield_gain

        insights.append(f"Avg. Grain Yield in 2020: {avg_yield_2020:.2f}")
        insights.append(f"Avg. Yield Gain (2018-2020): {avg_yield_gain:.2f}")

        fig1 = px.box(df, x='tillage', y='grain2020',
                      title="Impact of Tillage System on 2020 Grain Yield")
        visualizations["tillage_impact_on_2020_yield"] = fig1.to_json()
        
        metrics["yield_by_tillage"] = json.loads(df.groupby('tillage')['grain2020'].mean().to_json(orient="split"))

        fig2 = px.box(df, x='endingcroppingsystem', y='yield_gain_18_20',
                      title="Yield Gain (2018-2020) by Ending Cropping System",
                      labels={'endingcroppingsystem': 'Cropping System', 'yield_gain_18_20': 'Yield Gain'})
        visualizations["yield_gain_by_cropping_system"] = fig2.to_json()
        
        metrics["yield_gain_by_system"] = json.loads(df.groupby('endingcroppingsystem')['yield_gain_18_20'].mean().to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def soil_compaction_and_organic_matter_analysis(df):
    analysis_type = "soil_compaction_and_organic_matter_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Soil Compaction and Organic Matter Analysis ---"]

        expected = ['depth_upper', 'coneindex', 'tillage', 'om_perc', 'bd'] # BD = Bulk Density
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['depth_upper', 'coneindex', 'om_perc', 'bd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_cone_index = df['coneindex'].mean() # Cone index measures compaction
        avg_om = df['om_perc'].mean()
        avg_bd = df['bd'].mean()

        metrics["average_cone_index_compaction"] = avg_cone_index
        metrics["average_organic_matter_perc"] = avg_om
        metrics["average_bulk_density"] = avg_bd

        insights.append(f"Average Cone Index (Compaction): {avg_cone_index:.2f}")
        insights.append(f"Average Organic Matter: {avg_om:.2f}%")
        insights.append(f"Average Bulk Density: {avg_bd:.2f}")

        fig1 = px.scatter(df, x='om_perc', y='bd', color='tillage',
                          title="Organic Matter vs. Bulk Density by Tillage Type",
                          labels={'om_perc': 'Organic Matter %', 'bd': 'Bulk Density'})
        visualizations["om_vs_bulk_density"] = fig1.to_json()

        # Group data for line plot to avoid clutter
        df_grouped = df.groupby(['tillage', 'depth_upper'])['coneindex'].mean().reset_index()
        fig2 = px.line(df_grouped.sort_values('depth_upper'), x='depth_upper', y='coneindex', color='tillage',
                       title="Soil Compaction (Cone Index) by Depth and Tillage")
        visualizations["compaction_by_depth_tillage"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def monthly_temperature_variation_analysis(df):
    analysis_type = "monthly_temperature_variation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Monthly Temperature Variation Analysis ---"]

        # This is a very specific analysis, we try to find any month
        base_expected = ['year']
        month_cols = [col for col in df.columns if any(c in col.lower() for c in ['_tmin', '_tmax', '_tmean'])]
        if not month_cols:
             # Try a different pattern
             month_cols = [col for col in df.columns if any(c in col.lower() for c in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])]
        
        if not month_cols:
            insights.append("Could not find any monthly temperature columns (e.g., 'may_tmin_c').")
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = {}
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data
            
        expected = base_expected + month_cols
        matched = fuzzy_match_column(df, expected)
        df = df.rename(columns={v: k for k, v in matched.items() if v})

        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Analyze the first set of min/mean/max found
        tmin_col = next((col for col in month_cols if 'tmin' in col.lower()), None)
        tmax_col = next((col for col in month_cols if 'tmax' in col.lower()), None)
        tmean_col = next((col for col in month_cols if 'tmean' in col.lower()), None)
        
        if not (tmin_col and tmax_col and tmean_col):
             insights.append("Could not find a full set of Tmin, Tmax, and Tmean columns to analyze.")
             # We can just use the first month col
             tmean_col = month_cols[0]
             avg_mean_temp = df[tmean_col].mean()
             metrics[f"average_{tmean_col}"] = avg_mean_temp
             insights.append(f"Avg. {tmean_col}: {avg_mean_temp:.1f}°C")
             df_melted = df.melt(id_vars='year', value_vars=[tmean_col],
                                var_name='temp_type', value_name='temperature')
        else:
            avg_mean_temp = df[tmean_col].mean()
            max_temp = df[tmax_col].max()
            min_temp = df[tmin_col].min()

            metrics[f"average_{tmean_col}"] = avg_mean_temp
            metrics[f"highest_{tmax_col}"] = max_temp
            metrics[f"lowest_{tmin_col}"] = min_temp

            insights.append(f"Avg. Mean Temp ({tmean_col}): {avg_mean_temp:.1f}°C")
            insights.append(f"Highest Max Temp ({tmax_col}): {max_temp:.1f}°C")
            insights.append(f"Lowest Min Temp ({tmin_col}): {min_temp:.1f}°C")
            
            df_melted = df.melt(id_vars='year', value_vars=[tmin_col, tmean_col, tmax_col],
                                var_name='temp_type', value_name='temperature')

        fig1 = px.line(df_melted, x='year', y='temperature', color='temp_type',
                       title="Temperature Trends Over Years")
        visualizations["monthly_temp_trends"] = fig1.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def daily_precipitation_data_analysis(df):
    analysis_type = "daily_precipitation_data_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Daily Precipitation Data Analysis ---"]

        expected = ['date', 'doy', 'month', 'year', 'mm']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['mm'] = pd.to_numeric(df['mm'], errors='coerce')
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Check if 'month' is present, if not, create it
        if 'month' not in df.columns and 'date' in df.columns:
            df['month'] = df['date'].dt.month
            
        df.dropna(inplace=True)

        total_precip = df['mm'].sum()
        wettest_day_idx = df['mm'].idxmax()
        wettest_day = df.loc[wettest_day_idx]['date'].strftime('%Y-%m-%d')
        wettest_day_mm = df['mm'].max()

        metrics["total_precipitation_mm"] = total_precip
        metrics["wettest_day"] = wettest_day
        metrics["wettest_day_mm"] = wettest_day_mm

        insights.append(f"Total Precipitation Recorded: {total_precip:,.1f} mm")
        insights.append(f"Wettest Day: {wettest_day} with {wettest_day_mm:.1f} mm")

        yearly_precip = df.groupby('year')['mm'].sum().reset_index()
        fig1 = px.bar(yearly_precip, x='year', y='mm', title="Total Annual Precipitation")
        visualizations["total_annual_precipitation"] = fig1.to_json()

        monthly_precip = df.groupby('month')['mm'].mean().reset_index()
        # Map month number to month name for better plotting
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        monthly_precip['month_name'] = monthly_precip['month'].map(month_map)
        monthly_precip.sort_values(by='month', inplace=True)

        fig2 = px.bar(monthly_precip, x='month_name', y='mm', title="Average Daily Precipitation by Month")
        visualizations["average_daily_precipitation_by_month"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def soil_carbon_and_nitrogen_dynamics_analysis(df):
    analysis_type = "soil_carbon_and_nitrogen_dynamics_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Soil Carbon and Nitrogen Dynamics Analysis ---"]

        expected = ['tillage', 'depth', 'totaln_perc', 'totalc_perc', 'organicc_perc', 'ph']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['totaln_perc', 'totalc_perc', 'organicc_perc', 'ph']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        df['c_n_ratio'] = df['totalc_perc'] / df['totaln_perc']
        avg_c_n_ratio = df['c_n_ratio'].mean()
        avg_ph = df['ph'].mean()

        metrics["average_c_n_ratio"] = avg_c_n_ratio
        metrics["average_soil_ph"] = avg_ph
        
        metrics["avg_c_by_tillage"] = json.loads(df.groupby('tillage')['totalc_perc'].mean().to_json(orient="split"))
        metrics["avg_n_by_tillage"] = json.loads(df.groupby('tillage')['totaln_perc'].mean().to_json(orient="split"))


        insights.append(f"Average C:N Ratio: {avg_c_n_ratio:.2f}")
        insights.append(f"Average Soil pH: {avg_ph:.2f}")

        fig1 = px.box(df, x='tillage', y=['totalc_perc', 'totaln_perc'],
                      title="Total Carbon and Nitrogen Percentage by Tillage System")
        visualizations["c_n_by_tillage"] = fig1.to_json()

        fig2 = px.scatter(df, x='ph', y='organicc_perc', color='tillage',
                          trendline='ols',
                          title="Organic Carbon vs. pH by Tillage System",
                          labels={'ph': 'Soil pH', 'organicc_perc': 'Organic Carbon %'})
        visualizations["oc_vs_ph_by_tillage"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def soil_chemistry_and_nutrient_level_change_analysis(df):
    analysis_type = "soil_chemistry_and_nutrient_level_change_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Soil Chemistry and Nutrient Level Change Analysis ---"]

        expected = ['rotn', 'tillage', 'nrate', 'delta_ph', 'delta_kcmol', 'delta_cacmol']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['nrate', 'delta_ph', 'delta_kcmol', 'delta_cacmol']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_ph_change = df['delta_ph'].mean()
        avg_k_change = df['delta_kcmol'].mean()
        avg_ca_change = df['delta_cacmol'].mean()

        metrics["average_ph_change"] = avg_ph_change
        metrics["avg_potassium_change_cmol_kg"] = avg_k_change
        metrics["avg_calcium_change_cmol_kg"] = avg_ca_change

        insights.append(f"Average pH Change: {avg_ph_change:.3f}")
        insights.append(f"Avg. Potassium Change (cmol/kg): {avg_k_change:.3f}")
        insights.append(f"Avg. Calcium Change (cmol/kg): {avg_ca_change:.3f}")

        fig1 = px.box(df, x='tillage', y='delta_ph', color='rotn',
                      title="Change in Soil pH by Tillage and Rotation")
        visualizations["ph_change_by_tillage_rotation"] = fig1.to_json()

        fig2 = px.scatter(df, x='nrate', y='delta_ph', color='tillage',
                          trendline='ols',
                          title="Impact of Nitrogen Rate on pH Change",
                          labels={'nrate': 'Nitrogen Rate', 'delta_ph': 'Change in pH'})
        visualizations["nrate_impact_on_ph_change"] = fig2.to_json()
        
        nrate_ph_corr = df['nrate'].corr(df['delta_ph'])
        metrics["nrate_ph_change_correlation"] = nrate_ph_corr
        insights.append(f"Correlation between N Rate and pH Change: {nrate_ph_corr:.3f}")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def plant_dissection_and_larval_mortality_analysis(df):
    analysis_type = "plant_dissection_and_larval_mortality_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Plant Dissection and Larval Mortality Analysis ---"]

        expected = ['line', 'total_#plants', 'stunting_score_mean', 'total_dead_mean', 'total_live_mean', 'larval_mortality_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            if col != 'line':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_mortality = df['larval_mortality_perc'].mean()
        most_resistant_line = df.loc[df['larval_mortality_perc'].idxmax()]['line']
        most_stunted_line = df.loc[df['stunting_score_mean'].idxmax()]['line']

        metrics["average_larval_mortality_perc"] = avg_mortality
        metrics["most_resistant_line_by_mortality"] = str(most_resistant_line)
        metrics["most_stunted_line"] = str(most_stunted_line)

        insights.append(f"Average Larval Mortality: {avg_mortality:.2f}%")
        insights.append(f"Most Resistant Line (by Mortality): {most_resistant_line}")
        insights.append(f"Most Stunted Line: {most_stunted_line}")

        mortality_by_line = df.groupby('line')['larval_mortality_perc'].mean().sort_values(ascending=False).reset_index()
        fig1 = px.bar(mortality_by_line, x='line', y='larval_mortality_perc',
                      title="Average Larval Mortality by Plant Line")
        visualizations["mortality_by_line"] = fig1.to_json()
        metrics["mortality_by_line"] = json.loads(mortality_by_line.to_json(orient="split"))

        fig2 = px.scatter(df, x='stunting_score_mean', y='larval_mortality_perc', hover_name='line',
                          trendline='ols',
                          title="Stunting Score vs. Larval Mortality",
                          labels={'stunting_score_mean': 'Mean Stunting Score', 'larval_mortality_perc': 'Larval Mortality %'})
        visualizations["stunting_vs_mortality"] = fig2.to_json()
        
        stunting_mortality_corr = df['stunting_score_mean'].corr(df['larval_mortality_perc'])
        metrics["stunting_mortality_correlation"] = stunting_mortality_corr
        insights.append(f"Correlation between Stunting and Mortality: {stunting_mortality_corr:.3f}")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def detailed_pest_resistance_scoring_analysis(df):
    analysis_type = "detailed_pest_resistance_scoring_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Detailed Pest Resistance Scoring Analysis ---"]

        expected = ['line', 'plant#', 'stunting_score', 'resistance_score_leaves', 'total_live', 'total_dead', 'total_larvae']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            if col not in ['line', 'plant#']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_total_larvae = df['total_larvae'].mean()
        avg_resistance_score = df['resistance_score_leaves'].mean()

        metrics["average_larvae_count_per_plant"] = avg_total_larvae
        metrics["average_leaf_resistance_score"] = avg_resistance_score

        insights.append(f"Average Larvae Count per Plant: {avg_total_larvae:.1f}")
        insights.append(f"Average Leaf Resistance Score: {avg_resistance_score:.2f}")

        fig1 = px.density_heatmap(df, x="total_live", y="total_dead",
                                  marginal_x="histogram", marginal_y="histogram",
                                  title="Heatmap of Live vs. Dead Larvae Counts")
        visualizations["live_vs_dead_heatmap"] = fig1.to_json()

        fig2 = px.box(df, x='line', y='total_larvae',
                      title="Total Larvae Distribution by Plant Line")
        visualizations["larvae_by_line"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def coffee_plantation_area_analysis(df):
    analysis_type = "coffee_plantation_area_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Coffee Plantation Area Analysis (Arabica vs. Robusta) ---"]

        expected = ['year', 'arabica_in_hectares', 'robusta_in_hectares', 'total_in_hectares']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        df = df.sort_values('year')
        latest_year = df['year'].max()
        latest_data = df[df['year'] == latest_year].iloc[0] # Get first row for latest year
        latest_total = latest_data['total_in_hectares']
        latest_arabica_perc = (latest_data['arabica_in_hectares'] / latest_total * 100)
        
        first_year = df['year'].min()
        first_data = df[df['year'] == first_year].iloc[0]
        first_total = first_data['total_in_hectares']
        change_total = latest_total - first_total
        change_perc = (change_total / first_total) * 100

        metrics["latest_year"] = latest_year
        metrics["total_area_ha_latest_year"] = latest_total
        metrics["arabica_share_latest_year_perc"] = latest_arabica_perc
        metrics["total_area_change_since_first_year"] = change_total
        metrics["total_area_change_perc_since_first_year"] = change_perc


        insights.append(f"Most Recent Year: {latest_year}")
        insights.append(f"Total Area (ha) in {latest_year}: {latest_total:,.0f}")
        insights.append(f"Arabica Share in {latest_year}: {latest_arabica_perc:.1f}%")
        insights.append(f"Total area has changed by {change_total:,.0f} ha ({change_perc:.1f}%) since {first_year}.")

        df_melted = df.melt(id_vars='year', value_vars=['arabica_in_hectares', 'robusta_in_hectares'],
                            var_name='coffee_type', value_name='area_ha')
        fig1 = px.area(df_melted, x='year', y='area_ha', color='coffee_type',
                       title="Coffee Plantation Area by Type Over Time")
        visualizations["area_by_type_over_time"] = fig1.to_json()

        df['arabica_share'] = df['arabica_in_hectares'] / df['total_in_hectares'] * 100
        fig2 = px.line(df, x='year', y='arabica_share', title="Percentage Share of Arabica Over Time")
        visualizations["arabica_share_over_time"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def comprehensive_district_level_crop_production_and_yield_analysis(df):
    analysis_type = "comprehensive_district_level_crop_production_and_yield_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Comprehensive District-Level Crop Production and Yield Analysis ---"]

        expected = ['dist_name', 'rice_production_1000_tons', 'rice_yield_kg_per_ha', 'wheat_production_1000_tons', 'wheat_yield_kg_per_ha', 'maize_production_1000_tons']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            if 'dist_name' not in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        total_rice_prod = df['rice_production_1000_tons'].sum()
        avg_wheat_yield = df['wheat_yield_kg_per_ha'].mean()
        top_maize_dist = df.groupby('dist_name')['maize_production_1000_tons'].sum().idxmax()

        metrics["total_rice_production_k_tons"] = total_rice_prod
        metrics["average_wheat_yield_kg_ha"] = avg_wheat_yield
        metrics["top_maize_district"] = str(top_maize_dist)

        insights.append(f"Total Rice Production (k-tons): {total_rice_prod:,.0f}")
        insights.append(f"Avg. Wheat Yield (kg/ha): {avg_wheat_yield:,.1f}")
        insights.append(f"Top Maize District: {top_maize_dist}")

        top_districts_rice = df.groupby('dist_name')['rice_production_1000_tons'].sum().nlargest(10).reset_index()
        fig1 = px.bar(top_districts_rice, x='dist_name', y='rice_production_1000_tons',
                      title="Top 10 Districts by Rice Production")
        visualizations["top_10_districts_rice_production"] = fig1.to_json()
        metrics["top_10_districts_rice_production"] = json.loads(top_districts_rice.to_json(orient="split"))

        fig2 = px.scatter(df, x='rice_yield_kg_per_ha', y='wheat_yield_kg_per_ha',
                          hover_name='dist_name', title="Rice Yield vs. Wheat Yield by District")
        visualizations["rice_vs_wheat_yield"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def nutrient_retention_factor_analysis_in_foods(df):
    analysis_type = "nutrient_retention_factor_analysis_in_foods"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Nutrient Retention Factor Analysis in Foods ---"]

        expected = ['fdgrp_cd', 'nutrdesc', 'retn_factor', 'date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['retn_factor'] = pd.to_numeric(df['retn_factor'], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(inplace=True)

        avg_retention = df['retn_factor'].mean()
        most_retained_nutrient = df.loc[df['retn_factor'].idxmax()]['nutrdesc']
        least_retained_nutrient = df.loc[df['retn_factor'].idxmin()]['nutrdesc']
        
        metrics["average_retention_factor"] = avg_retention
        metrics["most_retained_nutrient"] = str(most_retained_nutrient)
        metrics["least_retained_nutrient"] = str(least_retained_nutrient)

        insights.append(f"Average Retention Factor: {avg_retention:.2f}")
        insights.append(f"Most Retained Nutrient: {most_retained_nutrient} (Factor: {df['retn_factor'].max():.2f})")
        insights.append(f"Least Retained Nutrient: {least_retained_nutrient} (Factor: {df['retn_factor'].min():.2f})")

        top_nutrients = df.groupby('nutrdesc')['retn_factor'].mean().nlargest(15).reset_index()
        fig1 = px.bar(top_nutrients, x='nutrdesc', y='retn_factor',
                      title="Top 15 Nutrients by Average Retention Factor")
        visualizations["top_15_nutrients_by_retention"] = fig1.to_json()
        metrics["top_15_nutrients_by_retention"] = json.loads(top_nutrients.to_json(orient="split"))

        fig2 = px.box(df, x='fdgrp_cd', y='retn_factor', title="Nutrient Retention by Food Group Code")
        visualizations["retention_by_food_group"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def food_cooking_yield_and_nutrient_change_analysis(df):
    analysis_type = "food_cooking_yield_and_nutrient_change_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Food Cooking Yield and Nutrient Change Analysis ---"]

        expected = ['food_group_code', 'preparation_method1', 'cooking_yield_perc', 'moisture_gain_loss_perc', 'fat_gain_loss_perc']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['cooking_yield_perc', 'moisture_gain_loss_perc', 'fat_gain_loss_perc']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_yield = df['cooking_yield_perc'].mean()
        avg_moisture_change = df['moisture_gain_loss_perc'].mean()
        avg_fat_change = df['fat_gain_loss_perc'].mean()

        metrics["average_cooking_yield_perc"] = avg_yield
        metrics["avg_moisture_change_perc"] = avg_moisture_change
        metrics["avg_fat_change_perc"] = avg_fat_change

        insights.append(f"Average Cooking Yield: {avg_yield:.1f}%")
        insights.append(f"Avg. Moisture Change: {avg_moisture_change:.1f}%")
        insights.append(f"Avg. Fat Change: {avg_fat_change:.1f}%")

        yield_by_prep = df.groupby('preparation_method1')['cooking_yield_perc'].mean().sort_values().reset_index()
        fig1 = px.bar(yield_by_prep, x='cooking_yield_perc', y='preparation_method1',
                      orientation='h', title="Average Cooking Yield by Preparation Method")
        visualizations["yield_by_preparation_method"] = fig1.to_json()
        metrics["yield_by_preparation_method"] = json.loads(yield_by_prep.to_json(orient="split"))

        fig2 = px.scatter(df, x='moisture_gain_loss_perc', y='fat_gain_loss_perc', color='preparation_method1',
                          title="Moisture vs. Fat Change During Cooking")
        visualizations["moisture_vs_fat_change"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def field_operation_and_tillage_log_analysis(df):
    analysis_type = "field_operation_and_tillage_log_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Field Operation and Tillage Log Analysis ---"]

        expected = ['siteid', 'plotid', 'date', 'cashcrop', 'tillage_type', 'depth']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
        df.dropna(inplace=True)

        common_tillage = df['tillage_type'].mode()[0]
        common_crop = df['cashcrop'].mode()[0]
        avg_depth = df['depth'].mean()

        metrics["most_common_tillage"] = str(common_tillage)
        metrics["most_common_crop"] = str(common_crop)
        metrics["average_tillage_depth"] = avg_depth

        insights.append(f"Most Common Tillage: {common_tillage}")
        insights.append(f"Most Common Crop: {common_crop}")
        insights.append(f"Average Tillage Depth: {avg_depth:.2f}")

        tillage_counts = df['tillage_type'].value_counts().reset_index()
        tillage_counts.columns = ['Tillage Type', 'Count']
        fig1 = px.pie(tillage_counts, names='Tillage Type', values='Count', title="Distribution of Tillage Types")
        visualizations["tillage_type_distribution"] = fig1.to_json()
        metrics["tillage_type_distribution"] = json.loads(tillage_counts.to_json(orient="split"))

        fig2 = px.box(df, x='tillage_type', y='depth', title="Tillage Depth by Type")
        visualizations["tillage_depth_by_type"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def fertilizer_and_manure_application_analysis(df):
    analysis_type = "fertilizer_and_manure_application_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Fertilizer and Manure Application Analysis ---"]

        expected = ['cashcrop', 'operation_type', 'fertilizer_form', 'fertilizer_rate', 'manure_source', 'manure_rate', 'nitrogen_elem']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['fertilizer_rate', 'manure_rate', 'nitrogen_elem']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        # We don't dropna here, as a row might have fertilizer OR manure
        df.fillna(0, inplace=True) # Assume NA means 0 for rates

        avg_n_rate = df['nitrogen_elem'].mean()
        avg_fert_rate = df[df['fertilizer_rate'] > 0]['fertilizer_rate'].mean()
        avg_manure_rate = df[df['manure_rate'] > 0]['manure_rate'].mean()

        metrics["average_nitrogen_rate"] = avg_n_rate
        metrics["average_fertilizer_rate_when_used"] = avg_fert_rate
        metrics["average_manure_rate_when_used"] = avg_manure_rate

        insights.append(f"Average Nitrogen Rate (elemental): {avg_n_rate:.2f}")
        insights.append(f"Average Fertilizer Rate (when used): {avg_fert_rate:.2f}")
        insights.append(f"Average Manure Rate (when used): {avg_manure_rate:.2f}")

        fig1 = px.scatter(df, x='fertilizer_rate', y='nitrogen_elem', color='cashcrop',
                          title="Elemental Nitrogen vs. Total Fertilizer Rate")
        visualizations["nitrogen_vs_fertilizer_rate"] = fig1.to_json()

        rate_by_crop = df.groupby('cashcrop')[['fertilizer_rate', 'manure_rate', 'nitrogen_elem']].mean().reset_index()
        fig2 = px.bar(rate_by_crop, x='cashcrop', y=['fertilizer_rate', 'manure_rate'],
                      title="Average Application Rates by Crop", barmode='group')
        visualizations["rates_by_crop"] = fig2.to_json()
        metrics["rates_by_crop"] = json.loads(rate_by_crop.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def no_till_farming_practice_impact_analysis(df):
    analysis_type = "no_till_farming_practice_impact_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- No-Till Farming Practice Impact Analysis ---"]

        expected = ['siteid', 'plotid', 'year_crop', 'crop', 'notill']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['notill'] = df['notill'].astype(str)

        # Standardize 'notill' column
        df['notill_std'] = df['notill'].str.lower().map({'yes': 'No-Till', '1': 'No-Till', 'true': 'No-Till',
                                                        'no': 'Conventional', '0': 'Conventional', 'false': 'Conventional'
                                                       }).fillna('Unknown')
        
        no_till_perc = (df['notill_std'] == 'No-Till').mean() * 100

        metrics["percentage_of_no_till_plots"] = no_till_perc
        insights.append(f"Percentage of No-Till Plots: {no_till_perc:.1f}%")
        insights.append("Note: This analysis is basic, as yield or soil data is not present in these columns. Combine with other datasets for impact analysis.")

        notill_counts = df['notill_std'].value_counts().reset_index()
        notill_counts.columns = ['Tillage Practice', 'Count']
        fig1 = px.pie(notill_counts, names='Tillage Practice', values='Count', title="Proportion of No-Till vs. Conventional Till Plots")
        visualizations["tillage_practice_distribution"] = fig1.to_json()
        metrics["tillage_practice_distribution"] = json.loads(notill_counts.to_json(orient="split"))

        crop_counts = df.groupby(['crop', 'notill_std']).size().reset_index(name='count')
        fig2 = px.bar(crop_counts, x='crop', y='count', color='notill_std',
                      title="Tillage Practice by Crop Type", barmode='group')
        visualizations["tillage_by_crop_type"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def cash_crop_planting_schedule_analysis(df):
    analysis_type = "cash_crop_planting_schedule_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Cash Crop Planting Schedule Analysis ---"]

        expected = ['plotid', 'year_calendar', 'date', 'cashcrop']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(inplace=True)
        df['day_of_year'] = df['date'].dt.dayofyear
        df['year_calendar'] = df['year_calendar'].astype(str) # Treat as category

        avg_planting_doy = df['day_of_year'].mean()
        metrics["average_planting_day_of_year"] = avg_planting_doy
        insights.append(f"Average Planting Day of Year: {avg_planting_doy:.0f}")

        fig1 = px.histogram(df, x='day_of_year', color='cashcrop', marginal='box',
                            barmode='overlay',
                            title="Distribution of Planting Dates by Crop")
        visualizations["planting_date_distribution"] = fig1.to_json()

        yearly_planting = df.groupby(['year_calendar', 'cashcrop'])['day_of_year'].mean().reset_index()
        fig2 = px.line(yearly_planting, x='year_calendar', y='day_of_year', color='cashcrop',
                       title="Trend of Average Planting Day Over Years")
        visualizations["planting_date_trend"] = fig2.to_json()
        metrics["planting_date_trend"] = json.loads(yearly_planting.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def plant_hybrid_and_seeding_rate_performance_analysis(df):
    analysis_type = "plant_hybrid_and_seeding_rate_performance_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Plant Hybrid and Seeding Rate Performance Analysis ---"]

        expected = ['cashcrop', 'plant_hybrid', 'plant_maturity', 'plant_rate', 'plant_rate_units']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['plant_maturity', 'plant_rate']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_seeding_rate = df['plant_rate'].mean()
        most_common_hybrid = df['plant_hybrid'].mode()[0]
        
        metrics["average_seeding_rate"] = avg_seeding_rate
        metrics["most_common_hybrid"] = str(most_common_hybrid)
        
        insights.append(f"Average Seeding Rate: {avg_seeding_rate:,.0f} {df['plant_rate_units'].mode()[0]}")
        insights.append(f"Most Common Hybrid: {most_common_hybrid}")
        insights.append("Note: Analysis is limited without yield/performance data linked to these rates.")

        # Limit to top 5 hybrids for visual clarity
        top_5_hybrids = df['plant_hybrid'].value_counts().nlargest(5).index
        df_top5 = df[df['plant_hybrid'].isin(top_5_hybrids)]

        fig1 = px.box(df_top5, x='cashcrop', y='plant_rate', color='plant_hybrid',
                      title="Seeding Rate by Crop and Hybrid (Top 5 Hybrids)")
        visualizations["seeding_rate_by_crop_hybrid"] = fig1.to_json()

        fig2 = px.scatter(df, x='plant_maturity', y='plant_rate', color='cashcrop',
                          title="Seeding Rate vs. Plant Maturity",
                          labels={'plant_maturity': 'Plant Maturity (days)', 'plant_rate': 'Seeding Rate'})
        visualizations["seeding_rate_vs_maturity"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def irrigation_scheduling_and_water_usage_analysis(df):
    analysis_type = "irrigation_scheduling_and_water_usage_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Irrigation Scheduling and Water Usage Analysis ---"]

        expected = ['irrigation_method', 'year_calendar', 'date_irrigation_start', 'date_irrigation_end', 'irrigation_amount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date_irrigation_start'] = pd.to_datetime(df['date_irrigation_start'], errors='coerce')
        df['irrigation_amount'] = pd.to_numeric(df['irrigation_amount'], errors='coerce')
        df['year_calendar'] = pd.to_numeric(df['year_calendar'], errors='coerce').astype(str) # Treat as category
        df.dropna(inplace=True)

        total_water_used = df['irrigation_amount'].sum()
        avg_water_per_event = df['irrigation_amount'].mean()
        most_common_method = df['irrigation_method'].mode()[0]

        metrics["total_water_used"] = total_water_used
        metrics["avg_water_per_event"] = avg_water_per_event
        metrics["most_common_irrigation_method"] = str(most_common_method)

        insights.append(f"Total Water Used: {total_water_used:,.2f}")
        insights.append(f"Avg. Water per Event: {avg_water_per_event:.2f}")
        insights.append(f"Most Common Method: {most_common_method}")

        water_by_year = df.groupby('year_calendar')['irrigation_amount'].sum().reset_index()
        fig1 = px.bar(water_by_year, x='year_calendar', y='irrigation_amount',
                      title="Total Irrigation Water Used Per Year")
        visualizations["total_water_by_year"] = fig1.to_json()
        metrics["total_water_by_year"] = json.loads(water_by_year.to_json(orient="split"))

        water_by_method = df.groupby('irrigation_method')['irrigation_amount'].sum().reset_index()
        fig2 = px.pie(water_by_method, names='irrigation_method', values='irrigation_amount',
                      title="Total Water Usage by Irrigation Method")
        visualizations["total_water_by_method"] = fig2.to_json()
        metrics["total_water_by_method"] = json.loads(water_by_method.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def water_drainage_control_structure_analysis(df):
    analysis_type = "water_drainage_control_structure_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Water Drainage Control Structure Analysis ---"]

        expected = ['plot_id', 'year_calendar', 'date', 'outlet_depth', 'outlet_height']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['outlet_depth', 'outlet_height']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_depth = df['outlet_depth'].mean()
        avg_height = df['outlet_height'].mean()

        metrics["average_outlet_depth"] = avg_depth
        metrics["average_outlet_height"] = avg_height
        
        insights.append(f"Average Outlet Depth: {avg_depth:.2f}")
        insights.append(f"Average Outlet Height: {avg_height:.2f}")

        # Resample to daily average to make plot readable
        df = df.sort_values('date')
        df_daily = df.set_index('date').resample('D')[['outlet_depth', 'outlet_height']].mean().reset_index()
        
        fig1 = px.line(df_daily, x='date', y=['outlet_depth', 'outlet_height'],
                       title="Drainage Outlet Settings Over Time (Daily Avg)")
        visualizations["outlet_settings_over_time"] = fig1.to_json()

        fig2 = px.scatter(df, x='outlet_depth', y='outlet_height',
                          title="Outlet Depth vs. Height Settings")
        visualizations["depth_vs_height_settings"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def agro_meteorological_data_and_evapotranspiration_analysis(df):
    analysis_type = "agro_meteorological_data_and_evapotranspiration_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Agro-Meteorological Data and Evapotranspiration Analysis ---"]

        expected = ['date', 'precipitation', 'air_temp_avg', 'solar_radiation', 'wind_speed', 'et']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['precipitation', 'air_temp_avg', 'solar_radiation', 'wind_speed', 'et']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_et = df['et'].mean() # Evapotranspiration
        avg_temp = df['air_temp_avg'].mean()
        total_precip = df['precipitation'].sum()

        metrics["average_et"] = avg_et
        metrics["average_air_temp_c"] = avg_temp
        metrics["total_precipitation"] = total_precip

        insights.append(f"Average Evapotranspiration (ET): {avg_et:.2f}")
        insights.append(f"Average Air Temp: {avg_temp:.2f}°C")
        insights.append(f"Total Precipitation: {total_precip:.2f}")

        # Resample to weekly to make plot readable
        df = df.sort_values('date')
        df_weekly = df.set_index('date').resample('W')[['et', 'precipitation', 'air_temp_avg']].mean().reset_index()

        fig1 = px.line(df_weekly, x='date', y=['et', 'precipitation'],
                       title="Evapotranspiration and Precipitation Over Time (Weekly Avg)")
        visualizations["et_precip_over_time"] = fig1.to_json()

        fig2 = px.scatter(df, x='air_temp_avg', y='et', color='solar_radiation',
                          title="Evapotranspiration vs. Air Temperature (Colored by Solar Radiation)",
                          labels={'air_temp_avg': 'Average Air Temp (°C)', 'et': 'Evapotranspiration'})
        visualizations["et_vs_temp_by_solar"] = fig2.to_json()
        
        et_temp_corr = df['air_temp_avg'].corr(df['et'])
        et_solar_corr = df['solar_radiation'].corr(df['et'])
        metrics["et_temp_correlation"] = et_temp_corr
        metrics["et_solar_radiation_correlation"] = et_solar_corr
        insights.append(f"Correlation (ET & Temp): {et_temp_corr:.3f}")
        insights.append(f"Correlation (ET & Solar Radiation): {et_solar_corr:.3f}")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def crop_cultivation_cost_and_profitability_analysis(df):
    analysis_type = "crop_cultivation_cost_and_profitability_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Cultivation Cost and Profitability Analysis ---"]

        expected = ['crop', 'state', 'cost_of_cultivation_hectare_c2', 'cost_of_production_quintal_c2', 'yield_quintal_hectare']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            if col not in ['crop', 'state']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Assuming price = 1.2 * cost_of_production (20% markup)
        df['revenue_per_hectare'] = (df['cost_of_production_quintal_c2'] * 1.2) * df['yield_quintal_hectare']
        df['profit_per_hectare'] = df['revenue_per_hectare'] - df['cost_of_cultivation_hectare_c2']
        insights.append("Calculated 'profit_per_hectare' assuming price is 20% over 'cost_of_production_quintal_c2'.")

        most_profitable_crop = df.groupby('crop')['profit_per_hectare'].mean().idxmax()
        avg_profit = df['profit_per_hectare'].mean()
        highest_cost_crop = df.groupby('crop')['cost_of_cultivation_hectare_c2'].mean().idxmax()

        metrics["most_profitable_crop"] = str(most_profitable_crop)
        metrics["average_profit_per_hectare"] = avg_profit
        metrics["highest_cost_crop"] = str(highest_cost_crop)

        insights.append(f"Most Profitable Crop (on avg): {most_profitable_crop}")
        insights.append(f"Average Profit per Hectare: ${avg_profit:,.2f}")
        insights.append(f"Highest Cost Crop (on avg): {highest_cost_crop}")

        profit_by_crop = df.groupby('crop')[['cost_of_cultivation_hectare_c2', 'revenue_per_hectare', 'profit_per_hectare']].mean().reset_index()
        profit_by_crop_melted = profit_by_crop.melt(id_vars='crop', value_vars=['cost_of_cultivation_hectare_c2', 'profit_per_hectare'],
                                                    var_name='Metric', value_name='Value')
        
        fig1 = px.bar(profit_by_crop_melted.sort_values('Value'),
                      x='crop', y='Value', color='Metric',
                      title="Cost and Profit per Hectare by Crop", barmode='stack')
        visualizations["cost_profit_by_crop"] = fig1.to_json()
        metrics["cost_profit_by_crop"] = json.loads(profit_by_crop.to_json(orient="split"))

        fig2 = px.scatter(df, x='cost_of_cultivation_hectare_c2', y='yield_quintal_hectare',
                          color='crop', title="Cost of Cultivation vs. Yield",
                          labels={'cost_of_cultivation_hectare_c2': 'Cost of Cultivation (per Hectare)',
                                  'yield_quintal_hectare': 'Yield (Quintal per Hectare)'})
        visualizations["cost_vs_yield"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def detailed_soil_physicochemical_property_analysis(df):
    analysis_type = "detailed_soil_physicochemical_property_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Detailed Soil Physicochemical Property Analysis ---"]

        expected = ['depth', 'percent_sand', 'percent_silt', 'percent_clay', 'bulk_density', 'ph_water', 'soc', 'total_n']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in expected:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_ph = df['ph_water'].mean()
        avg_bulk_density = df['bulk_density'].mean()
        avg_soc = df['soc'].mean()

        metrics["average_soil_ph"] = avg_ph
        metrics["average_bulk_density_g_cm3"] = avg_bulk_density
        metrics["average_soil_organic_carbon_perc"] = avg_soc

        insights.append(f"Average Soil pH: {avg_ph:.2f}")
        insights.append(f"Average Bulk Density: {avg_bulk_density:.2f} g/cm³")
        insights.append(f"Average Soil Organic Carbon: {avg_soc:.2f}%")

        insights.append("Correlation Between Soil Properties")
        corr_cols = ['percent_sand', 'percent_silt', 'percent_clay', 'bulk_density', 'ph_water', 'soc', 'total_n']
        corr_matrix = df[corr_cols].corr(numeric_only=True)
        fig1 = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                         title="Correlation Heatmap of Soil Properties")
        visualizations["soil_properties_correlation"] = fig1.to_json()
        metrics["soil_properties_correlation_matrix"] = json.loads(corr_matrix.to_json(orient="split"))

        fig2 = px.scatter(df, x='soc', y='total_n', size='bulk_density',
                          hover_name='depth',
                          title="Soil Organic Carbon vs. Total Nitrogen (Sized by Bulk Density)",
                          labels={'soc': 'Soil Organic Carbon (%)', 'total_n': 'Total Nitrogen (%)'})
        visualizations["soc_vs_total_n"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def soil_moisture_temperature_and_electrical_conductivity_analysis(df):
    analysis_type = "soil_moisture_temperature_and_electrical_conductivity_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Soil Moisture, Temperature, and EC Analysis ---"]

        expected = ['date', 'depth', 'soil_moisture', 'soil_temperature', 'soil_ec']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['depth', 'soil_moisture', 'soil_temperature', 'soil_ec']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_moisture = df['soil_moisture'].mean()
        avg_temp = df['soil_temperature'].mean()
        avg_ec = df['soil_ec'].mean()

        metrics["average_soil_moisture_m3_m3"] = avg_moisture
        metrics["average_soil_temperature_c"] = avg_temp
        metrics["average_soil_ec_ds_m"] = avg_ec

        insights.append(f"Average Soil Moisture: {avg_moisture:.3f} m³/m³")
        insights.append(f"Average Soil Temperature: {avg_temp:.2f}°C")
        insights.append(f"Average Soil EC: {avg_ec:.2f} dS/m")

        # Resample to daily average
        df = df.sort_values('date')
        df_daily_avg = df.set_index('date').resample('D')[['soil_moisture', 'soil_temperature']].mean().reset_index()
        
        fig1 = px.line(df_daily_avg, x='date', y=['soil_moisture', 'soil_temperature'],
                       title="Daily Average Soil Moisture and Temperature Over Time", facet_row='variable')
        fig1.update_yaxes(matches=None)
        visualizations["daily_moisture_temp_trend"] = fig1.to_json()

        df_depth_profile = df.groupby('depth')[['soil_moisture', 'soil_ec', 'soil_temperature']].mean().reset_index()
        fig2 = px.line(df_depth_profile, x='depth', y=['soil_moisture', 'soil_ec'],
                       title="Average Soil Moisture and EC by Depth", facet_row='variable')
        fig2.update_yaxes(matches=None)
        visualizations["profile_by_depth"] = fig2.to_json()
        metrics["profile_by_depth"] = json.loads(df_depth_profile.to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }


def crop_biomass_and_nutrient_content_analysis(df):
    analysis_type = "crop_biomass_and_nutrient_content_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Biomass and Nutrient Content Analysis ---"]

        expected = ['crop', 'crop_yield', 'vegetative_biomass', 'grain_biomass', 'vegetative_total_n', 'grain_total_n']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['crop_yield', 'vegetative_biomass', 'grain_biomass', 'vegetative_total_n', 'grain_total_n']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        total_grain_biomass = df['grain_biomass'].sum()
        avg_grain_n = df['grain_total_n'].mean()
        most_biomass_crop = df.groupby('crop')['grain_biomass'].sum().idxmax()

        metrics["total_grain_biomass"] = total_grain_biomass
        metrics["average_grain_total_nitrogen_perc"] = avg_grain_n
        metrics["crop_with_highest_grain_biomass"] = str(most_biomass_crop)

        insights.append(f"Total Grain Biomass: {total_grain_biomass:,.2f}")
        insights.append(f"Average Grain Total Nitrogen: {avg_grain_n:.2f}%")
        insights.append(f"Crop with Highest Grain Biomass: {most_biomass_crop}")

        biomass_by_crop = df.groupby('crop')['grain_biomass'].sum().reset_index()
        fig1 = px.bar(biomass_by_crop,
                      x='crop', y='grain_biomass', title="Total Grain Biomass by Crop Type")
        visualizations["grain_biomass_by_crop"] = fig1.to_json()
        metrics["grain_biomass_by_crop"] = json.loads(biomass_by_crop.to_json(orient="split"))

        fig2 = px.scatter(df, x='vegetative_total_n', y='grain_total_n', color='crop',
                          title="Vegetative N vs. Grain N Content by Crop",
                          labels={'vegetative_total_n': 'Vegetative Total Nitrogen (%)', 'grain_total_n': 'Grain Total Nitrogen (%)'})
        visualizations["veg_n_vs_grain_n"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def crop_growth_stage_monitoring_analysis(df):
    analysis_type = "crop_growth_stage_monitoring_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Crop Growth Stage Monitoring Analysis ---"]

        expected = ['plot_id', 'crop', 'date', 'growth_stage', 'biomass_g_per_m2', 'height_cm']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['biomass_g_per_m2', 'height_cm']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        num_plots = df['plot_id'].nunique()
        common_growth_stage = df['growth_stage'].mode()[0]
        avg_biomass = df['biomass_g_per_m2'].mean()

        metrics["number_of_plots_monitored"] = num_plots
        metrics["most_common_growth_stage"] = str(common_growth_stage)
        metrics["average_biomass_g_per_m2"] = avg_biomass

        insights.append(f"Number of Plots Monitored: {num_plots}")
        insights.append(f"Most Common Growth Stage: {common_growth_stage}")
        insights.append(f"Average Biomass: {avg_biomass:.2f} g/m²")

        # Group by date to make line plot readable
        df_daily_growth = df.sort_values('date').groupby(['date', 'crop'])[['biomass_g_per_m2', 'height_cm']].mean().reset_index()

        fig1 = px.line(df_daily_growth, x='date', y='biomass_g_per_m2', color='crop',
                       title="Biomass Growth Over Time by Crop Type (Daily Avg)")
        visualizations["biomass_growth_over_time"] = fig1.to_json()

        fig2 = px.box(df, x='growth_stage', y='height_cm', color='crop',
                      title="Crop Height Distribution by Growth Stage and Crop")
        visualizations["height_by_growth_stage"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def water_table_depth_fluctuation_analysis(df):
    analysis_type = "water_table_depth_fluctuation_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Water Table Depth Fluctuation Analysis ---"]

        expected = ['date', 'site_id', 'water_table_depth_cm', 'rainfall_mm', 'irrigation_amount_mm']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['water_table_depth_cm', 'rainfall_mm', 'irrigation_amount_mm']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['date', 'water_table_depth_cm'], inplace=True) # Only date and depth are essential

        avg_depth = df['water_table_depth_cm'].mean()
        max_depth = df['water_table_depth_cm'].max()
        min_depth = df['water_table_depth_cm'].min()

        metrics["average_water_table_depth_cm"] = avg_depth
        metrics["maximum_water_table_depth_cm"] = max_depth
        metrics["minimum_water_table_depth_cm"] = min_depth

        insights.append(f"Average Water Table Depth: {avg_depth:.2f} cm")
        insights.append(f"Maximum Water Table Depth: {max_depth:.2f} cm")
        insights.append(f"Minimum Water Table Depth: {min_depth:.2f} cm")

        # Group by date to make plot readable
        df_daily = df.sort_values('date').groupby(['date', 'site_id'])['water_table_depth_cm'].mean().reset_index()

        fig1 = px.line(df_daily, x='date', y='water_table_depth_cm', color='site_id',
                       title="Water Table Depth Fluctuation Over Time by Site (Daily Avg)")
        visualizations["water_table_over_time"] = fig1.to_json()

        if 'rainfall_mm' in df.columns:
            fig2 = px.scatter(df, x='rainfall_mm', y='water_table_depth_cm', color='site_id',
                              title="Rainfall vs. Water Table Depth",
                              labels={'rainfall_mm': 'Rainfall (mm)', 'water_table_depth_cm': 'Water Table Depth (cm)'})
            visualizations["rainfall_vs_water_table"] = fig2.to_json()
            
            rain_depth_corr = df['rainfall_mm'].corr(df['water_table_depth_cm'])
            metrics["rainfall_depth_correlation"] = rain_depth_corr
            insights.append(f"Correlation (Rainfall & Depth): {rain_depth_corr:.3f} (Note: negative correlation means rain makes table *shallower*)")


        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def agricultural_water_quality_monitoring_analysis(df):
    analysis_type = "agricultural_water_quality_monitoring_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Agricultural Water Quality Monitoring Analysis ---"]

        expected = ['sample_date', 'source_type', 'ph', 'ec_us_per_cm', 'nitrate_mg_per_l', 'phosphate_mg_per_l']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
        for col in ['ph', 'ec_us_per_cm', 'nitrate_mg_per_l', 'phosphate_mg_per_l']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_ph = df['ph'].mean()
        avg_ec = df['ec_us_per_cm'].mean()
        avg_nitrate = df['nitrate_mg_per_l'].mean()
        avg_phosphate = df['phosphate_mg_per_l'].mean()

        metrics["average_ph"] = avg_ph
        metrics["average_ec_us_cm"] = avg_ec
        metrics["average_nitrate_mg_l"] = avg_nitrate
        metrics["average_phosphate_mg_l"] = avg_phosphate

        insights.append(f"Average pH: {avg_ph:.2f}")
        insights.append(f"Average EC: {avg_ec:.2f} µS/cm")
        insights.append(f"Average Nitrate: {avg_nitrate:.2f} mg/L")
        insights.append(f"Average Phosphate: {avg_phosphate:.2f} mg/L")

        fig1 = px.box(df, x='source_type', y=['ph', 'ec_us_per_cm'],
                      title="Water pH and EC by Source Type", facet_row='variable')
        fig1.update_yaxes(matches=None)
        visualizations["ph_ec_by_source"] = fig1.to_json()

        fig2 = px.scatter(df, x='nitrate_mg_per_l', y='phosphate_mg_per_l', color='source_type',
                          title="Nitrate vs. Phosphate Levels by Source Type")
        visualizations["nitrate_vs_phosphate"] = fig2.to_json()
        
        metrics["metrics_by_source"] = json.loads(df.groupby('source_type')[['ph', 'ec_us_per_cm', 'nitrate_mg_per_l', 'phosphate_mg_per_l']].mean().to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def drain_flow_and_nutrient_load_analysis(df):
    analysis_type = "drain_flow_and_nutrient_load_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Drain Flow and Nutrient Load Analysis ---"]

        expected = ['date', 'drain_flow_l_per_sec', 'nitrate_load_kg', 'phosphate_load_kg', 'site_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['drain_flow_l_per_sec', 'nitrate_load_kg', 'phosphate_load_kg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        total_drain_flow_liters = df['drain_flow_l_per_sec'].sum() * (24*60*60) # This is likely incorrect, flow is a rate. Summing it is not useful.
        avg_drain_flow = df['drain_flow_l_per_sec'].mean()
        total_nitrate_load = df['nitrate_load_kg'].sum()
        total_phosphate_load = df['phosphate_load_kg'].sum()

        metrics["average_drain_flow_l_sec"] = avg_drain_flow
        metrics["total_nitrate_load_kg"] = total_nitrate_load
        metrics["total_phosphate_load_kg"] = total_phosphate_load

        insights.append(f"Average Drain Flow: {avg_drain_flow:,.2f} L/sec")
        insights.append(f"Total Nitrate Load (sum of samples): {total_nitrate_load:,.2f} kg")
        insights.append(f"Total Phosphate Load (sum of samples): {total_phosphate_load:,.2f} kg")

        df_daily = df.sort_values('date').groupby(['date', 'site_id'])[['drain_flow_l_per_sec', 'nitrate_load_kg', 'phosphate_load_kg']].mean().reset_index()

        fig1 = px.line(df_daily, x='date', y='drain_flow_l_per_sec', color='site_id',
                       title="Drain Flow Over Time by Site (Daily Avg)")
        visualizations["drain_flow_over_time"] = fig1.to_json()

        fig2 = px.scatter(df, x='nitrate_load_kg', y='phosphate_load_kg', color='site_id',
                          size='drain_flow_l_per_sec',
                          title="Nitrate Load vs. Phosphate Load in Drains (Sized by Flow)")
        visualizations["nitrate_vs_phosphate_load"] = fig2.to_json()
        
        metrics["loads_by_site"] = json.loads(df.groupby('site_id')[['nitrate_load_kg', 'phosphate_load_kg']].sum().to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def miticide_efficacy_analysis_for_varroa_destructor(df):
    analysis_type = "miticide_efficacy_analysis_for_varroa_destructor"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Miticide Efficacy Analysis for Varroa Destructor ---"]

        expected = ['treatment_group', 'colony_id', 'pre_treatment_mite_count', 'post_treatment_mite_count', 'efficacy_percent']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['pre_treatment_mite_count', 'post_treatment_mite_count', 'efficacy_percent']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_efficacy = df['efficacy_percent'].mean()
        avg_efficacy_by_treatment = df.groupby('treatment_group')['efficacy_percent'].mean()
        highest_efficacy_treatment = avg_efficacy_by_treatment.idxmax()

        metrics["average_efficacy_all_treatments_perc"] = avg_efficacy
        metrics["best_performing_treatment"] = str(highest_efficacy_treatment)
        metrics["average_efficacy_by_treatment"] = json.loads(avg_efficacy_by_treatment.to_json(orient="split"))

        insights.append(f"Average Miticide Efficacy (All Groups): {avg_efficacy:.2f}%")
        insights.append(f"Best Performing Treatment (Avg Efficacy): {highest_efficacy_treatment} ({avg_efficacy_by_treatment.max():.2f}%)")

        fig1 = px.box(df, x='treatment_group', y='efficacy_percent',
                      title="Miticide Efficacy Distribution by Treatment Group")
        visualizations["efficacy_by_treatment"] = fig1.to_json()

        fig2 = px.scatter(df, x='pre_treatment_mite_count', y='efficacy_percent', color='treatment_group',
                          trendline='ols',
                          title="Pre-Treatment Mite Count vs. Efficacy by Treatment",
                          labels={'pre_treatment_mite_count': 'Pre-Treatment Mite Count', 'efficacy_percent': 'Efficacy (%)'})
        visualizations["pre_count_vs_efficacy"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def groundwater_quality_and_suitability_analysis_for_irrigation(df):
    analysis_type = "groundwater_quality_and_suitability_analysis_for_irrigation"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Groundwater Quality & Suitability for Irrigation Analysis ---"]

        expected = ['well_id', 'sample_date', 'ph', 'ec_us_per_cm', 'sodium_mg_per_l', 'chloride_mg_per_l', 'suitability_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
        for col in ['ph', 'ec_us_per_cm', 'sodium_mg_per_l', 'chloride_mg_per_l']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_ph = df['ph'].mean()
        avg_ec = df['ec_us_per_cm'].mean()
        most_common_suitability = df['suitability_category'].mode()[0]

        metrics["average_ph"] = avg_ph
        metrics["average_ec_us_cm"] = avg_ec
        metrics["most_common_suitability"] = str(most_common_suitability)

        insights.append(f"Average pH: {avg_ph:.2f}")
        insights.append(f"Average EC: {avg_ec:.2f} µS/cm")
        insights.append(f"Most Common Suitability: {most_common_suitability}")

        fig1 = px.box(df, x='suitability_category', y=['ph', 'ec_us_per_cm'],
                      title="Water Quality Parameters by Suitability Category", facet_row='variable')
        fig1.update_yaxes(matches=None)
        visualizations["quality_by_suitability"] = fig1.to_json()

        fig2 = px.scatter(df, x='sodium_mg_per_l', y='chloride_mg_per_l', color='suitability_category',
                          title="Sodium vs. Chloride Levels by Suitability",
                          labels={'sodium_mg_per_l': 'Sodium (mg/L)', 'chloride_mg_per_l': 'Chloride (mg/L)'})
        visualizations["sodium_vs_chloride"] = fig2.to_json()
        
        metrics["metrics_by_suitability"] = json.loads(df.groupby('suitability_category')[['ph', 'ec_us_per_cm', 'sodium_mg_per_l', 'chloride_mg_per_l']].mean().to_json(orient="split"))


        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }

def soil_acidity_and_cation_exchange_capacity_analysis(df):
    analysis_type = "soil_acidity_and_cation_exchange_capacity_analysis"
    matched = {}
    try:
        visualizations = {}
        metrics = {}
        insights = ["--- Soil Acidity and Cation Exchange Capacity Analysis ---"]

        expected = ['site_id', 'depth', 'ph_h2o', 'cec_cmol_per_kg', 'exchangeable_calcium', 'exchangeable_magnesium']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            insights += get_missing_columns_message(missing, matched)
            fallback_data = general_insights_analysis(df, "General Analysis")
            fallback_data["analysis_type"] = analysis_type
            fallback_data["status"] = "fallback"
            fallback_data["message"] = "Required columns missing, returned general analysis."
            fallback_data["matched_columns"] = matched
            fallback_data["insights"] = insights + fallback_data["insights"]
            return fallback_data

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        for col in ['ph_h2o', 'cec_cmol_per_kg', 'exchangeable_calcium', 'exchangeable_magnesium']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        avg_ph = df['ph_h2o'].mean()
        avg_cec = df['cec_cmol_per_kg'].mean()
        avg_calcium = df['exchangeable_calcium'].mean()
        avg_magnesium = df['exchangeable_magnesium'].mean()

        metrics["average_ph_h2o"] = avg_ph
        metrics["average_cec_cmol_kg"] = avg_cec
        metrics["average_exchangeable_calcium"] = avg_calcium
        metrics["average_exchangeable_magnesium"] = avg_magnesium

        insights.append(f"Average pH (H2O): {avg_ph:.2f}")
        insights.append(f"Average CEC: {avg_cec:.2f} cmol/kg")
        insights.append(f"Average Exchangeable Calcium: {avg_calcium:.2f} cmol/kg")
        insights.append(f"Average Exchangeable Magnesium: {avg_magnesium:.2f} cmol/kg")
        
        # Treat depth as categorical for plotting
        if 'depth' in df.columns:
            df['depth'] = df['depth'].astype(str)

        fig1 = px.scatter(df, x='ph_h2o', y='cec_cmol_per_kg', color='depth',
                          trendline='ols',
                          title="Soil pH vs. CEC by Depth",
                          labels={'ph_h2o': 'pH (H2O)', 'cec_cmol_per_kg': 'CEC (cmol/kg)'})
        visualizations["ph_vs_cec_by_depth"] = fig1.to_json()

        fig2 = px.box(df, x='depth', y=['exchangeable_calcium', 'exchangeable_magnesium'],
                      title="Exchangeable Calcium and Magnesium by Depth", facet_row='variable')
        fig2.update_yaxes(matches=None)
        visualizations["ca_mg_by_depth"] = fig2.to_json()
        
        metrics["metrics_by_depth"] = json.loads(df.groupby('depth')[['ph_h2o', 'cec_cmol_per_kg', 'exchangeable_calcium', 'exchangeable_magnesium']].mean().to_json(orient="split"))

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "message": "Analysis completed successfully.",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_numpy_types(metrics),
            "insights": insights
        }

    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "message": f"An error occurred: {str(e)}",
            "matched_columns": matched,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {str(e)}"]
        }
def varroa_mite_population_assessment_in_beehives(df):
    """Varroa mite population assessment in beehives with structured return format"""
    result = {
        "analysis_type": "varroa_mite_population_assessment_in_beehives",
        "status": "success",
        "matched_columns": {},
        "metrics": {},
        "visualizations": {},
        "insights": []
    }
    
    try:
        print("\n--- Varroa Mite Population Assessment in Beehives ---")
        expected = ['colony_id', 'sample_date', 'mite_count_after_shake', 'mite_fall_24hr', 'mite_infestation_level']
        matched = fuzzy_match_column(df, expected)
        result["matched_columns"] = matched
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
            result["status"] = "fallback"
            result["insights"] += get_missing_columns_message(missing, matched)
            general_result = general_insights_analysis(df, "General Analysis")
            result.update(general_result)
            return result

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['sample_date'] = pd.to_datetime(df['sample_date'], errors='coerce')
        for col in ['mite_count_after_shake', 'mite_fall_24hr', 'mite_infestation_level']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)

        # Metrics
        avg_mite_shake = convert_to_native_types(df['mite_count_after_shake'].mean())
        avg_mite_fall = convert_to_native_types(df['mite_fall_24hr'].mean())
        highest_infestation_colony = convert_to_native_types(df.loc[df['mite_infestation_level'].idxmax()]['colony_id'])

        result["metrics"]["average_mite_shake"] = avg_mite_shake
        result["metrics"]["average_mite_fall"] = avg_mite_fall
        result["metrics"]["highest_infestation_colony"] = highest_infestation_colony
        
        result["insights"].append(f"Average Mite Count (Sugar Shake): {avg_mite_shake:.1f}")
        result["insights"].append(f"Average 24hr Mite Fall: {avg_mite_fall:.1f}")
        result["insights"].append(f"Colony with Highest Infestation: {highest_infestation_colony}")

        # Visualizations
        fig1 = px.line(df.sort_values('sample_date'), x='sample_date', y='mite_infestation_level', color='colony_id',
                       title="Mite Infestation Level Over Time by Colony")
        result["visualizations"]["infestation_over_time"] = fig1.to_json()

        fig2 = px.box(df, x='mite_infestation_level', y='mite_count_after_shake',
                      title="Mite Count (Shake) Distribution by Infestation Level")
        result["visualizations"]["mite_count_by_infestation"] = fig2.to_json()

        return result
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["insights"].append(f"Error during varroa mite population assessment: {str(e)}")
        return result


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
                    result = selected_function(df)
                    print(f"\nAnalysis completed with status: {result['status']}")
                    print(f"Found {len(result.get('visualizations', {}))} visualizations")
                    print(f"Generated {len(result.get('insights', []))} insights")
                    
                    # For demo purposes, print some key metrics
                    if result.get('metrics'):
                        print("\nKey Metrics:")
                        for key, value in result['metrics'].items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:,.2f}")
                            else:
                                print(f"  {key}: {value}")
                    
                    return result
                    
                except Exception as e:
                    print(f"\n[ERROR] An error occurred while running the analysis '{selected_analysis_key.replace('_', ' ').title()}':")
                    print(f"Error details: {e}")
                    return {
                        "analysis_type": selected_analysis_key,
                        "status": "error",
                        "error": str(e),
                        "insights": [f"Error during analysis: {str(e)}"]
                    }
            else:
                error_msg = f"Function for '{selected_analysis_key.replace('_', ' ').title()}' not found."
                print(f"\n[ERROR] {error_msg}")
                return {
                    "analysis_type": selected_analysis_key,
                    "status": "error",
                    "error": error_msg,
                    "insights": [error_msg]
                }
        elif choice_idx == len(all_analysis_names): # General Insights option
            result = general_insights_analysis(df, "Initial Data Overview")
            print(f"\nGeneral insights completed with {len(result.get('visualizations', {}))} visualizations")
            return result
        else:
            print("\nInvalid choice. Please enter a number within the given range.")
            result = general_insights_analysis(df, "Initial Data Overview")
            return result
    except ValueError:
        print("\nInvalid input. Please enter a number.")
        result = general_insights_analysis(df, "Initial Data Overview")
        return result


if __name__ == "__main__":
    result = main()