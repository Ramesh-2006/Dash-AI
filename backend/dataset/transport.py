import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, linregress
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "fleet_analysis",
    "route_analysis",
    "driver_analysis",
    "fuel_analysis",
    "maintenance_analysis",
    "delivery_analysis",
    "cost_analysis",
    "safety_analysis",
    "commuter_transportation_mode_choice_analysis",
    "public_bus_performance_and_delay_analysis",
    "global_air_transport_passenger_trends_analysis",
    "airline_directory_and_operational_status_analysis",
    "public_transit_fare_and_journey_type_analysis",
    "regional_vehicle_registration_trend_analysis",
    "transportation_user_survey_response_analysis",
    "bus_route_schedule_analysis",
    "public_transit_station_ridership_analysis",
    "county_level_transportation_infrastructure_and_commute_analysis",
    "transit_agency_information_analysis",
    "public_transit_route_definition_analysis",
    "transit_trip_schedule_and_accessibility_analysis",
    "transit_stop_location_and_information_analysis",
    "transit_stop_time_and_sequence_analysis",
    "transit_service_calendar_analysis",
    "transit_service_exception_and_holiday_schedule_analysis",
    "transit_fare_structure_analysis",
    "transit_fare_rule_and_zone_analysis",
    "transit_route_shape_and_path_geospatial_analysis",
    "transit_frequency_and_headway_analysis",
    "station_pathway_and_accessibility_analysis",
    "gtfs_feed_information_and_version_analysis",
    "real_time_vehicle_position_and_trip_update_analysis",
    "extended_transit_route_attribute_analysis",
    "transit_fare_zone_definition_analysis",
    "multi_level_station_and_platform_information_analysis",
    "transit_trip_stop_timepoint_analysis",
    "transit_trip_details_and_accessibility_features_analysis",
    "transit_stop_and_station_location_analysis",
    "public_transportation_route_details_analysis",
    "transportation_agency_contact_and_timezone_analysis",
    "transit_trip_planning_and_route_shape_analysis",
    "transit_service_schedule_definition",
    "stop_by_stop_transit_schedule_analysis",
    "public_transport_agency_directory_analysis",
    "transit_fare_attribute_analysis",
    "inter_stop_transfer_path_analysis",
    "geospatial_route_path_analysis",
    "trip_frequency_and_service_interval_analysis",
    "fare_cost_and_transfer_policy_analysis",
    "trip_service_detail_and_distance_analysis",
    "transit_data_feed_publisher_information_analysis",
    "pedestrian_pathway_analysis_in_transit_stations",
    "transit_route_information_analysis",
    "trip_accessibility_and_direction_analysis",
    "scheduled_stop_times_analysis_for_trips",
    "special_service_dates_and_schedule_exception_analysis",
    "General Insights"
]

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
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception:
        # Fallback for complex unhandled types
        return str(data)

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
        
        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / total_rows) * 100 if total_rows > 0 else 0
        columns_with_missing = missing_values[missing_values > 0]
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Create visualizations
        visualizations = {}
        
        # 1. Data types distribution
        try:
            dtype_counts = {
                'Numeric': len(numeric_cols),
                'Categorical': len(categorical_cols),
                'Datetime': len(datetime_cols),
                'Other': len(other_cols)
            }
            fig_dtypes = px.pie(
                values=list(dtype_counts.values()), 
                names=list(dtype_counts.keys()),
                title='Data Types Distribution'
            )
            visualizations["data_types_distribution"] = fig_dtypes.to_json()
        except Exception:
            pass
        
        # 2. Missing values visualization
        try:
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
                    title='Top 10 Columns with Missing Values (%)'
                )
                visualizations["missing_values"] = fig_missing.to_json()
            else:
                fig_no_missing = go.Figure()
                fig_no_missing.add_annotation(
                    text="No Missing Values Found!",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=20, color="green")
                )
                fig_no_missing.update_layout(title="Missing Values Analysis")
                visualizations["missing_values"] = fig_no_missing.to_json()
        except Exception:
            pass
        
        # 3. Numeric columns distributions
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:2]): # Show first 2 numeric histograms
                try:
                    fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception:
                    pass
        
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
            }
        }
        
        # Generate insights
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns",
        ]
        
        # Add missing columns warning if provided
        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append("The following columns are needed for the requested analysis but weren't found in your data:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"    - {col}{match_info}")
            insights.append("")
            insights.append("Showing General Analysis instead of the requested specific analysis.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data)")
        else:
            insights.append("No duplicate rows found")
        
        if len(columns_with_missing) > 0:
            insights.append(f"{len(columns_with_missing)} columns have missing values")
        else:
            insights.append("No missing values found in the dataset")
        
        insights.append(f"Generated {len(visualizations)} visualizations for data exploration")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights,
            "missing_columns": missing_cols or []
        }
        
    except Exception as e:
        # Ultra-safe fallback
        basic_insights = [
            f"Basic dataset info: {len(df)} rows, {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}",
            f"Limited analysis due to an error: {str(e)}"
        ]
        
        if missing_cols and len(missing_cols) > 0:
            basic_insights.insert(0, "⚠️ REQUIRED COLUMNS NOT FOUND - Showing General Analysis")
            basic_insights.insert(1, f"Missing columns: {', '.join(missing_cols)}")
        
        return {
            "analysis_type": analysis_type,
            "status": "success", # Still return success for basic info
            "matched_columns": matched_cols or {},
            "visualizations": {},
            "metrics": {
                "dataset_basic_info": {
                    "total_rows": len(df),
                    "total_columns": len(df.columns),
                    "column_names": df.columns.tolist()[:10]
                }
            },
            "insights": basic_insights,
            "missing_columns": missing_cols or []
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides general insights as a fallback.
    """
    print(f"--- ⚠️ Required Columns Not Found for {analysis_name} ---")
    print(f"Missing: {missing_cols}")
    print("Falling back to General Insights.")
    
    try:
        # Generate the fallback general insights
        general_insights_data = show_general_insights(
            df, 
            f"General Analysis (Fallback for {analysis_name})",
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
        
        if general_insights_data.get('status') == 'error':
            raise Exception(general_insights_data.get('error_message', 'General insights failed'))
            
    except Exception as fallback_error:
        print(f"General insights also failed: {fallback_error}")
        # Create a minimal fallback response
        general_insights_data = {
            "analysis_type": "General Insights",
            "status": "partial_success",
            "visualizations": {},
            "metrics": {
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()[:10]
                }
            },
            "insights": [
                f"⚠️ REQUIRED COLUMNS NOT FOUND for '{analysis_name}'",
                f"Missing columns: {', '.join(missing_cols)}",
                f"Dataset has {len(df)} rows and {len(df.columns)} columns",
                "Available columns: " + ", ".join(df.columns.tolist()[:8]) + ("..." if len(df.columns) > 8 else ""),
                "Showing General Analysis due to missing required columns."
            ],
            "missing_columns": missing_cols
        }

    # Create the specific error response
    missing_info = {}
    for col in missing_cols:
        match_info = f" (fuzzy matched to: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (No match found)"
        missing_info[col] = match_info

    return {
        "analysis_type": analysis_name,
        "status": "fallback",
        "message": f"Required columns were missing for '{analysis_name}'. Falling back to general insights.",
        "missing_columns_info": missing_info,
        "matched_columns": matched_cols,
        "visualizations": general_insights_data.get("visualizations", {}),
        "metrics": general_insights_data.get("metrics", {}),
        "insights": general_insights_data.get("insights", []),
        "missing_columns": missing_cols
    }

def fuzzy_match_column(df, target_columns):
    """Improved fuzzy matching with better handling"""
    matched = {}
    available = df.columns.tolist()
    
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
            
        # Standardize for better matching
        target_std = target.lower().replace("_", "").replace(" ", "").replace("-", "")
        available_std = {col.lower().replace("_", "").replace(" ", "").replace("-", ""): col for col in available}
        
        # Try exact match after standardization
        if target_std in available_std:
            matched[target] = available_std[target_std]
            continue

        # If no direct match, use fuzzy matching
        try:
            # Use process.extractOne to find the best match
            # score_cutoff=80 means it needs to be a good match
            result = process.extractOne(target, available, score_cutoff=80)
            if result:
                matched[target] = result[0] # result[0] is the matched string
            else:
                matched[target] = None
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    return df.rename(columns={v: k for k, v in matched.items() if v is not None})

# ========== DATA LOADING ==========
def load_data(file_path, encoding='utf-8'):
    """Load data from CSV or Excel file with robust encoding support"""
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    return df
                except UnicodeDecodeError:
                    continue
            print("[ERROR] Failed to decode file with common encodings.")
            return None
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            print("[ERROR] Unsupported file format. Please provide CSV or Excel file.")
            return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

# --- Transportation Analysis Functions ---

def fleet_analysis(df):
    analysis_name = "Fleet Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['VehicleID', 'VehicleType', 'PurchaseYear', 'Mileage', 'Status']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['PurchaseYear'] = pd.to_numeric(df['PurchaseYear'], errors='coerce')
        df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
        df = df.dropna(subset=['PurchaseYear', 'Mileage', 'Status'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_vehicles = len(df)
        avg_mileage = df['Mileage'].mean()
        most_common_type = df['VehicleType'].mode()[0]
        
        metrics = {
            "Total Vehicles": total_vehicles,
            "Average Mileage": avg_mileage,
            "Most Common Vehicle Type": most_common_type
        }
        
        insights.append(f"Total Vehicles in Fleet: {total_vehicles}")
        insights.append(f"Average Fleet Mileage: {avg_mileage:,.0f} miles")
        insights.append(f"Most Common Vehicle Type: {most_common_type}")
        
        fig1 = px.histogram(df, x='VehicleType', title='Vehicle Type Distribution')
        visualizations["Vehicle_Type_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='PurchaseYear', y='Mileage', color='VehicleType', hover_name='VehicleID',
                          title='Mileage by Purchase Year and Vehicle Type')
        visualizations["Mileage_by_Purchase_Year_Scatter"] = fig2.to_json()
        
        fig3 = px.pie(df, names='Status', title='Operational Status Distribution')
        visualizations["Operational_Status_Distribution_Pie"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def route_analysis(df):
    analysis_name = "Route Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'StartLocation', 'EndLocation', 'Distance', 'AverageTravelTime']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
        df['AverageTravelTime'] = pd.to_numeric(df['AverageTravelTime'], errors='coerce')
        df = df.dropna(subset=['Distance', 'AverageTravelTime'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_routes = len(df)
        avg_distance = df['Distance'].mean()
        longest_route = df.loc[df['Distance'].idxmax(), 'RouteID']
        
        metrics = {
            "Total Routes": total_routes,
            "Average Route Distance": avg_distance,
            "Longest Route": longest_route
        }
        
        insights.append(f"Total Routes: {total_routes}")
        insights.append(f"Average Route Distance: {avg_distance:.2f} miles")
        insights.append(f"Longest Route: {longest_route}")
        
        fig1 = px.histogram(df, x='Distance', nbins=20, title='Route Distance Distribution')
        visualizations["Route_Distance_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='Distance', y='AverageTravelTime', color='StartLocation', hover_name='RouteID',
                          title='Average Travel Time vs. Distance by Start Location')
        visualizations["Travel_Time_vs_Distance_Scatter"] = fig2.to_json()
        
        route_counts = df['StartLocation'].value_counts().reset_index()
        route_counts.columns = ['StartLocation', 'count']
        fig3 = px.bar(route_counts, x='StartLocation', y='count', title='Number of Routes Originating from Each Location')
        visualizations["Routes_by_Start_Location_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def driver_analysis(df):
    analysis_name = "Driver Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['DriverID', 'YearsExperience', 'SafetyScore', 'HoursDrivenLastWeek']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['YearsExperience'] = pd.to_numeric(df['YearsExperience'], errors='coerce')
        df['SafetyScore'] = pd.to_numeric(df['SafetyScore'], errors='coerce')
        df['HoursDrivenLastWeek'] = pd.to_numeric(df['HoursDrivenLastWeek'], errors='coerce')
        df = df.dropna(subset=['YearsExperience', 'SafetyScore', 'HoursDrivenLastWeek'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_drivers = len(df)
        avg_experience = df['YearsExperience'].mean()
        highest_safety_score_driver = df.loc[df['SafetyScore'].idxmax(), 'DriverID']
        
        metrics = {
            "Total Drivers": total_drivers,
            "Average Experience": avg_experience,
            "Highest Safety Score Driver": highest_safety_score_driver
        }
        
        insights.append(f"Total Drivers: {total_drivers}")
        insights.append(f"Average Driver Experience: {avg_experience:.1f} years")
        insights.append(f"Driver with Highest Safety Score: {highest_safety_score_driver}")
        
        fig1 = px.histogram(df, x='SafetyScore', nbins=20, title='Safety Score Distribution')
        visualizations["Safety_Score_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='YearsExperience', y='SafetyScore', hover_name='DriverID',
                          title='Safety Score vs. Years of Experience')
        visualizations["Safety_Score_vs_Experience_Scatter"] = fig2.to_json()
        
        fig3 = px.box(df, y='HoursDrivenLastWeek', title='Weekly Hours Driven Distribution')
        visualizations["Weekly_Hours_Driven_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def fuel_analysis(df):
    analysis_name = "Fuel Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['VehicleID', 'FuelType', 'FuelConsumedGallons', 'DistanceDriven', 'FuelCost']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FuelConsumedGallons'] = pd.to_numeric(df['FuelConsumedGallons'], errors='coerce')
        df['DistanceDriven'] = pd.to_numeric(df['DistanceDriven'], errors='coerce')
        df['FuelCost'] = pd.to_numeric(df['FuelCost'], errors='coerce')
        df = df.dropna(subset=['FuelConsumedGallons', 'DistanceDriven', 'FuelCost'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }
        
        # Avoid division by zero
        df_safe = df[df['FuelConsumedGallons'] > 0].copy()
        
        total_fuel_cost = df['FuelCost'].sum()
        avg_mpg = (df_safe['DistanceDriven'] / df_safe['FuelConsumedGallons']).mean() if not df_safe.empty else 0
        
        metrics = {
            "Total Fuel Cost": total_fuel_cost,
            "Average MPG": avg_mpg
        }
        
        insights.append(f"Total Fuel Cost: ${total_fuel_cost:,.2f}")
        insights.append(f"Average MPG: {avg_mpg:.2f}")
        
        fig1 = px.histogram(df, x='FuelType', title='Fuel Type Distribution')
        visualizations["Fuel_Type_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.scatter(df, x='DistanceDriven', y='FuelConsumedGallons', color='FuelType', hover_name='VehicleID',
                          title='Fuel Consumption vs. Distance Driven by Fuel Type')
        visualizations["Fuel_Consumption_vs_Distance_Scatter"] = fig2.to_json()
        
        fig3 = px.box(df, y='FuelCost', title='Fuel Cost Distribution')
        visualizations["Fuel_Cost_Distribution_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def maintenance_analysis(df):
    analysis_name = "Maintenance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['VehicleID', 'MaintenanceType', 'MaintenanceCost', 'DateOfMaintenance', 'MileageAtMaintenance']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['MaintenanceCost'] = pd.to_numeric(df['MaintenanceCost'], errors='coerce')
        df['MileageAtMaintenance'] = pd.to_numeric(df['MileageAtMaintenance'], errors='coerce')
        df['DateOfMaintenance'] = pd.to_datetime(df['DateOfMaintenance'], errors='coerce')
        df = df.dropna(subset=['MaintenanceCost', 'MileageAtMaintenance', 'DateOfMaintenance'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_maintenance_cost = df['MaintenanceCost'].sum()
        avg_maintenance_cost_per_vehicle = df.groupby('VehicleID')['MaintenanceCost'].sum().mean()
        most_common_maintenance_type = df['MaintenanceType'].mode()[0]
        
        metrics = {
            "Total Maintenance Cost": total_maintenance_cost,
            "Average Maintenance Cost per Vehicle": avg_maintenance_cost_per_vehicle,
            "Most Common Maintenance Type": most_common_maintenance_type
        }
        
        insights.append(f"Total Maintenance Cost: ${total_maintenance_cost:,.2f}")
        insights.append(f"Average Maintenance Cost per Vehicle: ${avg_maintenance_cost_per_vehicle:,.2f}")
        insights.append(f"Most Common Maintenance Type: {most_common_maintenance_type}")
        
        fig1 = px.histogram(df, x='MaintenanceType', title='Maintenance Type Frequency')
        visualizations["Maintenance_Type_Frequency_Histogram"] = fig1.to_json()
        
        monthly_data = df.set_index('DateOfMaintenance').resample('M')['MaintenanceCost'].sum().reset_index().rename(columns={'DateOfMaintenance': 'Month'})
        fig2 = px.line(monthly_data, x='Month', y='MaintenanceCost', title='Monthly Maintenance Cost Trend')
        visualizations["Monthly_Maintenance_Cost_Trend_Line"] = fig2.to_json()
        
        fig3 = px.box(df, y='MaintenanceCost', color='MaintenanceType', title='Maintenance Cost Distribution by Type')
        visualizations["Maintenance_Cost_Distribution_by_Type_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def delivery_analysis(df):
    analysis_name = "Delivery Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['DeliveryID', 'RouteID', 'DeliveryStatus', 'DeliveryTimeSeconds', 'CustomerRating']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['DeliveryTimeSeconds'] = pd.to_numeric(df['DeliveryTimeSeconds'], errors='coerce')
        df['CustomerRating'] = pd.to_numeric(df['CustomerRating'], errors='coerce')
        df = df.dropna(subset=['DeliveryTimeSeconds', 'CustomerRating'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_deliveries = len(df)
        avg_delivery_time_minutes = (df['DeliveryTimeSeconds'].mean() / 60) if df['DeliveryTimeSeconds'].mean() is not np.nan else 0
        avg_customer_rating = df['CustomerRating'].mean()
        
        metrics = {
            "Total Deliveries": total_deliveries,
            "Average Delivery Time (minutes)": avg_delivery_time_minutes,
            "Average Customer Rating": avg_customer_rating
        }
        
        insights.append(f"Total Deliveries: {total_deliveries}")
        insights.append(f"Average Delivery Time: {avg_delivery_time_minutes:.1f} minutes")
        insights.append(f"Average Customer Rating: {avg_customer_rating:.1f}")
        
        fig1 = px.histogram(df, x='DeliveryStatus', title='Delivery Status Distribution')
        visualizations["Delivery_Status_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, y='DeliveryTimeSeconds', title='Delivery Time Distribution (Seconds)')
        visualizations["Delivery_Time_Distribution_Box"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='CustomerRating', nbins=5, title='Customer Rating Distribution')
        visualizations["Customer_Rating_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def cost_analysis(df):
    analysis_name = "Cost Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['VehicleID', 'CostCategory', 'Amount', 'Month']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['VehicleID', 'CostCategory', 'Amount'] if matched[col] is None] # Month is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
        df = df.dropna(subset=['Amount', 'CostCategory'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_cost = df['Amount'].sum()
        avg_cost_per_vehicle = df.groupby('VehicleID')['Amount'].sum().mean()
        highest_cost_category = df.groupby('CostCategory')['Amount'].sum().idxmax()
        
        metrics = {
            "Total Overall Cost": total_cost,
            "Average Cost per Vehicle": avg_cost_per_vehicle,
            "Highest Cost Category": highest_cost_category
        }
        
        insights.append(f"Total Overall Cost: ${total_cost:,.2f}")
        insights.append(f"Average Cost per Vehicle: ${avg_cost_per_vehicle:,.2f}")
        insights.append(f"Highest Cost Category: {highest_cost_category}")
        
        fig1 = px.pie(df, names='CostCategory', values='Amount', title='Cost Distribution by Category')
        visualizations["Cost_Distribution_by_Category_Pie"] = fig1.to_json()
        
        if 'Month' in df.columns:
            monthly_costs = df.groupby('Month')['Amount'].sum().reset_index()
            fig2 = px.line(monthly_costs, x='Month', y='Amount', title='Monthly Cost Trend')
            visualizations["Monthly_Cost_Trend_Line"] = fig2.to_json()
        else:
            insights.append("Note: 'Month' column not found for monthly trend analysis.")
        
        fig3 = px.box(df, y='Amount', color='CostCategory', title='Cost Distribution by Category')
        visualizations["Cost_Distribution_by_Category_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def safety_analysis(df):
    analysis_name = "Safety Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['IncidentID', 'IncidentType', 'Date', 'Severity', 'Location']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['IncidentID', 'IncidentType', 'Date', 'Severity'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'IncidentType', 'Severity'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_incidents = len(df)
        most_common_incident_type = df['IncidentType'].mode()[0]
        
        avg_severity_val = "N/A"
        if pd.api.types.is_numeric_dtype(df['Severity']):
            avg_severity_val = df['Severity'].mean()
            insights.append(f"Average Incident Severity: {avg_severity_val:.1f}")
        else:
            insights.append(f"Average Incident Severity: {avg_severity_val} (column is non-numeric)")

            
        metrics = {
            "Total Incidents": total_incidents,
            "Most Common Incident Type": most_common_incident_type,
            "Average Incident Severity": avg_severity_val
        }
        
        insights.insert(0, f"Total Incidents Recorded: {total_incidents}")
        insights.insert(1, f"Most Common Incident Type: {most_common_incident_type}")
        
        fig1 = px.histogram(df, x='IncidentType', title='Incident Type Frequency')
        visualizations["Incident_Type_Frequency_Histogram"] = fig1.to_json()
        
        monthly_incidents = df.set_index('Date').resample('M').size().reset_index(name='Count')
        monthly_incidents['YearMonth'] = monthly_incidents['Date'].dt.to_period('M').astype(str)
        fig2 = px.line(monthly_incidents, x='YearMonth', y='Count', title='Monthly Incident Trend')
        visualizations["Monthly_Incident_Trend_Line"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='Severity', title='Incident Severity Distribution')
        visualizations["Incident_Severity_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def commuter_transportation_mode_choice_analysis(df):
    analysis_name = "Commuter Transportation Mode Choice Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['UserID', 'ModeChoice', 'CommuteDistance', 'CommuteTime', 'AgeGroup', 'IncomeLevel']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['UserID', 'ModeChoice', 'CommuteDistance', 'CommuteTime'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['CommuteDistance'] = pd.to_numeric(df['CommuteDistance'], errors='coerce')
        df['CommuteTime'] = pd.to_numeric(df['CommuteTime'], errors='coerce')
        df = df.dropna(subset=['ModeChoice', 'CommuteDistance', 'CommuteTime'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        most_popular_mode = df['ModeChoice'].mode()[0]
        avg_commute_distance = df['CommuteDistance'].mean()
        avg_commute_time = df['CommuteTime'].mean()
        
        metrics = {
            "Most Popular Commute Mode": most_popular_mode,
            "Average Commute Distance": avg_commute_distance,
            "Average Commute Time": avg_commute_time
        }
        
        insights.append(f"Most Popular Commute Mode: {most_popular_mode}")
        insights.append(f"Average Commute Distance: {avg_commute_distance:.2f} miles")
        insights.append(f"Average Commute Time: {avg_commute_time:.1f} minutes")
        
        fig1 = px.pie(df, names='ModeChoice', title='Commuter Mode Choice Distribution')
        visualizations["Mode_Choice_Distribution_Pie"] = fig1.to_json()
        
        fig2 = px.box(df, x='ModeChoice', y='CommuteTime', title='Commute Time Distribution by Mode')
        visualizations["Commute_Time_by_Mode_Box"] = fig2.to_json()
        
        fig3 = px.scatter(df, x='CommuteDistance', y='CommuteTime', color='ModeChoice', hover_name='UserID',
                          title='Commute Time vs. Distance by Mode')
        visualizations["Commute_Time_vs_Distance_Scatter"] = fig3.to_json()

        if 'AgeGroup' in df.columns:
            fig4 = px.histogram(df, x='AgeGroup', color='ModeChoice', barmode='group', title='Mode Choice by Age Group')
            visualizations["Mode_Choice_by_Age_Group_Histogram"] = fig4.to_json()
        
        if 'IncomeLevel' in df.columns:
            fig5 = px.histogram(df, x='IncomeLevel', color='ModeChoice', barmode='group', title='Mode Choice by Income Level')
            visualizations["Mode_Choice_by_Income_Level_Histogram"] = fig5.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_bus_performance_and_delay_analysis(df):
    analysis_name = "Public Bus Performance and Delay Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'BusID', 'ScheduledArrivalTime', 'ActualArrivalTime', 'TripDuration', 'PassengerCount']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['RouteID', 'ScheduledArrivalTime', 'ActualArrivalTime', 'TripDuration', 'PassengerCount'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')
        df['ActualArrivalTime'] = pd.to_datetime(df['ActualArrivalTime'], errors='coerce')
        df['TripDuration'] = pd.to_numeric(df['TripDuration'], errors='coerce')
        df['PassengerCount'] = pd.to_numeric(df['PassengerCount'], errors='coerce')
        df = df.dropna(subset=['ScheduledArrivalTime', 'ActualArrivalTime', 'TripDuration', 'PassengerCount'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        df['DelayMinutes'] = (df['ActualArrivalTime'] - df['ScheduledArrivalTime']).dt.total_seconds() / 60
        
        avg_delay = df['DelayMinutes'].mean()
        on_time_percentage = (df['DelayMinutes'] <= 5).mean() * 100 # Assuming <= 5 min delay is on-time
        avg_ridership = df['PassengerCount'].mean()
        
        metrics = {
            "Average Bus Delay": avg_delay,
            "On-Time Performance": on_time_percentage,
            "Average Ridership per Trip": avg_ridership
        }
        
        insights.append(f"Average Bus Delay: {avg_delay:.2f} minutes")
        insights.append(f"On-Time Performance: {on_time_percentage:.2f}%")
        insights.append(f"Average Ridership per Trip: {avg_ridership:.1f} passengers")
        
        fig1 = px.histogram(df, x='DelayMinutes', nbins=30, title='Distribution of Bus Delays (Minutes)')
        visualizations["Bus_Delays_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='RouteID', y='DelayMinutes', title='Delay Distribution by Route')
        visualizations["Delay_Distribution_by_Route_Box"] = fig2.to_json()
        
        fig3 = px.scatter(df, x='TripDuration', y='PassengerCount', color='RouteID', hover_name='BusID',
                          title='Passenger Count vs. Trip Duration by Route')
        visualizations["Passenger_Count_vs_Trip_Duration_Scatter"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def global_air_transport_passenger_trends_analysis(df):
    analysis_name = "Global Air Transport Passenger Trends Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['Country', 'Year', 'TotalPassengers', 'CargoTons']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['Country', 'Year', 'TotalPassengers'] if matched[col] is None] # Cargo is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['TotalPassengers'] = pd.to_numeric(df['TotalPassengers'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['TotalPassengers', 'Year'])
        df = df.sort_values(by='Year')

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_passengers_overall = df['TotalPassengers'].sum()
        latest_year_passengers = df.loc[df['Year'].idxmax(), 'TotalPassengers']
        
        metrics = {
            "Total Passengers Overall": total_passengers_overall,
            "Latest Year Passengers": latest_year_passengers,
            "Latest Year": df['Year'].max()
        }
        
        insights.append(f"Total Passengers (Overall): {total_passengers_overall:,.0f}")
        insights.append(f"Latest Year ({df['Year'].max()}) Passengers: {latest_year_passengers:,.0f}")
        
        fig1 = px.line(df, x='Year', y='TotalPassengers', color='Country', title='Global Passenger Trends by Country')
        visualizations["Global_Passenger_Trends_Line"] = fig1.to_json()
        
        yearly_global_passengers = df.groupby('Year')['TotalPassengers'].sum().reset_index()
        fig2 = px.bar(yearly_global_passengers, x='Year', y='TotalPassengers', title='Total Global Air Passengers by Year')
        visualizations["Total_Global_Air_Passengers_Bar"] = fig2.to_json()
        
        if 'CargoTons' in df.columns:
            df['CargoTons'] = pd.to_numeric(df['CargoTons'], errors='coerce')
            if not df['CargoTons'].dropna().empty:
                fig3 = px.line(df, x='Year', y='CargoTons', color='Country', title='Global Air Cargo Trends by Country')
                visualizations["Global_Air_Cargo_Trends_Line"] = fig3.to_json()
            else:
                insights.append("Note: 'CargoTons' column found but has no valid data.")
        else:
            insights.append("Note: 'CargoTons' column not found for cargo trend analysis.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def airline_directory_and_operational_status_analysis(df):
    analysis_name = "Airline Directory and Operational Status Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['AirlineID', 'AirlineName', 'Country', 'OperationalStatus', 'FleetSize']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['AirlineID', 'AirlineName', 'Country', 'OperationalStatus', 'FleetSize'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FleetSize'] = pd.to_numeric(df['FleetSize'], errors='coerce')
        df = df.dropna(subset=['OperationalStatus', 'FleetSize'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_airlines = len(df)
        active_airlines_count = (df['OperationalStatus'].str.lower() == 'active').sum()
        avg_fleet_size_active = df[df['OperationalStatus'].str.lower() == 'active']['FleetSize'].mean()
        
        metrics = {
            "Total Airlines": total_airlines,
            "Active Airlines": active_airlines_count,
            "Average Fleet Size of Active Airlines": avg_fleet_size_active
        }
        
        insights.append(f"Total Airlines in Directory: {total_airlines}")
        insights.append(f"Active Airlines: {active_airlines_count}")
        insights.append(f"Average Fleet Size of Active Airlines: {avg_fleet_size_active:.0f} aircraft")
        
        fig1 = px.pie(df, names='OperationalStatus', title='Airline Operational Status Distribution')
        visualizations["Airline_Operational_Status_Pie"] = fig1.to_json()
        
        airlines_by_country = df['Country'].value_counts().reset_index()
        airlines_by_country.columns = ['Country', 'Count']
        fig2 = px.bar(airlines_by_country.head(20), x='Country', y='Count', title='Number of Airlines by Country (Top 20)')
        visualizations["Airlines_by_Country_Bar"] = fig2.to_json()
        
        fig3 = px.box(df, x='OperationalStatus', y='FleetSize', title='Fleet Size Distribution by Operational Status')
        visualizations["Fleet_Size_by_Operational_Status_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_transit_fare_and_journey_type_analysis(df):
    analysis_name = "Public Transit Fare and Journey Type Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FareID', 'JourneyType', 'FareAmount', 'RouteID', 'RidershipCount']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['FareID', 'JourneyType', 'FareAmount', 'RidershipCount'] if matched[col] is None] # RouteID is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FareAmount'] = pd.to_numeric(df['FareAmount'], errors='coerce')
        df['RidershipCount'] = pd.to_numeric(df['RidershipCount'], errors='coerce')
        df = df.dropna(subset=['FareAmount', 'RidershipCount', 'JourneyType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_fare_amount = df['FareAmount'].mean()
        most_common_journey_type = df['JourneyType'].mode()[0]
        total_ridership_by_fare = df['RidershipCount'].sum()
        
        metrics = {
            "Average Fare Amount": avg_fare_amount,
            "Most Common Journey Type": most_common_journey_type,
            "Total Ridership by Fare Type": total_ridership_by_fare
        }
        
        insights.append(f"Average Fare Amount: ${avg_fare_amount:.2f}")
        insights.append(f"Most Common Journey Type: {most_common_journey_type}")
        insights.append(f"Total Ridership by Fare Type: {total_ridership_by_fare:,.0f}")
        
        fig1 = px.histogram(df, x='JourneyType', y='RidershipCount', title='Ridership by Journey Type')
        visualizations["Ridership_by_Journey_Type_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='JourneyType', y='FareAmount', title='Fare Amount Distribution by Journey Type')
        visualizations["Fare_Amount_by_Journey_Type_Box"] = fig2.to_json()
        
        if 'RouteID' in df.columns:
            ridership_by_route_fare = df.groupby(['RouteID', 'JourneyType'])['RidershipCount'].sum().reset_index()
            fig3 = px.bar(ridership_by_route_fare.sort_values('RidershipCount', ascending=False).head(20),
                           x='RouteID', y='RidershipCount', color='JourneyType', title='Top Routes by Ridership and Journey Type')
            visualizations["Top_Routes_by_Ridership_and_Journey_Type_Bar"] = fig3.to_json()
        else:
            insights.append("Note: 'RouteID' column not found for route-specific ridership analysis.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def regional_vehicle_registration_trend_analysis(df):
    analysis_name = "Regional Vehicle Registration Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['Region', 'Year', 'VehicleType', 'RegisteredVehiclesCount']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['RegisteredVehiclesCount'] = pd.to_numeric(df['RegisteredVehiclesCount'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['RegisteredVehiclesCount', 'Year', 'Region', 'VehicleType'])
        df = df.sort_values(by='Year')

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_registered_vehicles = df['RegisteredVehiclesCount'].sum()
        latest_year_registrations = df[df['Year'] == df['Year'].max()]['RegisteredVehiclesCount'].sum()
        
        metrics = {
            "Total Registered Vehicles Overall": total_registered_vehicles,
            "Latest Year Registrations": latest_year_registrations,
            "Latest Year": df['Year'].max()
        }
        
        insights.append(f"Total Registered Vehicles (Overall): {total_registered_vehicles:,.0f}")
        insights.append(f"Latest Year ({df['Year'].max()}) Registrations: {latest_year_registrations:,.0f}")
        
        fig1 = px.line(df, x='Year', y='RegisteredVehiclesCount', color='Region',
                       title='Vehicle Registration Trends by Region')
        visualizations["Vehicle_Registration_Trends_by_Region_Line"] = fig1.to_json()
        
        fig2 = px.bar(df.groupby('VehicleType')['RegisteredVehiclesCount'].sum().reset_index().sort_values('RegisteredVehiclesCount', ascending=False),
                       x='VehicleType', y='RegisteredVehiclesCount', title='Total Registered Vehicles by Type')
        visualizations["Total_Registered_Vehicles_by_Type_Bar"] = fig2.to_json()
        
        regional_yearly_registrations = df.groupby(['Region', 'Year'])['RegisteredVehiclesCount'].sum().reset_index()
        fig3 = px.line(regional_yearly_registrations, x='Year', y='RegisteredVehiclesCount', color='Region', line_group='Region',
                       title='Regional Vehicle Registration Trends Over Time')
        visualizations["Regional_Vehicle_Registration_Trends_Over_Time_Line"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transportation_user_survey_response_analysis(df):
    analysis_name = "Transportation User Survey Response Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['SurveyID', 'Age', 'Gender', 'SatisfactionRating', 'FrequencyOfUse', 'PrimaryTransportationMode']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['SurveyID', 'SatisfactionRating', 'PrimaryTransportationMode'] if matched[col] is None] # Age/Gender optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['SatisfactionRating'] = pd.to_numeric(df['SatisfactionRating'], errors='coerce')
        df = df.dropna(subset=['SatisfactionRating', 'PrimaryTransportationMode'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_responses = len(df)
        avg_satisfaction = df['SatisfactionRating'].mean()
        most_freq_mode_in_survey = df['PrimaryTransportationMode'].mode()[0]
        
        metrics = {
            "Total Survey Responses": total_responses,
            "Average Satisfaction Rating": avg_satisfaction,
            "Most Frequent Primary Transportation Mode": most_freq_mode_in_survey
        }
        
        insights.append(f"Total Survey Responses: {total_responses}")
        insights.append(f"Average Satisfaction Rating: {avg_satisfaction:.2f}")
        insights.append(f"Most Frequent Primary Transportation Mode in Survey: {most_freq_mode_in_survey}")
        
        fig1 = px.histogram(df, x='SatisfactionRating', nbins=5, title='Satisfaction Rating Distribution')
        visualizations["Satisfaction_Rating_Distribution_Histogram"] = fig1.to_json()
        
        if 'Gender' in df.columns:
            fig2 = px.histogram(df, x='PrimaryTransportationMode', color='Gender', barmode='group',
                                title='Primary Transportation Mode by Gender')
            visualizations["Primary_Transportation_Mode_by_Gender_Histogram"] = fig2.to_json()
        
        if 'Age' in df.columns:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
            if not df['Age'].isnull().all():
                fig3 = px.box(df, x='PrimaryTransportationMode', y='Age', title='Age Distribution by Primary Transportation Mode')
                visualizations["Age_Distribution_by_Primary_Transportation_Mode_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def bus_route_schedule_analysis(df):
    analysis_name = "Bus Route Schedule Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'TripID', 'DepartureTime', 'ArrivalTime', 'ServiceDay', 'HeadwayMinutes']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['RouteID', 'TripID', 'DepartureTime', 'ArrivalTime', 'HeadwayMinutes'] if matched[col] is None] # ServiceDay is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        # Handle time-only strings
        df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['HeadwayMinutes'] = pd.to_numeric(df['HeadwayMinutes'], errors='coerce')
        df = df.dropna(subset=['DepartureTime', 'ArrivalTime', 'HeadwayMinutes', 'RouteID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_trips = len(df)
        avg_headway = df['HeadwayMinutes'].mean()
        most_frequent_route = df.groupby('RouteID')['HeadwayMinutes'].mean().nsmallest(1).index[0]
        
        metrics = {
            "Total Scheduled Trips": total_trips,
            "Average Headway": avg_headway,
            "Most Frequent Route": most_frequent_route
        }
        
        insights.append(f"Total Scheduled Trips: {total_trips}")
        insights.append(f"Average Headway (Frequency) Across Routes: {avg_headway:.1f} minutes")
        insights.append(f"Most Frequent Route (lowest average headway): {most_frequent_route}")
        
        fig1 = px.histogram(df, x='HeadwayMinutes', nbins=20, title='Distribution of Headway Minutes')
        visualizations["Headway_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='RouteID', y='HeadwayMinutes', title='Headway Distribution by Route')
        visualizations["Headway_Distribution_by_Route_Box"] = fig2.to_json()
        
        if 'ServiceDay' in df.columns:
            fig3 = px.histogram(df, x='ServiceDay', title='Scheduled Trips by Service Day')
            visualizations["Scheduled_Trips_by_Service_Day_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_transit_station_ridership_analysis(df):
    analysis_name = "Public Transit Station Ridership Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['StationID', 'StationName', 'Date', 'DailyRidership', 'Line']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['StationID', 'StationName', 'Date', 'DailyRidership'] if matched[col] is None] # Line is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['DailyRidership'] = pd.to_numeric(df['DailyRidership'], errors='coerce')
        df = df.dropna(subset=['Date', 'DailyRidership', 'StationName'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_ridership = df['DailyRidership'].sum()
        busiest_station = df.groupby('StationName')['DailyRidership'].sum().idxmax()
        avg_daily_ridership = df['DailyRidership'].mean()
        
        metrics = {
            "Total Ridership Recorded": total_ridership,
            "Busiest Station": busiest_station,
            "Average Daily Ridership per Station": avg_daily_ridership
        }
        
        insights.append(f"Total Ridership Recorded: {total_ridership:,.0f}")
        insights.append(f"Busiest Station: {busiest_station}")
        insights.append(f"Average Daily Ridership per Station: {avg_daily_ridership:.0f}")
        
        fig1 = px.histogram(df, x='DailyRidership', nbins=30, title='Distribution of Daily Ridership')
        visualizations["Daily_Ridership_Distribution_Histogram"] = fig1.to_json()
        
        top_stations = df.groupby('StationName')['DailyRidership'].sum().nlargest(20).reset_index()
        fig2 = px.bar(top_stations, x='StationName', y='DailyRidership', title='Top 20 Busiest Stations by Total Ridership')
        visualizations["Top_20_Busiest_Stations_Bar"] = fig2.to_json()
        
        ridership_trend = df.set_index('Date').resample('M')['DailyRidership'].sum().reset_index().rename(columns={'Date':'Month'})
        fig3 = px.line(ridership_trend, x='Month', y='DailyRidership', title='Monthly Ridership Trend')
        visualizations["Monthly_Ridership_Trend_Line"] = fig3.to_json()

        if 'Line' in df.columns:
            fig4 = px.box(df, x='Line', y='DailyRidership', title='Daily Ridership by Transit Line')
            visualizations["Daily_Ridership_by_Transit_Line_Box"] = fig4.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def county_level_transportation_infrastructure_and_commute_analysis(df):
    analysis_name = "County-Level Transportation Infrastructure and Commute Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['CountyName', 'State', 'RoadMiles', 'PublicTransitAccessScore', 'AverageCommuteTime', 'Population']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['RoadMiles'] = pd.to_numeric(df['RoadMiles'], errors='coerce')
        df['PublicTransitAccessScore'] = pd.to_numeric(df['PublicTransitAccessScore'], errors='coerce')
        df['AverageCommuteTime'] = pd.to_numeric(df['AverageCommuteTime'], errors='coerce')
        df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
        df = df.dropna(subset=['RoadMiles', 'PublicTransitAccessScore', 'AverageCommuteTime', 'Population'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_road_miles = df['RoadMiles'].sum()
        avg_commute_time_overall = df['AverageCommuteTime'].mean()
        county_highest_transit_access = df.loc[df['PublicTransitAccessScore'].idxmax(), 'CountyName']
        
        metrics = {
            "Total Road Miles Overall": total_road_miles,
            "Average Commute Time Overall": avg_commute_time_overall,
            "County with Highest Public Transit Access": county_highest_transit_access
        }
        
        insights.append(f"Total Road Miles (Overall): {total_road_miles:,.0f} miles")
        insights.append(f"Average Commute Time (Overall): {avg_commute_time_overall:.1f} minutes")
        insights.append(f"County with Highest Public Transit Access: {county_highest_transit_access}")
        
        fig1 = px.scatter(df, x='RoadMiles', y='AverageCommuteTime', color='PublicTransitAccessScore', hover_name='CountyName',
                          title='Average Commute Time vs. Road Miles (colored by Transit Access)')
        visualizations["Commute_Time_vs_Road_Miles_Scatter"] = fig1.to_json()
        
        fig2 = px.box(df, x='State', y='AverageCommuteTime', title='Average Commute Time by State')
        visualizations["Average_Commute_Time_by_State_Box"] = fig2.to_json()
        
        fig3 = px.bar(df.sort_values('PublicTransitAccessScore', ascending=False).head(20), x='CountyName', y='PublicTransitAccessScore',
                      title='Top 20 Counties by Public Transit Access Score')
        visualizations["Top_20_Counties_by_Public_Transit_Access_Bar"] = fig3.to_json()
        
        fig4 = px.scatter(df, x='Population', y='RoadMiles', size='PublicTransitAccessScore', hover_name='CountyName',
                          title='Road Miles vs. Population (sized by Transit Access)')
        visualizations["Road_Miles_vs_Population_Scatter"] = fig4.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_agency_information_analysis(df):
    analysis_name = "Transit Agency Information Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['AgencyID', 'AgencyName', 'AgencyURL', 'AgencyTimezone', 'AgencyFareURL']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['AgencyID', 'AgencyName', 'AgencyURL', 'AgencyTimezone'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df.empty:
            insights.append("No data available for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_agencies = len(df)
        unique_timezones = df['AgencyTimezone'].nunique()
        
        metrics = {
            "Total Transit Agencies": total_agencies,
            "Unique Timezones Represented": unique_timezones
        }
        
        insights.append(f"Total Transit Agencies: {total_agencies}")
        insights.append(f"Unique Timezones Represented: {unique_timezones}")
        
        agency_counts_by_timezone = df['AgencyTimezone'].value_counts().reset_index()
        agency_counts_by_timezone.columns = ['Timezone', 'Count']
        fig1 = px.bar(agency_counts_by_timezone, x='Timezone', y='Count', title='Number of Agencies by Timezone')
        visualizations["Agencies_by_Timezone_Bar"] = fig1.to_json()
        
        sample_details = []
        if 'AgencyName' in df.columns and 'AgencyURL' in df.columns and 'AgencyTimezone' in df.columns:
            for index, row in df.head(10).iterrows():
                sample_details.append(f"{row['AgencyName']} | URL: {row['AgencyURL']} | Timezone: {row['AgencyTimezone']}")
        insights.append("Sample Agency Details:")
        insights.extend(sample_details)

        metrics["details"] = {
            "Sample Agency URLs": df['AgencyURL'].head().tolist() if 'AgencyURL' in df.columns else None
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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_transit_route_definition_analysis(df):
    analysis_name = "Public Transit Route Definition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'AgencyID', 'RouteShortName', 'RouteLongName', 'RouteType', 'RouteColor']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['RouteID', 'RouteShortName', 'RouteType', 'AgencyID'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_routes = len(df)
        unique_route_types = df['RouteType'].nunique()
        most_common_route_type = df['RouteType'].mode()[0]
        
        metrics = {
            "Total Defined Routes": total_routes,
            "Unique Route Types": unique_route_types,
            "Most Common Route Type": str(most_common_route_type) # Ensure native type
        }
        
        insights.append(f"Total Defined Routes: {total_routes}")
        insights.append(f"Unique Route Types: {unique_route_types}")
        insights.append(f"Most Common Route Type: {most_common_route_type}")
        
        fig1 = px.histogram(df, x='RouteType', title='Distribution of Route Types')
        visualizations["Route_Types_Distribution_Histogram"] = fig1.to_json()
        
        routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
        fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                      title='Number of Routes per Agency (Top 20)')
        visualizations["Routes_per_Agency_Bar"] = fig2.to_json()
        
        if 'RouteShortName' in df.columns and 'RouteLongName' in df.columns:
            insights.append("\nSample Routes:")
            for index, row in df.head(10).iterrows():
                insights.append(f"- {row['RouteShortName']}: {row['RouteLongName']} (Type: {row['RouteType']})")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_trip_schedule_and_accessibility_analysis(df):
    analysis_name = "Transit Trip Schedule and Accessibility Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'RouteID', 'ServiceCode', 'WheelchairAccessible', 'BikesAllowed', 'TripStartTime', 'TripEndTime']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'RouteID', 'WheelchairAccessible', 'BikesAllowed', 'TripStartTime', 'TripEndTime'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['TripStartTime'] = pd.to_datetime(df['TripStartTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['TripEndTime'] = pd.to_datetime(df['TripEndTime'], format='%H:%M:%S', errors='coerce').dt.time
        
        df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        
        df = df.dropna(subset=['TripID', 'RouteID', 'TripStartTime', 'TripEndTime', 'WheelchairAccessible', 'BikesAllowed'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_trips = len(df)
        accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
        bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
        
        metrics = {
            "Total Trips Scheduled": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        }
        
        insights.append(f"Total Trips Scheduled: {total_trips}")
        insights.append(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
        insights.append(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
        
        fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
        visualizations["Wheelchair_Accessibility_Pie"] = fig1.to_json()
        
        fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
        visualizations["Bike_Accessibility_Pie"] = fig2.to_json()
        
        if 'ServiceCode' in df.columns:
            trips_by_service = df.groupby('ServiceCode').size().reset_index(name='Count')
            fig3 = px.bar(trips_by_service, x='ServiceCode', y='Count', title='Number of Trips by Service Code')
            visualizations["Trips_by_Service_Code_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_stop_location_and_information_analysis(df):
    analysis_name = "Transit Stop Location and Information Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['StopID', 'StopName', 'StopLat', 'StopLon', 'LocationType', 'WheelchairBoarding']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['StopLat'] = pd.to_numeric(df['StopLat'], errors='coerce')
        df['StopLon'] = pd.to_numeric(df['StopLon'], errors='coerce')
        df['WheelchairBoarding'] = df['WheelchairBoarding'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['StopLat', 'StopLon', 'StopName', 'LocationType', 'WheelchairBoarding'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_stops = len(df)
        accessible_stops_pct = (df['WheelchairBoarding'] == 'Yes').mean() * 100
        unique_stop_types = df['LocationType'].nunique()
        
        metrics = {
            "Total Transit Stops": total_stops,
            "Wheelchair Accessible Stops (%)": accessible_stops_pct,
            "Unique Location Types for Stops": unique_stop_types
        }
        
        insights.append(f"Total Transit Stops: {total_stops}")
        insights.append(f"Percentage of Wheelchair Accessible Stops: {accessible_stops_pct:.2f}%")
        insights.append(f"Unique Location Types for Stops: {unique_stop_types}")
        
        fig1 = px.scatter_mapbox(df, lat='StopLat', lon='StopLon', hover_name='StopName', color='LocationType',
                                 zoom=10, title='Transit Stop Locations by Type',
                                 mapbox_style="carto-positron")
        visualizations["Transit_Stop_Locations_Map"] = fig1.to_json()
        
        fig2 = px.pie(df, names='WheelchairBoarding', title='Wheelchair Boarding Availability at Stops')
        visualizations["Wheelchair_Boarding_Availability_Pie"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='LocationType', title='Distribution of Stop Location Types')
        visualizations["Stop_Location_Types_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_stop_time_and_sequence_analysis(df):
    analysis_name = "Transit Stop Time and Sequence Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence', 'TravelTimeFromPreviousStop']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'] if matched[col] is None] # TravelTime is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
        df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_stop_times = len(df)
        avg_stops_per_trip = df.groupby('TripID').size().mean()
        
        metrics = {
            "Total Stop Times Records": total_stop_times,
            "Average Number of Stops per Trip": avg_stops_per_trip
        }
        
        insights.append(f"Total Stop Times Records: {total_stop_times}")
        insights.append(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
        
        fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences')
        visualizations["Stop_Sequences_Distribution_Histogram"] = fig1.to_json()
        
        # Calculate dwell time
        # Need to convert time objects to datetime objects to subtract
        today = datetime.now().date()
        arrival_dt = pd.to_datetime(df['ArrivalTime'].astype(str), format='%H:%M:%S', errors='coerce')
        departure_dt = pd.to_datetime(df['DepartureTime'].astype(str), format='%H:%M:%S', errors='coerce')
        
        df['DwellTimeSeconds'] = (departure_dt - arrival_dt).dt.total_seconds()
        df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan) # Handle overnight trips or data errors
        df_dwell = df.dropna(subset=['DwellTimeSeconds'])

        if not df_dwell.empty:
            fig2 = px.histogram(df_dwell, x='DwellTimeSeconds', nbins=30, title='Distribution of Dwell Times at Stops (Seconds)')
            visualizations["Dwell_Times_Distribution_Histogram"] = fig2.to_json()
        else:
            insights.append("Note: Dwell time calculation not possible or no valid dwell times found.")

        if 'TravelTimeFromPreviousStop' in df.columns:
            df['TravelTimeFromPreviousStop'] = pd.to_numeric(df['TravelTimeFromPreviousStop'], errors='coerce')
            df_travel_time = df.dropna(subset=['TravelTimeFromPreviousStop'])
            if not df_travel_time.empty:
                fig3 = px.histogram(df_travel_time, x='TravelTimeFromPreviousStop',
                                  title='Distribution of Travel Times Between Stops')
                visualizations["Travel_Times_Between_Stops_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_service_calendar_analysis(df):
    analysis_name = "Transit Service Calendar Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        expected_cols = ['ServiceID', 'StartDate', 'EndDate'] + days_of_week
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
        
        for day in days_of_week:
            df[day] = pd.to_numeric(df[day], errors='coerce').fillna(0).astype(bool) # Assuming 0/1
        
        df = df.dropna(subset=['StartDate', 'EndDate'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_service_ids = len(df)
        avg_service_duration_days = (df['EndDate'] - df['StartDate']).dt.days.mean()
        
        metrics = {
            "Total Service IDs Defined": total_service_ids,
            "Average Service Duration (days)": avg_service_duration_days
        }
        
        insights.append(f"Total Service IDs Defined: {total_service_ids}")
        insights.append(f"Average Service Duration: {avg_service_duration_days:.1f} days")
        
        service_days_counts = {day: df[day].sum() for day in days_of_week}
        service_days_df = pd.DataFrame(service_days_counts.items(), columns=['Day', 'ServiceCount'])
        fig1 = px.bar(service_days_df, x='Day', y='ServiceCount', title='Total Service Days per Day of Week',
                      category_orders={"Day": days_of_week})
        visualizations["Total_Service_Days_Per_Day_of_Week_Bar"] = fig1.to_json()
        
        fig2 = px.histogram(df, x=(df['EndDate'] - df['StartDate']).dt.days, title='Distribution of Service Durations (Days)')
        visualizations["Service_Durations_Distribution_Histogram"] = fig2.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_service_exception_and_holiday_schedule_analysis(df):
    analysis_name = "Transit Service Exception and Holiday Schedule Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['ServiceID', 'Date', 'ExceptionType', 'Description']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['ServiceID', 'Date', 'ExceptionType'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date', 'ExceptionType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_exceptions = len(df)
        most_common_exception_type = df['ExceptionType'].mode()[0]
        unique_exception_dates = df['Date'].nunique()
        
        metrics = {
            "Total Service Exceptions/Holidays": total_exceptions,
            "Most Common Exception Type": str(most_common_exception_type),
            "Unique Dates with Exceptions": unique_exception_dates
        }
        
        insights.append(f"Total Service Exceptions/Holidays: {total_exceptions}")
        insights.append(f"Most Common Exception Type: {most_common_exception_type}")
        insights.append(f"Unique Dates with Exceptions: {unique_exception_dates}")
        
        fig1 = px.histogram(df, x='ExceptionType', title='Distribution of Service Exception Types')
        visualizations["Service_Exception_Types_Distribution_Histogram"] = fig1.to_json()
        
        exceptions_by_month = df.set_index('Date').resample('M').size().reset_index(name='Count')
        exceptions_by_month.columns = ['Month', 'Count']
        fig2 = px.line(exceptions_by_month, x='Month', y='Count', title='Monthly Trend of Service Exceptions')
        visualizations["Monthly_Service_Exceptions_Trend_Line"] = fig2.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_fare_structure_analysis(df):
    analysis_name = "Transit Fare Structure Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FareID', 'Price', 'Currency', 'PaymentMethod', 'TransferPolicy']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['FareID', 'Price', 'Currency', 'PaymentMethod'] if matched[col] is None] # TransferPolicy is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_fare_price = df['Price'].mean()
        most_common_currency = df['Currency'].mode()[0]
        unique_payment_methods = df['PaymentMethod'].nunique()
        
        metrics = {
            "Average Fare Price": avg_fare_price,
            "Most Common Currency": most_common_currency,
            "Unique Payment Methods": unique_payment_methods
        }
        
        insights.append(f"Average Fare Price: ${avg_fare_price:.2f}")
        insights.append(f"Most Common Currency: {most_common_currency}")
        insights.append(f"Unique Payment Methods: {unique_payment_methods}")
        
        fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
        visualizations["Fare_Prices_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price Distribution by Payment Method')
        visualizations["Fare_Price_by_Payment_Method_Box"] = fig2.to_json()
        
        if 'TransferPolicy' in df.columns:
            fig3 = px.histogram(df, x='TransferPolicy', title='Distribution of Transfer Policies')
            visualizations["Transfer_Policies_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_fare_rule_and_zone_analysis(df):
    analysis_name = "Transit Fare Rule and Zone Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FareID', 'RouteID', 'OriginZoneID', 'DestinationZoneID', 'FarePrice']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['FareID', 'OriginZoneID', 'DestinationZoneID', 'FarePrice'] if matched[col] is None] # RouteID is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FarePrice'] = pd.to_numeric(df['FarePrice'], errors='coerce')
        df = df.dropna(subset=['FarePrice', 'OriginZoneID', 'DestinationZoneID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_fare_rules = len(df)
        unique_origin_zones = df['OriginZoneID'].nunique()
        avg_fare_price = df['FarePrice'].mean()
        
        metrics = {
            "Total Fare Rules Defined": total_fare_rules,
            "Unique Origin Zones": unique_origin_zones,
            "Average Fare Price": avg_fare_price
        }
        
        insights.append(f"Total Fare Rules Defined: {total_fare_rules}")
        insights.append(f"Unique Origin Zones: {unique_origin_zones}")
        insights.append(f"Average Fare Price (across rules): ${avg_fare_price:.2f}")
        
        fig1 = px.histogram(df, x='FarePrice', nbins=20, title='Distribution of Fare Prices per Rule')
        visualizations["Fare_Prices_per_Rule_Distribution_Histogram"] = fig1.to_json()
        
        df['ZonePair'] = df['OriginZoneID'].astype(str) + ' to ' + df['DestinationZoneID'].astype(str)
        top_fares = df.sort_values('FarePrice', ascending=False).head(10)
        fig2 = px.bar(top_fares, x='ZonePair', y='FarePrice', title='Top 10 Most Expensive Zone-to-Zone Fares')
        visualizations["Top_10_Most_Expensive_Zone_to_Zone_Fares_Bar"] = fig2.to_json()
        
        if 'RouteID' in df.columns:
            fares_by_route = df.groupby('RouteID')['FarePrice'].mean().reset_index()
            fig3 = px.bar(fares_by_route.sort_values('FarePrice', ascending=False).head(20),
                          x='RouteID', y='FarePrice', title='Average Fare Price by Route (Top 20)')
            visualizations["Average_Fare_Price_by_Route_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_route_shape_and_path_geospatial_analysis(df):
    analysis_name = "Transit Route Shape and Path Geospatial Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['ShapeID', 'ShapeLat', 'ShapeLon', 'ShapeSequence']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ShapeLat'] = pd.to_numeric(df['ShapeLat'], errors='coerce')
        df['ShapeLon'] = pd.to_numeric(df['ShapeLon'], errors='coerce')
        df['ShapeSequence'] = pd.to_numeric(df['ShapeSequence'], errors='coerce')
        df = df.dropna(subset=['ShapeLat', 'ShapeLon', 'ShapeSequence', 'ShapeID'])
        df = df.sort_values(by=['ShapeID', 'ShapeSequence'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_shape_points = len(df)
        unique_route_shapes = df['ShapeID'].nunique()
        
        metrics = {
            "Total Geospatial Shape Points": total_shape_points,
            "Unique Route Shapes": unique_route_shapes
        }
        
        insights.append(f"Total Geospatial Shape Points: {total_shape_points}")
        insights.append(f"Unique Route Shapes: {unique_route_shapes}")
        
        # Plot all shapes (or a sample if too many)
        plot_df = df
        if unique_route_shapes > 50: # Limit to 50 shapes for performance
            top_shapes = df['ShapeID'].value_counts().head(50).index
            plot_df = df[df['ShapeID'].isin(top_shapes)]
            insights.append("Note: Displaying a sample of 50 routes on the map due to high volume.")
            
        fig1 = px.line_mapbox(plot_df, lat='ShapeLat', lon='ShapeLon', color='ShapeID', line_group='ShapeID',
                              zoom=9, title='Geospatial Paths of Transit Routes',
                              mapbox_style="carto-positron")
        visualizations["Geospatial_Paths_of_Transit_Routes_Map"] = fig1.to_json()
        
        if unique_route_shapes > 0:
            sample_shape_id = df['ShapeID'].iloc[0]
            sample_shape_df = df[df['ShapeID'] == sample_shape_id]
            fig2 = px.line_mapbox(sample_shape_df, lat='ShapeLat', lon='ShapeLon',
                                  zoom=12, title=f'Geospatial Path of Sample Route Shape: {sample_shape_id}',
                                  mapbox_style="carto-positron")
            visualizations["Sample_Route_Shape_Geospatial_Path_Map"] = fig2.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_frequency_and_headway_analysis(df):
    analysis_name = "Transit Frequency and Headway Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'RouteID', 'StartTime', 'EndTime', 'HeadwaySeconds']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'RouteID', 'HeadwaySeconds'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['HeadwaySeconds'] = pd.to_numeric(df['HeadwaySeconds'], errors='coerce')
        df = df.dropna(subset=['HeadwaySeconds', 'RouteID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_headway_seconds = df['HeadwaySeconds'].mean()
        avg_headway_minutes = avg_headway_seconds / 60
        
        metrics = {
            "Average Headway (minutes)": avg_headway_minutes
        }
        
        insights.append(f"Average Headway: {avg_headway_minutes:.1f} minutes")
        
        df_plot = df.copy()
        df_plot['HeadwayMinutes'] = df_plot['HeadwaySeconds'] / 60
        
        fig1 = px.histogram(df_plot, x='HeadwayMinutes', nbins=30, title='Distribution of Headway (Minutes)')
        visualizations["Headway_Distribution_Histogram"] = fig1.to_json()
        
        avg_headway_by_route = df_plot.groupby('RouteID')['HeadwayMinutes'].mean().reset_index()
        
        fig2 = px.bar(avg_headway_by_route.sort_values('HeadwayMinutes', ascending=True).head(20),
                      x='RouteID', y='HeadwayMinutes', title='Top 20 Routes by Lowest Average Headway')
        visualizations["Top_20_Routes_by_Lowest_Avg_Headway_Bar"] = fig2.to_json()
        
        fig3 = px.box(df_plot, y='HeadwayMinutes', title='Headway Distribution (Minutes)')
        visualizations["Headway_Distribution_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def station_pathway_and_accessibility_analysis(df):
    analysis_name = "Station Pathway and Accessibility Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FromStopID', 'ToStopID', 'PathwayID', 'PathwayMode', 'IsAccessible']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['PathwayMode', 'IsAccessible'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_pathways = len(df)
        accessible_pathways_pct = (df['IsAccessible'] == 'Yes').mean() * 100
        most_common_pathway_mode = df['PathwayMode'].mode()[0]
        
        metrics = {
            "Total Pathways Defined": total_pathways,
            "Accessible Pathways (%)": accessible_pathways_pct,
            "Most Common Pathway Mode": most_common_pathway_mode
        }
        
        insights.append(f"Total Pathways Defined: {total_pathways}")
        insights.append(f"Percentage of Accessible Pathways: {accessible_pathways_pct:.2f}%")
        insights.append(f"Most Common Pathway Mode: {most_common_pathway_mode}")
        
        fig1 = px.pie(df, names='IsAccessible', title='Accessibility of Station Pathways')
        visualizations["Accessibility_of_Station_Pathways_Pie"] = fig1.to_json()
        
        fig2 = px.histogram(df, x='PathwayMode', color='IsAccessible', barmode='group',
                            title='Pathway Mode Distribution by Accessibility')
        visualizations["Pathway_Mode_Distribution_by_Accessibility_Histogram"] = fig2.to_json()
        
        if 'FromStopID' in df.columns and 'ToStopID' in df.columns:
            pathways_per_stop = df.groupby('FromStopID').size().reset_index(name='Count')
            fig3 = px.bar(pathways_per_stop.sort_values('Count', ascending=False).head(20),
                          x='FromStopID', y='Count', title='Stops with Most Outgoing Pathways (Top 20)')
            visualizations["Stops_with_Most_Outgoing_Pathways_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def gtfs_feed_information_and_version_analysis(df):
    analysis_name = "GTFS Feed Information and Version Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['PublisherName', 'PublisherURL', 'Lang', 'FeedStartDate', 'FeedEndDate', 'FeedVersion']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['PublisherName', 'PublisherURL', 'Lang', 'FeedStartDate', 'FeedEndDate'] if matched[col] is None] # Version is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FeedStartDate'] = pd.to_datetime(df['FeedStartDate'], errors='coerce')
        df['FeedEndDate'] = pd.to_datetime(df['FeedEndDate'], errors='coerce')
        df = df.dropna(subset=['PublisherName', 'PublisherURL', 'Lang', 'FeedStartDate', 'FeedEndDate'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_feeds = len(df)
        most_common_language = df['Lang'].mode()[0]
        avg_feed_validity_days = (df['FeedEndDate'] - df['FeedStartDate']).dt.days.mean()
        
        metrics = {
            "Total GTFS Feeds": total_feeds,
            "Most Common Language": most_common_language,
            "Average Feed Validity Duration (days)": avg_feed_validity_days
        }
        
        insights.append(f"Total GTFS Feeds: {total_feeds}")
        insights.append(f"Most Common Language: {most_common_language}")
        insights.append(f"Average Feed Validity Duration: {avg_feed_validity_days:.1f} days")
        
        fig1 = px.histogram(df, x='Lang', title='Distribution of Feed Languages')
        visualizations["Feed_Languages_Distribution_Histogram"] = fig1.to_json()
        
        if 'FeedVersion' in df.columns:
            fig2 = px.histogram(df, x='FeedVersion', title='Distribution of GTFS Feed Versions')
            visualizations["GTFS_Feed_Versions_Distribution_Histogram"] = fig2.to_json()
        else:
            insights.append("Note: 'FeedVersion' column not found.")
        
        if 'PublisherName' in df.columns:
            publisher_counts = df['PublisherName'].value_counts().reset_index()
            publisher_counts.columns = ['PublisherName', 'Count']
            fig3 = px.bar(publisher_counts.head(20), x='PublisherName', y='Count', title='Top 20 Feed Publishers')
            visualizations["Top_20_Feed_Publishers_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def real_time_vehicle_position_and_trip_update_analysis(df):
    analysis_name = "Real-Time Vehicle Position and Trip Update Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['VehicleID', 'TripID', 'Latitude', 'Longitude', 'Timestamp', 'DelaySeconds']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df['DelaySeconds'] = pd.to_numeric(df['DelaySeconds'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude', 'Timestamp', 'DelaySeconds', 'VehicleID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_updates = len(df)
        avg_delay_seconds = df['DelaySeconds'].mean()
        avg_delay_minutes = avg_delay_seconds / 60
        
        metrics = {
            "Total Real-Time Updates": total_updates,
            "Average Delay (minutes)": avg_delay_minutes
        }
        
        insights.append(f"Total Real-Time Updates: {total_updates}")
        insights.append(f"Average Delay: {avg_delay_minutes:.2f} minutes")
        
        # Sample for map performance
        plot_df = df.sample(n=min(5000, len(df))) # Sample up to 5000 points
        
        fig1 = px.scatter_mapbox(plot_df, lat='Latitude', lon='Longitude', color='DelaySeconds',
                                 size='DelaySeconds', hover_name='VehicleID',
                                 zoom=10, title='Vehicle Positions by Delay (Sampled)',
                                 mapbox_style="carto-positron")
        visualizations["Vehicle_Positions_by_Delay_Map"] = fig1.to_json()
        
        df_plot = df.copy()
        df_plot['DelayMinutes'] = df_plot['DelaySeconds'] / 60
        fig2 = px.histogram(df_plot, x='DelayMinutes', nbins=30, title='Distribution of Delays (Minutes)')
        visualizations["Delays_Distribution_Histogram"] = fig2.to_json()
        
        if 'TripID' in df.columns:
            delay_by_trip = df_plot.groupby('TripID')['DelayMinutes'].mean().reset_index()
            fig3 = px.box(delay_by_trip, y='DelayMinutes', title='Delay Distribution by Trip (Minutes)')
            visualizations["Delay_Distribution_by_Trip_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def extended_transit_route_attribute_analysis(df):
    analysis_name = "Extended Transit Route Attribute Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'RouteShortName', 'RouteLongName', 'RouteType', 'RouteDesc', 'RouteURL', 'RouteColor', 'RouteTextColor']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['RouteID', 'RouteType'] if matched[col] is None] # Others are optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['RouteID', 'RouteType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_routes = len(df)
        unique_route_types = df['RouteType'].nunique()
        
        metrics = {
            "Total Routes with Extended Attributes": total_routes,
            "Unique Route Types": unique_route_types
        }
        
        insights.append(f"Total Routes with Extended Attributes: {total_routes}")
        insights.append(f"Unique Route Types: {unique_route_types}")
        
        fig1 = px.histogram(df, x='RouteType', title='Distribution of Route Types (Extended Attributes)')
        visualizations["Route_Types_Distribution_Histogram_Extended"] = fig1.to_json()
        
        if 'RouteColor' in df.columns and not df['RouteColor'].dropna().empty:
            color_counts = df['RouteColor'].value_counts().reset_index()
            color_counts.columns = ['RouteColor', 'Count']
            fig2 = px.bar(color_counts.head(10), x='RouteColor', y='Count', title='Top 10 Most Common Route Colors')
            visualizations["Top_10_Most_Common_Route_Colors_Bar"] = fig2.to_json()
        else:
            insights.append("Note: 'RouteColor' column not found for color analysis.")

        if 'RouteURL' in df.columns:
            insights.append("\nSample Route URLs:")
            for url in df['RouteURL'].head().tolist():
                insights.append(f"- {url}")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_fare_zone_definition_analysis(df):
    analysis_name = "Transit Fare Zone Definition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['ZoneID', 'ZoneName', 'ZoneCenterLat', 'ZoneCenterLon', 'ZoneType']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['ZoneID', 'ZoneName', 'ZoneCenterLat', 'ZoneCenterLon'] if matched[col] is None] # ZoneType is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ZoneCenterLat'] = pd.to_numeric(df['ZoneCenterLat'], errors='coerce')
        df['ZoneCenterLon'] = pd.to_numeric(df['ZoneCenterLon'], errors='coerce')
        df = df.dropna(subset=['ZoneID', 'ZoneName', 'ZoneCenterLat', 'ZoneCenterLon'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_zones = len(df)
        unique_zone_types = df['ZoneType'].nunique() if 'ZoneType' in df.columns and not df['ZoneType'].isnull().all() else 0
        
        metrics = {
            "Total Fare Zones Defined": total_zones,
            "Unique Zone Types": unique_zone_types
        }
        
        insights.append(f"Total Fare Zones Defined: {total_zones}")
        insights.append(f"Unique Zone Types: {unique_zone_types}")
        
        fig1 = px.scatter_mapbox(df, lat='ZoneCenterLat', lon='ZoneCenterLon', hover_name='ZoneName', 
                                 color='ZoneType' if 'ZoneType' in df.columns else None,
                                 zoom=9, title='Geospatial Locations of Fare Zones',
                                 mapbox_style="carto-positron")
        visualizations["Fare_Zones_Geospatial_Map"] = fig1.to_json()
        
        if 'ZoneType' in df.columns:
            fig2 = px.histogram(df, x='ZoneType', title='Distribution of Fare Zone Types')
            visualizations["Fare_Zone_Types_Distribution_Histogram"] = fig2.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def multi_level_station_and_platform_information_analysis(df):
    analysis_name = "Multi-Level Station and Platform Information Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['StationID', 'StationName', 'LevelID', 'LevelName', 'PlatformCount', 'HasElevator']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['StationID', 'StationName', 'LevelID', 'PlatformCount'] if matched[col] is None] # HasElevator is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['PlatformCount'] = pd.to_numeric(df['PlatformCount'], errors='coerce')
        if 'HasElevator' in df.columns:
            df['HasElevator'] = df['HasElevator'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        
        df = df.dropna(subset=['StationID', 'StationName', 'LevelID', 'PlatformCount'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_levels = len(df)
        total_platforms = df['PlatformCount'].sum()
        stations_with_elevators_pct = (df['HasElevator'] == 'Yes').mean() * 100 if 'HasElevator' in df.columns else np.nan
        
        metrics = {
            "Total Levels Defined": total_levels,
            "Total Platforms Across All Levels": total_platforms,
            "Percentage of Levels with Elevator Access": stations_with_elevators_pct
        }
        
        insights.append(f"Total Levels Defined: {total_levels}")
        insights.append(f"Total Platforms Across All Levels: {total_platforms:,.0f}")
        if not np.isnan(stations_with_elevators_pct):
            insights.append(f"Percentage of Levels with Elevator Access: {stations_with_elevators_pct:.2f}%")
        
        fig1 = px.histogram(df, x='PlatformCount', title='Distribution of Platform Counts per Level')
        visualizations["Platform_Counts_per_Level_Histogram"] = fig1.to_json()
        
        if 'HasElevator' in df.columns:
            fig2 = px.pie(df, names='HasElevator', title='Elevator Availability Across Levels')
            visualizations["Elevator_Availability_Pie"] = fig2.to_json()
        
        stations_with_multiple_levels = df.groupby('StationID').filter(lambda x: x['LevelID'].nunique() > 1)
        if not stations_with_multiple_levels.empty:
            insights.append("\nStations with multiple levels:")
            for station_id in stations_with_multiple_levels['StationID'].unique():
                station_data = df[df['StationID'] == station_id]
                insights.append(f"- Station: {station_data['StationName'].iloc[0]} (ID: {station_id})")
                for index, row in station_data.iterrows():
                    insights.append(f"  Level: {row['LevelName']} (ID: {row['LevelID']}), Platforms: {row['PlatformCount']}, Elevator: {row.get('HasElevator', 'N/A')}")
        else:
            insights.append("No stations with multiple levels found.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_trip_stop_timepoint_analysis(df):
    analysis_name = "Transit Trip Stop Timepoint Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence', 'Timepoint']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in expected_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
        df['Timepoint'] = df['Timepoint'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence', 'Timepoint'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_stop_times = len(df)
        total_timepoints = (df['Timepoint'] == 'Yes').sum()
        
        metrics = {
            "Total Stop Times Records": total_stop_times,
            "Total Designated Timepoints": total_timepoints,
            "Percentage of Timepoints": (total_timepoints / total_stop_times * 100) if total_stop_times > 0 else 0
        }
        
        insights.append(f"Total Stop Times Records: {total_stop_times}")
        insights.append(f"Total Designated Timepoints: {total_timepoints}")
        if total_stop_times > 0:
            insights.append(f"Percentage of Timepoints: {total_timepoints / total_stop_times * 100:.2f}%")
        
        fig1 = px.pie(df, names='Timepoint', title='Distribution of Timepoints vs. Regular Stops')
        visualizations["Timepoints_vs_Regular_Stops_Pie"] = fig1.to_json()
        
        timepoints_per_trip = df[df['Timepoint'] == 'Yes'].groupby('TripID').size().reset_index(name='NumTimepoints')
        if not timepoints_per_trip.empty:
            fig2 = px.histogram(timepoints_per_trip, x='NumTimepoints', title='Distribution of Number of Timepoints per Trip')
            visualizations["Num_Timepoints_per_Trip_Distribution_Histogram"] = fig2.to_json()
        
        # Calculate dwell time
        arrival_dt = pd.to_datetime(df['ArrivalTime'].astype(str), format='%H:%M:%S', errors='coerce')
        departure_dt = pd.to_datetime(df['DepartureTime'].astype(str), format='%H:%M:%S', errors='coerce')
        df['DwellTimeSeconds'] = (departure_dt - arrival_dt).dt.total_seconds()
        df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan)
        
        df_dwell = df[(df['Timepoint'] == 'Yes') & (df['DwellTimeSeconds'].notna())]
        
        if not df_dwell.empty:
            fig3 = px.box(df_dwell, y='DwellTimeSeconds', title='Dwell Time Distribution at Timepoints (Seconds)')
            visualizations["Dwell_Time_at_Timepoints_Box"] = fig3.to_json()
        else:
            insights.append("Note: No valid dwell times for timepoints found.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_trip_details_and_accessibility_features_analysis(df):
    analysis_name = "Transit Trip Details and Accessibility Features Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'RouteID', 'ServiceID', 'TripHeadsign', 'TripShortName', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['DirectionID'] = pd.to_numeric(df['DirectionID'], errors='coerce')
        df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_trips = len(df)
        accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
        bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
        
        metrics = {
            "Total Trip Records": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        }
        
        insights.append(f"Total Trip Records: {total_trips}")
        insights.append(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
        insights.append(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
        
        fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
        visualizations["Wheelchair_Accessibility_Pie"] = fig1.to_json()
        
        fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
        visualizations["Bike_Accessibility_Pie"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
        visualizations["Trip_Directions_Distribution_Histogram"] = fig3.to_json()

        if 'TripHeadsign' in df.columns:
            headsign_counts = df['TripHeadsign'].value_counts().reset_index()
            headsign_counts.columns = ['TripHeadsign', 'Count']
            fig4 = px.bar(headsign_counts.head(20), x='TripHeadsign', y='Count', title='Top 20 Most Common Trip Headsigns')
            visualizations["Top_20_Most_Common_Trip_Headsigns_Bar"] = fig4.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_stop_and_station_location_analysis(df):
    analysis_name = "Transit Stop and Station Location Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['StopID', 'StopName', 'StopLat', 'StopLon', 'LocationType', 'ParentStationID']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['StopID', 'StopName', 'StopLat', 'StopLon', 'LocationType'] if matched[col] is None] # ParentStationID is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['StopLat'] = pd.to_numeric(df['StopLat'], errors='coerce')
        df['StopLon'] = pd.to_numeric(df['StopLon'], errors='coerce')
        df = df.dropna(subset=['StopLat', 'StopLon', 'StopName', 'LocationType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_locations = len(df)
        # LocationType might be numeric (e.g., 0 for stop, 1 for station) or string
        df['LocationType_str'] = df['LocationType'].astype(str).str.lower()
        total_stations = df[df['LocationType_str'].isin(['1', 'station'])].shape[0]
        total_stops = df[df['LocationType_str'].isin(['0', 'stop'])].shape[0]
        
        metrics = {
            "Total Locations": total_locations,
            "Total Stations": total_stations,
            "Total Stops": total_stops
        }
        
        insights.append(f"Total Locations (Stops/Stations): {total_locations}")
        insights.append(f"Total Stations: {total_stations}")
        insights.append(f"Total Stops: {total_stops}")
        
        fig1 = px.scatter_mapbox(df, lat='StopLat', lon='StopLon', hover_name='StopName', color='LocationType',
                                 zoom=9, title='Transit Stop and Station Locations',
                                 mapbox_style="carto-positron")
        visualizations["Transit_Stop_and_Station_Locations_Map"] = fig1.to_json()
        
        fig2 = px.histogram(df, x='LocationType', title='Distribution of Location Types (Stop/Station)')
        visualizations["Location_Types_Distribution_Histogram"] = fig2.to_json()
        
        if 'ParentStationID' in df.columns and not df['ParentStationID'].dropna().empty:
            stops_per_station = df.groupby('ParentStationID').size().reset_index(name='NumStops')
            fig3 = px.histogram(stops_per_station, x='NumStops', title='Distribution of Number of Stops per Station')
            visualizations["Num_Stops_per_Station_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_transportation_route_details_analysis(df):
    analysis_name = "Public Transportation Route Details Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['RouteID', 'RouteShortName', 'RouteLongName', 'RouteDesc', 'RouteType', 'AgencyID']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['RouteID', 'RouteShortName', 'RouteType', 'AgencyID'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_routes = len(df)
        unique_route_types = df['RouteType'].nunique()
        
        metrics = {
            "Total Public Transportation Routes": total_routes,
            "Unique Route Types": unique_route_types
        }
        
        insights.append(f"Total Public Transportation Routes: {total_routes}")
        insights.append(f"Unique Route Types: {unique_route_types}")
        
        fig1 = px.histogram(df, x='RouteType', title='Distribution of Public Transportation Route Types')
        visualizations["Public_Transportation_Route_Types_Distribution_Histogram"] = fig1.to_json()
        
        if 'AgencyID' in df.columns:
            routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
            fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                          title='Number of Routes per Agency (Top 20)')
            visualizations["Routes_per_Agency_Bar"] = fig2.to_json()

        if 'RouteLongName' in df.columns and 'RouteShortName' in df.columns:
            insights.append("\nSample Route Definitions:")
            for index, row in df.head(10).iterrows():
                insights.append(f"- {row['RouteShortName']}: {row['RouteLongName']} (Type: {row['RouteType']})")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transportation_agency_contact_and_timezone_analysis(df):
    analysis_name = "Transportation Agency Contact and Timezone Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['AgencyID', 'AgencyName', 'AgencyURL', 'AgencyTimezone', 'AgencyLang', 'AgencyPhone']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['AgencyID', 'AgencyName', 'AgencyTimezone'] if matched[col] is None] # Others are optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['AgencyID', 'AgencyName', 'AgencyTimezone'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_agencies = len(df)
        unique_timezones = df['AgencyTimezone'].nunique()
        most_common_timezone = df['AgencyTimezone'].mode()[0]
        
        metrics = {
            "Total Transportation Agencies": total_agencies,
            "Unique Timezones": unique_timezones,
            "Most Common Timezone": most_common_timezone
        }
        
        insights.append(f"Total Transportation Agencies: {total_agencies}")
        insights.append(f"Unique Timezones: {unique_timezones}")
        insights.append(f"Most Common Timezone: {most_common_timezone}")
        
        fig1 = px.histogram(df, x='AgencyTimezone', title='Distribution of Agency Timezones')
        visualizations["Agency_Timezones_Distribution_Histogram"] = fig1.to_json()
        
        if 'AgencyLang' in df.columns:
            fig2 = px.pie(df, names='AgencyLang', title='Distribution of Agency Languages')
            visualizations["Agency_Languages_Distribution_Pie"] = fig2.to_json()

        if 'AgencyPhone' in df.columns:
            agencies_with_phone = df['AgencyPhone'].notna().sum()
            insights.append(f"Agencies with Phone Numbers: {agencies_with_phone} ({agencies_with_phone / total_agencies * 100:.2f}%)")
            metrics["agencies_with_phone_count"] = agencies_with_phone
            metrics["agencies_with_phone_percent"] = (agencies_with_phone / total_agencies * 100)

        insights.append("\nSample Agency Contact Info:")
        if 'AgencyName' in df.columns and 'AgencyURL' in df.columns and 'AgencyPhone' in df.columns:
            for index, row in df.head(5).iterrows():
                insights.append(f"- {row['AgencyName']} | URL: {row['AgencyURL']} | Phone: {row['AgencyPhone']} | Timezone: {row['AgencyTimezone']}")
        else:
            insights.append("Not enough columns to display detailed contact info for sample agencies.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_trip_planning_and_route_shape_analysis(df):
    analysis_name = "Transit Trip Planning and Route Shape Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'RouteID', 'ShapeID', 'DirectionID', 'BlockID', 'TripHeadsign']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'RouteID', 'ShapeID', 'DirectionID'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['TripID', 'RouteID', 'ShapeID', 'DirectionID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_trips = len(df)
        unique_route_shapes_used = df['ShapeID'].nunique()
        
        metrics = {
            "Total Trips with Planning Data": total_trips,
            "Unique Route Shapes Used": unique_route_shapes_used
        }
        
        insights.append(f"Total Trips with Planning Data: {total_trips}")
        insights.append(f"Unique Route Shapes Used: {unique_route_shapes_used}")
        
        fig1 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
        visualizations["Trip_Directions_Distribution_Histogram"] = fig1.to_json()
        
        trips_per_shape = df.groupby('ShapeID').size().reset_index(name='TripCount')
        fig2 = px.bar(trips_per_shape.sort_values('TripCount', ascending=False).head(20),
                      x='ShapeID', y='TripCount', title='Top 20 Most Used Route Shapes by Trip Count')
        visualizations["Top_20_Most_Used_Route_Shapes_Bar"] = fig2.to_json()
        
        if 'TripHeadsign' in df.columns:
            headsign_counts = df['TripHeadsign'].value_counts().reset_index()
            headsign_counts.columns = ['TripHeadsign', 'Count']
            fig3 = px.bar(headsign_counts.head(20), x='TripHeadsign', y='Count', title='Top 20 Most Common Trip Headsigns')
            visualizations["Top_20_Most_Common_Trip_Headsigns_Bar"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_service_schedule_definition(df):
    analysis_name = "Transit Service Schedule Definition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        expected_cols = ['ServiceId', 'RouteId', 'TripId', 'StartDate', 'EndDate'] + days_of_week
        matched = fuzzy_match_column(df, expected_cols)
        # 'RouteId', 'TripId' are optional for basic calendar analysis
        missing = [col for col in ['ServiceId', 'StartDate', 'EndDate'] + days_of_week if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
        df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
        
        for day in days_of_week:
            df[day] = pd.to_numeric(df[day], errors='coerce').fillna(0).astype(bool)
        
        df = df.dropna(subset=['ServiceId', 'StartDate', 'EndDate'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_service_ids = len(df)
        avg_duration_days = (df['EndDate'] - df['StartDate']).dt.days.mean()
        
        metrics = {
            "Total Service Schedules Defined": total_service_ids,
            "Average Schedule Duration (days)": avg_duration_days
        }
        
        insights.append(f"Total Service Schedules Defined: {total_service_ids}")
        insights.append(f"Average Schedule Duration: {avg_duration_days:.1f} days")
        
        service_day_counts = {day: df[day].sum() for day in days_of_week}
        service_day_df = pd.DataFrame(service_day_counts.items(), columns=['Day', 'ServiceCount'])
        fig1 = px.bar(service_day_df, x='Day', y='ServiceCount', title='Number of Service Days per Weekday',
                      category_orders={"Day": days_of_week})
        visualizations["Service_Days_per_Weekday_Bar"] = fig1.to_json()
        
        if 'RouteId' in df.columns:
            routes_per_service = df.groupby('ServiceId')['RouteId'].nunique().reset_index(name='NumRoutes')
            fig2 = px.histogram(routes_per_service, x='NumRoutes', title='Distribution of Number of Routes per Service ID')
            visualizations["Num_Routes_per_Service_ID_Histogram"] = fig2.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def stop_by_stop_transit_schedule_analysis(df):
    analysis_name = "Stop-by-Stop Transit Schedule Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence', 'TravelTimeSeconds']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'] if matched[col] is None] # TravelTime is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
        df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
        
        df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_stop_times = len(df)
        avg_stops_per_trip = df.groupby('TripID').size().mean()
        
        metrics = {
            "Total Stop-by-Stop Schedule Records": total_stop_times,
            "Average Number of Stops per Trip": avg_stops_per_trip
        }
        
        insights.append(f"Total Stop-by-Stop Schedule Records: {total_stop_times}")
        insights.append(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
        
        fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences')
        visualizations["Stop_Sequences_Distribution_Histogram"] = fig1.to_json()
        
        avg_travel_time_between_stops = np.nan
        if 'TravelTimeSeconds' in df.columns:
            df['TravelTimeSeconds'] = pd.to_numeric(df['TravelTimeSeconds'], errors='coerce')
            df_travel = df.dropna(subset=['TravelTimeSeconds'])
            if not df_travel.empty:
                avg_travel_time_between_stops = df_travel['TravelTimeSeconds'].mean()
                insights.append(f"Average Travel Time Between Stops: {avg_travel_time_between_stops:.1f} seconds")
                fig2 = px.histogram(df_travel, x='TravelTimeSeconds', nbins=30, title='Distribution of Travel Times Between Stops (Seconds)')
                visualizations["Travel_Times_Between_Stops_Histogram"] = fig2.to_json()
            else:
                insights.append("Note: 'TravelTimeSeconds' column has no valid data.")
        else:
            insights.append("Note: 'TravelTimeSeconds' column not available for analysis.")
            
        metrics["Average Travel Time Between Stops (seconds)"] = avg_travel_time_between_stops

        # Calculate dwell time
        arrival_dt = pd.to_datetime(df['ArrivalTime'].astype(str), format='%H:%M:%S', errors='coerce')
        departure_dt = pd.to_datetime(df['DepartureTime'].astype(str), format='%H:%M:%S', errors='coerce')
        df['DwellTimeSeconds'] = (departure_dt - arrival_dt).dt.total_seconds()
        df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan)
        df_dwell = df.dropna(subset=['DwellTimeSeconds'])
        
        if not df_dwell.empty:
            fig3 = px.box(df_dwell, y='DwellTimeSeconds', title='Dwell Time Distribution at Stops (Seconds)')
            visualizations["Dwell_Time_at_Stops_Box"] = fig3.to_json()
        else:
            insights.append("Note: Dwell time calculation not possible or no valid dwell times found.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def public_transport_agency_directory_analysis(df):
    analysis_name = "Public Transport Agency Directory Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['AgencyID', 'AgencyName', 'AgencyURL', 'AgencyTimezone', 'AgencyLang', 'AgencyPhone']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['AgencyID', 'AgencyName', 'AgencyTimezone'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df = df.dropna(subset=['AgencyID', 'AgencyName', 'AgencyTimezone'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_agencies = len(df)
        unique_timezones = df['AgencyTimezone'].nunique()
        
        metrics = {
            "Total Public Transport Agencies": total_agencies,
            "Unique Timezones": unique_timezones
        }
        
        insights.append(f"Total Public Transport Agencies: {total_agencies}")
        insights.append(f"Unique Timezones: {unique_timezones}")
        
        fig1 = px.histogram(df, x='AgencyTimezone', title='Distribution of Agency Timezones')
        visualizations["Agency_Timezones_Distribution_Histogram"] = fig1.to_json()
        
        if 'AgencyLang' in df.columns:
            fig2 = px.pie(df, names='AgencyLang', title='Distribution of Agency Languages')
            visualizations["Agency_Languages_Distribution_Pie"] = fig2.to_json()

        insights.append("\nSample Agency Details:")
        if 'AgencyURL' in df.columns and 'AgencyPhone' in df.columns:
            for index, row in df.head(5).iterrows():
                insights.append(f"- {row['AgencyName']} (ID: {row['AgencyID']}) | URL: {row['AgencyURL']} | Phone: {row['AgencyPhone']}")
        else:
            insights.append("Not enough columns to display detailed agency info.")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_fare_attribute_analysis(df):
    analysis_name = "Transit Fare Attribute Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FareID', 'Price', 'Currency', 'PaymentMethod', 'Transfers', 'TransferDuration']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['FareID', 'Price', 'Currency', 'PaymentMethod', 'Transfers'] if matched[col] is None] # TransferDuration is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['Transfers'] = pd.to_numeric(df['Transfers'], errors='coerce')
        
        df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod', 'Transfers'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_price = df['Price'].mean()
        most_common_payment_method = df['PaymentMethod'].mode()[0]
        avg_transfer_duration_minutes = np.nan
        
        metrics = {
            "Average Fare Price": avg_price,
            "Most Common Payment Method": most_common_payment_method
        }
        
        insights.append(f"Average Fare Price: ${avg_price:.2f}")
        insights.append(f"Most Common Payment Method: {most_common_payment_method}")
        
        if 'TransferDuration' in df.columns:
            df['TransferDuration'] = pd.to_numeric(df['TransferDuration'], errors='coerce')
            df_duration = df.dropna(subset=['TransferDuration'])
            if not df_duration.empty:
                avg_transfer_duration_minutes = (df_duration['TransferDuration'].mean() / 60)
                metrics["Average Transfer Duration (minutes)"] = avg_transfer_duration_minutes
                insights.append(f"Average Transfer Duration: {avg_transfer_duration_minutes:.1f} minutes")
                
                df_duration_plot = df_duration.copy()
                df_duration_plot['TransferDurationMinutes'] = df_duration_plot['TransferDuration'] / 60
                fig4 = px.histogram(df_duration_plot, x='TransferDurationMinutes', title='Distribution of Transfer Durations (Minutes)')
                visualizations["Transfer_Durations_Distribution_Histogram"] = fig4.to_json()
            else:
                 insights.append("Note: 'TransferDuration' column has no valid data.")
        else:
            insights.append("Note: 'TransferDuration' column not found.")
            metrics["Average Transfer Duration (minutes)"] = avg_transfer_duration_minutes # Add nan to metrics
            
        fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
        visualizations["Fare_Prices_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price by Payment Method')
        visualizations["Fare_Price_by_Payment_Method_Box"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='Transfers', title='Distribution of Allowed Transfers')
        visualizations["Allowed_Transfers_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def inter_stop_transfer_path_analysis(df):
    analysis_name = "Inter-Stop Transfer Path Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['FromStopID', 'ToStopID', 'PathwayType', 'TransferTimeSeconds', 'MinTransferTime', 'IsAccessible']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['FromStopID', 'ToStopID', 'PathwayType', 'TransferTimeSeconds'] if matched[col] is None] # Others optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['TransferTimeSeconds'] = pd.to_numeric(df['TransferTimeSeconds'], errors='coerce')
        if 'MinTransferTime' in df.columns:
            df['MinTransferTime'] = pd.to_numeric(df['MinTransferTime'], errors='coerce')
        if 'IsAccessible' in df.columns:
            df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        
        df = df.dropna(subset=['FromStopID', 'ToStopID', 'PathwayType', 'TransferTimeSeconds'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_paths = len(df)
        avg_transfer_time_seconds = df['TransferTimeSeconds'].mean()
        avg_transfer_time_minutes = avg_transfer_time_seconds / 60
        accessible_paths_pct = (df['IsAccessible'] == 'Yes').mean() * 100 if 'IsAccessible' in df.columns and not df['IsAccessible'].isnull().all() else np.nan
        
        metrics = {
            "Total Inter-Stop Transfer Paths": total_paths,
            "Average Transfer Time (minutes)": avg_transfer_time_minutes,
            "Accessible Transfer Paths (%)": accessible_paths_pct
        }
        
        insights.append(f"Total Inter-Stop Transfer Paths: {total_paths}")
        insights.append(f"Average Transfer Time: {avg_transfer_time_minutes:.1f} minutes")
        if not np.isnan(accessible_paths_pct):
            insights.append(f"Percentage of Accessible Transfer Paths: {accessible_paths_pct:.2f}%")
        
        df_plot = df.copy()
        df_plot['TransferTimeMinutes'] = df_plot['TransferTimeSeconds'] / 60
        
        fig1 = px.histogram(df_plot, x='TransferTimeMinutes', nbins=30, title='Distribution of Transfer Times (Minutes)')
        visualizations["Transfer_Times_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df_plot, x='PathwayType', y='TransferTimeMinutes', title='Transfer Time by Pathway Type (Minutes)')
        visualizations["Transfer_Time_by_Pathway_Type_Box"] = fig2.to_json()
        
        if 'IsAccessible' in df.columns and not df['IsAccessible'].isnull().all():
            fig3 = px.pie(df, names='IsAccessible', title='Accessibility of Transfer Paths')
            visualizations["Accessibility_of_Transfer_Paths_Pie"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def geospatial_route_path_analysis(df):
    analysis_name = "Geospatial Route Path Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['ShapeID', 'Latitude', 'Longitude', 'Sequence', 'RouteID']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['ShapeID', 'Latitude', 'Longitude', 'Sequence'] if matched[col] is None] # RouteID is optional

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
        df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
        df['Sequence'] = pd.to_numeric(df['Sequence'], errors='coerce')
        df = df.dropna(subset=['Latitude', 'Longitude', 'Sequence', 'ShapeID'])
        df = df.sort_values(by=['ShapeID', 'Sequence'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        total_path_points = len(df)
        unique_route_shapes = df['ShapeID'].nunique()
        
        metrics = {
            "Total Geospatial Path Points": total_path_points,
            "Unique Route Shapes": unique_route_shapes
        }
        
        insights.append(f"Total Geospatial Path Points: {total_path_points}")
        insights.append(f"Unique Route Shapes: {unique_route_shapes}")
        
        # Sample for map performance
        plot_df = df
        if unique_route_shapes > 50:
            top_shapes = df['ShapeID'].value_counts().head(50).index
            plot_df = df[df['ShapeID'].isin(top_shapes)]
            insights.append("Note: Displaying a sample of 50 routes on the map due to high volume.")
            
        fig1 = px.line_mapbox(plot_df, lat='Latitude', lon='Longitude', color='ShapeID', line_group='ShapeID',
                              zoom=9, title='Geospatial Paths of Transit Routes (Sampled)',
                              mapbox_style="carto-positron")
        visualizations["Geospatial_Paths_of_Transit_Routes_Map"] = fig1.to_json()
        
        if unique_route_shapes > 0:
            sample_shape_id = df['ShapeID'].iloc[0]
            sample_shape_df = df[df['ShapeID'] == sample_shape_id]
            fig2 = px.line_mapbox(sample_shape_df, lat='Latitude', lon='Longitude',
                                  zoom=12, title=f'Geospatial Path of Sample Route Shape: {sample_shape_id}',
                                  mapbox_style="carto-positron")
            visualizations["Sample_Route_Shape_Geospatial_Path_Map"] = fig2.to_json()
        
        if 'RouteID' in df.columns:
            shapes_per_route = df.groupby('RouteID')['ShapeID'].nunique().reset_index(name='NumShapes')
            fig3 = px.histogram(shapes_per_route, x='NumShapes', title='Distribution of Number of Shapes per Route')
            visualizations["Num_Shapes_per_Route_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def trip_frequency_and_service_interval_analysis(df):
    analysis_name = "Trip Frequency and Service Interval Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected_cols = ['TripID', 'RouteID', 'ServiceID', 'StartTime', 'EndTime', 'FrequencySeconds']
        matched = fuzzy_match_column(df, expected_cols)
        missing = [col for col in ['TripID', 'RouteID', 'ServiceID', 'FrequencySeconds'] if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['FrequencySeconds'] = pd.to_numeric(df['FrequencySeconds'], errors='coerce')
        if 'StartTime' in df.columns:
            df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S', errors='coerce').dt.time
        if 'EndTime' in df.columns:
            df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S', errors='coerce').dt.time
        
        df = df.dropna(subset=['FrequencySeconds', 'RouteID', 'ServiceID'])

        if df.empty:
            insights.append("No sufficient data after cleaning for this analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "partial_success",
                "message": "No data available after cleaning.",
                "matched_columns": matched,
                "visualizations": {},
                "metrics": {},
                "insights": insights
            }

        avg_frequency_seconds = df['FrequencySeconds'].mean()
        avg_frequency_minutes = avg_frequency_seconds / 60
        
        metrics = {
            "Average Service Interval (minutes)": avg_frequency_minutes
        }
        
        insights.append(f"Average Service Interval (Frequency): {avg_frequency_minutes:.1f} minutes")
        
        df_plot = df.copy()
        df_plot['FrequencyMinutes'] = df_plot['FrequencySeconds'] / 60
        
        fig1 = px.histogram(df_plot, x='FrequencyMinutes', nbins=30, title='Distribution of Service Intervals (Minutes)')
        visualizations["Service_Intervals_Distribution_Histogram"] = fig1.to_json()
        
        avg_frequency_by_route = df_plot.groupby('RouteID')['FrequencyMinutes'].mean().reset_index()
        fig2 = px.bar(avg_frequency_by_route.sort_values('FrequencyMinutes', ascending=True).head(20),
                      x='RouteID', y='FrequencyMinutes', title='Top 20 Routes by Lowest Average Service Interval')
        visualizations["Top_20_Routes_by_Lowest_Avg_Service_Interval_Bar"] = fig2.to_json()
        
        if 'ServiceID' in df.columns:
            fig3 = px.box(df_plot, x='ServiceID', y='FrequencyMinutes', title='Service Interval by Service ID (Minutes)')
            visualizations["Service_Interval_by_Service_ID_Box"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def fare_cost_and_transfer_policy_analysis(df):
    analysis_name = "Fare Cost and Transfer Policy Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    fig4_json = None # Placeholder for optional figure

    try:
        expected = {
            'FareID': ['FareID', 'ID'],
            'Price': ['Price', 'FareAmount', 'Cost'],
            'Currency': ['Currency', 'CurrencyType'],
            'PaymentMethod': ['PaymentMethod', 'Method', 'PaymentOption'],
            'TransfersAllowed': ['TransfersAllowed', 'Transfers', 'NumTransfers'],
            'TransferDurationSeconds': ['TransferDurationSeconds', 'TransferTimeSeconds', 'DurationSeconds']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['Price', 'Currency', 'PaymentMethod', 'TransfersAllowed'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = safe_rename(df, matched)
        
        df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
        df['TransfersAllowed'] = pd.to_numeric(df['TransfersAllowed'], errors='coerce')
        if 'TransferDurationSeconds' in df.columns:
             df['TransferDurationSeconds'] = pd.to_numeric(df['TransferDurationSeconds'], errors='coerce')
             
        df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod', 'TransfersAllowed'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for fare and transfer analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        avg_fare_price = df['Price'].mean()
        most_common_payment_method = df['PaymentMethod'].mode()[0]
        
        metrics = {
            "Average Fare Price": avg_fare_price,
            "Most Common Payment Method": most_common_payment_method
        }
        
        insights.append(f"Average Fare Price: ${avg_fare_price:.2f}")
        insights.append(f"Most Common Payment Method: {most_common_payment_method}")
        
        fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
        visualizations["Fare_Prices_Distribution_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price by Payment Method')
        visualizations["Fare_Price_by_Payment_Method_Box"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='TransfersAllowed', title='Distribution of Allowed Transfers')
        visualizations["Allowed_Transfers_Distribution_Histogram"] = fig3.to_json()

        if 'TransferDurationSeconds' in df.columns and not df['TransferDurationSeconds'].dropna().empty:
            df['TransferDurationMinutes'] = df['TransferDurationSeconds'] / 60
            fig4 = px.histogram(df, x='TransferDurationMinutes', title='Distribution of Transfer Durations (Minutes)')
            fig4_json = fig4.to_json()
            insights.append("Generated histogram for transfer durations.")
        
        visualizations["Transfer_Durations_Distribution_Histogram"] = fig4_json

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def trip_service_detail_and_distance_analysis(df):
    analysis_name = "Trip Service Detail and Distance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    fig3_json = None # Placeholder for optional figure

    try:
        expected = {
            'TripID': ['TripID', 'ID'],
            'RouteID': ['RouteID', 'Route_ID'],
            'ServiceID': ['ServiceID', 'Service_ID'],
            'TripDistanceMiles': ['TripDistanceMiles', 'DistanceMiles', 'Distance'],
            'TripDurationMinutes': ['TripDurationMinutes', 'DurationMinutes', 'TravelTime'],
            'WheelchairAccessible': ['WheelchairAccessible', 'Accessible'],
            'BikesAllowed': ['BikesAllowed', 'BikeAccess']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['TripID', 'RouteID', 'TripDistanceMiles', 'TripDurationMinutes'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['TripDistanceMiles'] = pd.to_numeric(df['TripDistanceMiles'], errors='coerce')
        df['TripDurationMinutes'] = pd.to_numeric(df['TripDurationMinutes'], errors='coerce')
        
        if 'WheelchairAccessible' in df.columns:
            df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        if 'BikesAllowed' in df.columns:
            df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
            
        df = df.dropna(subset=['TripID', 'RouteID', 'TripDistanceMiles', 'TripDurationMinutes'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for trip detail analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        avg_trip_distance = df['TripDistanceMiles'].mean()
        avg_trip_duration = df['TripDurationMinutes'].mean()
        accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100 if 'WheelchairAccessible' in df.columns else np.nan
        
        metrics = {
            "Average Trip Distance": avg_trip_distance,
            "Average Trip Duration": avg_trip_duration,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct
        }
        
        insights.append(f"Average Trip Distance: {avg_trip_distance:.2f} miles")
        insights.append(f"Average Trip Duration: {avg_trip_duration:.1f} minutes")
        if not np.isnan(accessible_trips_pct):
            insights.append(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
        
        fig1 = px.scatter(df, x='TripDistanceMiles', y='TripDurationMinutes', hover_name='TripID',
                          title='Trip Duration vs. Distance')
        visualizations["Trip_Duration_vs_Distance_Scatter"] = fig1.to_json()
        
        fig2 = px.histogram(df, x='TripDistanceMiles', nbins=20, title='Distribution of Trip Distances')
        visualizations["Trip_Distances_Distribution_Histogram"] = fig2.to_json()
        
        if 'WheelchairAccessible' in df.columns:
            fig3 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
            fig3_json = fig3.to_json()
            insights.append("Generated pie chart for wheelchair accessibility.")

        visualizations["Wheelchair_Accessibility_of_Trips_Pie"] = fig3_json

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_data_feed_publisher_information_analysis(df):
    analysis_name = "Transit Data Feed Publisher Information Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    fig2_json = None # Placeholder for optional figure

    try:
        expected = {
            'PublisherName': ['PublisherName', 'Name', 'FeedPublisher'],
            'PublisherURL': ['PublisherURL', 'URL', 'FeedURL'],
            'PublisherLang': ['PublisherLang', 'Language', 'Lang'],
            'PublisherTimezone': ['PublisherTimezone', 'Timezone', 'FeedTimezone']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['PublisherName', 'PublisherURL', 'PublisherLang'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df = safe_rename(df, matched)
        
        df = df.dropna(subset=['PublisherName', 'PublisherURL', 'PublisherLang'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for publisher analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_publishers = len(df)
        unique_languages = df['PublisherLang'].nunique()
        most_common_language = df['PublisherLang'].mode()[0]
        
        metrics = {
            "Total Feed Publishers": total_publishers,
            "Unique Languages Used by Publishers": unique_languages,
            "Most Common Publisher Language": most_common_language
        }
        
        insights.append(f"Total Feed Publishers: {total_publishers}")
        insights.append(f"Unique Languages Used by Publishers: {unique_languages}")
        insights.append(f"Most Common Publisher Language: {most_common_language}")
        
        fig1 = px.histogram(df, x='PublisherLang', title='Distribution of Publisher Languages')
        visualizations["Publisher_Languages_Distribution_Histogram"] = fig1.to_json()
        
        if 'PublisherTimezone' in df.columns:
            fig2 = px.histogram(df, x='PublisherTimezone', title='Distribution of Publisher Timezones')
            fig2_json = fig2.to_json()
            insights.append("Generated histogram for publisher timezones.")
        
        visualizations["Publisher_Timezones_Distribution_Histogram"] = fig2_json
        
        if 'PublisherURL' in df.columns and 'PublisherTimezone' in df.columns:
            insights.append("Sample Publisher Information:")
            for index, row in df.head(5).iterrows():
                insights.append(f"- {row['PublisherName']} | URL: {row['PublisherURL']} | Lang: {row['PublisherLang']} | Timezone: {row['PublisherTimezone']}")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def pedestrian_pathway_analysis_in_transit_stations(df):
    analysis_name = "Pedestrian Pathway Analysis in Transit Stations"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = {
            'PathwayID': ['PathwayID', 'ID'],
            'FromStopID': ['FromStopID', 'OriginStopID'],
            'ToStopID': ['ToStopID', 'DestinationStopID'],
            'PathwayMode': ['PathwayMode', 'Type', 'Mode'],
            'LengthMeters': ['LengthMeters', 'Length', 'Distance'],
            'IsAccessible': ['IsAccessible', 'Accessible', 'WheelchairAccessible']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['PathwayMode', 'LengthMeters', 'IsAccessible'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['LengthMeters'] = pd.to_numeric(df['LengthMeters'], errors='coerce')
        df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['PathwayMode', 'LengthMeters', 'IsAccessible'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for pathway analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_pathways = len(df)
        avg_pathway_length = df['LengthMeters'].mean()
        accessible_pathways_pct = (df['IsAccessible'] == 'Yes').mean() * 100
        
        metrics = {
            "Total Pedestrian Pathways": total_pathways,
            "Average Pathway Length (meters)": avg_pathway_length,
            "Accessible Pathways (%)": accessible_pathways_pct
        }
        
        insights.append(f"Total Pedestrian Pathways: {total_pathways}")
        insights.append(f"Average Pathway Length: {avg_pathway_length:.2f} meters")
        insights.append(f"Percentage of Accessible Pathways: {accessible_pathways_pct:.2f}%")
        
        fig1 = px.histogram(df, x='PathwayMode', color='IsAccessible', barmode='group',
                                  title='Pathway Mode Distribution by Accessibility')
        visualizations["Pathway_Mode_Distribution_by_Accessibility_Histogram"] = fig1.to_json()
        
        fig2 = px.box(df, x='PathwayMode', y='LengthMeters', title='Pathway Length Distribution by Mode')
        visualizations["Pathway_Length_Distribution_by_Mode_Box"] = fig2.to_json()
        
        fig3 = px.pie(df, names='IsAccessible', title='Accessibility of Pedestrian Pathways')
        visualizations["Accessibility_of_Pedestrian_Pathways_Pie"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def transit_route_information_analysis(df):
    analysis_name = "Transit Route Information Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = {
            'RouteID': ['RouteID', 'ID', 'Route_ID'],
            'AgencyID': ['AgencyID', 'Agency_ID'],
            'RouteShortName': ['RouteShortName', 'ShortName', 'RouteNum'],
            'RouteLongName': ['RouteLongName', 'LongName', 'RouteDescription'],
            'RouteType': ['RouteType', 'Type', 'TransitType'],
            'RouteURL': ['RouteURL', 'URL'],
            'RouteColor': ['RouteColor', 'Color']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['RouteID', 'RouteShortName', 'RouteType', 'AgencyID'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType', 'AgencyID'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for route information analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_routes = len(df)
        unique_route_types = df['RouteType'].nunique()
        most_common_route_type = df['RouteType'].mode()[0]
        
        metrics = {
            "Total Transit Routes": total_routes,
            "Unique Route Types": unique_route_types,
            "Most Common Route Type": most_common_route_type
        }
        
        insights.append(f"Total Transit Routes: {total_routes}")
        insights.append(f"Unique Route Types: {unique_route_types}")
        insights.append(f"Most Common Route Type: {most_common_route_type}")
        
        fig1 = px.histogram(df, x='RouteType', title='Distribution of Transit Route Types')
        visualizations["Transit_Route_Types_Distribution_Histogram"] = fig1.to_json()
        
        routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
        fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                      title='Number of Routes per Agency (Top 20)')
        visualizations["Routes_per_Agency_Bar"] = fig2.to_json()
        
        if 'RouteURL' in df.columns:
            insights.append("\nSample Route URLs:")
            for url in df['RouteURL'].head().tolist():
                insights.append(f"- {url}")

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def trip_accessibility_and_direction_analysis(df):
    analysis_name = "Trip Accessibility and Direction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = {
            'TripID': ['TripID', 'ID'],
            'RouteID': ['RouteID', 'Route_ID'],
            'DirectionID': ['DirectionID', 'Direction'],
            'WheelchairAccessible': ['WheelchairAccessible', 'Accessible', 'Wheelchair'],
            'BikesAllowed': ['BikesAllowed', 'BikeAccess', 'Bikes']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['DirectionID'] = pd.to_numeric(df['DirectionID'], errors='coerce')
        df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
        df = df.dropna(subset=['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for trip accessibility analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_trips = len(df)
        accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
        bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
        
        metrics = {
            "Total Trips Analyzed": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        }
        
        insights.append(f"Total Trips Analyzed: {total_trips}")
        insights.append(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
        insights.append(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
        
        fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
        visualizations["Wheelchair_Accessibility_Pie"] = fig1.to_json()
        
        fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
        visualizations["Bike_Accessibility_Pie"] = fig2.to_json()
        
        fig3 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
        visualizations["Trip_Directions_Distribution_Histogram"] = fig3.to_json()

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def scheduled_stop_times_analysis_for_trips(df):
    analysis_name = "Scheduled Stop Times Analysis for Trips"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    fig2_json = None
    fig3_json = None
    fig4_json = None

    try:
        expected = {
            'TripID': ['TripID', 'Trip_ID'],
            'StopID': ['StopID', 'Stop_ID'],
            'ArrivalTime': ['ArrivalTime', 'ScheduledArrival', 'Arrival'],
            'DepartureTime': ['DepartureTime', 'ScheduledDeparture', 'Departure'],
            'StopSequence': ['StopSequence', 'Sequence', 'Order'],
            'PickupType': ['PickupType', 'PickupRule'],
            'DropOffType': ['DropOffType', 'DropOffRule']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        # Handle H:M:S strings that might be > 23:59:59
        def clean_time(time_str):
            try:
                parts = str(time_str).split(':')
                if len(parts) == 3:
                    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
                    if h >= 24:
                        h = h % 24 # Wrap around for datetime conversion
                    return f"{h:02d}:{m:02d}:{s:02d}"
            except:
                pass
            return None

        df['ArrivalTime_Clean'] = df['ArrivalTime'].apply(clean_time)
        df['DepartureTime_Clean'] = df['DepartureTime'].apply(clean_time)
        
        df['ArrivalTimeDT'] = pd.to_datetime(df['ArrivalTime_Clean'], format='%H:%M:%S', errors='coerce')
        df['DepartureTimeDT'] = pd.to_datetime(df['DepartureTime_Clean'], format='%H:%M:%S', errors='coerce')

        df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
        df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTimeDT', 'DepartureTimeDT', 'StopSequence'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for stop times analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_scheduled_stops = len(df)
        avg_stops_per_trip = df.groupby('TripID').size().mean()
        
        metrics = {
            "Total Scheduled Stop Times Records": total_scheduled_stops,
            "Average Number of Stops per Trip": avg_stops_per_trip
        }
        
        insights.append(f"Total Scheduled Stop Times Records: {total_scheduled_stops}")
        insights.append(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
        
        fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences within Trips')
        visualizations["Stop_Sequences_Distribution_Histogram"] = fig1.to_json()
        
        if 'PickupType' in df.columns:
            fig2 = px.histogram(df, x='PickupType', title='Distribution of Pickup Types')
            fig2_json = fig2.to_json()
            insights.append("Generated histogram for pickup types.")
        
        if 'DropOffType' in df.columns:
            fig3 = px.histogram(df, x='DropOffType', title='Distribution of Drop-Off Types')
            fig3_json = fig3.to_json()
            insights.append("Generated histogram for drop-off types.")

        # Calculate dwell time
        df['DwellTimeSeconds'] = (df['DepartureTimeDT'] - df['ArrivalTimeDT']).dt.total_seconds()
        # Handle overnight wrap-around (e.g., arrival 23:59, departure 00:01)
        df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else x + 86400)
        # Filter out unreasonable dwell times (e.g., > 1 hour)
        df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if (x >= 0 and x < 3600) else np.nan)
        
        if not df['DwellTimeSeconds'].dropna().empty:
            fig4 = px.box(df, y='DwellTimeSeconds', title='Dwell Time Distribution at Stops (Seconds)')
            fig4_json = fig4.to_json()
            insights.append(f"Average dwell time at stops: {df['DwellTimeSeconds'].mean():.1f} seconds.")
            metrics["Average Dwell Time (Seconds)"] = df['DwellTimeSeconds'].mean()

        visualizations["Pickup_Types_Distribution_Histogram"] = fig2_json
        visualizations["Drop_Off_Types_Distribution_Histogram"] = fig3_json
        visualizations["Dwell_Time_at_Stops_Box"] = fig4_json

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

def special_service_dates_and_schedule_exception_analysis(df):
    analysis_name = "Special Service Dates and Schedule Exception Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    fig3_json = None

    try:
        expected = {
            'ServiceID': ['ServiceID', 'ID', 'Service_ID'],
            'Date': ['Date', 'ServiceDate', 'ExceptionDate'],
            'ExceptionType': ['ExceptionType', 'Type', 'AddedRemoved'], # 1 for added, 2 for removed
            'Description': ['Description', 'Note', 'Reason']
        }
        matched = fuzzy_match_column(df, expected.keys())
        critical_missing = [col for col in ['Date', 'ExceptionType'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = safe_rename(df, matched)
        
        # Date can be in YYYYMMDD format
        df['Date'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        df['ExceptionType'] = pd.to_numeric(df['ExceptionType'], errors='coerce')
        df = df.dropna(subset=['Date', 'ExceptionType'])

        if df.empty:
            insights.append("No sufficient data found after cleaning for schedule exception analysis.")
            return {
                "analysis_type": analysis_name,
                "status": "success",
                "matched_columns": matched,
                "visualizations": visualizations,
                "metrics": metrics,
                "insights": insights,
                "message": "No sufficient data after cleaning."
            }

        total_exceptions = len(df)
        added_services = (df['ExceptionType'] == 1).sum()
        removed_services = (df['ExceptionType'] == 2).sum()
        
        metrics = {
            "Total Special Service Dates/Exceptions": total_exceptions,
            "Added Services": added_services,
            "Removed Services": removed_services
        }
        
        insights.append(f"Total Special Service Dates/Exceptions: {total_exceptions}")
        insights.append(f"Number of Added Services (Type 1): {added_services}")
        insights.append(f"Number of Removed Services (Type 2): {removed_services}")
        
        # Map exception type for clarity in visualization
        df['ExceptionCategory'] = df['ExceptionType'].map({1: '1: Added Service', 2: '2: Removed Service'}).fillna('Other')
        fig1 = px.histogram(df, x='ExceptionCategory', title='Distribution of Exception Types')
        visualizations["Exception_Types_Distribution_Histogram"] = fig1.to_json()
        
        df['Month'] = df['Date'].dt.to_period('M').astype(str) # Convert Period to string for JSON
        exceptions_by_month = df.groupby('Month').size().reset_index(name='Count')
        
        fig2 = px.line(exceptions_by_month, x='Month', y='Count', title='Monthly Trend of Service Exceptions')
        visualizations["Monthly_Service_Exceptions_Trend_Line"] = fig2.to_json()

        if 'ServiceID' in df.columns:
            exceptions_per_service = df.groupby('ServiceID').size().reset_index(name='NumExceptions')
            fig3 = px.histogram(exceptions_per_service, x='NumExceptions', title='Distribution of Number of Exceptions per Service ID')
            fig3_json = fig3.to_json()
            insights.append("Generated histogram for exceptions per service ID.")
        
        visualizations["Num_Exceptions_per_Service_ID_Histogram"] = fig3_json

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
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

# ========== UPDATED MAIN FUNCTIONS ==========

def main_backend(file_path, encoding='utf-8', selected_analysis_name=None):
    """
    Main function to run transportation data analysis for a backend API.
    
    Parameters:
    - file_path: path to the data file (CSV or Excel)
    - encoding: file encoding (default: 'utf-8')
    - selected_analysis_name: The name of the specific analysis to run.
    
    Returns:
    - Dictionary with analysis results.
    """
    
    # Load data
    df = load_data(file_path, encoding)
    if df is None:
        return {
            "analysis_type": "Data Loading",
            "status": "error", 
            "error_message": "Failed to load data file. Check file path and encoding."
        }
    
    # Mapping of all analysis functions
    # Note: Functions not provided (e.g., fleet_analysis) will fail if called.
    # We only define the ones that exist in this script.
    analysis_functions = {
        "General Insights": show_general_insights,
        "fare_cost_and_transfer_policy_analysis": fare_cost_and_transfer_policy_analysis,
        "trip_service_detail_and_distance_analysis": trip_service_detail_and_distance_analysis,
        "transit_data_feed_publisher_information_analysis": transit_data_feed_publisher_information_analysis,
        "pedestrian_pathway_analysis_in_transit_stations": pedestrian_pathway_analysis_in_transit_stations,
        "transit_route_information_analysis": transit_route_information_analysis,
        "trip_accessibility_and_direction_analysis": trip_accessibility_and_direction_analysis,
        "scheduled_stop_times_analysis_for_trips": scheduled_stop_times_analysis_for_trips,
        "special_service_dates_and_schedule_exception_analysis": special_service_dates_and_schedule_exception_analysis,
        
        # --- Placeholder for functions not yet refactored ---
        # "fleet_analysis": fleet_analysis,
        # "route_analysis": route_analysis,
        # ... etc.
    }
    
    # Determine which analysis to run
    if selected_analysis_name in analysis_functions:
        # Call the selected analysis function
        result = analysis_functions[selected_analysis_name](df.copy())
    elif selected_analysis_name == "General Insights" or selected_analysis_name is None:
        # Default to general insights
        result = show_general_insights(df.copy())
    else:
        # Analysis exists in the list but function is not implemented
        return {
            "analysis_type": selected_analysis_name,
            "status": "error",
            "error_message": f"Analysis '{selected_analysis_name}' is not implemented or available.",
            "insights": [f"The requested analysis '{selected_analysis_name}' could not be found or is not yet implemented."]
        }
    
    return result

def main():
    """Main function for command-line usage"""
    print("🚛 Transportation Analytics Dashboard")

    # File path and encoding input
    file_path = input("Enter path to your transportation data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252), or press Enter for utf-8: ")
    if not encoding:
        encoding = 'utf-8'

    df = load_data(file_path, encoding)

    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    
    # Display available columns
    print(f"\n📋 YOUR DATASET COLUMNS ({len(df.columns)} total):")
    for i, col in enumerate(df.columns):
        print(f"  {i+1}. '{col}'")
    
    print(f"\n📊 DATASET SHAPE: {df.shape}")

    # Analysis selection
    # Using the global analysis_options list
    print("\nSelect Analysis to Perform:")
    for i, option in enumerate(analysis_options):
        print(f"{i}: {option}")
    
    choice = input(f"Enter the option number (0-{len(analysis_options)-1}): ")
    
    try:
        choice_num = int(choice)
        if 0 <= choice_num < len(analysis_options):
            selected = analysis_options[choice_num]
        else:
            print("Invalid number. Showing General Insights.")
            selected = "General Insights"
    except ValueError:
        print("Invalid input. Showing General Insights.")
        selected = "General Insights"

    print(f"\nRunning analysis: {selected}...")
    
    # Execute analysis based on selection using the main_backend function
    result = main_backend(file_path, encoding, selected_analysis_name=selected)

    # Display results
    if result:
        print("\n" + "="*60)
        print(f"📈 ANALYSIS RESULTS: {result.get('analysis_type', 'Unknown Analysis')}")
        print("="*60)
        
        # Status and basic info
        status = result.get('status', 'unknown')
        status_emoji = "✅" if status == "success" else "⚠️" if status == "fallback" else "❌"
        print(f"Status: {status_emoji} {status.upper()}")
        
        if status == "error":
            print(f"Error: {result.get('error_message', 'Unknown error')}")
            return
        elif status == "fallback":
            print(f"Message: {result.get('message', 'Falling back to general insights')}")

        # Matched columns
        matched_cols = result.get('matched_columns', {})
        if matched_cols:
            print(f"\n🔍 Matched Columns:")
            for target, actual in matched_cols.items():
                match_indicator = " ✅" if actual else " ❌"
                print(f"  - {target}: {actual if actual else 'Not found'}{match_indicator}")

        # Insights
        insights = result.get('insights', [])
        if insights:
            print(f"\n💡 Key Insights:")
            for insight in insights:
                print(f"  • {insight}")

        # Metrics
        metrics = result.get('metrics', {})
        if metrics:
            print(f"\n📊 Key Metrics:")
            def print_metrics(data, indent=0):
                for key, value in data.items():
                    if isinstance(value, dict):
                        print("  " * indent + f"  {key}:")
                        print_metrics(value, indent + 1)
                    else:
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if abs(value) >= 1000:
                                formatted_value = f"{value:,.0f}"
                            elif isinstance(value, float):
                                formatted_value = f"{value:.2f}"
                            else:
                                formatted_value = str(value)
                        else:
                            formatted_value = str(value)
                        print("  " * indent + f"  - {key}: {formatted_value}")
            
            print_metrics(metrics)

        # Visualizations info
        visualizations = result.get('visualizations', {})
        if visualizations:
            print(f"\n📈 Generated Visualizations (as JSON): {len(visualizations)}")
            for viz_name in visualizations.keys():
                if visualizations[viz_name] is not None:
                    print(f"  - {viz_name}")
                else:
                    print(f"  - {viz_name} (No data to display)")

        # Save results option
        save_option = input("\n💾 Would you like to save the full results to a JSON file? (y/n): ").lower()
        if save_option in ['y', 'yes']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_name_clean = result.get('analysis_type', 'analysis').replace(' ', '_').lower()
            filename = f"transport_analytics_{analysis_name_clean}_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(convert_to_native_types(result), f, indent=2)
                print(f"✅ Results saved to: {filename}")
            except Exception as e:
                print(f"❌ Error saving file: {e}")

        print(f"\n🎉 Analysis completed successfully!")

    else:
        print("❌ No results returned from analysis.")


if __name__ == "__main__":
    main()