import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import datetime
import warnings
import json

# Suppress warnings if desired
warnings.filterwarnings('ignore')

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "shipment_analysis",
    "inventory_analysis",
    "transportation_analysis",
    "warehouse_analysis",
    "supplier_analysis",
    "route_optimization_analysis",
    "demand_forecasting_analysis",
    "cost_analysis",
    "delivery_performance_analysis",
    "trip_route_and_schedule_performance_analysis",
    "shipping_carrier_and_cost_optimization_analysis",
    "shipment_dispatch_and_delivery_time_analysis",
    "logistics_carrier_rate_and_service_analysis",
    "warehouse_capacity_and_operational_analysis",
    "warehouse_stock_movement_and_inventory_analysis",
    "purchase_order_and_supplier_delivery_performance_analysis",
    "purchase_order_line_item_cost_analysis",
    "order_fulfillment_process_analysis",
    "inventory_quantity_on_hand_analysis",
    "shipment_on_time_delivery_performance_analysis",
    "late_delivery_and_order_value_correlation_analysis",
    "driver_trip_performance_and_fuel_efficiency_analysis",
    "barcode_scan_and_shipment_tracking_analysis",
    "logistics_route_optimization_analysis",
    "package_delivery_delay_analysis",
    "delivery_performance_and_delay_root_cause_analysis",
    "warehouse_inventory_reorder_level_analysis",
    "supplier_lead_time_and_reliability_analysis",
    "freight_haulage_and_truck_load_analysis",
    "inbound_logistics_and_vendor_quality_analysis",
    "vehicle_route_and_on_time_performance_analysis",
    "vehicle_fleet_maintenance_and_capacity_analysis",
    "sales_order_and_pricing_analysis",
    "shipment_tracking_and_status_update_analysis",
    "package_volumetric_weight_and_zone_analysis",
    "inter_warehouse_stock_transfer_analysis",
    "shipment_manifest_and_trip_planning_analysis",
    "last_mile_delivery_confirmation_analysis",
    "order_delivery_time_estimation_by_shipping_zone",
    "order_fulfillment_cycle_time_analysis",
    "truck_loading_efficiency_analysis"
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
    visualizations = {}
    metrics = {}
    insights = []
    
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
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
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            try:
                numeric_stats = df[numeric_cols].describe().to_dict()
            except Exception as e:
                insights.append(f"Could not generate numeric stats: {e}")

        # Categorical columns analysis
        categorical_stats = {}
        if categorical_cols:
            for col in categorical_cols[:5]:  # Limit to first 5 for brevity
                try:
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(5).to_dict()
                    categorical_stats[col] = {
                        "unique_count": int(unique_count),
                        "top_values": convert_to_native_types(top_values)
                    }
                except Exception as e:
                    insights.append(f"Could not analyze categorical col {col}: {e}")
        
        # --- Create visualizations ---
        
        # 1. Data types distribution
        try:
            dtype_counts = {
                'Numeric': len(numeric_cols),
                'Categorical/Bool': len(categorical_cols),
                'Datetime': len(datetime_cols),
                'Other': len(other_cols)
            }
            fig_dtypes = px.pie(
                values=list(dtype_counts.values()), 
                names=list(dtype_counts.keys()),
                title='Data Types Distribution'
            )
            visualizations["data_types_distribution"] = fig_dtypes.to_json()
        except Exception as e:
            insights.append(f"Could not create dtype pie chart: {e}")
        
        # 2. Missing values visualization
        try:
            if len(columns_with_missing) > 0:
                missing_df = pd.DataFrame({
                    'column': columns_with_missing.index,
                    'missing_count': columns_with_missing.values,
                    'missing_percentage': missing_percentage[columns_with_missing.index]
                }).sort_values('missing_percentage', ascending=False)
                
                fig_missing = px.bar(
                    missing_df.head(15), 
                    x='column', 
                    y='missing_percentage',
                    title='Top 15 Columns with Missing Values (%)'
                )
                visualizations["missing_values"] = fig_missing.to_json()
            else:
                insights.append("No missing values found in the dataset.")
        except Exception as e:
            insights.append(f"Could not create missing values chart: {e}")

        # 3. Numeric columns distributions (first 2)
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:2]):
                try:
                    fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception as e:
                    insights.append(f"Could not create histogram for {col}: {e}")
        
        # 4. Correlation heatmap
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     title="Numeric Feature Correlation Heatmap")
                visualizations["correlation_heatmap"] = fig_corr.to_json()
            except Exception as e:
                insights.append(f"Could not create correlation heatmap: {e}")

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
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "datetime_columns_count": len(datetime_cols),
                "other_columns_count": len(other_cols)
            },
            "data_quality": {
                "total_missing_values": int(missing_values.sum()),
                "columns_with_missing": len(columns_with_missing),
                "complete_columns": len(df.columns) - len(columns_with_missing)
            },
            "numeric_stats_sample": {k: v for k, v in numeric_stats.items() if k in numeric_cols[:2]}, # Sample stats
            "categorical_stats_sample": categorical_stats
        }
        
        # Generate insights - INCLUDING MISSING COLUMNS WARNINGS
        insights.insert(0, f"Dataset contains {total_rows:,} rows and {total_columns} columns.")
        insights.insert(1, f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns.")
        
        # Add missing columns warning if provided
        if missing_cols and len(missing_cols) > 0:
            insights.append("---")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND")
            insights.append(f"The following columns were needed for the '{analysis_name}' but weren't found:")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f"  - {col}{match_info}")
            insights.append("Showing General Analysis instead.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data).")
        
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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched_cols or {},
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred during general insights: {e}"],
            "missing_columns": missing_cols or []
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides general insights as a fallback.
    """
    general_insights_data = show_general_insights(
        df, 
        analysis_name, # Pass the original analysis name for the warning
        missing_cols=missing_cols,
        matched_cols=matched_cols
    )

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
            # Use a slightly lower threshold for logistics terms which can be very different
            match, score = process.extractOne(target, available)
            matched[target] = match if score >= 70 else None
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    return df.rename(columns={v: k for k, v in matched.items() if v is not None and v in df.columns})

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
            # If all fail
            return None
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        print(f"[Error] Data loading failed: {e}") # Keep one print for catastrophic failure
        return None

# ========== ANALYSIS FUNCTIONS (Refactored) ==========

def shipment_analysis(df):
    analysis_name = "Shipment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['shipment_id', 'origin', 'destination', 'shipment_date', 'delivery_date',
                    'weight', 'volume', 'cost', 'status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)

        # Convert dates
        date_cols = ['shipment_date', 'delivery_date']
        for col in date_cols:
            if col in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        # Drop rows where critical date conversions failed
        df_copy.dropna(subset=['shipment_date', 'delivery_date'], inplace=True)

        # Calculate transit time
        if 'shipment_date' in df_copy and 'delivery_date' in df_copy:
            df_copy['transit_time'] = (df_copy['delivery_date'] - df_copy['shipment_date']).dt.days
        
        # Metrics
        total_shipments = len(df_copy)
        avg_cost = df_copy['cost'].mean() if 'cost' in df_copy and not df_copy['cost'].isnull().all() else 0
        # Assuming 'Delivered' is the positive status
        on_time = (df_copy['status'].str.lower() == 'delivered').mean() * 100 if 'status' in df_copy else 0
        avg_transit = df_copy['transit_time'].mean() if 'transit_time' in df_copy else 0

        metrics = {
            "Total Shipments": total_shipments,
            "Avg Cost": avg_cost,
            "On-Time Delivery (Assumed 'Delivered' status)": on_time,
            "Avg Transit Time (Days)": avg_transit
        }
        
        insights.append(f"Analyzed {total_shipments} shipments.")
        insights.append(f"Average cost per shipment: ${avg_cost:,.2f}.")
        insights.append(f"Average transit time: {avg_transit:.1f} days.")

        # Visualizations
        if 'transit_time' in df_copy and not df_copy['transit_time'].isnull().all():
            fig1 = px.histogram(df_copy, x='transit_time',
                                title="Transit Time Distribution")
            visualizations['transit_time_distribution'] = fig1.to_json()
        
        if 'origin' in df_copy and 'destination' in df_copy:
            route_counts = df_copy.groupby(['origin', 'destination']).size().reset_index(name='count')
            fig2 = px.bar(route_counts.sort_values('count', ascending=False).head(10),
                          x='origin', y='count', color='destination',
                          title="Top 10 Shipping Routes")
            visualizations['top_shipping_routes'] = fig2.to_json()

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

def inventory_analysis(df):
    analysis_name = "Inventory Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['product_id', 'product_name', 'category', 'current_stock',
                    'min_stock', 'max_stock', 'warehouse']
        matched = fuzzy_match_column(df, expected)
        # Critical for this analysis
        critical_missing = [col for col in ['current_stock', 'min_stock', 'max_stock'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        # Ensure numeric columns are numeric
        for col in ['current_stock', 'min_stock', 'max_stock']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        # Drop rows if critical stock levels are NaN
        df_copy.dropna(subset=['current_stock', 'min_stock', 'max_stock'], inplace=True) 

        # Calculate stock status
        df_copy['stock_status'] = np.where(
            df_copy['current_stock'] < df_copy['min_stock'], 'Low Stock',
            np.where(df_copy['current_stock'] > df_copy['max_stock'], 'Overstock', 'Normal')
        )
        
        # Metrics
        total_products = len(df_copy)
        low_stock_count = (df_copy['stock_status'] == 'Low Stock').sum()
        overstock_count = (df_copy['stock_status'] == 'Overstock').sum()
        low_stock_percent = (low_stock_count / total_products) * 100 if total_products > 0 else 0
        overstock_percent = (overstock_count / total_products) * 100 if total_products > 0 else 0

        metrics = {
            "Total Products": total_products,
            "Low Stock Items (Count)": low_stock_count,
            "Overstock Items (Count)": overstock_count,
            "Low Stock Items (%)": low_stock_percent,
            "Overstock Items (%)": overstock_percent
        }
        
        insights.append(f"Analyzed {total_products} product stock levels.")
        insights.append(f"{low_stock_count} items ({low_stock_percent:.1f}%) are below minimum stock.")
        insights.append(f"{overstock_count} items ({overstock_percent:.1f}%) are above maximum stock.")

        # Visualizations
        if 'stock_status' in df_copy and not df_copy['stock_status'].isnull().all():
            fig1 = px.pie(df_copy, names='stock_status',
                          title="Inventory Status Distribution")
            visualizations['inventory_status_distribution'] = fig1.to_json()
        
        if 'category' in df_copy.columns and 'current_stock' in df_copy and not df_copy['current_stock'].isnull().all():
            fig2 = px.box(df_copy, x='category', y='current_stock',
                          title="Stock Levels by Category")
            visualizations['stock_levels_by_category'] = fig2.to_json()

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

def transportation_analysis(df):
    analysis_name = "Transportation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['vehicle_id', 'vehicle_type', 'capacity', 'fuel_efficiency',
                    'maintenance_cost', 'status']
        matched = fuzzy_match_column(df, expected)
        # Check for at least some useful columns
        critical_missing = [col for col in ['vehicle_id', 'vehicle_type', 'fuel_efficiency'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['capacity', 'fuel_efficiency', 'maintenance_cost']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Metrics
        total_vehicles = len(df_copy)
        avg_fuel = df_copy['fuel_efficiency'].mean() if 'fuel_efficiency' in df_copy and not df_copy['fuel_efficiency'].isnull().all() else 0
        under_maintenance = 0
        if 'status' in df_copy.columns:
             under_maintenance = (df_copy['status'].str.lower() == 'maintenance').sum()

        metrics = {
            "Total Vehicles": total_vehicles,
            "Avg Fuel Efficiency (mpg)": avg_fuel,
            "Vehicles Under Maintenance": under_maintenance
        }
        
        insights.append(f"Analyzed {total_vehicles} vehicles.")
        insights.append(f"Average fuel efficiency: {avg_fuel:.1f} mpg.")
        insights.append(f"{under_maintenance} vehicles currently under maintenance.")

        # Visualizations
        if 'vehicle_type' in df_copy and not df_copy['vehicle_type'].isnull().all():
            fig1 = px.bar(df_copy['vehicle_type'].value_counts().reset_index(name='count'), x='vehicle_type', y='count',
                          title="Vehicle Type Distribution")
            visualizations['vehicle_type_distribution'] = fig1.to_json()
        
        if 'fuel_efficiency' in df_copy and 'maintenance_cost' in df_copy and \
           not df_copy['fuel_efficiency'].isnull().all() and not df_copy['maintenance_cost'].isnull().all():
            fig2 = px.scatter(df_copy, x='fuel_efficiency', y='maintenance_cost',
                              color='vehicle_type' if 'vehicle_type' in df_copy.columns else None,
                              title="Fuel Efficiency vs Maintenance Cost")
            visualizations['fuel_efficiency_vs_maintenance_cost'] = fig2.to_json()

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

def warehouse_analysis(df):
    analysis_name = "Warehouse Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['warehouse_id', 'location', 'capacity', 'current_utilization',
                    'operating_cost', 'employees']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['capacity', 'current_utilization'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['capacity', 'current_utilization', 'operating_cost', 'employees']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        df_copy.dropna(subset=['capacity', 'current_utilization'], inplace=True) # Crucial for utilization calculation

        # Calculate utilization percentage
        df_copy['utilization_pct'] = np.nan
        if (df_copy['capacity'] != 0).any():
            df_copy['utilization_pct'] = (df_copy['current_utilization'] / df_copy['capacity']) * 100
            df_copy.loc[df_copy['capacity'] == 0, 'utilization_pct'] = np.nan # Handle division by zero
        
        # Metrics
        total_warehouses = len(df_copy)
        avg_utilization = df_copy['utilization_pct'].mean() if not df_copy['utilization_pct'].isnull().all() else 0
        high_utilization_threshold = 85
        high_utilization_count = (df_copy['utilization_pct'] > high_utilization_threshold).sum() if not df_copy['utilization_pct'].isnull().all() else 0

        metrics = {
            "Total Warehouses": total_warehouses,
            "Avg Utilization (%)": avg_utilization,
            f"High Utilization (>{high_utilization_threshold}%) Count": high_utilization_count
        }
        
        insights.append(f"Analyzed {total_warehouses} warehouses.")
        insights.append(f"Average utilization: {avg_utilization:.1f}%.")
        insights.append(f"{high_utilization_count} warehouses are over {high_utilization_threshold}% utilization.")

        # Visualizations
        if 'location' in df_copy.columns and 'utilization_pct' in df_copy and not df_copy['utilization_pct'].isnull().all():
            fig1 = px.bar(df_copy, x='location', y='utilization_pct',
                          title="Warehouse Utilization by Location")
            visualizations['warehouse_utilization_by_location'] = fig1.to_json()
        
        if 'operating_cost' in df_copy and 'current_utilization' in df_copy and \
           not df_copy['operating_cost'].isnull().all() and not df_copy['current_utilization'].isnull().all():
            fig2 = px.scatter(df_copy, x='current_utilization', y='operating_cost',
                              title="Utilization vs Operating Cost")
            visualizations['utilization_vs_operating_cost'] = fig2.to_json()

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

def supplier_analysis(df):
    analysis_name = "Supplier Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['supplier_id', 'supplier_name', 'lead_time', 'defect_rate',
                    'unit_cost', 'delivery_reliability']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['supplier_name', 'lead_time', 'defect_rate', 'delivery_reliability'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['lead_time', 'defect_rate', 'unit_cost', 'delivery_reliability']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        # Metrics
        total_suppliers = len(df_copy)
        avg_lead_time = df_copy['lead_time'].mean() if 'lead_time' in df_copy and not df_copy['lead_time'].isnull().all() else 0
        avg_defect_rate = df_copy['defect_rate'].mean() * 100 if 'defect_rate' in df_copy and not df_copy['defect_rate'].isnull().all() else 0
        avg_reliability = df_copy['delivery_reliability'].mean() * 100 if 'delivery_reliability' in df_copy and not df_copy['delivery_reliability'].isnull().all() else 0


        metrics = {
            "Total Suppliers": total_suppliers,
            "Avg Lead Time (Days)": avg_lead_time,
            "Avg Defect Rate (%)": avg_defect_rate,
            "Avg Delivery Reliability (%)": avg_reliability
        }
        
        insights.append(f"Analyzed {total_suppliers} suppliers.")
        insights.append(f"Average lead time: {avg_lead_time:.1f} days.")
        insights.append(f"Average defect rate: {avg_defect_rate:.1f}%.")

        # Visualizations
        if 'supplier_name' in df_copy and 'delivery_reliability' in df_copy and not df_copy['delivery_reliability'].isnull().all():
            fig1 = px.bar(df_copy.sort_values('delivery_reliability', ascending=False).head(10),
                          x='supplier_name', y='delivery_reliability',
                          title="Top 10 Suppliers by Delivery Reliability")
            visualizations['top_suppliers_by_delivery_reliability'] = fig1.to_json()
        
        if 'lead_time' in df_copy and 'defect_rate' in df_copy and \
           not df_copy['lead_time'].isnull().all() and not df_copy['defect_rate'].isnull().all():
            fig2 = px.scatter(df_copy, x='lead_time', y='defect_rate',
                              hover_name='supplier_name' if 'supplier_name' in df_copy.columns else None,
                              title="Lead Time vs Defect Rate")
            visualizations['lead_time_vs_defect_rate'] = fig2.to_json()

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

def route_optimization_analysis(df):
    analysis_name = "Route Optimization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['route_id', 'origin', 'destination', 'distance', 'transit_time',
                    'cost', 'vehicle_type']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['distance', 'transit_time', 'cost'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['distance', 'transit_time', 'cost']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

        df_copy.dropna(subset=['distance', 'cost'], inplace=True) # Crucial for cost per mile

        # Calculate cost per mile
        df_copy['cost_per_mile'] = np.nan
        if (df_copy['distance'] != 0).any():
            df_copy['cost_per_mile'] = df_copy['cost'] / df_copy['distance']
            df_copy.loc[df_copy['distance'] == 0, 'cost_per_mile'] = np.nan # Handle division by zero
        
        # Metrics
        total_routes = len(df_copy)
        avg_transit_time = df_copy['transit_time'].mean() if 'transit_time' in df_copy and not df_copy['transit_time'].isnull().all() else 0
        avg_cost_per_mile = df_copy['cost_per_mile'].mean() if not df_copy['cost_per_mile'].isnull().all() else 0

        metrics = {
            "Total Routes": total_routes,
            "Avg Transit Time (Days)": avg_transit_time,
            "Avg Cost per Mile": avg_cost_per_mile
        }
        
        insights.append(f"Analyzed {total_routes} routes.")
        insights.append(f"Average transit time: {avg_transit_time:.1f} days.")
        insights.append(f"Average cost per mile: ${avg_cost_per_mile:.2f}.")

        # Visualizations
        if 'distance' in df_copy and 'transit_time' in df_copy and \
           not df_copy['distance'].isnull().all() and not df_copy['transit_time'].isnull().all():
            fig1 = px.scatter(df_copy, x='distance', y='transit_time',
                              color='vehicle_type' if 'vehicle_type' in df_copy.columns else None,
                              title="Distance vs Transit Time")
            visualizations['distance_vs_transit_time'] = fig1.to_json()
        
        if 'origin' in df_copy and 'destination' in df_copy and 'cost' in df_copy and not df_copy['cost'].isnull().all():
            route_costs = df_copy.groupby(['origin', 'destination'])['cost'].mean().reset_index()
            fig2 = px.bar(route_costs.sort_values('cost', ascending=False).head(10),
                          x='origin', y='cost', color='destination',
                          title="Top 10 Most Expensive Routes (by Avg. Cost)")
            visualizations['most_expensive_routes'] = fig2.to_json()

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

def demand_forecasting_analysis(df):
    analysis_name = "Demand Forecasting Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['product_id', 'date', 'demand', 'sales', 'inventory_level']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['date', 'demand', 'sales'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        # Convert date if needed
        if 'date' in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        
        for col in ['demand', 'sales', 'inventory_level']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['date', 'demand', 'sales'], inplace=True) 

        # Metrics
        total_products = df_copy['product_id'].nunique() if 'product_id' in df_copy else 0
        avg_demand = df_copy['demand'].mean() if not df_copy['demand'].isnull().all() else 0
        avg_sales = df_copy['sales'].mean() if not df_copy['sales'].isnull().all() else 0
        total_demand = df_copy['demand'].sum()
        total_sales = df_copy['sales'].sum()

        metrics = {
            "Total Unique Products": total_products,
            "Avg Daily Demand": avg_demand,
            "Avg Daily Sales": avg_sales,
            "Total Demand": total_demand,
            "Total Sales": total_sales
        }
        
        insights.append(f"Analyzed demand for {total_products} unique products.")
        insights.append(f"Total demand: {total_demand:,.0f} units. Total sales: {total_sales:,.0f} units.")
        insights.append(f"Average daily demand: {avg_demand:.1f} units.")

        # Visualizations
        if 'date' in df_copy and 'demand' in df_copy and not df_copy['demand'].isnull().all():
            time_df = df_copy.groupby('date')[['demand', 'sales']].sum().reset_index()
            fig1 = px.line(time_df, x='date', y=['demand', 'sales'],
                           title="Demand and Sales Over Time")
            visualizations['demand_sales_over_time'] = fig1.to_json()
        
        if 'product_id' in df_copy and 'inventory_level' in df_copy and \
           not df_copy['demand'].isnull().all() and not df_copy['inventory_level'].isnull().all():
            product_df = df_copy.groupby('product_id')[['demand', 'inventory_level']].mean().reset_index()
            fig2 = px.scatter(product_df, x='demand', y='inventory_level',
                              hover_name='product_id',
                              title="Average Demand vs. Inventory Levels by Product")
            visualizations['demand_vs_inventory_levels'] = fig2.to_json()

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
        expected = ['cost_id', 'category', 'amount', 'date', 'description']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['category', 'amount', 'date'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        if 'date' in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
            df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
        
        if 'amount' in df_copy.columns:
            df_copy['amount'] = pd.to_numeric(df_copy['amount'], errors='coerce')
        
        df_copy.dropna(subset=['amount', 'date', 'category'], inplace=True)

        # Metrics
        total_costs = df_copy['amount'].sum() if not df_copy['amount'].isnull().all() else 0
        avg_daily_cost = df_copy.groupby(df_copy['date'].dt.date)['amount'].sum().mean() if not df_copy.empty else 0
        top_category = df_copy.groupby('category')['amount'].sum().idxmax() if not df_copy.empty else "N/A"
        top_category_cost = df_copy.groupby('category')['amount'].sum().max() if not df_copy.empty else 0

        metrics = {
            "Total Costs": total_costs,
            "Avg Daily Cost": avg_daily_cost,
            "Largest Cost Category": top_category,
            "Largest Category Cost": top_category_cost
        }
        
        insights.append(f"Total recorded costs: ${total_costs:,.2f}.")
        insights.append(f"Average daily cost: ${avg_daily_cost:,.2f}.")
        insights.append(f"The largest cost category is '{top_category}', accounting for ${top_category_cost:,.2f}.")

        # Visualizations
        if 'category' in df_copy and 'amount' in df_copy and not df_copy['amount'].isnull().all():
            cost_by_category = df_copy.groupby('category')['amount'].sum().reset_index()
            fig1 = px.bar(cost_by_category.sort_values('amount', ascending=False),
                          x='category', y='amount',
                          title="Costs by Category")
            visualizations['costs_by_category'] = fig1.to_json()
        
        if 'date' in df_copy and 'amount' in df_copy and not df_copy['amount'].isnull().all():
            cost_over_time = df_copy.groupby(df_copy['date'].dt.to_period('M').astype(str))['amount'].sum().reset_index()
            fig2 = px.line(cost_over_time, x='date', y='amount',
                           title="Costs Over Time (Monthly)")
            visualizations['costs_over_time'] = fig2.to_json()

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

def delivery_performance_analysis(df):
    analysis_name = "Delivery Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['delivery_id', 'promised_date', 'actual_date', 'status',
                    'customer_id', 'delay_reason']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['promised_date', 'actual_date', 'status'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
        
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        # Convert dates
        date_cols = ['promised_date', 'actual_date']
        for col in date_cols:
            if col in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['promised_date', 'actual_date'], inplace=True)

        # Calculate delay days
        df_copy['delay_days'] = (df_copy['actual_date'] - df_copy['promised_date']).dt.days
        df_copy['on_time'] = df_copy['delay_days'] <= 0
        
        # Metrics
        total_deliveries = len(df_copy)
        on_time_rate = df_copy['on_time'].mean() * 100
        avg_delay = df_copy[df_copy['delay_days'] > 0]['delay_days'].mean() if (df_copy['delay_days'] > 0).any() else 0
        top_reason = "N/A"
        if 'delay_reason' in df_copy.columns:
            top_reason = df_copy[df_copy['on_time'] == False]['delay_reason'].mode()[0] if not df_copy[df_copy['on_time'] == False]['delay_reason'].empty else "N/A"

        metrics = {
            "Total Deliveries": total_deliveries,
            "On-Time Rate (%)": on_time_rate,
            "Avg Delay (when late, in days)": avg_delay,
            "Top Delay Reason": top_reason
        }
        
        insights.append(f"Analyzed {total_deliveries} deliveries.")
        insights.append(f"Overall on-time delivery rate: {on_time_rate:.1f}%.")
        insights.append(f"When deliveries are late, the average delay is {avg_delay:.1f} days.")
        if top_reason != "N/A":
            insights.append(f"The most common reason for delay is: {top_reason}.")

        # Visualizations
        if 'delay_days' in df_copy and not df_copy['delay_days'].isnull().all():
            fig1 = px.histogram(df_copy, x='delay_days',
                                title="Delivery Delay/Earliness Distribution (Days)")
            visualizations['delay_days_distribution'] = fig1.to_json()
        
        if 'delay_reason' in df_copy and not df_copy['delay_reason'].isnull().all() and (df_copy['on_time'] == False).any():
            reason_counts = df_copy[df_copy['on_time'] == False]['delay_reason'].value_counts().reset_index(name='count')
            fig2 = px.bar(reason_counts.head(10), x='delay_reason', y='count',
                          title="Top 10 Delay Reasons")
            visualizations['top_delay_reasons'] = fig2.to_json()

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

def trip_route_and_schedule_performance_analysis(df):
    analysis_name = "Trip Route and Schedule Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['trip_uuid', 'route_type', 'source_name', 'destination_name', 'trip_creation_time', 'scheduled_arrival_time', 'actual_arrival_time']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['scheduled_arrival_time', 'actual_arrival_time'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['trip_creation_time', 'scheduled_arrival_time', 'actual_arrival_time']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['scheduled_arrival_time', 'actual_arrival_time'], inplace=True)

        # Metrics
        df_copy['delay_minutes'] = (df_copy['actual_arrival_time'] - df_copy['scheduled_arrival_time']).dt.total_seconds() / 60
        df_copy['on_time'] = df_copy['delay_minutes'] <= 0
        on_time_rate = df_copy['on_time'].mean() * 100
        avg_delay = df_copy[df_copy['delay_minutes'] > 0]['delay_minutes'].mean() if (df_copy['delay_minutes'] > 0).any() else 0
        
        worst_route_series = pd.Series(dtype=float)
        if 'source_name' in df_copy.columns and 'destination_name' in df_copy.columns:
             worst_route_series = df_copy.groupby(['source_name', 'destination_name'])['delay_minutes'].mean()
        
        worst_route = worst_route_series.idxmax() if not worst_route_series.empty else ("N/A", "N/A")
        worst_route_delay = worst_route_series.max() if not worst_route_series.empty else 0

        metrics = {
            "On-Time Arrival Rate (%)": on_time_rate,
            "Average Delay (for late trips, mins)": avg_delay,
            "Route with Highest Avg. Delay": f"{worst_route[0]} to {worst_route[1]}",
            "Highest Avg. Delay (mins)": worst_route_delay
        }
        
        insights.append(f"Overall on-time arrival rate: {on_time_rate:.2f}%.")
        insights.append(f"Late trips were delayed by {avg_delay:.1f} minutes on average.")
        insights.append(f"The worst route is '{worst_route[0]} to {worst_route[1]}' with an average delay of {worst_route_delay:.1f} minutes.")

        # Visualizations
        if not df_copy['delay_minutes'].isnull().all():
            fig1 = px.histogram(df_copy, x='delay_minutes', title="Distribution of Arrival Delays (in minutes)")
            visualizations['arrival_delays_distribution'] = fig1.to_json()

        if 'route_type' in df_copy.columns and not df_copy['delay_minutes'].isnull().all():
            delay_by_route_type = df_copy.groupby('route_type')['delay_minutes'].mean().reset_index()
            fig2 = px.bar(delay_by_route_type, x='route_type', y='delay_minutes', title="Average Delay by Route Type")
            visualizations['avg_delay_by_route_type'] = fig2.to_json()

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

def shipping_carrier_and_cost_optimization_analysis(df):
    analysis_name = "Shipping Carrier and Cost Optimization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'carrier_id', 'shipping_cost']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['carrier_id', 'shipping_cost'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['shipping_cost'] = pd.to_numeric(df_copy['shipping_cost'], errors='coerce')
        df_copy.dropna(subset=['shipping_cost', 'carrier_id'], inplace=True) 

        # Metrics
        total_shipping_cost = df_copy['shipping_cost'].sum()
        avg_shipping_cost = df_copy['shipping_cost'].mean()
        most_used_carrier = df_copy['carrier_id'].mode()[0] if not df_copy['carrier_id'].empty else "N/A"
        
        metrics = {
            "Total Shipping Cost": total_shipping_cost,
            "Average Shipping Cost": avg_shipping_cost,
            "Most Used Carrier": most_used_carrier
        }
        
        insights.append(f"Total shipping cost: ${total_shipping_cost:,.2f}.")
        insights.append(f"Average shipping cost: ${avg_shipping_cost:,.2f}.")
        insights.append(f"The most frequently used carrier is: {most_used_carrier}.")

        # Visualizations
        if 'carrier_id' in df_copy and not df_copy['shipping_cost'].isnull().all():
            cost_by_carrier = df_copy.groupby('carrier_id')['shipping_cost'].agg(['sum', 'mean']).reset_index()
            fig1 = px.bar(cost_by_carrier.sort_values('sum', ascending=False), 
                          x='carrier_id', y='sum', title="Total Shipping Cost by Carrier")
            visualizations['total_shipping_cost_by_carrier'] = fig1.to_json()
        
        if 'carrier_id' in df_copy and not df_copy['shipping_cost'].isnull().all():
            fig2 = px.box(df_copy, x='carrier_id', y='shipping_cost', title="Distribution of Shipping Costs by Carrier")
            visualizations['shipping_costs_distribution_by_carrier'] = fig2.to_json()

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

def shipment_dispatch_and_delivery_time_analysis(df):
    analysis_name = "Shipment Dispatch and Delivery Time Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shipment_id', 'warehouse_id', 'dispatch_time', 'expected_delivery_time', 'actual_delivery_time']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['dispatch_time', 'expected_delivery_time'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['dispatch_time'] = pd.to_datetime(df_copy['dispatch_time'], errors='coerce')
        df_copy['expected_delivery_time'] = pd.to_datetime(df_copy['expected_delivery_time'], errors='coerce')
        df_copy.dropna(subset=['dispatch_time', 'expected_delivery_time'], inplace=True)
        
        df_copy['planned_transit_hours'] = (df_copy['expected_delivery_time'] - df_copy['dispatch_time']).dt.total_seconds() / 3600
        
        # Metrics
        avg_planned_transit = df_copy['planned_transit_hours'].mean() if not df_copy['planned_transit_hours'].isnull().all() else 0
        
        metrics = {
            "Average Planned Transit Time (Hours)": avg_planned_transit
        }
        
        insights.append(f"Average planned transit time (dispatch to expected delivery): {avg_planned_transit:.2f} hours.")

        # Visualizations
        if not df_copy['planned_transit_hours'].isnull().all():
            fig1 = px.histogram(df_copy, x='planned_transit_hours', title="Distribution of Planned Transit Times")
            visualizations['planned_transit_times_distribution'] = fig1.to_json()

        if 'warehouse_id' in df_copy.columns and not df_copy['planned_transit_hours'].isnull().all():
            transit_by_warehouse = df_copy.groupby('warehouse_id')['planned_transit_hours'].mean().reset_index()
            fig2 = px.bar(transit_by_warehouse, x='warehouse_id', y='planned_transit_hours', title="Average Planned Transit Time by Warehouse")
            visualizations['avg_planned_transit_by_warehouse'] = fig2.to_json()

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

def logistics_carrier_rate_and_service_analysis(df):
    analysis_name = "Logistics Carrier Rate and Service Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['carrier_name', 'service_type', 'rate_per_kg', 'max_weight_limit']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['rate_per_kg', 'max_weight_limit']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)

        if df_copy.empty:
            raise Exception("No valid data after cleaning for rate/weight analysis.")

        # Metrics
        cheapest_carrier_idx = df_copy['rate_per_kg'].idxmin()
        cheapest_carrier = df_copy.loc[cheapest_carrier_idx, 'carrier_name']
        cheapest_rate = df_copy.loc[cheapest_carrier_idx, 'rate_per_kg']
        
        highest_capacity_idx = df_copy['max_weight_limit'].idxmax()
        highest_capacity_carrier = df_copy.loc[highest_capacity_idx, 'carrier_name']
        highest_capacity = df_copy.loc[highest_capacity_idx, 'max_weight_limit']
        
        metrics = {
            "Cheapest Carrier (by rate/kg)": cheapest_carrier,
            "Cheapest Rate": cheapest_rate,
            "Highest Capacity Carrier": highest_capacity_carrier,
            "Highest Capacity (kg)": highest_capacity
        }
        
        insights.append(f"Cheapest carrier by rate: {cheapest_carrier} at ${cheapest_rate:.2f}/kg.")
        insights.append(f"Highest capacity carrier: {highest_capacity_carrier} at {highest_capacity:,.0f} kg.")
        
        # Visualizations
        if 'carrier_name' in df_copy and 'service_type' in df_copy and not df_copy['rate_per_kg'].isnull().all():
            fig1 = px.bar(df_copy.sort_values('rate_per_kg'), x='carrier_name', y='rate_per_kg', color='service_type',
                          title="Rate per Kg by Carrier and Service Type")
            visualizations['rate_per_kg_by_carrier_service'] = fig1.to_json()
        
        if 'rate_per_kg' in df_copy and 'max_weight_limit' in df_copy and \
           not df_copy['rate_per_kg'].isnull().all() and not df_copy['max_weight_limit'].isnull().all():
            fig2 = px.scatter(df_copy, x='rate_per_kg', y='max_weight_limit', color='carrier_name',
                              title="Max Weight Limit vs. Rate per Kg")
            visualizations['max_weight_vs_rate_per_kg'] = fig2.to_json()

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

def warehouse_capacity_and_operational_analysis(df):
    analysis_name = "Warehouse Capacity and Operational Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['warehouse_name', 'capacity_orders_per_day', 'operational_since']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['capacity_orders_per_day'] = pd.to_numeric(df_copy['capacity_orders_per_day'], errors='coerce')
        df_copy['operational_since'] = pd.to_datetime(df_copy['operational_since'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for capacity/operational analysis.")
            
        # Metrics
        total_capacity = df_copy['capacity_orders_per_day'].sum()
        oldest_warehouse_idx = df_copy['operational_since'].idxmin()
        oldest_warehouse = df_copy.loc[oldest_warehouse_idx, 'warehouse_name']
        oldest_date = df_copy.loc[oldest_warehouse_idx, 'operational_since']
        
        metrics = {
            "Total Network Capacity (Orders/Day)": total_capacity,
            "Oldest Warehouse": oldest_warehouse,
            "Oldest Warehouse Operational Since": oldest_date.strftime('%Y-%m-%d')
        }
        
        insights.append(f"Total network capacity: {total_capacity:,.0f} orders per day.")
        insights.append(f"The oldest warehouse is {oldest_warehouse}, operational since {oldest_date.date()}.")

        # Visualizations
        if 'warehouse_name' in df_copy and not df_copy['capacity_orders_per_day'].isnull().all():
            fig1 = px.pie(df_copy, names='warehouse_name', values='capacity_orders_per_day',
                          title="Share of Network Capacity by Warehouse")
            visualizations['network_capacity_by_warehouse'] = fig1.to_json()

        if 'operational_since' in df_copy and not df_copy['capacity_orders_per_day'].isnull().all():
            df_copy['years_operational'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['operational_since']).dt.days / 365.25
            fig2 = px.scatter(df_copy, x='years_operational', y='capacity_orders_per_day',
                              hover_name='warehouse_name', title="Capacity vs. Years Operational")
            visualizations['capacity_vs_years_operational'] = fig2.to_json()

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

def warehouse_stock_movement_and_inventory_analysis(df):
    analysis_name = "Warehouse Stock Movement and Inventory Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['warehouse_id', 'material_id', 'movement_type', 'quantity', 'movement_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['movement_date'] = pd.to_datetime(df_copy['movement_date'], errors='coerce')
        df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
        df_copy.dropna(inplace=True)

        # Metrics
        total_inbound = df_copy[df_copy['movement_type'].str.lower().isin(['in', 'inbound', 'receipt'])]['quantity'].sum()
        total_outbound = df_copy[df_copy['movement_type'].str.lower().isin(['out', 'outbound', 'shipment'])]['quantity'].sum()
        net_flow = total_inbound - total_outbound
        
        metrics = {
            "Total Inbound Quantity": total_inbound,
            "Total Outbound Quantity": total_outbound,
            "Net Stock Flow": net_flow
        }
        
        insights.append(f"Total quantity received (inbound): {total_inbound:,.0f} units.")
        insights.append(f"Total quantity shipped (outbound): {total_outbound:,.0f} units.")
        insights.append(f"Net stock flow: {net_flow:,.0f} units.")
        
        # Visualizations
        if 'movement_date' in df_copy and 'movement_type' in df_copy and not df_copy['quantity'].isnull().all():
            # Standardize movement types for grouping
            df_copy['movement_group'] = np.where(df_copy['movement_type'].str.lower().isin(['in', 'inbound', 'receipt']), 'Inbound',
                                        np.where(df_copy['movement_type'].str.lower().isin(['out', 'outbound', 'shipment']), 'Outbound', 'Other'))
                                        
            movement_over_time = df_copy.groupby([df_copy['movement_date'].dt.to_period('M').astype(str), 'movement_group'])['quantity'].sum().unstack().fillna(0)
            if 'Inbound' not in movement_over_time: movement_over_time['Inbound'] = 0
            if 'Outbound' not in movement_over_time: movement_over_time['Outbound'] = 0

            fig1 = px.bar(movement_over_time, title="Inbound vs. Outbound Stock Movement Over Time (Monthly)")
            visualizations['inbound_outbound_movement_over_time'] = fig1.to_json()
        
        if 'material_id' in df_copy and not df_copy['quantity'].isnull().all():
            movement_by_material = df_copy.groupby('material_id')['quantity'].sum().nlargest(15).reset_index()
            fig2 = px.bar(movement_by_material, x='material_id', y='quantity', title="Top 15 Materials by Movement Volume")
            visualizations['top_materials_by_movement_volume'] = fig2.to_json()

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

def purchase_order_and_supplier_delivery_performance_analysis(df):
    analysis_name = "Purchase Order and Supplier Delivery Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['po_id', 'supplier_id', 'order_date', 'expected_delivery_date', 'total_cost', 'actual_delivery_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['total_cost'] = pd.to_numeric(df_copy['total_cost'], errors='coerce')
        df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], errors='coerce')
        df_copy['expected_delivery_date'] = pd.to_datetime(df_copy['expected_delivery_date'], errors='coerce')
        df_copy['actual_delivery_date'] = pd.to_datetime(df_copy['actual_delivery_date'], errors='coerce')
        
        df_copy.dropna(subset=['total_cost', 'order_date', 'expected_delivery_date', 'actual_delivery_date', 'supplier_id'], inplace=True)

        if df_copy.empty:
            raise Exception("No valid data after cleaning for PO analysis.")

        # Metrics
        total_spend = df_copy['total_cost'].sum()
        top_supplier = df_copy.groupby('supplier_id')['total_cost'].sum().idxmax()
        top_supplier_spend = df_copy.groupby('supplier_id')['total_cost'].sum().max()
        
        df_copy['on_time'] = df_copy['actual_delivery_date'] <= df_copy['expected_delivery_date']
        overall_otd = df_copy['on_time'].mean() * 100
        
        metrics = {
            "Total Purchase Order Spend": total_spend,
            "Top Supplier by Spend": top_supplier,
            "Top Supplier Spend": top_supplier_spend,
            "Overall On-Time Delivery Rate (%)": overall_otd
        }
        
        insights.append(f"Total PO spend analyzed: ${total_spend:,.2f}.")
        insights.append(f"Top supplier by spend: {top_supplier} (${top_supplier_spend:,.2f}).")
        insights.append(f"Overall supplier on-time delivery rate: {overall_otd:.1f}%.")
        
        # Visualizations
        if 'supplier_id' in df_copy and not df_copy['total_cost'].isnull().all():
            spend_by_supplier = df_copy.groupby('supplier_id')['total_cost'].sum().nlargest(15).reset_index()
            fig1 = px.bar(spend_by_supplier, x='supplier_id', y='total_cost', title="Top 15 Suppliers by PO Spend")
            visualizations['top_suppliers_by_po_spend'] = fig1.to_json()
        
        if 'actual_delivery_date' in df_copy.columns and 'expected_delivery_date' in df_copy.columns:
            otd_by_supplier = df_copy.groupby('supplier_id')['on_time'].mean().mul(100).reset_index()
            fig2 = px.bar(otd_by_supplier.sort_values('on_time', ascending=False), 
                          x='supplier_id', y='on_time', title="On-Time Delivery Rate by Supplier")
            visualizations['otd_rate_by_supplier'] = fig2.to_json()

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

def purchase_order_line_item_cost_analysis(df):
    analysis_name = "Purchase Order Line Item Cost Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['po_id', 'material_id', 'quantity', 'unit_cost']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
        df_copy['unit_cost'] = pd.to_numeric(df_copy['unit_cost'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for line item analysis.")
            
        df_copy['line_total'] = df_copy['quantity'] * df_copy['unit_cost']
        
        # Metrics
        avg_unit_cost = df_copy['unit_cost'].mean()
        most_expensive_material = df_copy.groupby('material_id')['line_total'].sum().idxmax()
        most_expensive_material_spend = df_copy.groupby('material_id')['line_total'].sum().max()
        
        metrics = {
            "Average Unit Cost": avg_unit_cost,
            "Most Expensive Material (Total Spend)": most_expensive_material,
            "Most Expensive Material Spend": most_expensive_material_spend,
            "Total Line Item Spend": df_copy['line_total'].sum()
        }
        
        insights.append(f"Average unit cost across all items: ${avg_unit_cost:,.2f}.")
        insights.append(f"Material with highest total spend: {most_expensive_material} (${most_expensive_material_spend:,.2f}).")
        
        # Visualizations
        if 'material_id' in df_copy and not df_copy['line_total'].isnull().all():
            cost_by_material = df_copy.groupby('material_id')['line_total'].sum().nlargest(20).reset_index()
            fig1 = px.bar(cost_by_material, x='material_id', y='line_total', title="Top 20 Materials by Total Spend")
            visualizations['top_materials_by_total_spend'] = fig1.to_json()
        
        if not df_copy['quantity'].isnull().all() and not df_copy['unit_cost'].isnull().all():
            fig2 = px.scatter(df_copy, x='quantity', y='unit_cost', hover_name='material_id',
                              title="Unit Cost vs. Quantity Ordered (Volume Discount Analysis)")
            visualizations['unit_cost_vs_quantity_ordered'] = fig2.to_json()

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

def order_fulfillment_process_analysis(df):
    analysis_name = "Order Fulfillment Process Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['fulfillment_id', 'sales_order_id', 'shipment_id']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        # Metrics
        num_fulfillments = df_copy['fulfillment_id'].nunique()
        num_orders = df_copy['sales_order_id'].nunique()
        num_shipments = df_copy['shipment_id'].nunique()
        shipments_per_order = num_shipments / num_orders if num_orders > 0 else 0
        
        metrics = {
            "Total Fulfillments": num_fulfillments,
            "Total Sales Orders": num_orders,
            "Total Unique Shipments": num_shipments,
            "Avg. Shipments per Order": shipments_per_order
        }
        
        insights.append(f"Tracked {num_fulfillments} fulfillments for {num_orders} unique sales orders.")
        insights.append(f"These orders resulted in {num_shipments} unique shipments.")
        insights.append(f"On average, each order is fulfilled in {shipments_per_order:.2f} shipments.")
        
        # Visualizations
        if 'sales_order_id' in df_copy.columns:
            shipments_per_order_counts = df_copy.groupby('sales_order_id')['shipment_id'].nunique().value_counts().reset_index()
            shipments_per_order_counts.columns = ['num_shipments', 'order_count']
            fig1 = px.bar(shipments_per_order_counts, x='num_shipments', y='order_count',
                          title="Distribution of Shipments per Order")
            visualizations['shipments_per_order_distribution'] = fig1.to_json()

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

def inventory_quantity_on_hand_analysis(df):
    analysis_name = "Inventory Quantity on Hand Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['material_id', 'warehouse_id', 'quantity_on_hand', 'last_stocktake_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['material_id', 'warehouse_id', 'quantity_on_hand'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['quantity_on_hand'] = pd.to_numeric(df_copy['quantity_on_hand'], errors='coerce')
        df_copy.dropna(subset=['quantity_on_hand', 'material_id', 'warehouse_id'], inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for QOH analysis.")
            
        # Metrics
        total_inventory = df_copy['quantity_on_hand'].sum()
        top_material = df_copy.groupby('material_id')['quantity_on_hand'].sum().idxmax()
        top_material_qty = df_copy.groupby('material_id')['quantity_on_hand'].sum().max()
        top_warehouse = df_copy.groupby('warehouse_id')['quantity_on_hand'].sum().idxmax()
        top_warehouse_qty = df_copy.groupby('warehouse_id')['quantity_on_hand'].sum().max()
        
        metrics = {
            "Total Quantity on Hand": total_inventory,
            "Material with Highest Stock": top_material,
            "Material with Highest Stock (Qty)": top_material_qty,
            "Warehouse with Highest Stock": top_warehouse,
            "Warehouse with Highest Stock (Qty)": top_warehouse_qty
        }
        
        insights.append(f"Total quantity on hand across all warehouses: {total_inventory:,.0f} units.")
        insights.append(f"Material with most stock: {top_material} ({top_material_qty:,.0f} units).")
        insights.append(f"Warehouse with most stock: {top_warehouse} ({top_warehouse_qty:,.0f} units).")
        
        # Visualizations
        if 'warehouse_id' in df_copy and not df_copy['quantity_on_hand'].isnull().all():
            stock_by_warehouse = df_copy.groupby('warehouse_id')['quantity_on_hand'].sum().reset_index()
            fig1 = px.pie(stock_by_warehouse, names='warehouse_id', values='quantity_on_hand', title="Inventory Distribution by Warehouse")
            visualizations['inventory_distribution_by_warehouse'] = fig1.to_json()
        
        if 'material_id' in df_copy and not df_copy['quantity_on_hand'].isnull().all():
            top_items = df_copy.groupby('material_id')['quantity_on_hand'].sum().nlargest(20).reset_index()
            fig2 = px.bar(top_items, x='material_id', y='quantity_on_hand', title="Top 20 Materials by Quantity on Hand")
            visualizations['top_materials_by_quantity_on_hand'] = fig2.to_json()

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

def shipment_on_time_delivery_performance_analysis(df):
    analysis_name = "Shipment On-Time Delivery Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['carrier_id', 'origin', 'destination', 'ship_date', 'expected_delivery_date', 'actual_delivery_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['carrier_id', 'expected_delivery_date', 'actual_delivery_date'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['ship_date', 'expected_delivery_date', 'actual_delivery_date']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['expected_delivery_date', 'actual_delivery_date', 'carrier_id'], inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for OTD analysis.")
            
        df_copy['on_time'] = df_copy['actual_delivery_date'] <= df_copy['expected_delivery_date']
        
        # Metrics
        overall_otd_rate = df_copy['on_time'].mean() * 100
        best_carrier_series = df_copy.groupby('carrier_id')['on_time'].mean().mul(100)
        best_carrier = best_carrier_series.idxmax()
        best_carrier_rate = best_carrier_series.max()
        worst_carrier = best_carrier_series.idxmin()
        worst_carrier_rate = best_carrier_series.min()
        
        metrics = {
            "Overall On-Time Delivery Rate (%)": overall_otd_rate,
            "Best Carrier by OTD Rate": best_carrier,
            "Best Carrier OTD Rate (%)": best_carrier_rate,
            "Worst Carrier by OTD Rate": worst_carrier,
            "Worst Carrier OTD Rate (%)": worst_carrier_rate
        }
        
        insights.append(f"Overall On-Time Delivery (OTD) Rate: {overall_otd_rate:.2f}%.")
        insights.append(f"Best performing carrier: {best_carrier} ({best_carrier_rate:.2f}% OTD).")
        insights.append(f"Worst performing carrier: {worst_carrier} ({worst_carrier_rate:.2f}% OTD).")
        
        # Visualizations
        if 'carrier_id' in df_copy and not df_copy['on_time'].isnull().all():
            otd_by_carrier = best_carrier_series.reset_index()
            fig1 = px.bar(otd_by_carrier.sort_values('on_time', ascending=False), 
                          x='carrier_id', y='on_time', title="On-Time Delivery Rate by Carrier")
            visualizations['otd_rate_by_carrier'] = fig1.to_json()

        if 'actual_delivery_date' in df_copy.columns and not df_copy['on_time'].isnull().all():
            df_copy['delivery_day_of_week'] = df_copy['actual_delivery_date'].dt.day_name()
            otd_by_day = df_copy.groupby('delivery_day_of_week')['on_time'].mean().mul(100).reset_index()
            day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            otd_by_day['delivery_day_of_week'] = pd.Categorical(otd_by_day['delivery_day_of_week'], categories=day_order, ordered=True)
            otd_by_day = otd_by_day.sort_values('delivery_day_of_week')
            fig2 = px.bar(otd_by_day, x='delivery_day_of_week', y='on_time', title="On-Time Delivery Rate by Day of the Week")
            visualizations['otd_rate_by_day_of_week'] = fig2.to_json()

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

def late_delivery_and_order_value_correlation_analysis(df):
    analysis_name = "Late Delivery and Order Value Correlation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shipping_mode', 'late_delivery_flag', 'order_value']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['late_delivery_flag'] = pd.to_numeric(df_copy['late_delivery_flag'], errors='coerce') 
        df_copy['order_value'] = pd.to_numeric(df_copy['order_value'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Metrics
        late_rate = df_copy['late_delivery_flag'].mean() * 100
        avg_value_late = df_copy[df_copy['late_delivery_flag'] == 1]['order_value'].mean()
        avg_value_on_time = df_copy[df_copy['late_delivery_flag'] == 0]['order_value'].mean()
        
        metrics = {
            "Overall Late Delivery Rate (%)": late_rate,
            "Avg. Order Value (Late)": avg_value_late,
            "Avg. Order Value (On-Time)": avg_value_on_time
        }
        
        insights.append(f"Overall late delivery rate: {late_rate:.2f}%.")
        insights.append(f"Average value of late orders: ${avg_value_late:,.2f}.")
        insights.append(f"Average value of on-time orders: ${avg_value_on_time:,.2f}.")
        
        # Visualizations
        if not df_copy['late_delivery_flag'].isnull().all() and not df_copy['order_value'].isnull().all():
            fig1 = px.box(df_copy, x='late_delivery_flag', y='order_value', title="Order Value by Delivery Status")
            fig1.update_xaxes(tickvals=[0, 1], ticktext=['On-Time', 'Late']) # Improve readability
            visualizations['order_value_by_delivery_status'] = fig1.to_json()
        
        if 'shipping_mode' in df_copy and not df_copy['late_delivery_flag'].isnull().all():
            late_by_mode = df_copy.groupby('shipping_mode')['late_delivery_flag'].mean().mul(100).reset_index()
            fig2 = px.bar(late_by_mode.sort_values('late_delivery_flag', ascending=False), 
                          x='shipping_mode', y='late_delivery_flag', title="Late Delivery Rate by Shipping Mode")
            visualizations['late_delivery_rate_by_shipping_mode'] = fig2.to_json()

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

def driver_trip_performance_and_fuel_efficiency_analysis(df):
    analysis_name = "Driver Trip Performance and Fuel Efficiency Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['driver_id', 'start_time', 'end_time', 'distance_km', 'fuel_consumed_liters']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['distance_km', 'fuel_consumed_liters']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Metrics
        # Avoid division by zero for km_per_liter
        df_copy['km_per_liter'] = df_copy.apply(lambda row: row['distance_km'] / row['fuel_consumed_liters'] if row['fuel_consumed_liters'] > 0 else np.nan, axis=1)
        
        avg_efficiency = df_copy['km_per_liter'].mean() if not df_copy['km_per_liter'].isnull().all() else 0
        
        most_efficient_driver_series = df_copy.groupby('driver_id')['km_per_liter'].mean()
        most_efficient_driver = most_efficient_driver_series.idxmax() if not most_efficient_driver_series.empty else "N/A"
        best_efficiency = most_efficient_driver_series.max() if not most_efficient_driver_series.empty else 0
        
        metrics = {
            "Average Fuel Efficiency (km/L)": avg_efficiency,
            "Most Efficient Driver": most_efficient_driver,
            "Most Efficient Driver Avg (km/L)": best_efficiency
        }
        
        insights.append(f"Average fuel efficiency across all drivers: {avg_efficiency:.2f} km/L.")
        insights.append(f"Most efficient driver: {most_efficient_driver} (avg {best_efficiency:.2f} km/L).")
        
        # Visualizations
        if 'driver_id' in df_copy and not df_copy['km_per_liter'].isnull().all():
            efficiency_by_driver = most_efficient_driver_series.nlargest(15).reset_index()
            fig1 = px.bar(efficiency_by_driver, x='driver_id', y='km_per_liter', title="Top 15 Most Fuel-Efficient Drivers")
            visualizations['top_fuel_efficient_drivers'] = fig1.to_json()
        
        if not df_copy['distance_km'].isnull().all() and not df_copy['fuel_consumed_liters'].isnull().all():
            fig2 = px.scatter(df_copy, x='distance_km', y='fuel_consumed_liters', hover_name='driver_id',
                              title="Fuel Consumed vs. Distance Traveled",
                              trendline="ols", trendline_scope="overall")
            visualizations['fuel_consumed_vs_distance_traveled'] = fig2.to_json()

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

def barcode_scan_and_shipment_tracking_analysis(df):
    analysis_name = "Barcode Scan and Shipment Tracking Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shipment_id', 'barcode', 'scan_time', 'scan_type', 'location_center']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['scan_time'] = pd.to_datetime(df_copy['scan_time'], errors='coerce')
        df_copy.dropna(inplace=True)

        # Metrics
        total_scans = len(df_copy)
        unique_shipments_tracked = df_copy['shipment_id'].nunique()
        unique_locations = df_copy['location_center'].nunique()
        
        metrics = {
            "Total Scans Recorded": total_scans,
            "Unique Shipments Tracked": unique_shipments_tracked,
            "Unique Scan Locations": unique_locations
        }
        
        insights.append(f"Recorded {total_scans:,} scans across {unique_locations} locations.")
        insights.append(f"Tracking {unique_shipments_tracked:,} unique shipments.")
        
        # Sample trace for insights
        if not df_copy['shipment_id'].empty:
            sample_shipment_to_trace = df_copy['shipment_id'].iloc[0]
            shipment_journey = df_copy[df_copy['shipment_id'] == sample_shipment_to_trace].sort_values('scan_time')
            insights.append(f"Sample journey for shipment '{sample_shipment_to_trace}': {len(shipment_journey)} scans.")
            
        # Visualizations
        if 'location_center' in df_copy and not df_copy['location_center'].isnull().all():
            scans_by_location = df_copy['location_center'].value_counts().nlargest(15).reset_index(name='count')
            fig1 = px.bar(scans_by_location, x='location_center', y='count', title="Top 15 Scan Locations by Volume")
            visualizations['scans_by_location'] = fig1.to_json()

        if 'scan_type' in df_copy and not df_copy['scan_type'].isnull().all():
            scans_by_type = df_copy['scan_type'].value_counts().reset_index(name='count')
            fig2 = px.pie(scans_by_type, names='scan_type', values='count', title="Distribution of Scan Types")
            visualizations['scan_types_distribution'] = fig2.to_json()

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

def logistics_route_optimization_analysis(df):
    analysis_name = "Logistics Route Optimization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['route_id', 'start_center', 'end_center', 'distance_km', 'travel_time_min', 'route_type']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['distance_km', 'travel_time_min'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['distance_km', 'travel_time_min']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Metrics
        df_copy['speed_kmh'] = df_copy.apply(lambda row: row['distance_km'] / (row['travel_time_min'] / 60) if row['travel_time_min'] > 0 else np.nan, axis=1)
        
        avg_speed = df_copy['speed_kmh'].mean() if not df_copy['speed_kmh'].isnull().all() else 0
        avg_distance = df_copy['distance_km'].mean()
        avg_travel_time = df_copy['travel_time_min'].mean()
        
        metrics = {
            "Average Route Speed (km/h)": avg_speed,
            "Average Route Distance (km)": avg_distance,
            "Average Route Travel Time (min)": avg_travel_time
        }
        
        insights.append(f"Analyzed {len(df_copy)} routes.")
        insights.append(f"Average route: {avg_distance:.1f} km, {avg_travel_time:.1f} min.")
        insights.append(f"Average effective speed: {avg_speed:.1f} km/h.")
        
        # Visualizations
        if not df_copy['distance_km'].isnull().all() and not df_copy['travel_time_min'].isnull().all():
            fig1 = px.scatter(df_copy, x='distance_km', y='travel_time_min', 
                              color='route_type' if 'route_type' in df_copy.columns else None,
                              hover_name='route_id' if 'route_id' in df_copy.columns else None, 
                              title="Travel Time vs. Distance by Route Type")
            visualizations['travel_time_vs_distance'] = fig1.to_json()

        if 'route_type' in df_copy and not df_copy['speed_kmh'].isnull().all():
            speed_by_type = df_copy.groupby('route_type')['speed_kmh'].mean().reset_index()
            fig2 = px.bar(speed_by_type, x='route_type', y='speed_kmh', title="Average Speed by Route Type")
            visualizations['avg_speed_by_route_type'] = fig2.to_json()
        
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

def package_delivery_delay_analysis(df):
    analysis_name = "Package Delivery Delay Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['package_id', 'delivery_time', 'pickup_time', 'courier_id', 'delay_minutes']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['courier_id', 'delay_minutes'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['delay_minutes'] = pd.to_numeric(df_copy['delay_minutes'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Metrics
        avg_delay = df_copy[df_copy['delay_minutes'] > 0]['delay_minutes'].mean() if (df_copy['delay_minutes'] > 0).any() else 0
        on_time_rate = (df_copy['delay_minutes'] <= 0).mean() * 100
        
        worst_courier_series = df_copy.groupby('courier_id')['delay_minutes'].mean()
        worst_courier = worst_courier_series.idxmax() if not worst_courier_series.empty else "N/A"
        worst_courier_delay = worst_courier_series.max() if not worst_courier_series.empty else 0
        
        metrics = {
            "On-Time Rate (%)": on_time_rate,
            "Average Delay (for late packages, mins)": avg_delay,
            "Courier with Highest Avg. Delay": worst_courier,
            "Highest Avg. Delay (mins)": worst_courier_delay
        }
        
        insights.append(f"Overall on-time rate: {on_time_rate:.2f}%.")
        insights.append(f"Late packages are delayed by an average of {avg_delay:.1f} minutes.")
        insights.append(f"Courier with the highest average delay: {worst_courier} ({worst_courier_delay:.1f} mins).")
        
        # Visualizations
        if not df_copy['delay_minutes'].isnull().all():
            fig1 = px.histogram(df_copy, x='delay_minutes', title="Distribution of Delivery Delays (mins)")
            visualizations['delivery_delays_distribution'] = fig1.to_json()
        
        if 'courier_id' in df_copy and not df_copy['delay_minutes'].isnull().all():
            delay_by_courier = worst_courier_series.reset_index()
            fig2 = px.bar(delay_by_courier.sort_values('delay_minutes', ascending=False), 
                          x='courier_id', y='delay_minutes', title="Average Delay by Courier")
            visualizations['avg_delay_by_courier'] = fig2.to_json()

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

def delivery_performance_and_delay_root_cause_analysis(df):
    analysis_name = "Delivery Performance and Delay Root Cause Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['delivery_id', 'delivery_date', 'delayed_flag', 'delay_reason']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['delayed_flag'] = pd.to_numeric(df_copy['delayed_flag'], errors='coerce') # 1=Delayed, 0=On-Time
        df_copy.dropna(subset=['delayed_flag', 'delay_reason'], inplace=True)
        
        # Metrics
        delay_rate = df_copy['delayed_flag'].mean() * 100
        top_reason = df_copy[df_copy['delayed_flag'] == 1]['delay_reason'].mode()[0] if (df_copy['delayed_flag'] == 1).any() else "N/A"
        top_reason_count = df_copy[df_copy['delayed_flag'] == 1]['delay_reason'].value_counts().max() if (df_copy['delayed_flag'] == 1).any() else 0
        
        metrics = {
            "Overall Delay Rate (%)": delay_rate,
            "Top Reason for Delays": top_reason,
            "Top Reason Count": top_reason_count
        }
        
        insights.append(f"Overall delay rate: {delay_rate:.2f}%.")
        insights.append(f"The top reason for delays is '{top_reason}', occurring {top_reason_count} times.")
        
        # Visualizations
        if 'delayed_flag' in df_copy and 'delay_reason' in df_copy and (df_copy['delayed_flag'] == 1).any():
            delay_reason_counts = df_copy[df_copy['delayed_flag'] == 1]['delay_reason'].value_counts().reset_index(name='count')
            fig1 = px.pie(delay_reason_counts.head(10), names='delay_reason', values='count', title="Top 10 Distribution of Delay Reasons")
            visualizations['delay_reasons_distribution'] = fig1.to_json()
        
        if 'delivery_date' in df_copy.columns and not df_copy['delayed_flag'].isnull().all():
            df_copy['delivery_date'] = pd.to_datetime(df_copy['delivery_date'], errors='coerce')
            df_copy.dropna(subset=['delivery_date'], inplace=True)
            if not df_copy.empty:
                delays_over_time = df_copy.groupby(df_copy['delivery_date'].dt.to_period('W').astype(str))['delayed_flag'].mean().mul(100).reset_index()
                fig2 = px.line(delays_over_time, x='delivery_date', y='delayed_flag', title="Delay Rate Over Time (Weekly)")
                visualizations['delay_rate_over_time'] = fig2.to_json()

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

def warehouse_inventory_reorder_level_analysis(df):
    analysis_name = "Warehouse Inventory Reorder Level Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['warehouse_id', 'product_id', 'on_hand', 'reorder_level', 'reorder_qty']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['on_hand', 'reorder_level'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['on_hand', 'reorder_level', 'reorder_qty']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)
        
        df_copy['needs_reorder'] = df_copy['on_hand'] <= df_copy['reorder_level']
        
        # Metrics
        items_to_reorder = df_copy['needs_reorder'].sum()
        total_items = len(df_copy)
        perc_to_reorder = (items_to_reorder / total_items) * 100 if total_items > 0 else 0
        
        metrics = {
            "Total SKUs Analyzed": total_items,
            "Number of Items to Reorder": items_to_reorder,
            "% of SKUs Needing Reorder": perc_to_reorder
        }
        
        insights.append(f"Analyzed {total_items} SKUs.")
        insights.append(f"{items_to_reorder} items ({perc_to_reorder:.2f}%) are at or below their reorder level.")
        
        # Visualizations
        if not df_copy['needs_reorder'].isnull().all():
            reorder_status = df_copy['needs_reorder'].value_counts().reset_index()
            reorder_status['needs_reorder'] = reorder_status['needs_reorder'].map({True: 'Needs Reorder', False: 'Stock OK'})
            fig1 = px.pie(reorder_status, names='needs_reorder', values='count', title="Stock Status vs. Reorder Level")
            visualizations['stock_status_vs_reorder_level'] = fig1.to_json()
        
        if 'warehouse_id' in df_copy and not df_copy['needs_reorder'].isnull().all():
            reorder_by_warehouse = df_copy.groupby('warehouse_id')['needs_reorder'].sum().reset_index()
            fig2 = px.bar(reorder_by_warehouse.sort_values('needs_reorder', ascending=False), 
                          x='warehouse_id', y='needs_reorder', title="Number of Items to Reorder by Warehouse")
            visualizations['items_to_reorder_by_warehouse'] = fig2.to_json()

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

def supplier_lead_time_and_reliability_analysis(df):
    analysis_name = "Supplier Lead Time and Reliability Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['supplier_name', 'lead_time_days', 'on_time_delivery_rate']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['lead_time_days', 'on_time_delivery_rate']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for supplier analysis.")
            
        # Metrics
        avg_lead_time = df_copy['lead_time_days'].mean()
        avg_otd_rate = df_copy['on_time_delivery_rate'].mean()
        
        best_supplier_otd = df_copy.loc[df_copy['on_time_delivery_rate'].idxmax()]
        fastest_supplier = df_copy.loc[df_copy['lead_time_days'].idxmin()]
        
        metrics = {
            "Average Lead Time (Days)": avg_lead_time,
            "Average On-Time Delivery Rate (%)": avg_otd_rate,
            "Best Supplier (OTD)": best_supplier_otd['supplier_name'],
            "Best Supplier OTD Rate (%)": best_supplier_otd['on_time_delivery_rate'],
            "Fastest Supplier (Lead Time)": fastest_supplier['supplier_name'],
            "Fastest Supplier Lead Time (Days)": fastest_supplier['lead_time_days']
        }
        
        insights.append(f"Average lead time across all suppliers: {avg_lead_time:.1f} days.")
        insights.append(f"Average on-time delivery rate: {avg_otd_rate:.2f}%.")
        insights.append(f"Best OTD: {best_supplier_otd['supplier_name']} ({best_supplier_otd['on_time_delivery_rate']:.2f}%).")
        insights.append(f"Fastest Lead Time: {fastest_supplier['supplier_name']} ({fastest_supplier['lead_time_days']:.1f} days).")
        
        # Visualizations
        if not df_copy['lead_time_days'].isnull().all() and not df_copy['on_time_delivery_rate'].isnull().all():
            fig1 = px.scatter(df_copy, x='lead_time_days', y='on_time_delivery_rate', hover_name='supplier_name',
                              title="On-Time Rate vs. Lead Time by Supplier",
                              trendline="ols", trendline_scope="overall")
            visualizations['otd_vs_lead_time_by_supplier'] = fig1.to_json()
        
        if 'supplier_name' in df_copy and not df_copy['on_time_delivery_rate'].isnull().all():
            top_suppliers_otd = df_copy.nlargest(15, 'on_time_delivery_rate')
            fig2 = px.bar(top_suppliers_otd, x='supplier_name', y='on_time_delivery_rate', title="Top 15 Suppliers by On-Time Delivery Rate")
            visualizations['top_suppliers_by_otd'] = fig2.to_json()

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

def freight_haulage_and_truck_load_analysis(df):
    analysis_name = "Freight Haulage and Truck Load Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['truck_id', 'driver_id', 'departure_time', 'arrival_time', 'load_weight_tonnes']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['departure_time'] = pd.to_datetime(df_copy['departure_time'], errors='coerce')
        df_copy['arrival_time'] = pd.to_datetime(df_copy['arrival_time'], errors='coerce')
        df_copy['load_weight_tonnes'] = pd.to_numeric(df_copy['load_weight_tonnes'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        df_copy['trip_duration_hours'] = (df_copy['arrival_time'] - df_copy['departure_time']).dt.total_seconds() / 3600
        
        # Metrics
        total_weight_hauled = df_copy['load_weight_tonnes'].sum()
        avg_load_weight = df_copy['load_weight_tonnes'].mean()
        avg_trip_duration = df_copy['trip_duration_hours'].mean()
        
        metrics = {
            "Total Weight Hauled (Tonnes)": total_weight_hauled,
            "Average Load Weight (Tonnes)": avg_load_weight,
            "Average Trip Duration (Hours)": avg_trip_duration
        }
        
        insights.append(f"Total weight hauled: {total_weight_hauled:,.2f} tonnes.")
        insights.append(f"Average load weight per trip: {avg_load_weight:.2f} tonnes.")
        insights.append(f"Average trip duration: {avg_trip_duration:.2f} hours.")
        
        # Visualizations
        if 'truck_id' in df_copy and not df_copy['load_weight_tonnes'].isnull().all():
            weight_by_truck = df_copy.groupby('truck_id')['load_weight_tonnes'].sum().nlargest(15).reset_index()
            fig1 = px.bar(weight_by_truck, x='truck_id', y='load_weight_tonnes', title="Top 15 Trucks by Total Weight Hauled")
            visualizations['top_trucks_by_weight_hauled'] = fig1.to_json()
        
        if not df_copy['trip_duration_hours'].isnull().all() and not df_copy['load_weight_tonnes'].isnull().all():
            fig2 = px.scatter(df_copy, x='trip_duration_hours', y='load_weight_tonnes', 
                              hover_name='truck_id' if 'truck_id' in df_copy.columns else None,
                              title="Load Weight vs. Trip Duration")
            visualizations['load_weight_vs_trip_duration'] = fig2.to_json()

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

def inbound_logistics_and_vendor_quality_analysis(df):
    analysis_name = "Inbound Logistics and Vendor Quality Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['vendor_id', 'warehouse_id', 'order_date', 'receive_date', 'quantity_received', 'quality_check_flag']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], errors='coerce')
        df_copy['receive_date'] = pd.to_datetime(df_copy['receive_date'], errors='coerce')
        df_copy['quality_check_flag'] = pd.to_numeric(df_copy['quality_check_flag'], errors='coerce') # 1=Pass, 0=Fail
        df_copy.dropna(inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid data after cleaning for inbound analysis.")
            
        # Metrics
        df_copy['receipt_lead_time'] = (df_copy['receive_date'] - df_copy['order_date']).dt.days
        avg_lead_time = df_copy['receipt_lead_time'].mean() if not df_copy['receipt_lead_time'].isnull().all() else 0
        quality_pass_rate = df_copy['quality_check_flag'].mean() * 100
        
        metrics = {
            "Average Receipt Lead Time (Days)": avg_lead_time,
            "Quality Check Pass Rate (%)": quality_pass_rate
        }
        
        insights.append(f"Average receipt lead time (order to receive): {avg_lead_time:.1f} days.")
        insights.append(f"Overall quality check pass rate: {quality_pass_rate:.2f}%.")
        
        # Visualizations
        if 'vendor_id' in df_copy and not df_copy['quality_check_flag'].isnull().all():
            pass_rate_by_vendor = df_copy.groupby('vendor_id')['quality_check_flag'].mean().mul(100).reset_index()
            fig1 = px.bar(pass_rate_by_vendor.sort_values('quality_check_flag', ascending=False), 
                          x='vendor_id', y='quality_check_flag', title="Quality Pass Rate by Vendor")
            visualizations['quality_pass_rate_by_vendor'] = fig1.to_json()
        
        if 'vendor_id' in df_copy and not df_copy['receipt_lead_time'].isnull().all():
            lead_time_by_vendor = df_copy.groupby('vendor_id')['receipt_lead_time'].mean().reset_index()
            fig2 = px.bar(lead_time_by_vendor.sort_values('receipt_lead_time', ascending=True), 
                          x='vendor_id', y='receipt_lead_time', title="Average Receipt Lead Time by Vendor")
            visualizations['avg_receipt_lead_time_by_vendor'] = fig2.to_json()

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

def vehicle_route_and_on_time_performance_analysis(df):
    analysis_name = "Vehicle Route and On-Time Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['vehicle_id', 'trip_date', 'start_hub', 'end_hub', 'total_distance_km', 'on_time_delivery']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['on_time_delivery'] = pd.to_numeric(df_copy['on_time_delivery'], errors='coerce') # 1=On-time, 0=Late
        df_copy.dropna(subset=['on_time_delivery', 'start_hub', 'end_hub'], inplace=True)
        
        # Metrics
        otd_rate = df_copy['on_time_delivery'].mean() * 100
        df_copy['route'] = df_copy['start_hub'] + ' to ' + df_copy['end_hub']
        route_performance = df_copy.groupby('route')['on_time_delivery'].mean().mul(100)
        best_route = route_performance.idxmax() if not route_performance.empty else "N/A"
        best_route_rate = route_performance.max() if not route_performance.empty else 0
        
        metrics = {
            "Overall On-Time Delivery Rate (%)": otd_rate,
            "Best Route by OTD": best_route,
            "Best Route OTD Rate (%)": best_route_rate
        }
        
        insights.append(f"Overall on-time delivery rate for vehicles: {otd_rate:.2f}%.")
        insights.append(f"The best performing route is '{best_route}' with a {best_route_rate:.2f}% OTD rate.")
        
        # Visualizations
        if 'route' in df_copy and not df_copy['on_time_delivery'].isnull().all():
            otd_by_route = route_performance.nlargest(15).reset_index()
            fig1 = px.bar(otd_by_route, x='route', y='on_time_delivery', title="Top 15 Routes by On-Time Performance")
            visualizations['top_routes_by_otd'] = fig1.to_json()
        
        if 'vehicle_id' in df_copy and not df_copy['on_time_delivery'].isnull().all():
            otd_by_vehicle = df_copy.groupby('vehicle_id')['on_time_delivery'].mean().mul(100).reset_index()
            fig2 = px.histogram(otd_by_vehicle, x='on_time_delivery', title="Distribution of On-Time Performance Across Vehicles")
            visualizations['otd_distribution_across_vehicles'] = fig2.to_json()

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

def vehicle_fleet_maintenance_and_capacity_analysis(df):
    analysis_name = "Vehicle Fleet Maintenance and Capacity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['vehicle_id', 'vehicle_type', 'capacity_tonnes', 'last_service_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['last_service_date'] = pd.to_datetime(df_copy['last_service_date'], errors='coerce')
        df_copy['capacity_tonnes'] = pd.to_numeric(df_copy['capacity_tonnes'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        df_copy['days_since_service'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['last_service_date']).dt.days
        
        # Metrics
        total_capacity = df_copy['capacity_tonnes'].sum()
        avg_days_since_service = df_copy['days_since_service'].mean()
        needs_service_count = (df_copy['days_since_service'] > 90).sum() # Assuming 90 days service interval
        
        metrics = {
            "Total Fleet Capacity (Tonnes)": total_capacity,
            "Average Days Since Last Service": avg_days_since_service,
            "Vehicles Needing Service (>90 Days)": needs_service_count
        }
        
        insights.append(f"Total fleet capacity: {total_capacity:,.2f} tonnes.")
        insights.append(f"Average days since last service: {avg_days_since_service:.1f} days.")
        insights.append(f"{needs_service_count} vehicles are overdue for service (> 90 days).")
        
        # Visualizations
        if 'vehicle_type' in df_copy and not df_copy['capacity_tonnes'].isnull().all():
            capacity_by_type = df_copy.groupby('vehicle_type')['capacity_tonnes'].sum().reset_index()
            fig1 = px.pie(capacity_by_type, names='vehicle_type', values='capacity_tonnes', title="Fleet Capacity by Vehicle Type")
            visualizations['fleet_capacity_by_vehicle_type'] = fig1.to_json()
        
        if 'last_service_date' in df_copy.columns and not df_copy['capacity_tonnes'].isnull().all():
            fig2 = px.scatter(df_copy, x='days_since_service', y='capacity_tonnes', 
                              color='vehicle_type' if 'vehicle_type' in df_copy.columns else None,
                              title="Capacity vs. Days Since Last Service")
            visualizations['capacity_vs_days_since_service'] = fig2.to_json()

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

def sales_order_and_pricing_analysis(df):
    analysis_name = "Sales Order and Pricing Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'customer_id', 'product_id', 'quantity', 'price_per_unit', 'total_price', 'discount']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['order_id', 'product_id', 'quantity', 'total_price'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['quantity', 'price_per_unit', 'total_price', 'discount']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(subset=['order_id', 'product_id', 'quantity', 'total_price'], inplace=True)
        
        # Metrics
        total_sales = df_copy['total_price'].sum()
        avg_order_value = df_copy.groupby('order_id')['total_price'].sum().mean()
        avg_discount = df_copy['discount'].mean() * 100 if 'discount' in df_copy.columns and not df_copy['discount'].isnull().all() else 0
        top_product = df_copy.groupby('product_id')['total_price'].sum().idxmax()
        top_product_sales = df_copy.groupby('product_id')['total_price'].sum().max()
        
        metrics = {
            "Total Sales": total_sales,
            "Average Order Value": avg_order_value,
            "Average Discount (%)": avg_discount,
            "Top Product by Sales": top_product,
            "Top Product Sales": top_product_sales
        }
        
        insights.append(f"Total sales analyzed: ${total_sales:,.2f}.")
        insights.append(f"Average order value: ${avg_order_value:,.2f}.")
        if avg_discount > 0:
            insights.append(f"Average discount: {avg_discount:.2f}%.")
        insights.append(f"Top product by sales: {top_product} (${top_product_sales:,.2f}).")
        
        # Visualizations
        if 'product_id' in df_copy and not df_copy['total_price'].isnull().all():
            sales_by_product = df_copy.groupby('product_id')['total_price'].sum().nlargest(15).reset_index()
            fig1 = px.bar(sales_by_product, x='product_id', y='total_price', title="Top 15 Products by Sales Revenue")
            visualizations['top_products_by_sales_revenue'] = fig1.to_json()
        
        if 'quantity' in df_copy and 'price_per_unit' in df_copy and 'discount' in df_copy and \
           not df_copy['quantity'].isnull().all() and not df_copy['price_per_unit'].isnull().all() and not df_copy['discount'].isnull().all():
            
            fig2 = px.scatter(df_copy, x='quantity', y='price_per_unit', color='discount',
                              title="Price per Unit vs. Quantity (Colored by Discount)")
            visualizations['price_per_unit_vs_quantity_by_discount'] = fig2.to_json()

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

def shipment_tracking_and_status_update_analysis(df):
    analysis_name = "Shipment Tracking and Status Update Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['tracking_number', 'carrier', 'origin', 'destination', 'last_update_time']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['last_update_time'] = pd.to_datetime(df_copy['last_update_time'], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Analysis
        df_copy['hours_since_update'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['last_update_time']).dt.total_seconds() / 3600
        stale_shipments_threshold = 72 # 3 days
        stale_shipments = df_copy[df_copy['hours_since_update'] > stale_shipments_threshold]
        
        metrics = {
            "Total Shipments Tracked": len(df_copy),
            "Stale Shipments (No update > 72h)": len(stale_shipments),
            "Stale Shipments (%)": (len(stale_shipments) / len(df_copy)) * 100 if len(df_copy) > 0 else 0
        }
        
        insights.append(f"Tracking {len(df_copy)} shipments.")
        insights.append(f"{len(stale_shipments)} shipments ({metrics['Stale Shipments (%)']:.1f}%) have not had a tracking update in over 72 hours.")
        
        # Visualizations
        if 'carrier' in df_copy and not df_copy['carrier'].isnull().all():
            shipments_by_carrier = df_copy['carrier'].value_counts().reset_index(name='count')
            fig = px.pie(shipments_by_carrier, names='carrier', values='count', title="Shipment Volume by Carrier")
            visualizations['shipment_volume_by_carrier'] = fig.to_json()

        if not df_copy['hours_since_update'].isnull().all():
            fig2 = px.histogram(df_copy, x='hours_since_update', title="Distribution of Hours Since Last Scan")
            visualizations['hours_since_last_scan_distribution'] = fig2.to_json()
            
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

def package_volumetric_weight_and_zone_analysis(df):
    analysis_name = "Package Volumetric Weight and Zone Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['weight_g', 'length_cm', 'width_cm', 'height_cm', 'volumetric_weight', 'destination_zone']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['weight_g', 'volumetric_weight', 'destination_zone'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['weight_g', 'length_cm', 'width_cm', 'height_cm', 'volumetric_weight']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        df_copy.dropna(inplace=True)
        
        # Metrics
        df_copy['actual_weight_kg'] = df_copy['weight_g'] / 1000
        df_copy['chargeable_weight'] = df_copy[['actual_weight_kg', 'volumetric_weight']].max(axis=1)
        
        avg_chargeable_weight = df_copy['chargeable_weight'].mean()
        avg_actual_weight = df_copy['actual_weight_kg'].mean()
        avg_volumetric_weight = df_copy['volumetric_weight'].mean()
        
        metrics = {
            "Average Chargeable Weight (kg)": avg_chargeable_weight,
            "Average Actual Weight (kg)": avg_actual_weight,
            "Average Volumetric Weight (kg)": avg_volumetric_weight
        }
        
        insights.append(f"Average actual weight: {avg_actual_weight:.2f} kg.")
        insights.append(f"Average volumetric weight: {avg_volumetric_weight:.2f} kg.")
        insights.append(f"Average chargeable weight: {avg_chargeable_weight:.2f} kg.")
        
        # Visualizations
        if not df_copy['actual_weight_kg'].isnull().all() and not df_copy['volumetric_weight'].isnull().all():
            fig1 = px.scatter(df_copy, x='actual_weight_kg', y='volumetric_weight', color='destination_zone',
                              title="Actual Weight vs. Volumetric Weight by Zone")
            visualizations['actual_vs_volumetric_weight'] = fig1.to_json()
        
        if 'destination_zone' in df_copy and not df_copy['actual_weight_kg'].isnull().all() and not df_copy['volumetric_weight'].isnull().all():
            weight_by_zone = df_copy.groupby('destination_zone')[['actual_weight_kg', 'volumetric_weight', 'chargeable_weight']].mean().reset_index()
            fig2 = px.bar(weight_by_zone, x='destination_zone', y=['actual_weight_kg', 'volumetric_weight', 'chargeable_weight'],
                          barmode='group', title="Average Weights by Destination Zone")
            visualizations['avg_weights_by_destination_zone'] = fig2.to_json()

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

def inter_warehouse_stock_transfer_analysis(df):
    analysis_name = "Inter-Warehouse Stock Transfer Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['transfer_id', 'source_warehouse', 'destination_warehouse', 'product_id', 'quantity', 'transfer_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in expected if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
        df_copy['transfer_date'] = pd.to_datetime(df_copy['transfer_date'], errors='coerce')
        df_copy.dropna(inplace=True)

        if df_copy.empty:
            raise Exception("No valid data after cleaning for transfer analysis.")
            
        # Metrics
        total_quantity = df_copy['quantity'].sum()
        avg_transfer_quantity = df_copy['quantity'].mean()
        top_source = df_copy.groupby('source_warehouse')['quantity'].sum().idxmax()
        top_destination = df_copy.groupby('destination_warehouse')['quantity'].sum().idxmax()
        
        metrics = {
            "Total Quantity Transferred": total_quantity,
            "Average Transfer Quantity": avg_transfer_quantity,
            "Top Source Warehouse": top_source,
            "Top Destination Warehouse": top_destination
        }
        
        insights.append(f"Total quantity transferred between warehouses: {total_quantity:,.0f} units.")
        insights.append(f"Top source warehouse: {top_source}.")
        insights.append(f"Top destination warehouse: {top_destination}.")
        
        # Visualizations
        if not df_copy.empty:
            transfers_by_route = df_copy.groupby(['source_warehouse', 'destination_warehouse'])['quantity'].sum().nlargest(15).reset_index()
            fig1 = px.bar(transfers_by_route, x='source_warehouse', y='quantity', color='destination_warehouse',
                          title="Top 15 Transfer Routes by Quantity")
            visualizations['top_transfer_routes'] = fig1.to_json()
            
        if not df_copy.empty:
            df_copy['month'] = df_copy['transfer_date'].dt.to_period('M').astype(str)
            monthly_trends = df_copy.groupby('month')['quantity'].sum().reset_index()
            fig2 = px.line(monthly_trends, x='month', y='quantity', title="Total Transfer Volume Over Time (Monthly)")
            visualizations['transfer_volume_over_time'] = fig2.to_json()

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

def shipment_manifest_and_trip_planning_analysis(df):
    analysis_name = "Shipment Manifest and Trip Planning Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shipment_id', 'trip_id', 'origin', 'destination', 'planned_departure', 'planned_arrival', 'actual_departure', 'actual_arrival', 'total_weight_kg', 'total_volume_m3']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['planned_departure', 'actual_departure', 'planned_arrival', 'actual_arrival'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for time_col in ['planned_departure', 'planned_arrival', 'actual_departure', 'actual_arrival']:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
        
        for numeric_col in ['total_weight_kg', 'total_volume_m3']:
            if numeric_col in df_copy.columns:
                df_copy[numeric_col] = pd.to_numeric(df_copy[numeric_col], errors='coerce')
        
        df_copy.dropna(subset=['planned_departure', 'actual_departure', 'planned_arrival', 'actual_arrival'], inplace=True)
        
        # Calculate delays
        df_copy['departure_delay_minutes'] = (df_copy['actual_departure'] - df_copy['planned_departure']).dt.total_seconds() / 60
        df_copy['arrival_delay_minutes'] = (df_copy['actual_arrival'] - df_copy['planned_arrival']).dt.total_seconds() / 60
        
        avg_departure_delay = df_copy['departure_delay_minutes'].mean()
        avg_arrival_delay = df_copy['arrival_delay_minutes'].mean()
        total_weight = df_copy['total_weight_kg'].sum() if 'total_weight_kg' in df_copy.columns else 0
        
        metrics = {
            "Average Departure Delay (minutes)": avg_departure_delay,
            "Average Arrival Delay (minutes)": avg_arrival_delay,
            "Total Weight Transported (kg)": total_weight
        }
        
        insights.append(f"Average departure delay: {avg_departure_delay:.1f} minutes.")
        insights.append(f"Average arrival delay: {avg_arrival_delay:.1f} minutes.")
        
        # Visualizations
        fig1 = px.histogram(df_copy, x='departure_delay_minutes', title="Distribution of Departure Delays (Minutes)")
        visualizations['departure_delay_distribution'] = fig1.to_json()
        
        fig2 = px.histogram(df_copy, x='arrival_delay_minutes', title="Distribution of Arrival Delays (Minutes)")
        visualizations['arrival_delay_distribution'] = fig2.to_json()
        
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

def last_mile_delivery_confirmation_analysis(df):
    analysis_name = "Last-Mile Delivery Confirmation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['delivery_id', 'order_id', 'delivery_status', 'confirmation_time', 'delivery_time', 'delivery_agent_id']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['delivery_status', 'confirmation_time', 'delivery_time'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['confirmation_time'] = pd.to_datetime(df_copy['confirmation_time'], errors='coerce')
        df_copy['delivery_time'] = pd.to_datetime(df_copy['delivery_time'], errors='coerce')
        
        # Filter for confirmed deliveries
        confirmed_statuses = ['confirmed', 'delivered', 'completed']
        confirmed_deliveries = df_copy[df_copy['delivery_status'].str.lower().isin(confirmed_statuses)].copy()
        confirmed_deliveries.dropna(subset=['confirmation_time', 'delivery_time'], inplace=True)
        
        # Calculate confirmation delay
        confirmed_deliveries['confirmation_delay_minutes'] = (confirmed_deliveries['confirmation_time'] - confirmed_deliveries['delivery_time']).dt.total_seconds() / 60
        
        avg_confirmation_delay = confirmed_deliveries['confirmation_delay_minutes'].mean() if not confirmed_deliveries.empty else 0
        confirmation_rate = len(confirmed_deliveries) / len(df_copy) * 100 if len(df_copy) > 0 else 0
        
        metrics = {
            "Average Confirmation Delay (minutes)": avg_confirmation_delay,
            "Delivery Confirmation Rate (%)": confirmation_rate
        }
        
        insights.append(f"Delivery confirmation rate (status is 'confirmed', 'delivered', or 'completed'): {confirmation_rate:.2f}%.")
        insights.append(f"Average time from delivery to confirmation: {avg_confirmation_delay:.2f} minutes.")
        
        # Visualizations
        status_counts = df_copy['delivery_status'].value_counts().reset_index(name='count')
        fig1 = px.pie(status_counts, names='delivery_status', values='count', title="Overall Delivery Status Distribution")
        visualizations['delivery_status_distribution'] = fig1.to_json()
        
        if not confirmed_deliveries.empty:
            fig2 = px.histogram(confirmed_deliveries, x='confirmation_delay_minutes', title="Distribution of Confirmation Delays (Minutes)")
            visualizations['confirmation_delay_distribution'] = fig2.to_json()
            
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

def order_delivery_time_estimation_by_shipping_zone(df):
    analysis_name = "Order Delivery Time Estimation by Shipping Zone"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'shipping_zone', 'order_placed_time', 'order_delivered_time', 'delivery_status']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['shipping_zone', 'order_placed_time', 'order_delivered_time'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['order_placed_time'] = pd.to_datetime(df_copy['order_placed_time'], errors='coerce')
        df_copy['order_delivered_time'] = pd.to_datetime(df_copy['order_delivered_time'], errors='coerce')
        
        # Filter for delivered orders only if status is available
        if 'delivery_status' in df_copy.columns:
            df_copy = df_copy[df_copy['delivery_status'].str.lower().isin(['delivered', 'completed'])]
        
        df_copy.dropna(subset=['order_placed_time', 'order_delivered_time', 'shipping_zone'], inplace=True)
        
        if df_copy.empty:
            raise Exception("No valid delivered orders found after cleaning.")
            
        # Calculate delivery duration
        df_copy['delivery_duration_hours'] = (df_copy['order_delivered_time'] - df_copy['order_placed_time']).dt.total_seconds() / 3600
        
        # Analysis by shipping zone
        zone_analysis = df_copy.groupby('shipping_zone').agg(
            avg_hours=('delivery_duration_hours', 'mean'),
            median_hours=('delivery_duration_hours', 'median'),
            min_hours=('delivery_duration_hours', 'min'),
            max_hours=('delivery_duration_hours', 'max'),
            total_orders=('order_id', 'count')
        ).round(2).reset_index()
        
        overall_avg = df_copy['delivery_duration_hours'].mean()
        total_orders = len(df_copy)
        
        fastest_zone = zone_analysis.loc[zone_analysis['avg_hours'].idxmin()]
        slowest_zone = zone_analysis.loc[zone_analysis['avg_hours'].idxmax()]
        
        metrics = {
            "total_orders_analyzed": total_orders,
            "overall_avg_delivery_hours": overall_avg,
            "overall_avg_delivery_days": overall_avg / 24,
            "zone_analysis": zone_analysis.to_dict(orient='records'),
            "fastest_zone": fastest_zone['shipping_zone'],
            "fastest_zone_avg_hours": fastest_zone['avg_hours'],
            "slowest_zone": slowest_zone['shipping_zone'],
            "slowest_zone_avg_hours": slowest_zone['avg_hours']
        }
        
        insights.append(f"Analyzed {total_orders} delivered orders.")
        insights.append(f"Overall average delivery time: {overall_avg:.1f} hours ({overall_avg/24:.1f} days).")
        insights.append(f"Fastest Zone: {fastest_zone['shipping_zone']} ({fastest_zone['avg_hours']:.1f} hours).")
        insights.append(f"Slowest Zone: {slowest_zone['shipping_zone']} ({slowest_zone['avg_hours']:.1f} hours).")

        # Visualizations
        fig1 = px.box(df_copy, x='shipping_zone', y='delivery_duration_hours', title="Delivery Duration by Shipping Zone")
        visualizations['delivery_duration_by_zone_box'] = fig1.to_json()
        
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

def order_fulfillment_cycle_time_analysis(df):
    analysis_name = "Order Fulfillment Cycle Time Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'order_received_time', 'order_fulfilled_time', 'warehouse_id', 'fulfillment_status']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['order_received_time', 'order_fulfilled_time'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        df_copy['order_received_time'] = pd.to_datetime(df_copy['order_received_time'], errors='coerce')
        df_copy['order_fulfilled_time'] = pd.to_datetime(df_copy['order_fulfilled_time'], errors='coerce')
        df_copy.dropna(subset=['order_received_time', 'order_fulfilled_time'], inplace=True)
        
        # Calculate cycle time
        df_copy['fulfillment_cycle_time_hours'] = (df_copy['order_fulfilled_time'] - df_copy['order_received_time']).dt.total_seconds() / 3600
        
        avg_cycle_time = df_copy['fulfillment_cycle_time_hours'].mean()
        median_cycle_time = df_copy['fulfillment_cycle_time_hours'].median()
        
        metrics = {
            "Average Fulfillment Cycle Time (hours)": avg_cycle_time,
            "Median Fulfillment Cycle Time (hours)": median_cycle_time,
            "Total Orders Analyzed": len(df_copy)
        }
        
        insights.append(f"Analyzed {len(df_copy)} fulfilled orders.")
        insights.append(f"Average fulfillment cycle time (received to fulfilled): {avg_cycle_time:.2f} hours.")
        insights.append(f"Median fulfillment cycle time: {median_cycle_time:.2f} hours.")
        
        # Visualizations
        fig1 = px.histogram(df_copy, x='fulfillment_cycle_time_hours', title="Distribution of Fulfillment Cycle Times (Hours)")
        visualizations['cycle_time_distribution'] = fig1.to_json()
        
        if 'warehouse_id' in df_copy.columns:
            warehouse_performance = df_copy.groupby('warehouse_id')['fulfillment_cycle_time_hours'].agg(['mean', 'median', 'count']).reset_index()
            fig2 = px.bar(warehouse_performance.sort_values('mean', ascending=True), 
                          x='warehouse_id', y='mean', title="Average Fulfillment Cycle Time by Warehouse")
            visualizations['cycle_time_by_warehouse'] = fig2.to_json()
            
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

def truck_loading_efficiency_analysis(df):
    analysis_name = "Truck Loading Efficiency Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['truck_id', 'loading_start_time', 'loading_end_time', 'loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg'] if matched[col] is None]
        
        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df_copy = df.copy()
        df_copy = safe_rename(df_copy, matched)
        
        for col in ['loading_start_time', 'loading_end_time']:
            if col in df_copy.columns:
                df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        
        for col in ['loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg']:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
        
        df_copy.dropna(subset=['loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg'], inplace=True)
        
        # Calculate efficiency metrics
        df_copy['volume_utilization'] = (df_copy['loaded_volume_m3'] / df_copy['truck_capacity_volume_m3']) * 100
        df_copy['weight_utilization'] = (df_copy['loaded_weight_kg'] / df_copy['truck_capacity_weight_kg']) * 100
        # Efficiency is often limited by the first constraint hit
        df_copy['loading_efficiency'] = df_copy[['volume_utilization', 'weight_utilization']].max(axis=1)
        
        avg_volume_utilization = df_copy['volume_utilization'].mean()
        avg_weight_utilization = df_copy['weight_utilization'].mean()
        avg_loading_efficiency = df_copy['loading_efficiency'].mean()
        
        metrics = {
            "Average Volume Utilization (%)": avg_volume_utilization,
            "Average Weight Utilization (%)": avg_weight_utilization,
            "Average Overall Loading Efficiency (%)": avg_loading_efficiency
        }
        
        if 'loading_start_time' in df_copy.columns and 'loading_end_time' in df_copy.columns:
            df_copy.dropna(subset=['loading_start_time', 'loading_end_time'], inplace=True)
            df_copy['loading_duration_minutes'] = (df_copy['loading_end_time'] - df_copy['loading_start_time']).dt.total_seconds() / 60
            metrics["Average Loading Duration (minutes)"] = df_copy['loading_duration_minutes'].mean()
            
        insights.append(f"Average volume utilization: {avg_volume_utilization:.2f}%.")
        insights.append(f"Average weight utilization: {avg_weight_utilization:.2f}%.")
        insights.append(f"Average overall loading efficiency (max of vol/weight): {avg_loading_efficiency:.2f}%.")
        
        # Visualizations
        fig1 = px.histogram(df_copy, x='loading_efficiency', title="Distribution of Overall Loading Efficiency")
        visualizations['loading_efficiency_distribution'] = fig1.to_json()
        
        fig2 = px.scatter(df_copy, x='volume_utilization', y='weight_utilization', 
                          hover_name='truck_id' if 'truck_id' in df_copy.columns else None,
                          title="Volume Utilization vs. Weight Utilization")
        fig2.add_shape(type="line", x0=0, y0=0, x1=100, y1=100, line=dict(color="Red", dash="dash"))
        visualizations['volume_vs_weight_utilization'] = fig2.to_json()

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


# ========== MAIN FUNCTIONS ==========

def main_backend(file_path, encoding='utf-8', analysis_name=None):
    """
    Main function to run logistics data analysis for a backend API.
    
    Parameters:
    - file_path: path to the data file (CSV or Excel)
    - encoding: file encoding (default: 'utf-8')
    - analysis_name: The string name of the analysis to run.
    
    Returns:
    - Dictionary with analysis results.
    """
    
    df = load_data(file_path, encoding)
    if df is None:
        return {
            "analysis_type": "Data Loading",
            "status": "error", 
            "error_message": f"Failed to load data file from {file_path}. Check file path and encoding."
        }
    
    # Mapping of all analysis functions
    analysis_function_map = {
        "General Insights": show_general_insights,
        "shipment_analysis": shipment_analysis,
        "inventory_analysis": inventory_analysis,
        "transportation_analysis": transportation_analysis,
        "warehouse_analysis": warehouse_analysis,
        "supplier_analysis": supplier_analysis,
        "route_optimization_analysis": route_optimization_analysis,
        "demand_forecasting_analysis": demand_forecasting_analysis,
        "cost_analysis": cost_analysis,
        "delivery_performance_analysis": delivery_performance_analysis,
        "trip_route_and_schedule_performance_analysis": trip_route_and_schedule_performance_analysis,
        "shipping_carrier_and_cost_optimization_analysis": shipping_carrier_and_cost_optimization_analysis,
        "shipment_dispatch_and_delivery_time_analysis": shipment_dispatch_and_delivery_time_analysis,
        "logistics_carrier_rate_and_service_analysis": logistics_carrier_rate_and_service_analysis,
        "warehouse_capacity_and_operational_analysis": warehouse_capacity_and_operational_analysis,
        "warehouse_stock_movement_and_inventory_analysis": warehouse_stock_movement_and_inventory_analysis,
        "purchase_order_and_supplier_delivery_performance_analysis": purchase_order_and_supplier_delivery_performance_analysis,
        "purchase_order_line_item_cost_analysis": purchase_order_line_item_cost_analysis,
        "order_fulfillment_process_analysis": order_fulfillment_process_analysis,
        "inventory_quantity_on_hand_analysis": inventory_quantity_on_hand_analysis,
        "shipment_on_time_delivery_performance_analysis": shipment_on_time_delivery_performance_analysis,
        "late_delivery_and_order_value_correlation_analysis": late_delivery_and_order_value_correlation_analysis,
        "driver_trip_performance_and_fuel_efficiency_analysis": driver_trip_performance_and_fuel_efficiency_analysis,
        "barcode_scan_and_shipment_tracking_analysis": barcode_scan_and_shipment_tracking_analysis,
        "logistics_route_optimization_analysis": logistics_route_optimization_analysis,
        "package_delivery_delay_analysis": package_delivery_delay_analysis,
        "delivery_performance_and_delay_root_cause_analysis": delivery_performance_and_delay_root_cause_analysis,
        "warehouse_inventory_reorder_level_analysis": warehouse_inventory_reorder_level_analysis,
        "supplier_lead_time_and_reliability_analysis": supplier_lead_time_and_reliability_analysis,
        "freight_haulage_and_truck_load_analysis": freight_haulage_and_truck_load_analysis,
        "inbound_logistics_and_vendor_quality_analysis": inbound_logistics_and_vendor_quality_analysis,
        "vehicle_route_and_on_time_performance_analysis": vehicle_route_and_on_time_performance_analysis,
        "vehicle_fleet_maintenance_and_capacity_analysis": vehicle_fleet_maintenance_and_capacity_analysis,
        "sales_order_and_pricing_analysis": sales_order_and_pricing_analysis,
        "shipment_tracking_and_status_update_analysis": shipment_tracking_and_status_update_analysis,
        "package_volumetric_weight_and_zone_analysis": package_volumetric_weight_and_zone_analysis,
        "inter_warehouse_stock_transfer_analysis": inter_warehouse_stock_transfer_analysis,
        "shipment_manifest_and_trip_planning_analysis": shipment_manifest_and_trip_planning_analysis,
        "last_mile_delivery_confirmation_analysis": last_mile_delivery_confirmation_analysis,
        "order_delivery_time_estimation_by_shipping_zone": order_delivery_time_estimation_by_shipping_zone,
        "order_fulfillment_cycle_time_analysis": order_fulfillment_cycle_time_analysis,
        "truck_loading_efficiency_analysis": truck_loading_efficiency_analysis
    }
    
    # Determine which analysis to run
    if analysis_name in analysis_function_map:
        result = analysis_function_map[analysis_name](df)
    else:
        # Default to general insights if no or invalid analysis is specified
        result = show_general_insights(df, "General Insights (Default)", missing_cols=[f"Analysis '{analysis_name}' not found"], matched_cols={})
    
    return result

def main():
    """Main function for command-line usage"""
    print("🚚 Logistics Analytics Dashboard (CLI Mode)")

    # File path and encoding input
    file_path = input("Enter path to your logistics data file (e.g., data.csv): ")
    encoding = input("Enter file encoding (default=utf-8): ")
    if not encoding:
        encoding = 'utf-8'

    df_check = load_data(file_path, encoding)
    if df_check is None:
        print("Failed to load data. Exiting.")
        return
    print("Data loaded successfully!")
    
    # Analysis selection
    print("\nSelect Analysis to Perform:")
    print("0: General Insights")
    for i, option in enumerate(analysis_options):
        print(f"{i+1}: {option}")

    choice_str = input(f"Enter the option number (0-{len(analysis_options)}): ")
    try:
        choice = int(choice_str)
        if choice == 0:
            selected_analysis = "General Insights"
        elif 1 <= choice <= len(analysis_options):
            selected_analysis = analysis_options[choice - 1]
        else:
            print("Invalid choice. Running General Insights.")
            selected_analysis = "General Insights"
    except ValueError:
        print("Invalid input. Running General Insights.")
        selected_analysis = "General Insights"

    # Run the selected analysis using the backend function
    result = main_backend(file_path, encoding=encoding, analysis_name=selected_analysis)

    # Display results
    if result:
        print("\n" + "="*60)
        print(f"📈 ANALYSIS RESULTS: {result.get('analysis_type', 'Unknown Analysis')}")
        print("="*60)
        
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
            # Simple print for CLI
            for key, value in metrics.items():
                if isinstance(value, dict) or isinstance(value, list):
                     print(f"  - {key}: (see JSON output for details)")
                else:
                    if isinstance(value, float):
                        print(f"  - {key}: {value:,.2f}")
                    else:
                        print(f"  - {key}: {value}")

        # Visualizations info
        visualizations = result.get('visualizations', {})
        if visualizations:
            print(f"\n📈 Generated Visualizations: {len(visualizations)}")
            for viz_name in visualizations.keys():
                print(f"  - {viz_name}")
        print("\nAnalysis complete.")

