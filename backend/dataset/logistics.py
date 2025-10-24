import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import datetime
import warnings

# Suppress warnings if desired
warnings.filterwarnings('ignore')

# ========== UTILITY FUNCTIONS (Adapted for console/return values) ==========
def check_and_rename_columns(df, expected_columns_map):
    """
    Checks if expected columns exist in the DataFrame and renames them if alternative
    names are provided and found.

    Args:
        df (pd.DataFrame): The input DataFrame.
        expected_columns_map (dict): A dictionary where keys are the desired column names
                                     and values are lists of possible column names (including the desired one).
                                     Example: {'DesiredName': ['DesiredName', 'AltName1', 'AltName2']}

    Returns:
        tuple: A tuple containing:
               - pd.DataFrame: The DataFrame with columns renamed.
               - list: A list of desired columns that were not found.
    """
    missing_columns = []
    renamed_columns = {}

    for desired_name, possible_names in expected_columns_map.items():
        found = False
        for col_name in possible_names:
            if col_name in df.columns:
                if col_name != desired_name:
                    renamed_columns[col_name] = desired_name
                found = True
                break
        if not found:
            missing_columns.append(desired_name)

    if renamed_columns:
        df = df.rename(columns=renamed_columns)

    return df, missing_columns

def show_missing_columns_warning(missing_cols, matched_cols=None):
    """
    Displays a warning message for missing required columns to the console.

    Args:
        missing_cols (list): A list of desired columns that were not found.
        matched_cols (dict, optional): Dictionary of matched columns. Defaults to None.
    """
    warning_message = f"⚠️ Required Columns Not Found: The following columns are needed for this analysis but weren't found in your data:\n"
    for col in missing_cols:
        warning_message += f"  - {col}"
        if matched_cols and matched_cols.get(col):
            warning_message += f" (matched to: {matched_cols[col]})"
        warning_message += "\n"
    print(warning_message)

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

def show_general_insights(df, title="General Insights"):
    """
    Show general data insights and basic visualizations (prints to console).
    """
    print(f"\n--- {title} ---")

    # Key Metrics
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns # Include bool for consistency

    print(f"Total Records: {total_records}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

    # Numeric columns analysis
    if len(numeric_cols) > 0:
        print("\nNumeric Features Analysis:")
        for col in numeric_cols:
            print(f"\n  -- Column: '{col}' --")
            print(df[col].describe()) # Basic descriptive statistics
            print(f"    (Would typically show a histogram and box plot for '{col}')")
    else:
        print("\nNo numeric columns found for analysis.")

    # Correlation heatmap if enough numeric columns
    if len(numeric_cols) >= 2:
        print("\nFeature Correlations:")
        corr = df[numeric_cols].corr()
        print(corr.round(2))
        print("(Would typically show a correlation heatmap)")

    # Categorical columns analysis
    if len(categorical_cols) > 0:
        print("\nCategorical Features Analysis:")
        for col in categorical_cols:
            print(f"\n  -- Column: '{col}' --")
            value_counts = df[col].value_counts()
            print(f"Value Counts (Top 10):\n{value_counts.head(10)}")
            print(f"    (Would typically show a bar chart for '{col}')")
    else:
        print("\nNo categorical columns found for analysis.")

    return {
        "total_records": total_records,
        "total_columns": len(df.columns),
        "numeric_features_count": len(numeric_cols),
        "categorical_features_count": len(categorical_cols),
        "numeric_descriptive_stats": {col: df[col].describe().to_dict() for col in numeric_cols},
        "correlations": corr.to_dict() if len(numeric_cols) >= 2 else {},
        "categorical_value_counts": {col: df[col].value_counts().head(10).to_dict() for col in categorical_cols}
    }


# ========== ANALYSIS FUNCTIONS (Adapted for console output and returning figures as JSON) ==========

def shipment_analysis(df):
    print("\n--- Shipment Analysis ---")
    expected = ['shipment_id', 'origin', 'destination', 'shipment_date', 'delivery_date',
                'weight', 'volume', 'cost', 'status']
    df_copy = df.copy() # Work on a copy to avoid modifying original DataFrame unexpectedly
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis") # Fallback to general insights
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates
    date_cols = ['shipment_date', 'delivery_date']
    for col in date_cols:
        if col in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    # Drop rows where critical date conversions failed
    df_copy.dropna(subset=['shipment_date', 'delivery_date'], inplace=True)

    # Calculate transit time if both dates available
    if 'shipment_date' in df_copy and 'delivery_date' in df_copy:
        df_copy['transit_time'] = (df_copy['delivery_date'] - df_copy['shipment_date']).dt.days
    
    # Metrics
    total_shipments = len(df_copy)
    avg_cost = df_copy['cost'].mean() if 'cost' in df_copy and not df_copy['cost'].isnull().all() else 0
    on_time = (df_copy['status'] == 'Delivered').mean() * 100 if 'status' in df_copy else 0

    print(f"Total Shipments: {total_shipments}")
    print(f"Avg Cost: ${avg_cost:,.2f}")
    print(f"On-Time Delivery: {on_time:.1f}%")
    
    figures = {}

    # Visualizations
    if 'transit_time' in df_copy and not df_copy['transit_time'].isnull().all():
        fig1 = px.histogram(df_copy, x='transit_time',
                            title="Transit Time Distribution")
        print("\nFigure Generated: Transit Time Distribution (Histogram)")
        figures['transit_time_distribution'] = fig1.to_json()
    
    if 'origin' in df_copy and 'destination' in df_copy:
        route_counts = df_copy.groupby(['origin', 'destination']).size().reset_index(name='count')
        fig2 = px.bar(route_counts.sort_values('count', ascending=False).head(10),
                      x='origin', y='count', color='destination',
                      title="Top Shipping Routes")
        print("Figure Generated: Top Shipping Routes (Bar Chart)")
        figures['top_shipping_routes'] = fig2.to_json()

    return {
        "metrics": {
            "Total Shipments": total_shipments,
            "Avg Cost": avg_cost,
            "On-Time Delivery": on_time
        },
        "figures": figures
    }

def inventory_analysis(df):
    print("\n--- Inventory Analysis ---")
    expected = ['product_id', 'product_name', 'category', 'current_stock',
                'min_stock', 'max_stock', 'warehouse']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    # Ensure numeric columns are numeric
    for col in ['current_stock', 'min_stock', 'max_stock']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    df_copy.dropna(subset=['current_stock', 'min_stock', 'max_stock'], inplace=True) # Drop rows if critical stock levels are NaN

    # Calculate stock status
    if 'current_stock' in df_copy and 'min_stock' in df_copy and 'max_stock' in df_copy:
        df_copy['stock_status'] = np.where(
            df_copy['current_stock'] < df_copy['min_stock'], 'Low Stock',
            np.where(df_copy['current_stock'] > df_copy['max_stock'], 'Overstock', 'Normal')
        )
    else:
        df_copy['stock_status'] = 'Unknown' # Default if columns missing after rename

    # Metrics
    total_products = len(df_copy)
    low_stock = (df_copy['stock_status'] == 'Low Stock').sum() if 'stock_status' in df_copy else 0
    overstock = (df_copy['stock_status'] == 'Overstock').sum() if 'stock_status' in df_copy else 0

    print(f"Total Products: {total_products}")
    print(f"Low Stock Items: {low_stock}")
    print(f"Overstock Items: {overstock}")
    
    figures = {}

    # Visualizations
    if 'stock_status' in df_copy and not df_copy['stock_status'].isnull().all():
        fig1 = px.pie(df_copy, names='stock_status',
                      title="Inventory Status Distribution")
        print("\nFigure Generated: Inventory Status Distribution (Pie Chart)")
        figures['inventory_status_distribution'] = fig1.to_json()
    
    if 'category' in df_copy and 'current_stock' in df_copy and not df_copy['current_stock'].isnull().all():
        fig2 = px.box(df_copy, x='category', y='current_stock',
                      title="Stock Levels by Category")
        print("Figure Generated: Stock Levels by Category (Box Plot)")
        figures['stock_levels_by_category'] = fig2.to_json()

    return {
        "metrics": {
            "Total Products": total_products,
            "Low Stock Items": low_stock,
            "Overstock Items": overstock
        },
        "figures": figures
    }

def transportation_analysis(df):
    print("\n--- Transportation Analysis ---")
    expected = ['vehicle_id', 'vehicle_type', 'capacity', 'fuel_efficiency',
                'maintenance_cost', 'status']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    for col in ['capacity', 'fuel_efficiency', 'maintenance_cost']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Metrics
    total_vehicles = len(df_copy)
    avg_fuel = df_copy['fuel_efficiency'].mean() if 'fuel_efficiency' in df_copy and not df_copy['fuel_efficiency'].isnull().all() else 0
    under_maintenance = (df_copy['status'] == 'Maintenance').sum() if 'status' in df_copy else 0

    print(f"Total Vehicles: {total_vehicles}")
    print(f"Avg Fuel Efficiency: {avg_fuel:.1f} mpg")
    print(f"Under Maintenance: {under_maintenance}")
    
    figures = {}

    # Visualizations
    if 'vehicle_type' in df_copy and not df_copy['vehicle_type'].isnull().all():
        fig1 = px.bar(df_copy['vehicle_type'].value_counts().reset_index(name='count'), x='index', y='count',
                      title="Vehicle Type Distribution")
        print("\nFigure Generated: Vehicle Type Distribution (Bar Chart)")
        figures['vehicle_type_distribution'] = fig1.to_json()
    
    if 'fuel_efficiency' in df_copy and 'maintenance_cost' in df_copy and \
       not df_copy['fuel_efficiency'].isnull().all() and not df_copy['maintenance_cost'].isnull().all():
        fig2 = px.scatter(df_copy, x='fuel_efficiency', y='maintenance_cost',
                          color='vehicle_type',
                          title="Fuel Efficiency vs Maintenance Cost")
        print("Figure Generated: Fuel Efficiency vs Maintenance Cost (Scatter Plot)")
        figures['fuel_efficiency_vs_maintenance_cost'] = fig2.to_json()

    return {
        "metrics": {
            "Total Vehicles": total_vehicles,
            "Avg Fuel Efficiency": avg_fuel,
            "Under Maintenance": under_maintenance
        },
        "figures": figures
    }

def warehouse_analysis(df):
    print("\n--- Warehouse Analysis ---")
    expected = ['warehouse_id', 'location', 'capacity', 'current_utilization',
                'operating_cost', 'employees']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    for col in ['capacity', 'current_utilization', 'operating_cost', 'employees']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    df_copy.dropna(subset=['capacity', 'current_utilization'], inplace=True) # Crucial for utilization calculation

    # Calculate utilization percentage
    if 'capacity' in df_copy and 'current_utilization' in df_copy and (df_copy['capacity'] != 0).any():
        df_copy['utilization_pct'] = (df_copy['current_utilization'] / df_copy['capacity']) * 100
        df_copy.loc[df_copy['capacity'] == 0, 'utilization_pct'] = np.nan # Handle division by zero
    else:
        df_copy['utilization_pct'] = np.nan
    
    # Metrics
    total_warehouses = len(df_copy)
    avg_utilization = df_copy['utilization_pct'].mean() if 'utilization_pct' in df_copy and not df_copy['utilization_pct'].isnull().all() else 0
    high_utilization = (df_copy['utilization_pct'] > 85).sum() if 'utilization_pct' in df_copy and not df_copy['utilization_pct'].isnull().all() else 0

    print(f"Total Warehouses: {total_warehouses}")
    print(f"Avg Utilization: {avg_utilization:.1f}%")
    print(f"High Utilization (>85%): {high_utilization}")
    
    figures = {}

    # Visualizations
    if 'location' in df_copy and 'utilization_pct' in df_copy and not df_copy['utilization_pct'].isnull().all():
        fig1 = px.bar(df_copy, x='location', y='utilization_pct',
                      title="Warehouse Utilization by Location")
        print("\nFigure Generated: Warehouse Utilization by Location (Bar Chart)")
        figures['warehouse_utilization_by_location'] = fig1.to_json()
    
    if 'operating_cost' in df_copy and 'current_utilization' in df_copy and \
       not df_copy['operating_cost'].isnull().all() and not df_copy['current_utilization'].isnull().all():
        fig2 = px.scatter(df_copy, x='current_utilization', y='operating_cost',
                          title="Utilization vs Operating Cost")
        print("Figure Generated: Utilization vs Operating Cost (Scatter Plot)")
        figures['utilization_vs_operating_cost'] = fig2.to_json()

    return {
        "metrics": {
            "Total Warehouses": total_warehouses,
            "Avg Utilization": avg_utilization,
            "High Utilization (>85%)": high_utilization
        },
        "figures": figures
    }

def supplier_analysis(df):
    print("\n--- Supplier Analysis ---")
    expected = ['supplier_id', 'supplier_name', 'lead_time', 'defect_rate',
                'unit_cost', 'delivery_reliability']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    for col in ['lead_time', 'defect_rate', 'unit_cost', 'delivery_reliability']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    # Metrics
    total_suppliers = len(df_copy)
    avg_lead_time = df_copy['lead_time'].mean() if 'lead_time' in df_copy and not df_copy['lead_time'].isnull().all() else 0
    avg_defect_rate = df_copy['defect_rate'].mean() * 100 if 'defect_rate' in df_copy and not df_copy['defect_rate'].isnull().all() else 0

    print(f"Total Suppliers: {total_suppliers}")
    print(f"Avg Lead Time: {avg_lead_time:.1f} days")
    print(f"Avg Defect Rate: {avg_defect_rate:.1f}%")
    
    figures = {}

    # Visualizations
    if 'supplier_name' in df_copy and 'delivery_reliability' in df_copy and not df_copy['delivery_reliability'].isnull().all():
        fig1 = px.bar(df_copy.sort_values('delivery_reliability', ascending=False).head(10),
                      x='supplier_name', y='delivery_reliability',
                      title="Top Suppliers by Delivery Reliability")
        print("\nFigure Generated: Top Suppliers by Delivery Reliability (Bar Chart)")
        figures['top_suppliers_by_delivery_reliability'] = fig1.to_json()
    
    if 'lead_time' in df_copy and 'defect_rate' in df_copy and \
       not df_copy['lead_time'].isnull().all() and not df_copy['defect_rate'].isnull().all():
        fig2 = px.scatter(df_copy, x='lead_time', y='defect_rate',
                          title="Lead Time vs Defect Rate")
        print("Figure Generated: Lead Time vs Defect Rate (Scatter Plot)")
        figures['lead_time_vs_defect_rate'] = fig2.to_json()

    return {
        "metrics": {
            "Total Suppliers": total_suppliers,
            "Avg Lead Time": avg_lead_time,
            "Avg Defect Rate": avg_defect_rate
        },
        "figures": figures
    }

def route_optimization_analysis(df):
    print("\n--- Route Optimization Analysis ---")
    expected = ['route_id', 'origin', 'destination', 'distance', 'transit_time',
                'cost', 'vehicle_type']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    for col in ['distance', 'transit_time', 'cost']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')

    df_copy.dropna(subset=['distance', 'cost'], inplace=True) # Crucial for cost per mile

    # Calculate cost per mile
    if 'cost' in df_copy and 'distance' in df_copy and (df_copy['distance'] != 0).any():
        df_copy['cost_per_mile'] = df_copy['cost'] / df_copy['distance']
        df_copy.loc[df_copy['distance'] == 0, 'cost_per_mile'] = np.nan # Handle division by zero
    else:
        df_copy['cost_per_mile'] = np.nan
    
    # Metrics
    total_routes = len(df_copy)
    avg_transit_time = df_copy['transit_time'].mean() if 'transit_time' in df_copy and not df_copy['transit_time'].isnull().all() else 0
    avg_cost_per_mile = df_copy['cost_per_mile'].mean() if 'cost_per_mile' in df_copy and not df_copy['cost_per_mile'].isnull().all() else 0

    print(f"Total Routes: {total_routes}")
    print(f"Avg Transit Time: {avg_transit_time:.1f} days")
    print(f"Avg Cost per Mile: ${avg_cost_per_mile:.2f}")
    
    figures = {}

    # Visualizations
    if 'distance' in df_copy and 'transit_time' in df_copy and \
       not df_copy['distance'].isnull().all() and not df_copy['transit_time'].isnull().all():
        fig1 = px.scatter(df_copy, x='distance', y='transit_time',
                          color='vehicle_type',
                          title="Distance vs Transit Time")
        print("\nFigure Generated: Distance vs Transit Time (Scatter Plot)")
        figures['distance_vs_transit_time'] = fig1.to_json()
    
    if 'origin' in df_copy and 'destination' in df_copy and 'cost' in df_copy and not df_copy['cost'].isnull().all():
        fig2 = px.bar(df_copy.groupby(['origin', 'destination'])['cost'].mean().reset_index()
                      .sort_values('cost', ascending=False).head(10),
                      x='origin', y='cost', color='destination',
                      title="Most Expensive Routes")
        print("Figure Generated: Most Expensive Routes (Bar Chart)")
        figures['most_expensive_routes'] = fig2.to_json()

    return {
        "metrics": {
            "Total Routes": total_routes,
            "Avg Transit Time": avg_transit_time,
            "Avg Cost per Mile": avg_cost_per_mile
        },
        "figures": figures
    }

def demand_forecasting_analysis(df):
    print("\n--- Demand Forecasting Analysis ---")
    expected = ['product_id', 'date', 'demand', 'sales', 'inventory_level']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    # Convert date if needed
    if 'date' in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    
    for col in ['demand', 'sales', 'inventory_level']:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    df_copy.dropna(subset=['date', 'demand', 'sales', 'inventory_level'], inplace=True) # All are crucial here

    # Metrics
    total_products = df_copy['product_id'].nunique() if 'product_id' in df_copy else 0
    avg_demand = df_copy['demand'].mean() if 'demand' in df_copy and not df_copy['demand'].isnull().all() else 0
    avg_sales = df_copy['sales'].mean() if 'sales' in df_copy and not df_copy['sales'].isnull().all() else 0

    print(f"Total Products: {total_products}")
    print(f"Avg Daily Demand: {avg_demand:.1f}")
    print(f"Avg Daily Sales: {avg_sales:.1f}")
    
    figures = {}

    # Visualizations
    if 'date' in df_copy and 'demand' in df_copy and not df_copy['demand'].isnull().all():
        time_df = df_copy.groupby('date')['demand'].sum().reset_index()
        fig1 = px.line(time_df, x='date', y='demand',
                       title="Demand Over Time")
        print("\nFigure Generated: Demand Over Time (Line Chart)")
        figures['demand_over_time'] = fig1.to_json()
    
    if 'product_id' in df_copy and 'demand' in df_copy and 'inventory_level' in df_copy and \
       not df_copy['demand'].isnull().all() and not df_copy['inventory_level'].isnull().all():
        product_df = df_copy.groupby('product_id')[['demand', 'inventory_level']].mean().reset_index()
        fig2 = px.scatter(product_df, x='demand', y='inventory_level',
                          title="Demand vs Inventory Levels")
        print("Figure Generated: Demand vs Inventory Levels (Scatter Plot)")
        figures['demand_vs_inventory_levels'] = fig2.to_json()

    return {
        "metrics": {
            "Total Products": total_products,
            "Avg Daily Demand": avg_demand,
            "Avg Daily Sales": avg_sales
        },
        "figures": figures
    }

def cost_analysis(df):
    print("\n--- Cost Analysis ---")
    expected = ['cost_id', 'category', 'amount', 'date', 'description']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "Cost Analysis (Partial)")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    # Convert date if needed
    if 'date' in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy['date']):
        df_copy['date'] = pd.to_datetime(df_copy['date'], errors='coerce')
    
    if 'amount' in df_copy.columns:
        df_copy['amount'] = pd.to_numeric(df_copy['amount'], errors='coerce')
    
    df_copy.dropna(subset=['amount', 'date'], inplace=True) # Amount and date are crucial

    # Metrics
    total_costs = df_copy['amount'].sum() if 'amount' in df_copy and not df_copy['amount'].isnull().all() else 0
    avg_daily_cost = df_copy.groupby('date')['amount'].sum().mean() if 'amount' in df_copy and 'date' in df_copy else 0
    top_category = df_copy.groupby('category')['amount'].sum().idxmax() if 'category' in df_copy and 'amount' in df_copy and not df_copy['amount'].isnull().all() else "N/A"

    print(f"Total Costs: ${total_costs:,.2f}")
    print(f"Avg Daily Cost: ${avg_daily_cost:,.2f}")
    print(f"Largest Cost Category: {top_category}")
    
    figures = {}

    # Visualizations
    if 'category' in df_copy and 'amount' in df_copy and not df_copy['amount'].isnull().all():
        cost_by_category = df_copy.groupby('category')['amount'].sum().reset_index()
        fig1 = px.bar(cost_by_category.sort_values('amount', ascending=False),
                      x='category', y='amount',
                      title="Costs by Category")
        print("\nFigure Generated: Costs by Category (Bar Chart)")
        figures['costs_by_category'] = fig1.to_json()
    
    if 'date' in df_copy and 'amount' in df_copy and not df_copy['amount'].isnull().all():
        cost_over_time = df_copy.groupby('date')['amount'].sum().reset_index()
        fig2 = px.line(cost_over_time, x='date', y='amount',
                       title="Costs Over Time")
        print("Figure Generated: Costs Over Time (Line Chart)")
        figures['costs_over_time'] = fig2.to_json()

    return {
        "metrics": {
            "Total Costs": total_costs,
            "Avg Daily Cost": avg_daily_cost,
            "Largest Cost Category": top_category
        },
        "figures": figures
    }

def delivery_performance_analysis(df):
    print("\n--- Delivery Performance Analysis ---")
    expected = ['delivery_id', 'promised_date', 'actual_date', 'status',
                'customer_id', 'delay_reason']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    
    df_copy = df_copy.rename(columns={v:k for k,v in matched.items() if v})
    
    # Convert dates
    date_cols = ['promised_date', 'actual_date']
    for col in date_cols:
        if col in df_copy and not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    
    df_copy.dropna(subset=['promised_date', 'actual_date'], inplace=True)

    # Calculate delay days if both dates available
    if 'promised_date' in df_copy and 'actual_date' in df_copy:
        df_copy['delay_days'] = (df_copy['actual_date'] - df_copy['promised_date']).dt.days
    
    # Metrics
    total_deliveries = len(df_copy)
    on_time = (df_copy['status'] == 'On Time').mean() * 100 if 'status' in df_copy else 0
    avg_delay = df_copy[df_copy['delay_days'] > 0]['delay_days'].mean() if 'delay_days' in df_copy and (df_copy['delay_days'] > 0).any() else 0

    print(f"Total Deliveries: {total_deliveries}")
    print(f"On-Time Rate: {on_time:.1f}%")
    print(f"Avg Delay (when late): {avg_delay:.1f} days")
    
    figures = {}

    # Visualizations
    if 'delay_days' in df_copy and not df_copy['delay_days'].isnull().all():
        fig1 = px.histogram(df_copy[df_copy['delay_days'] > 0], x='delay_days',
                            title="Delay Days Distribution")
        print("\nFigure Generated: Delay Days Distribution (Histogram)")
        figures['delay_days_distribution'] = fig1.to_json()
    
    if 'delay_reason' in df_copy and not df_copy['delay_reason'].isnull().all():
        reason_counts = df_copy['delay_reason'].value_counts().reset_index(name='count')
        fig2 = px.bar(reason_counts, x='index', y='count',
                      title="Top Delay Reasons")
        print("Figure Generated: Top Delay Reasons (Bar Chart)")
        figures['top_delay_reasons'] = fig2.to_json()

    return {
        "metrics": {
            "Total Deliveries": total_deliveries,
            "On-Time Rate": on_time,
            "Avg Delay (when late)": avg_delay
        },
        "figures": figures
    }

# Extra functions
def trip_route_and_schedule_performance_analysis(df):
    print("\n--- Trip Route and Schedule Performance Analysis ---")
    expected = ['trip_uuid', 'route_type', 'source_name', 'destination_name', 'trip_creation_time', 'scheduled_arrival_time', 'actual_arrival_time']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['trip_creation_time', 'scheduled_arrival_time', 'actual_arrival_time']:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    df_copy.dropna(subset=['trip_creation_time', 'scheduled_arrival_time', 'actual_arrival_time'], inplace=True)

    # Metrics
    df_copy['delay_minutes'] = (df_copy['actual_arrival_time'] - df_copy['scheduled_arrival_time']).dt.total_seconds() / 60
    df_copy['on_time'] = df_copy['delay_minutes'] <= 0
    on_time_rate = df_copy['on_time'].mean() * 100
    avg_delay = df_copy[df_copy['delay_minutes'] > 0]['delay_minutes'].mean() if (df_copy['delay_minutes'] > 0).any() else 0
    
    worst_route_series = df_copy.groupby(['source_name', 'destination_name'])['delay_minutes'].mean()
    worst_route = worst_route_series.idxmax() if not worst_route_series.empty else ("N/A", "N/A")

    print(f"On-Time Arrival Rate: {on_time_rate:.2f}%")
    print(f"Average Delay (for late trips): {avg_delay:.1f} min")
    print(f"Route with Highest Avg. Delay: {worst_route[0]} to {worst_route[1]}")
    
    figures = {}

    # Visualizations
    if not df_copy['delay_minutes'].isnull().all():
        fig1 = px.histogram(df_copy, x='delay_minutes', title="Distribution of Arrival Delays (in minutes)")
        print("\nFigure Generated: Distribution of Arrival Delays (Histogram)")
        figures['arrival_delays_distribution'] = fig1.to_json()

    if 'route_type' in df_copy and not df_copy['delay_minutes'].isnull().all():
        delay_by_route_type = df_copy.groupby('route_type')['delay_minutes'].mean().reset_index()
        fig2 = px.bar(delay_by_route_type, x='route_type', y='delay_minutes', title="Average Delay by Route Type")
        print("Figure Generated: Average Delay by Route Type (Bar Chart)")
        figures['avg_delay_by_route_type'] = fig2.to_json()

    return {
        "metrics": {
            "On-Time Arrival Rate": on_time_rate,
            "Average Delay (for late trips)": avg_delay,
            "Route with Highest Avg. Delay": worst_route
        },
        "figures": figures
    }

def shipping_carrier_and_cost_optimization_analysis(df):
    print("\n--- Shipping Carrier and Cost Optimization Analysis ---")
    expected = ['order_id', 'carrier_id', 'shipping_cost']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['shipping_cost'] = pd.to_numeric(df_copy['shipping_cost'], errors='coerce')
    df_copy.dropna(subset=['shipping_cost', 'carrier_id'], inplace=True) # Ensure critical columns are clean

    # Metrics
    total_shipping_cost = df_copy['shipping_cost'].sum()
    avg_shipping_cost = df_copy['shipping_cost'].mean()
    most_used_carrier = df_copy['carrier_id'].mode()[0] if not df_copy['carrier_id'].empty else "N/A"
    
    print(f"Total Shipping Cost: ${total_shipping_cost:,.2f}")
    print(f"Average Shipping Cost: ${avg_shipping_cost:,.2f}")
    print(f"Most Used Carrier: {most_used_carrier}")

    figures = {}

    # Visualizations
    if 'carrier_id' in df_copy and not df_copy['shipping_cost'].isnull().all():
        cost_by_carrier = df_copy.groupby('carrier_id')['shipping_cost'].agg(['sum', 'mean']).reset_index()
        fig1 = px.bar(cost_by_carrier, x='carrier_id', y='sum', title="Total Shipping Cost by Carrier")
        print("\nFigure Generated: Total Shipping Cost by Carrier (Bar Chart)")
        figures['total_shipping_cost_by_carrier'] = fig1.to_json()
    
    if 'carrier_id' in df_copy and not df_copy['shipping_cost'].isnull().all():
        fig2 = px.box(df_copy, x='carrier_id', y='shipping_cost', title="Distribution of Shipping Costs by Carrier")
        print("Figure Generated: Distribution of Shipping Costs by Carrier (Box Plot)")
        figures['shipping_costs_distribution_by_carrier'] = fig2.to_json()

    return {
        "metrics": {
            "Total Shipping Cost": total_shipping_cost,
            "Average Shipping Cost": avg_shipping_cost,
            "Most Used Carrier": most_used_carrier
        },
        "figures": figures
    }

def shipment_dispatch_and_delivery_time_analysis(df):
    print("\n--- Shipment Dispatch and Delivery Time Analysis ---")
    expected = ['shipment_id', 'warehouse_id', 'dispatch_time', 'expected_delivery_time', 'actual_delivery_time']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    # Only critical missing for this analysis: 'shipment_id', 'dispatch_time', 'expected_delivery_time'
    missing = [col for col in ['shipment_id', 'dispatch_time', 'expected_delivery_time'] if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['dispatch_time'] = pd.to_datetime(df_copy['dispatch_time'], errors='coerce')
    df_copy['expected_delivery_time'] = pd.to_datetime(df_copy['expected_delivery_time'], errors='coerce')
    df_copy.dropna(subset=['dispatch_time', 'expected_delivery_time'], inplace=True)
    
    df_copy['planned_transit_hours'] = (df_copy['expected_delivery_time'] - df_copy['dispatch_time']).dt.total_seconds() / 3600
    
    # Metrics
    avg_planned_transit = df_copy['planned_transit_hours'].mean() if not df_copy['planned_transit_hours'].isnull().all() else 0
    print(f"Average Planned Transit Time (Hours): {avg_planned_transit:.2f}")

    figures = {}

    # Visualizations
    if not df_copy['planned_transit_hours'].isnull().all():
        fig1 = px.histogram(df_copy, x='planned_transit_hours', title="Distribution of Planned Transit Times")
        print("\nFigure Generated: Distribution of Planned Transit Times (Histogram)")
        figures['planned_transit_times_distribution'] = fig1.to_json()

    if matched.get('warehouse_id') and 'warehouse_id' in df_copy.columns and not df_copy['planned_transit_hours'].isnull().all():
        transit_by_warehouse = df_copy.groupby('warehouse_id')['planned_transit_hours'].mean().reset_index()
        fig2 = px.bar(transit_by_warehouse, x='warehouse_id', y='planned_transit_hours', title="Average Planned Transit Time by Warehouse")
        print("Figure Generated: Average Planned Transit Time by Warehouse (Bar Chart)")
        figures['avg_planned_transit_by_warehouse'] = fig2.to_json()

    return {
        "metrics": {
            "Average Planned Transit Time (Hours)": avg_planned_transit
        },
        "figures": figures
    }

def logistics_carrier_rate_and_service_analysis(df):
    print("\n--- Logistics Carrier Rate and Service Analysis ---")
    expected = ['carrier_name', 'service_type', 'rate_per_kg', 'max_weight_limit']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['rate_per_kg', 'max_weight_limit']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)

    # Metrics
    cheapest_carrier = df_copy.loc[df_copy['rate_per_kg'].idxmin()]['carrier_name'] if not df_copy['rate_per_kg'].isnull().all() else "N/A"
    highest_capacity_carrier = df_copy.loc[df_copy['max_weight_limit'].idxmax()]['carrier_name'] if not df_copy['max_weight_limit'].isnull().all() else "N/A"
    
    print(f"Cheapest Carrier (by rate/kg): {cheapest_carrier}")
    print(f"Highest Capacity Carrier: {highest_capacity_carrier}")
    
    figures = {}

    # Visualizations
    if 'carrier_name' in df_copy and 'service_type' in df_copy and not df_copy['rate_per_kg'].isnull().all():
        fig1 = px.bar(df_copy.sort_values('rate_per_kg'), x='carrier_name', y='rate_per_kg', color='service_type',
                      title="Rate per Kg by Carrier and Service Type")
        print("\nFigure Generated: Rate per Kg by Carrier and Service Type (Bar Chart)")
        figures['rate_per_kg_by_carrier_service'] = fig1.to_json()
    
    if 'rate_per_kg' in df_copy and 'max_weight_limit' in df_copy and \
       not df_copy['rate_per_kg'].isnull().all() and not df_copy['max_weight_limit'].isnull().all():
        fig2 = px.scatter(df_copy, x='rate_per_kg', y='max_weight_limit', color='carrier_name',
                          title="Max Weight Limit vs. Rate per Kg")
        print("Figure Generated: Max Weight Limit vs. Rate per Kg (Scatter Plot)")
        figures['max_weight_vs_rate_per_kg'] = fig2.to_json()

    return {
        "metrics": {
            "Cheapest Carrier (by rate/kg)": cheapest_carrier,
            "Highest Capacity Carrier": highest_capacity_carrier
        },
        "figures": figures
    }

def warehouse_capacity_and_operational_analysis(df):
    print("\n--- Warehouse Capacity and Operational Analysis ---")
    expected = ['warehouse_name', 'capacity_orders_per_day', 'operational_since']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['capacity_orders_per_day'] = pd.to_numeric(df_copy['capacity_orders_per_day'], errors='coerce')
    df_copy['operational_since'] = pd.to_datetime(df_copy['operational_since'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    total_capacity = df_copy['capacity_orders_per_day'].sum()
    oldest_warehouse = df_copy.loc[df_copy['operational_since'].idxmin()]['warehouse_name'] if not df_copy['operational_since'].empty else "N/A"
    
    print(f"Total Network Capacity (Orders/Day): {total_capacity:,.0f}")
    print(f"Oldest Warehouse: {oldest_warehouse}")

    figures = {}

    # Visualizations
    if 'warehouse_name' in df_copy and not df_copy['capacity_orders_per_day'].isnull().all():
        fig1 = px.pie(df_copy, names='warehouse_name', values='capacity_orders_per_day',
                      title="Share of Network Capacity by Warehouse")
        print("\nFigure Generated: Share of Network Capacity by Warehouse (Pie Chart)")
        figures['network_capacity_by_warehouse'] = fig1.to_json()

    if 'operational_since' in df_copy and not df_copy['capacity_orders_per_day'].isnull().all():
        # Ensure datetime.now() is available and robustly used
        df_copy['years_operational'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['operational_since']).dt.days / 365.25
        fig2 = px.scatter(df_copy, x='years_operational', y='capacity_orders_per_day',
                          hover_name='warehouse_name', title="Capacity vs. Years Operational")
        print("Figure Generated: Capacity vs. Years Operational (Scatter Plot)")
        figures['capacity_vs_years_operational'] = fig2.to_json()

    return {
        "metrics": {
            "Total Network Capacity (Orders/Day)": total_capacity,
            "Oldest Warehouse": oldest_warehouse
        },
        "figures": figures
    }

def warehouse_stock_movement_and_inventory_analysis(df):
    print("\n--- Warehouse Stock Movement and Inventory Analysis ---")
    expected = ['warehouse_id', 'material_id', 'movement_type', 'quantity', 'movement_date']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['movement_date'] = pd.to_datetime(df_copy['movement_date'], errors='coerce')
    df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
    df_copy.dropna(inplace=True)

    # Metrics
    total_inbound = df_copy[df_copy['movement_type'].str.lower().isin(['in', 'inbound', 'receipt'])]['quantity'].sum()
    total_outbound = df_copy[df_copy['movement_type'].str.lower().isin(['out', 'outbound', 'shipment'])]['quantity'].sum()
    
    print(f"Total Inbound Quantity: {total_inbound:,.0f}")
    print(f"Total Outbound Quantity: {total_outbound:,.0f}")
    
    figures = {}

    # Visualizations
    if 'movement_date' in df_copy and 'movement_type' in df_copy and not df_copy['quantity'].isnull().all():
        movement_over_time = df_copy.groupby([df_copy['movement_date'].dt.to_period('D').astype(str), 'movement_type'])['quantity'].sum().unstack().fillna(0)
        fig1 = px.bar(movement_over_time, title="Inbound vs. Outbound Stock Movement Over Time")
        print("\nFigure Generated: Inbound vs. Outbound Stock Movement Over Time (Bar Chart)")
        figures['inbound_outbound_movement_over_time'] = fig1.to_json()
    
    if 'material_id' in df_copy and not df_copy['quantity'].isnull().all():
        movement_by_material = df_copy.groupby('material_id')['quantity'].sum().nlargest(15).reset_index()
        fig2 = px.bar(movement_by_material, x='material_id', y='quantity', title="Top 15 Materials by Movement Volume")
        print("Figure Generated: Top 15 Materials by Movement Volume (Bar Chart)")
        figures['top_materials_by_movement_volume'] = fig2.to_json()

    return {
        "metrics": {
            "Total Inbound Quantity": total_inbound,
            "Total Outbound Quantity": total_outbound
        },
        "figures": figures
    }

def purchase_order_and_supplier_delivery_performance_analysis(df):
    print("\n--- Purchase Order and Supplier Delivery Performance Analysis ---")
    expected = ['po_id', 'supplier_id', 'order_date', 'expected_delivery_date', 'total_cost', 'actual_delivery_date']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None] # Check all columns
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['total_cost'] = pd.to_numeric(df_copy['total_cost'], errors='coerce')
    df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], errors='coerce')
    df_copy['expected_delivery_date'] = pd.to_datetime(df_copy['expected_delivery_date'], errors='coerce')
    df_copy['actual_delivery_date'] = pd.to_datetime(df_copy['actual_delivery_date'], errors='coerce')
    
    df_copy.dropna(subset=['total_cost', 'order_date', 'expected_delivery_date', 'actual_delivery_date', 'supplier_id'], inplace=True) # Ensure critical columns are clean

    # Metrics
    total_spend = df_copy['total_cost'].sum()
    top_supplier = df_copy.groupby('supplier_id')['total_cost'].sum().idxmax() if not df_copy['supplier_id'].empty else "N/A"
    
    print(f"Total Purchase Order Spend: ${total_spend:,.2f}")
    print(f"Top Supplier by Spend: {top_supplier}")

    figures = {}

    # Visualizations
    if 'supplier_id' in df_copy and not df_copy['total_cost'].isnull().all():
        spend_by_supplier = df_copy.groupby('supplier_id')['total_cost'].sum().nlargest(15).reset_index()
        fig1 = px.bar(spend_by_supplier, x='supplier_id', y='total_cost', title="Top 15 Suppliers by PO Spend")
        print("\nFigure Generated: Top 15 Suppliers by PO Spend (Bar Chart)")
        figures['top_suppliers_by_po_spend'] = fig1.to_json()
    
    if 'actual_delivery_date' in df_copy.columns and 'expected_delivery_date' in df_copy.columns:
        df_copy['on_time'] = df_copy['actual_delivery_date'] <= df_copy['expected_delivery_date']
        otd_by_supplier = df_copy.groupby('supplier_id')['on_time'].mean().mul(100).reset_index()
        fig2 = px.bar(otd_by_supplier, x='supplier_id', y='on_time', title="On-Time Delivery Rate by Supplier")
        print("Figure Generated: On-Time Delivery Rate by Supplier (Bar Chart)")
        figures['otd_rate_by_supplier'] = fig2.to_json()
    else:
        print("\n'actual_delivery_date' column is missing for On-Time Delivery analysis.")

    return {
        "metrics": {
            "Total Purchase Order Spend": total_spend,
            "Top Supplier by Spend": top_supplier
        },
        "figures": figures
    }

def purchase_order_line_item_cost_analysis(df):
    print("\n--- Purchase Order Line Item Cost Analysis ---")
    expected = ['po_id', 'material_id', 'quantity', 'unit_cost']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
    df_copy['unit_cost'] = pd.to_numeric(df_copy['unit_cost'], errors='coerce')
    df_copy.dropna(inplace=True)
    df_copy['line_total'] = df_copy['quantity'] * df_copy['unit_cost']
    
    # Metrics
    avg_unit_cost = df_copy['unit_cost'].mean()
    most_expensive_material = df_copy.groupby('material_id')['line_total'].sum().idxmax() if 'material_id' in df_copy and not df_copy['line_total'].isnull().all() else "N/A"
    
    print(f"Average Unit Cost: ${avg_unit_cost:,.2f}")
    print(f"Most Expensive Material (Total Spend): {most_expensive_material}")
    
    figures = {}

    # Visualizations
    if 'material_id' in df_copy and not df_copy['line_total'].isnull().all():
        cost_by_material = df_copy.groupby('material_id')['line_total'].sum().nlargest(20).reset_index()
        fig1 = px.bar(cost_by_material, x='material_id', y='line_total', title="Top 20 Materials by Total Spend")
        print("\nFigure Generated: Top 20 Materials by Total Spend (Bar Chart)")
        figures['top_materials_by_total_spend'] = fig1.to_json()
    
    if not df_copy['quantity'].isnull().all() and not df_copy['unit_cost'].isnull().all():
        fig2 = px.scatter(df_copy, x='quantity', y='unit_cost', hover_name='material_id',
                          title="Unit Cost vs. Quantity Ordered (Volume Discount Analysis)")
        print("Figure Generated: Unit Cost vs. Quantity Ordered (Scatter Plot)")
        figures['unit_cost_vs_quantity_ordered'] = fig2.to_json()

    return {
        "metrics": {
            "Average Unit Cost": avg_unit_cost,
            "Most Expensive Material (Total Spend)": most_expensive_material
        },
        "figures": figures
    }

def order_fulfillment_process_analysis(df):
    print("\n--- Order Fulfillment Process Analysis ---")
    expected = ['fulfillment_id', 'sales_order_id', 'shipment_id']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    
    # Metrics
    num_fulfillments = df_copy['fulfillment_id'].nunique()
    num_orders = df_copy['sales_order_id'].nunique()
    shipments_per_order = df_copy['shipment_id'].nunique() / num_orders if num_orders > 0 else 0
    
    print(f"Total Fulfillments: {num_fulfillments:,}")
    print(f"Total Sales Orders: {num_orders:,}")
    print(f"Avg. Shipments per Order: {shipments_per_order:.2f}")

    figures = {}
    print("\nThis analysis is primarily about linking IDs and counts. Visualizations would typically be more insightful with additional time-series or cost data.")
    print("Sample Data Head:")
    print(df_copy.head().to_string())

    return {
        "metrics": {
            "Total Fulfillments": num_fulfillments,
            "Total Sales Orders": num_orders,
            "Avg. Shipments per Order": shipments_per_order
        },
        "figures": figures
    }

def inventory_quantity_on_hand_analysis(df):
    print("\n--- Inventory Quantity on Hand Analysis ---")
    expected = ['material_id', 'warehouse_id', 'quantity_on_hand', 'last_stocktake_date']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['quantity_on_hand'] = pd.to_numeric(df_copy['quantity_on_hand'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    total_inventory = df_copy['quantity_on_hand'].sum()
    top_material = df_copy.groupby('material_id')['quantity_on_hand'].sum().idxmax() if 'material_id' in df_copy and not df_copy['quantity_on_hand'].isnull().all() else "N/A"
    
    print(f"Total Quantity on Hand: {total_inventory:,.0f}")
    print(f"Material with Highest Stock: {top_material}")
    
    figures = {}

    # Visualizations
    if 'warehouse_id' in df_copy and not df_copy['quantity_on_hand'].isnull().all():
        stock_by_warehouse = df_copy.groupby('warehouse_id')['quantity_on_hand'].sum().reset_index()
        fig1 = px.pie(stock_by_warehouse, names='warehouse_id', values='quantity_on_hand', title="Inventory Distribution by Warehouse")
        print("\nFigure Generated: Inventory Distribution by Warehouse (Pie Chart)")
        figures['inventory_distribution_by_warehouse'] = fig1.to_json()
    
    if 'material_id' in df_copy and not df_copy['quantity_on_hand'].isnull().all():
        top_items = df_copy.groupby('material_id')['quantity_on_hand'].sum().nlargest(20).reset_index()
        fig2 = px.bar(top_items, x='material_id', y='quantity_on_hand', title="Top 20 Materials by Quantity on Hand")
        print("Figure Generated: Top 20 Materials by Quantity on Hand (Bar Chart)")
        figures['top_materials_by_quantity_on_hand'] = fig2.to_json()

    return {
        "metrics": {
            "Total Quantity on Hand": total_inventory,
            "Material with Highest Stock": top_material
        },
        "figures": figures
    }

def shipment_on_time_delivery_performance_analysis(df):
    print("\n--- Shipment On-Time Delivery Performance Analysis ---")
    expected = ['carrier_id', 'origin', 'destination', 'ship_date', 'expected_delivery_date', 'actual_delivery_date']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['ship_date', 'expected_delivery_date', 'actual_delivery_date']:
        df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
    df_copy.dropna(subset=['expected_delivery_date', 'actual_delivery_date'], inplace=True)
    
    df_copy['on_time'] = df_copy['actual_delivery_date'] <= df_copy['expected_delivery_date']
    
    # Metrics
    overall_otd_rate = df_copy['on_time'].mean() * 100
    best_carrier_series = df_copy.groupby('carrier_id')['on_time'].mean()
    best_carrier = best_carrier_series.idxmax() if not best_carrier_series.empty else "N/A"
    
    print(f"Overall On-Time Delivery Rate: {overall_otd_rate:.2f}%")
    print(f"Best Carrier by OTD Rate: {best_carrier}")
    
    figures = {}

    # Visualizations
    if 'carrier_id' in df_copy and not df_copy['on_time'].isnull().all():
        otd_by_carrier = df_copy.groupby('carrier_id')['on_time'].mean().mul(100).reset_index()
        fig1 = px.bar(otd_by_carrier, x='carrier_id', y='on_time', title="On-Time Delivery Rate by Carrier")
        print("\nFigure Generated: On-Time Delivery Rate by Carrier (Bar Chart)")
        figures['otd_rate_by_carrier'] = fig1.to_json()

    if 'actual_delivery_date' in df_copy.columns and not df_copy['on_time'].isnull().all():
        df_copy['delivery_day_of_week'] = df_copy['actual_delivery_date'].dt.day_name()
        otd_by_day = df_copy.groupby('delivery_day_of_week')['on_time'].mean().mul(100).reset_index()
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        otd_by_day['delivery_day_of_week'] = pd.Categorical(otd_by_day['delivery_day_of_week'], categories=day_order, ordered=True)
        otd_by_day = otd_by_day.sort_values('delivery_day_of_week')
        fig2 = px.bar(otd_by_day, x='delivery_day_of_week', y='on_time', title="On-Time Delivery Rate by Day of the Week")
        print("Figure Generated: On-Time Delivery Rate by Day of the Week (Bar Chart)")
        figures['otd_rate_by_day_of_week'] = fig2.to_json()

    return {
        "metrics": {
            "Overall On-Time Delivery Rate": overall_otd_rate,
            "Best Carrier by OTD Rate": best_carrier
        },
        "figures": figures
    }

def late_delivery_and_order_value_correlation_analysis(df):
    print("\n--- Late Delivery and Order Value Correlation Analysis ---")
    expected = ['shipping_mode', 'late_delivery_flag', 'order_value']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['late_delivery_flag'] = pd.to_numeric(df_copy['late_delivery_flag'], errors='coerce') # Ensure 0/1 or True/False
    df_copy['order_value'] = pd.to_numeric(df_copy['order_value'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    late_rate = df_copy['late_delivery_flag'].mean() * 100
    avg_value_late = df_copy[df_copy['late_delivery_flag'] == 1]['order_value'].mean()
    avg_value_on_time = df_copy[df_copy['late_delivery_flag'] == 0]['order_value'].mean()
    
    print(f"Overall Late Delivery Rate: {late_rate:.2f}%")
    print(f"Avg. Order Value (Late): ${avg_value_late:,.2f}")
    print(f"Avg. Order Value (On-Time): ${avg_value_on_time:,.2f}")
    
    figures = {}

    # Visualizations
    if not df_copy['late_delivery_flag'].isnull().all() and not df_copy['order_value'].isnull().all():
        fig1 = px.box(df_copy, x='late_delivery_flag', y='order_value', title="Order Value by Delivery Status")
        fig1.update_xaxes(tickvals=[0, 1], ticktext=['On-Time', 'Late']) # Improve readability
        print("\nFigure Generated: Order Value by Delivery Status (Box Plot)")
        figures['order_value_by_delivery_status'] = fig1.to_json()
    
    if 'shipping_mode' in df_copy and not df_copy['late_delivery_flag'].isnull().all():
        late_by_mode = df_copy.groupby('shipping_mode')['late_delivery_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(late_by_mode, x='shipping_mode', y='late_delivery_flag', title="Late Delivery Rate by Shipping Mode")
        print("Figure Generated: Late Delivery Rate by Shipping Mode (Bar Chart)")
        figures['late_delivery_rate_by_shipping_mode'] = fig2.to_json()

    return {
        "metrics": {
            "Overall Late Delivery Rate": late_rate,
            "Avg. Order Value (Late)": avg_value_late,
            "Avg. Order Value (On-Time)": avg_value_on_time
        },
        "figures": figures
    }

def driver_trip_performance_and_fuel_efficiency_analysis(df):
    print("\n--- Driver Trip Performance and Fuel Efficiency Analysis ---")
    expected = ['driver_id', 'start_time', 'end_time', 'distance_km', 'fuel_consumed_liters']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['distance_km', 'fuel_consumed_liters']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    # Avoid division by zero for km_per_liter
    df_copy['km_per_liter'] = df_copy.apply(lambda row: row['distance_km'] / row['fuel_consumed_liters'] if row['fuel_consumed_liters'] > 0 else np.nan, axis=1)
    
    avg_efficiency = df_copy['km_per_liter'].mean() if not df_copy['km_per_liter'].isnull().all() else 0
    most_efficient_driver_series = df_copy.groupby('driver_id')['km_per_liter'].mean()
    most_efficient_driver = most_efficient_driver_series.idxmax() if not most_efficient_driver_series.empty else "N/A"
    
    print(f"Average Fuel Efficiency (km/L): {avg_efficiency:.2f}")
    print(f"Most Efficient Driver: {most_efficient_driver}")
    
    figures = {}

    # Visualizations
    if 'driver_id' in df_copy and not df_copy['km_per_liter'].isnull().all():
        efficiency_by_driver = df_copy.groupby('driver_id')['km_per_liter'].mean().nlargest(15).reset_index()
        fig1 = px.bar(efficiency_by_driver, x='driver_id', y='km_per_liter', title="Top 15 Most Fuel-Efficient Drivers")
        print("\nFigure Generated: Top 15 Most Fuel-Efficient Drivers (Bar Chart)")
        figures['top_fuel_efficient_drivers'] = fig1.to_json()
    
    if not df_copy['distance_km'].isnull().all() and not df_copy['fuel_consumed_liters'].isnull().all():
        # Removed trendline='ols' as requested (no ML)
        fig2 = px.scatter(df_copy, x='distance_km', y='fuel_consumed_liters', hover_name='driver_id',
                          title="Fuel Consumed vs. Distance Traveled")
        print("Figure Generated: Fuel Consumed vs. Distance Traveled (Scatter Plot)")
        figures['fuel_consumed_vs_distance_traveled'] = fig2.to_json()

    return {
        "metrics": {
            "Average Fuel Efficiency (km/L)": avg_efficiency,
            "Most Efficient Driver": most_efficient_driver
        },
        "figures": figures
    }

def barcode_scan_and_shipment_tracking_analysis(df):
    print("\n--- Barcode Scan and Shipment Tracking Analysis ---")
    expected = ['shipment_id', 'barcode', 'scan_time', 'scan_type', 'location_center']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['scan_time'] = pd.to_datetime(df_copy['scan_time'], errors='coerce')
    df_copy.dropna(inplace=True)

    # Metrics are more qualitative here, based on counts and unique values
    total_scans = len(df_copy)
    unique_shipments_tracked = df_copy['shipment_id'].nunique()
    
    print(f"Total Scans Recorded: {total_scans}")
    print(f"Unique Shipments Tracked: {unique_shipments_tracked}")

    # UI for tracing a shipment (adapted for console)
    print("\n--- Trace a Shipment (Sample Output) ---")
    if not df_copy['shipment_id'].empty:
        # Just take the first unique shipment ID for console demo
        sample_shipment_to_trace = df_copy['shipment_id'].iloc[0]
        shipment_journey = df_copy[df_copy['shipment_id'] == sample_shipment_to_trace].sort_values('scan_time')
        print(f"Journey for Sample Shipment ID: {sample_shipment_to_trace}")
        print(shipment_journey[['scan_time', 'scan_type', 'location_center']].to_string())
    else:
        print("No shipments available to trace.")

    figures = {}

    # Visualizations
    if 'location_center' in df_copy and not df_copy['location_center'].isnull().all():
        scans_by_location = df_copy['location_center'].value_counts().reset_index(name='count')
        fig1 = px.bar(scans_by_location, x='index', y='count', title="Number of Scans by Location")
        print("\nFigure Generated: Number of Scans by Location (Bar Chart)")
        figures['scans_by_location'] = fig1.to_json()

    if 'scan_type' in df_copy and not df_copy['scan_type'].isnull().all():
        scans_by_type = df_copy['scan_type'].value_counts().reset_index(name='count')
        fig2 = px.pie(scans_by_type, names='index', values='count', title="Distribution of Scan Types")
        print("Figure Generated: Distribution of Scan Types (Pie Chart)")
        figures['scan_types_distribution'] = fig2.to_json()

    return {
        "metrics": {
            "Total Scans Recorded": total_scans,
            "Unique Shipments Tracked": unique_shipments_tracked
        },
        "figures": figures
    }

def logistics_route_optimization_analysis(df):
    print("\n--- Logistics Route Optimization Analysis ---")
    expected = ['route_id', 'start_center', 'end_center', 'distance_km', 'travel_time_min', 'route_type']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['distance_km', 'travel_time_min']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    # Handle division by zero for speed_kmh
    df_copy['speed_kmh'] = df_copy.apply(lambda row: row['distance_km'] / (row['travel_time_min'] / 60) if row['travel_time_min'] > 0 else np.nan, axis=1)
    
    avg_speed = df_copy['speed_kmh'].mean() if not df_copy['speed_kmh'].isnull().all() else 0
    longest_route_id = df_copy.loc[df_copy['distance_km'].idxmax()]['route_id'] if not df_copy['distance_km'].isnull().all() else "N/A"
    
    print(f"Average Route Speed (km/h): {avg_speed:.2f}")
    print(f"Longest Route by Distance: {longest_route_id}")
    
    figures = {}

    # Visualizations
    if not df_copy['distance_km'].isnull().all() and not df_copy['travel_time_min'].isnull().all():
        fig1 = px.scatter(df_copy, x='distance_km', y='travel_time_min', color='route_type',
                          hover_name='route_id', title="Travel Time vs. Distance by Route Type")
        print("\nFigure Generated: Travel Time vs. Distance by Route Type (Scatter Plot)")
        figures['travel_time_vs_distance'] = fig1.to_json()

    if 'route_type' in df_copy and not df_copy['speed_kmh'].isnull().all():
        speed_by_type = df_copy.groupby('route_type')['speed_kmh'].mean().reset_index()
        fig2 = px.bar(speed_by_type, x='route_type', y='speed_kmh', title="Average Speed by Route Type")
        print("Figure Generated: Average Speed by Route Type (Bar Chart)")
        figures['avg_speed_by_route_type'] = fig2.to_json()
    
    return {
        "metrics": {
            "Average Route Speed (km/h)": avg_speed,
            "Longest Route by Distance": longest_route_id
        },
        "figures": figures
    }

def package_delivery_delay_analysis(df):
    print("\n--- Package Delivery Delay Analysis ---")
    expected = ['package_id', 'delivery_time', 'pickup_time', 'courier_id', 'delay_minutes']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['delay_minutes'] = pd.to_numeric(df_copy['delay_minutes'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    avg_delay = df_copy[df_copy['delay_minutes'] > 0]['delay_minutes'].mean() if (df_copy['delay_minutes'] > 0).any() else 0
    on_time_rate = (df_copy['delay_minutes'] <= 0).mean() * 100
    worst_courier_series = df_copy.groupby('courier_id')['delay_minutes'].mean()
    worst_courier = worst_courier_series.idxmax() if not worst_courier_series.empty else "N/A"
    
    print(f"On-Time Rate: {on_time_rate:.2f}%")
    print(f"Average Delay (for late packages): {avg_delay:.1f} min")
    print(f"Courier with Highest Avg. Delay: {worst_courier}")
    
    figures = {}

    # Visualizations
    if not df_copy['delay_minutes'].isnull().all():
        fig1 = px.histogram(df_copy, x='delay_minutes', title="Distribution of Delivery Delays")
        print("\nFigure Generated: Distribution of Delivery Delays (Histogram)")
        figures['delivery_delays_distribution'] = fig1.to_json()
    
    if 'courier_id' in df_copy and not df_copy['delay_minutes'].isnull().all():
        delay_by_courier = df_copy.groupby('courier_id')['delay_minutes'].mean().reset_index()
        fig2 = px.bar(delay_by_courier, x='courier_id', y='delay_minutes', title="Average Delay by Courier")
        print("Figure Generated: Average Delay by Courier (Bar Chart)")
        figures['avg_delay_by_courier'] = fig2.to_json()

    return {
        "metrics": {
            "On-Time Rate": on_time_rate,
            "Average Delay (for late packages)": avg_delay,
            "Courier with Highest Avg. Delay": worst_courier
        },
        "figures": figures
    }

def delivery_performance_and_delay_root_cause_analysis(df):
    print("\n--- Delivery Performance and Delay Root Cause Analysis ---")
    expected = ['delivery_id', 'delivery_date', 'delayed_flag', 'delay_reason']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['delayed_flag'] = pd.to_numeric(df_copy['delayed_flag'], errors='coerce') # 1=Delayed, 0=On-Time
    df_copy.dropna(subset=['delayed_flag'], inplace=True)
    
    # Metrics
    delay_rate = df_copy['delayed_flag'].mean() * 100
    top_reason = df_copy[df_copy['delayed_flag'] == 1]['delay_reason'].mode()[0] if (df_copy['delayed_flag'] == 1).any() else "N/A"
    
    print(f"Overall Delay Rate: {delay_rate:.2f}%")
    print(f"Top Reason for Delays: {top_reason}")
    
    figures = {}

    # Visualizations
    if 'delayed_flag' in df_copy and 'delay_reason' in df_copy and (df_copy['delayed_flag'] == 1).any():
        delay_reason_counts = df_copy[df_copy['delayed_flag'] == 1]['delay_reason'].value_counts().reset_index(name='count')
        fig1 = px.pie(delay_reason_counts, names='index', values='count', title="Distribution of Delay Reasons")
        print("\nFigure Generated: Distribution of Delay Reasons (Pie Chart)")
        figures['delay_reasons_distribution'] = fig1.to_json()
    
    if 'delivery_date' in df_copy.columns and not df_copy['delayed_flag'].isnull().all():
        df_copy['delivery_date'] = pd.to_datetime(df_copy['delivery_date'], errors='coerce')
        df_copy.dropna(subset=['delivery_date'], inplace=True)
        if not df_copy.empty:
            delays_over_time = df_copy.groupby(df_copy['delivery_date'].dt.to_period('W').astype(str))['delayed_flag'].mean().mul(100).reset_index()
            fig2 = px.line(delays_over_time, x='delivery_date', y='delayed_flag', title="Delay Rate Over Time (Weekly)")
            print("Figure Generated: Delay Rate Over Time (Line Chart)")
            figures['delay_rate_over_time'] = fig2.to_json()

    return {
        "metrics": {
            "Overall Delay Rate": delay_rate,
            "Top Reason for Delays": top_reason
        },
        "figures": figures
    }

def warehouse_inventory_reorder_level_analysis(df):
    print("\n--- Warehouse Inventory Reorder Level Analysis ---")
    expected = ['warehouse_id', 'product_id', 'on_hand', 'reorder_level', 'reorder_qty']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['on_hand', 'reorder_level', 'reorder_qty']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    df_copy['needs_reorder'] = df_copy['on_hand'] <= df_copy['reorder_level']
    
    # Metrics
    items_to_reorder = df_copy['needs_reorder'].sum()
    perc_to_reorder = df_copy['needs_reorder'].mean() * 100
    
    print(f"Number of Items to Reorder: {items_to_reorder:,}")
    print(f"% of SKU's Needing Reorder: {perc_to_reorder:.2f}%")
    
    figures = {}

    # Visualizations
    if not df_copy['needs_reorder'].isnull().all():
        reorder_status = df_copy['needs_reorder'].value_counts().reset_index(name='count')
        fig1 = px.pie(reorder_status, names='index', values='count', title="Stock Status vs. Reorder Level")
        fig1.update_traces(hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent:.1%}<extra></extra>')
        print("\nFigure Generated: Stock Status vs. Reorder Level (Pie Chart)")
        figures['stock_status_vs_reorder_level'] = fig1.to_json()
    
    if 'warehouse_id' in df_copy and not df_copy['needs_reorder'].isnull().all():
        reorder_by_warehouse = df_copy.groupby('warehouse_id')['needs_reorder'].sum().reset_index()
        fig2 = px.bar(reorder_by_warehouse, x='warehouse_id', y='needs_reorder', title="Number of Items to Reorder by Warehouse")
        print("Figure Generated: Number of Items to Reorder by Warehouse (Bar Chart)")
        figures['items_to_reorder_by_warehouse'] = fig2.to_json()

    return {
        "metrics": {
            "Number of Items to Reorder": items_to_reorder,
            "% of SKU's Needing Reorder": perc_to_reorder
        },
        "figures": figures
    }

def supplier_lead_time_and_reliability_analysis(df):
    print("\n--- Supplier Lead Time and Reliability Analysis ---")
    expected = ['supplier_name', 'lead_time_days', 'on_time_delivery_rate']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['lead_time_days', 'on_time_delivery_rate']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    avg_lead_time = df_copy['lead_time_days'].mean()
    avg_otd_rate = df_copy['on_time_delivery_rate'].mean()
    
    print(f"Average Lead Time (Days): {avg_lead_time:.1f}")
    print(f"Average On-Time Delivery Rate: {avg_otd_rate:.2f}%")
    
    figures = {}

    # Visualizations
    if not df_copy['lead_time_days'].isnull().all() and not df_copy['on_time_delivery_rate'].isnull().all():
        fig1 = px.scatter(df_copy, x='lead_time_days', y='on_time_delivery_rate', hover_name='supplier_name',
                          title="On-Time Rate vs. Lead Time by Supplier")
        print("\nFigure Generated: On-Time Rate vs. Lead Time by Supplier (Scatter Plot)")
        figures['otd_vs_lead_time_by_supplier'] = fig1.to_json()
    
    if 'supplier_name' in df_copy and not df_copy['on_time_delivery_rate'].isnull().all():
        top_suppliers_otd = df_copy.nlargest(15, 'on_time_delivery_rate')
        fig2 = px.bar(top_suppliers_otd, x='supplier_name', y='on_time_delivery_rate', title="Top 15 Suppliers by On-Time Delivery Rate")
        print("Figure Generated: Top 15 Suppliers by On-Time Delivery Rate (Bar Chart)")
        figures['top_suppliers_by_otd'] = fig2.to_json()

    return {
        "metrics": {
            "Average Lead Time (Days)": avg_lead_time,
            "Average On-Time Delivery Rate": avg_otd_rate
        },
        "figures": figures
    }

def freight_haulage_and_truck_load_analysis(df):
    print("\n--- Freight Haulage and Truck Load Analysis ---")
    expected = ['truck_id', 'driver_id', 'departure_time', 'arrival_time', 'load_weight_tonnes']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['departure_time'] = pd.to_datetime(df_copy['departure_time'], errors='coerce')
    df_copy['arrival_time'] = pd.to_datetime(df_copy['arrival_time'], errors='coerce')
    df_copy['load_weight_tonnes'] = pd.to_numeric(df_copy['load_weight_tonnes'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    total_weight_hauled = df_copy['load_weight_tonnes'].sum()
    avg_load_weight = df_copy['load_weight_tonnes'].mean()
    
    print(f"Total Weight Hauled (Tonnes): {total_weight_hauled:,.2f}")
    print(f"Average Load Weight (Tonnes): {avg_load_weight:.2f}")
    
    figures = {}

    # Visualizations
    if 'truck_id' in df_copy and not df_copy['load_weight_tonnes'].isnull().all():
        weight_by_truck = df_copy.groupby('truck_id')['load_weight_tonnes'].sum().nlargest(15).reset_index()
        fig1 = px.bar(weight_by_truck, x='truck_id', y='load_weight_tonnes', title="Top 15 Trucks by Total Weight Hauled")
        print("\nFigure Generated: Top 15 Trucks by Total Weight Hauled (Bar Chart)")
        figures['top_trucks_by_weight_hauled'] = fig1.to_json()
    
    if 'departure_time' in df_copy.columns and 'arrival_time' in df_copy.columns and not df_copy['load_weight_tonnes'].isnull().all():
        df_copy['trip_duration_hours'] = (df_copy['arrival_time'] - df_copy['departure_time']).dt.total_seconds() / 3600
        fig2 = px.scatter(df_copy, x='trip_duration_hours', y='load_weight_tonnes', hover_name='truck_id',
                          title="Load Weight vs. Trip Duration")
        print("Figure Generated: Load Weight vs. Trip Duration (Scatter Plot)")
        figures['load_weight_vs_trip_duration'] = fig2.to_json()

    return {
        "metrics": {
            "Total Weight Hauled (Tonnes)": total_weight_hauled,
            "Average Load Weight (Tonnes)": avg_load_weight
        },
        "figures": figures
    }

def inbound_logistics_and_vendor_quality_analysis(df):
    print("\n--- Inbound Logistics and Vendor Quality Analysis ---")
    expected = ['vendor_id', 'warehouse_id', 'order_date', 'receive_date', 'quantity_received', 'quality_check_flag']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['order_date'] = pd.to_datetime(df_copy['order_date'], errors='coerce')
    df_copy['receive_date'] = pd.to_datetime(df_copy['receive_date'], errors='coerce')
    df_copy['quality_check_flag'] = pd.to_numeric(df_copy['quality_check_flag'], errors='coerce') # 1=Pass, 0=Fail
    df_copy.dropna(inplace=True)
    
    # Metrics
    df_copy['receipt_lead_time'] = (df_copy['receive_date'] - df_copy['order_date']).dt.days
    avg_lead_time = df_copy['receipt_lead_time'].mean() if not df_copy['receipt_lead_time'].isnull().all() else 0
    quality_pass_rate = df_copy['quality_check_flag'].mean() * 100
    
    print(f"Average Receipt Lead Time (Days): {avg_lead_time:.1f}")
    print(f"Quality Check Pass Rate: {quality_pass_rate:.2f}%")
    
    figures = {}

    # Visualizations
    if 'vendor_id' in df_copy and not df_copy['quality_check_flag'].isnull().all():
        pass_rate_by_vendor = df_copy.groupby('vendor_id')['quality_check_flag'].mean().mul(100).reset_index()
        fig1 = px.bar(pass_rate_by_vendor, x='vendor_id', y='quality_check_flag', title="Quality Pass Rate by Vendor")
        print("\nFigure Generated: Quality Pass Rate by Vendor (Bar Chart)")
        figures['quality_pass_rate_by_vendor'] = fig1.to_json()
    
    if 'vendor_id' in df_copy and not df_copy['receipt_lead_time'].isnull().all():
        lead_time_by_vendor = df_copy.groupby('vendor_id')['receipt_lead_time'].mean().reset_index()
        fig2 = px.bar(lead_time_by_vendor, x='vendor_id', y='receipt_lead_time', title="Average Receipt Lead Time by Vendor")
        print("Figure Generated: Average Receipt Lead Time by Vendor (Bar Chart)")
        figures['avg_receipt_lead_time_by_vendor'] = fig2.to_json()

    return {
        "metrics": {
            "Average Receipt Lead Time (Days)": avg_lead_time,
            "Quality Check Pass Rate": quality_pass_rate
        },
        "figures": figures
    }

def vehicle_route_and_on_time_performance_analysis(df):
    print("\n--- Vehicle Route and On-Time Performance Analysis ---")
    expected = ['vehicle_id', 'trip_date', 'start_hub', 'end_hub', 'total_distance_km', 'on_time_delivery']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['on_time_delivery'] = pd.to_numeric(df_copy['on_time_delivery'], errors='coerce') # 1=On-time, 0=Late
    df_copy.dropna(subset=['on_time_delivery'], inplace=True)
    
    # Metrics
    otd_rate = df_copy['on_time_delivery'].mean() * 100
    print(f"Overall On-Time Delivery Rate: {otd_rate:.2f}%")
    
    figures = {}

    # Visualizations
    if 'start_hub' in df_copy and 'end_hub' in df_copy and not df_copy['on_time_delivery'].isnull().all():
        df_copy['route'] = df_copy['start_hub'] + ' to ' + df_copy['end_hub']
        otd_by_route = df_copy.groupby('route')['on_time_delivery'].mean().mul(100).nlargest(15).reset_index()
        fig1 = px.bar(otd_by_route, x='route', y='on_time_delivery', title="Top 15 Routes by On-Time Performance")
        print("\nFigure Generated: Top 15 Routes by On-Time Performance (Bar Chart)")
        figures['top_routes_by_otd'] = fig1.to_json()
    
    if 'vehicle_id' in df_copy and not df_copy['on_time_delivery'].isnull().all():
        otd_by_vehicle = df_copy.groupby('vehicle_id')['on_time_delivery'].mean().mul(100).reset_index()
        fig2 = px.histogram(otd_by_vehicle, x='on_time_delivery', title="Distribution of On-Time Performance Across Vehicles")
        print("Figure Generated: Distribution of On-Time Performance Across Vehicles (Histogram)")
        figures['otd_distribution_across_vehicles'] = fig2.to_json()

    return {
        "metrics": {
            "Overall On-Time Delivery Rate": otd_rate
        },
        "figures": figures
    }

def vehicle_fleet_maintenance_and_capacity_analysis(df):
    print("\n--- Vehicle Fleet Maintenance and Capacity Analysis ---")
    expected = ['vehicle_id', 'vehicle_type', 'capacity_tonnes', 'last_service_date']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['last_service_date'] = pd.to_datetime(df_copy['last_service_date'], errors='coerce')
    df_copy['capacity_tonnes'] = pd.to_numeric(df_copy['capacity_tonnes'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    total_capacity = df_copy['capacity_tonnes'].sum()
    print(f"Total Fleet Capacity (Tonnes): {total_capacity:,.2f}")
    
    figures = {}

    # Visualizations
    if 'vehicle_type' in df_copy and not df_copy['capacity_tonnes'].isnull().all():
        capacity_by_type = df_copy.groupby('vehicle_type')['capacity_tonnes'].sum().reset_index()
        fig1 = px.pie(capacity_by_type, names='vehicle_type', values='capacity_tonnes', title="Fleet Capacity by Vehicle Type")
        print("\nFigure Generated: Fleet Capacity by Vehicle Type (Pie Chart)")
        figures['fleet_capacity_by_vehicle_type'] = fig1.to_json()
    
    if 'last_service_date' in df_copy.columns and not df_copy['capacity_tonnes'].isnull().all():
        df_copy['days_since_service'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['last_service_date']).dt.days
        fig2 = px.scatter(df_copy, x='days_since_service', y='capacity_tonnes', color='vehicle_type',
                          title="Capacity vs. Days Since Last Service")
        print("Figure Generated: Capacity vs. Days Since Last Service (Scatter Plot)")
        figures['capacity_vs_days_since_service'] = fig2.to_json()

    return {
        "metrics": {
            "Total Fleet Capacity (Tonnes)": total_capacity
        },
        "figures": figures
    }

def sales_order_and_pricing_analysis(df):
    print("\n--- Sales Order and Pricing Analysis ---")
    expected = ['order_id', 'customer_id', 'product_id', 'quantity', 'price_per_unit', 'total_price', 'discount']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['quantity', 'price_per_unit', 'total_price', 'discount']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    total_sales = df_copy['total_price'].sum()
    avg_order_value = df_copy.groupby('order_id')['total_price'].sum().mean() if 'order_id' in df_copy else 0
    avg_discount = df_copy['discount'].mean() * 100
    
    print(f"Total Sales: ${total_sales:,.2f}")
    print(f"Average Order Value: ${avg_order_value:,.2f}")
    print(f"Average Discount: {avg_discount:.2f}%")
    
    figures = {}

    # Visualizations
    if 'product_id' in df_copy and not df_copy['total_price'].isnull().all():
        sales_by_product = df_copy.groupby('product_id')['total_price'].sum().nlargest(15).reset_index()
        fig1 = px.bar(sales_by_product, x='product_id', y='total_price', title="Top 15 Products by Sales Revenue")
        print("\nFigure Generated: Top 15 Products by Sales Revenue (Bar Chart)")
        figures['top_products_by_sales_revenue'] = fig1.to_json()
    
    if not df_copy['quantity'].isnull().all() and not df_copy['price_per_unit'].isnull().all() and not df_copy['discount'].isnull().all():
        fig2 = px.scatter(df_copy, x='quantity', y='price_per_unit', color='discount',
                          title="Price per Unit vs. Quantity (Colored by Discount)")
        print("Figure Generated: Price per Unit vs. Quantity (Scatter Plot)")
        figures['price_per_unit_vs_quantity_by_discount'] = fig2.to_json()

    return {
        "metrics": {
            "Total Sales": total_sales,
            "Average Order Value": avg_order_value,
            "Average Discount": avg_discount
        },
        "figures": figures
    }

def shipment_tracking_and_status_update_analysis(df):
    print("\n--- Shipment Tracking and Status Update Analysis ---")
    expected = ['tracking_number', 'carrier', 'origin', 'destination', 'last_update_time']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    df_copy['last_update_time'] = pd.to_datetime(df_copy['last_update_time'], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Analysis
    df_copy['hours_since_update'] = (pd.Timestamp(datetime.datetime.now()) - df_copy['last_update_time']).dt.total_seconds() / 3600
    stale_shipments = df_copy[df_copy['hours_since_update'] > 72] # Example: no update in 3 days
    
    print(f"Number of Stale Shipments (>72h since update): {len(stale_shipments)}")
    if not stale_shipments.empty:
        print("Stale Shipments (head):")
        print(stale_shipments.head().to_string())
    
    figures = {}

    # Visualizations
    if 'carrier' in df_copy and not df_copy['carrier'].isnull().all():
        shipments_by_carrier = df_copy['carrier'].value_counts().reset_index(name='count')
        fig = px.pie(shipments_by_carrier, names='index', values='count', title="Shipment Volume by Carrier")
        print("\nFigure Generated: Shipment Volume by Carrier (Pie Chart)")
        figures['shipment_volume_by_carrier'] = fig.to_json()

    return {
        "metrics": {
            "Number of Stale Shipments (>72h since update)": len(stale_shipments)
        },
        "figures": figures
    }

def package_volumetric_weight_and_zone_analysis(df):
    print("\n--- Package Volumetric Weight and Zone Analysis ---")
    expected = ['weight_g', 'length_cm', 'width_cm', 'height_cm', 'volumetric_weight', 'destination_zone']
    df_copy = df.copy()
    matched = fuzzy_match_column(df_copy, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        return show_general_insights(df_copy, "General Analysis")
    df_copy = df_copy.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['weight_g', 'length_cm', 'width_cm', 'height_cm', 'volumetric_weight']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    df_copy.dropna(inplace=True)
    
    # Metrics
    df_copy['actual_weight_kg'] = df_copy['weight_g'] / 1000
    df_copy['chargeable_weight'] = df_copy[['actual_weight_kg', 'volumetric_weight']].max(axis=1)
    
    avg_chargeable_weight = df_copy['chargeable_weight'].mean()
    print(f"Average Chargeable Weight (kg): {avg_chargeable_weight:.2f}")

    figures = {}

    # Visualizations
    if not df_copy['actual_weight_kg'].isnull().all() and not df_copy['volumetric_weight'].isnull().all():
        fig1 = px.scatter(df_copy, x='actual_weight_kg', y='volumetric_weight', color='destination_zone',
                          title="Actual Weight vs. Volumetric Weight")
        print("\nFigure Generated: Actual Weight vs. Volumetric Weight (Scatter Plot)")
        figures['actual_vs_volumetric_weight'] = fig1.to_json()
    
    if 'destination_zone' in df_copy and not df_copy['actual_weight_kg'].isnull().all() and not df_copy['volumetric_weight'].isnull().all():
        weight_by_zone = df_copy.groupby('destination_zone')[['actual_weight_kg', 'volumetric_weight']].mean().reset_index()
        fig2 = px.bar(weight_by_zone, x='destination_zone', y=['actual_weight_kg', 'volumetric_weight'],
                      barmode='group', title="Average Weights by Destination Zone")
        print("Figure Generated: Average Weights by Destination Zone (Grouped Bar Chart)")
        figures['avg_weights_by_destination_zone'] = fig2.to_json()

    return {
        "metrics": {
            "Average Chargeable Weight (kg)": avg_chargeable_weight
        },
        "figures": figures
    }

def inter_warehouse_stock_transfer_analysis(df):
    print("\n--- Inter-Warehouse Stock Transfer Analysis ---")
    expected = ['transfer_id', 'source_warehouse', 'destination_warehouse', 'product_id', 'quantity', 'transfer_date']
    df_copy = df.copy()
    
    # Check for required columns
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return show_general_insights(df_copy, "General Analysis")
    
    # Data cleaning and conversion
    df_copy['quantity'] = pd.to_numeric(df_copy['quantity'], errors='coerce')
    df_copy['transfer_date'] = pd.to_datetime(df_copy['transfer_date'], errors='coerce')
    df_copy.dropna(subset=['quantity', 'transfer_date'], inplace=True)
    
    # Metrics
    total_quantity = df_copy['quantity'].sum()
    avg_transfer_quantity = df_copy['quantity'].mean()
    
    print(f"Total Quantity Transferred: {total_quantity:.2f}")
    print(f"Average Transfer Quantity: {avg_transfer_quantity:.2f}")
    
    # Transfer patterns by warehouse
    transfers_by_source = df_copy.groupby('source_warehouse')['quantity'].sum().reset_index()
    transfers_by_destination = df_copy.groupby('destination_warehouse')['quantity'].sum().reset_index()
    
    print("\nTop Source Warehouses:")
    print(transfers_by_source.sort_values('quantity', ascending=False).head())
    
    figures = {}
    
    # Monthly transfer trends
    df_copy['month'] = df_copy['transfer_date'].dt.to_period('M')
    monthly_trends = df_copy.groupby('month')['quantity'].sum().reset_index()
    monthly_trends['month'] = monthly_trends['month'].dt.to_timestamp()
    
    return {
        "metrics": {
            "Total Quantity Transferred": total_quantity,
            "Average Transfer Quantity": avg_transfer_quantity
        },
        "figures": figures
    }
def shipment_manifest_and_trip_planning_analysis(df):
    print("\n--- Shipment Manifest and Trip Planning Analysis ---")
    expected = ['shipment_id', 'trip_id', 'origin', 'destination', 'planned_departure', 'planned_arrival', 'actual_departure', 'actual_arrival', 'total_weight_kg', 'total_volume_m3']
    df_copy = df.copy()
    
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return show_general_insights(df_copy, "General Analysis")
    
    # Data conversion
    for time_col in ['planned_departure', 'planned_arrival', 'actual_departure', 'actual_arrival']:
        df_copy[time_col] = pd.to_datetime(df_copy[time_col], errors='coerce')
    
    for numeric_col in ['total_weight_kg', 'total_volume_m3']:
        df_copy[numeric_col] = pd.to_numeric(df_copy[numeric_col], errors='coerce')
    
    df_copy.dropna(subset=['planned_departure', 'actual_departure'], inplace=True)
    
    # Calculate delays
    df_copy['departure_delay_minutes'] = (df_copy['actual_departure'] - df_copy['planned_departure']).dt.total_seconds() / 60
    df_copy['arrival_delay_minutes'] = (df_copy['actual_arrival'] - df_copy['planned_arrival']).dt.total_seconds() / 60
    
    avg_departure_delay = df_copy['departure_delay_minutes'].mean()
    avg_arrival_delay = df_copy['arrival_delay_minutes'].mean()
    total_weight = df_copy['total_weight_kg'].sum()
    
    print(f"Average Departure Delay (minutes): {avg_departure_delay:.2f}")
    print(f"Average Arrival Delay (minutes): {avg_arrival_delay:.2f}")
    print(f"Total Weight Transported (kg): {total_weight:.2f}")
    
    return {
        "metrics": {
            "Average Departure Delay (minutes)": avg_departure_delay,
            "Average Arrival Delay (minutes)": avg_arrival_delay,
            "Total Weight Transported (kg)": total_weight
        },
        "figures": {}
    }
def last_mile_delivery_confirmation_analysis(df):
    print("\n--- Last-Mile Delivery Confirmation Analysis ---")
    expected = ['delivery_id', 'order_id', 'delivery_status', 'confirmation_time', 'delivery_time', 'delivery_agent_id']
    df_copy = df.copy()
    
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return show_general_insights(df_copy, "General Analysis")
    
    # Data conversion
    df_copy['confirmation_time'] = pd.to_datetime(df_copy['confirmation_time'], errors='coerce')
    df_copy['delivery_time'] = pd.to_datetime(df_copy['delivery_time'], errors='coerce')
    
    # Filter for confirmed deliveries
    confirmed_deliveries = df_copy[df_copy['delivery_status'].isin(['confirmed', 'delivered', 'completed'])]
    confirmed_deliveries.dropna(subset=['confirmation_time', 'delivery_time'], inplace=True)
    
    # Calculate confirmation delay
    confirmed_deliveries['confirmation_delay_minutes'] = (confirmed_deliveries['confirmation_time'] - confirmed_deliveries['delivery_time']).dt.total_seconds() / 60
    
    avg_confirmation_delay = confirmed_deliveries['confirmation_delay_minutes'].mean()
    confirmation_rate = len(confirmed_deliveries) / len(df_copy) * 100
    
    print(f"Average Confirmation Delay (minutes): {avg_confirmation_delay:.2f}")
    print(f"Delivery Confirmation Rate: {confirmation_rate:.2f}%")
    
    # Status distribution
    status_counts = df_copy['delivery_status'].value_counts()
    print("\nDelivery Status Distribution:")
    print(status_counts)
    
    return {
        "metrics": {
            "Average Confirmation Delay (minutes)": avg_confirmation_delay,
            "Delivery Confirmation Rate (%)": confirmation_rate
        },
        "figures": {}
    }
import pandas as pd

def load_data(file_path, encoding='utf-8'):
    """
    Load data from a file path (csv or excel) with given encoding.
    Returns a pandas DataFrame or None if loading fails.
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, encoding=encoding)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            print("Unsupported file type. Please use CSV or Excel files.")
            return None
        
        print(f"Data loaded from {file_path} successfully.")
        print(f"Dataset shape: {df.shape}")
        return df
        
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except UnicodeDecodeError:
        print(f"Encoding error. Try different encoding (e.g., latin1, cp1252)")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
def order_delivery_time_estimation_by_shipping_zone(df):
    print("\n--- Order Delivery Time Estimation by Shipping Zone ---")
    
    expected = ['order_id', 'shipping_zone', 'order_placed_time', 'order_delivered_time', 'delivery_status']
    df_copy = df.copy()
    
    # Check for required columns
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        print("Analysis requires: order_id, shipping_zone, order_placed_time, order_delivered_time")
        return {"message": "Missing required columns for delivery time analysis"}
    
    # Data cleaning and conversion
    df_copy['order_placed_time'] = pd.to_datetime(df_copy['order_placed_time'], errors='coerce')
    df_copy['order_delivered_time'] = pd.to_datetime(df_copy['order_delivered_time'], errors='coerce')
    
    # Filter for delivered orders only
    if 'delivery_status' in df_copy.columns:
        df_copy = df_copy[df_copy['delivery_status'].isin(['delivered', 'completed'])]
    
    # Remove rows with missing data
    df_copy.dropna(subset=['order_placed_time', 'order_delivered_time', 'shipping_zone'], inplace=True)
    
    if df_copy.empty:
        print("No valid data found after cleaning.")
        return {"message": "No valid data after cleaning"}
    
    # Calculate delivery duration
    df_copy['delivery_duration_hours'] = (df_copy['order_delivered_time'] - df_copy['order_placed_time']).dt.total_seconds() / 3600
    df_copy['delivery_duration_days'] = df_copy['delivery_duration_hours'] / 24
    
    # Analysis by shipping zone
    zone_analysis = df_copy.groupby('shipping_zone').agg({
        'delivery_duration_hours': ['mean', 'median', 'std', 'min', 'max'],
        'order_id': 'count'
    }).round(2)
    
    zone_analysis.columns = ['avg_hours', 'median_hours', 'std_hours', 'min_hours', 'max_hours', 'total_orders']
    zone_analysis = zone_analysis.reset_index()
    
    # Convert hours to days for better readability
    zone_analysis['avg_days'] = (zone_analysis['avg_hours'] / 24).round(2)
    zone_analysis['median_days'] = (zone_analysis['median_hours'] / 24).round(2)
    
    print("\n📊 Delivery Time Analysis by Shipping Zone:")
    print("="*60)
    for _, row in zone_analysis.iterrows():
        print(f"\n🌎 Zone: {row['shipping_zone']}")
        print(f"   Orders: {row['total_orders']}")
        print(f"   Average: {row['avg_hours']:.1f} hours ({row['avg_days']:.1f} days)")
        print(f"   Median: {row['median_hours']:.1f} hours ({row['median_days']:.1f} days)")
        print(f"   Range: {row['min_hours']:.1f} - {row['max_hours']:.1f} hours")
    
    # Overall statistics
    overall_avg = df_copy['delivery_duration_hours'].mean()
    overall_median = df_copy['delivery_duration_hours'].median()
    total_orders = len(df_copy)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Total Delivered Orders: {total_orders}")
    print(f"   Overall Average Delivery Time: {overall_avg:.1f} hours ({overall_avg/24:.1f} days)")
    print(f"   Overall Median Delivery Time: {overall_median:.1f} hours ({overall_median/24:.1f} days)")
    
    # Fastest and slowest zones
    fastest_zone = zone_analysis.loc[zone_analysis['avg_hours'].idxmin()]
    slowest_zone = zone_analysis.loc[zone_analysis['avg_hours'].idxmax()]
    
    print(f"\n🏆 Performance Insights:")
    print(f"   Fastest Zone: {fastest_zone['shipping_zone']} ({fastest_zone['avg_hours']:.1f} hrs)")
    print(f"   Slowest Zone: {slowest_zone['shipping_zone']} ({slowest_zone['avg_hours']:.1f} hrs)")
    
    return {
        "metrics": {
            "total_orders_analyzed": total_orders,
            "overall_avg_delivery_hours": overall_avg,
            "overall_median_delivery_hours": overall_median,
            "zone_analysis": zone_analysis.to_dict(orient='records'),
            "fastest_zone": fastest_zone['shipping_zone'],
            "slowest_zone": slowest_zone['shipping_zone']
        },
        "figures": {}
    }

def order_fulfillment_cycle_time_analysis(df):
    print("\n--- Order Fulfillment Cycle Time Analysis ---")
    expected = ['order_id', 'order_received_time', 'order_fulfilled_time', 'warehouse_id', 'fulfillment_status']
    df_copy = df.copy()
    
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return show_general_insights(df_copy, "General Analysis")
    
    # Data conversion
    df_copy['order_received_time'] = pd.to_datetime(df_copy['order_received_time'], errors='coerce')
    df_copy['order_fulfilled_time'] = pd.to_datetime(df_copy['order_fulfilled_time'], errors='coerce')
    df_copy.dropna(subset=['order_received_time', 'order_fulfilled_time'], inplace=True)
    
    # Calculate cycle time
    df_copy['fulfillment_cycle_time_hours'] = (df_copy['order_fulfilled_time'] - df_copy['order_received_time']).dt.total_seconds() / 3600
    
    avg_cycle_time = df_copy['fulfillment_cycle_time_hours'].mean()
    median_cycle_time = df_copy['fulfillment_cycle_time_hours'].median()
    
    print(f"Average Fulfillment Cycle Time (hours): {avg_cycle_time:.2f}")
    print(f"Median Fulfillment Cycle Time (hours): {median_cycle_time:.2f}")
    
    # Performance by warehouse
    warehouse_performance = df_copy.groupby('warehouse_id')['fulfillment_cycle_time_hours'].agg(['mean', 'median', 'count']).reset_index()
    print("\nWarehouse Performance:")
    print(warehouse_performance)
    
    return {
        "metrics": {
            "Average Fulfillment Cycle Time (hours)": avg_cycle_time,
            "Median Fulfillment Cycle Time (hours)": median_cycle_time
        },
        "figures": {}
    }
def truck_loading_efficiency_analysis(df):
    print("\n--- Truck Loading Efficiency Analysis ---")
    expected = ['truck_id', 'loading_start_time', 'loading_end_time', 'loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg']
    df_copy = df.copy()
    
    missing = [col for col in expected if col not in df_copy.columns]
    if missing:
        print(f"Missing required columns: {missing}")
        return show_general_insights(df_copy, "General Analysis")
    
    # Data conversion
    df_copy['loading_start_time'] = pd.to_datetime(df_copy['loading_start_time'], errors='coerce')
    df_copy['loading_end_time'] = pd.to_datetime(df_copy['loading_end_time'], errors='coerce')
    
    for col in ['loaded_volume_m3', 'loaded_weight_kg', 'truck_capacity_volume_m3', 'truck_capacity_weight_kg']:
        df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
    
    df_copy.dropna(subset=['loading_start_time', 'loading_end_time'], inplace=True)
    
    # Calculate efficiency metrics
    df_copy['loading_duration_minutes'] = (df_copy['loading_end_time'] - df_copy['loading_start_time']).dt.total_seconds() / 60
    df_copy['volume_utilization'] = df_copy['loaded_volume_m3'] / df_copy['truck_capacity_volume_m3']
    df_copy['weight_utilization'] = df_copy['loaded_weight_kg'] / df_copy['truck_capacity_weight_kg']
    df_copy['loading_efficiency'] = df_copy[['volume_utilization', 'weight_utilization']].max(axis=1)
    
    avg_loading_time = df_copy['loading_duration_minutes'].mean()
    avg_volume_utilization = df_copy['volume_utilization'].mean()
    avg_weight_utilization = df_copy['weight_utilization'].mean()
    avg_loading_efficiency = df_copy['loading_efficiency'].mean()
    
    print(f"Average Loading Duration (minutes): {avg_loading_time:.2f}")
    print(f"Average Volume Utilization: {avg_volume_utilization:.2%}")
    print(f"Average Weight Utilization: {avg_weight_utilization:.2%}")
    print(f"Average Loading Efficiency: {avg_loading_efficiency:.2%}")
    
    return {
        "metrics": {
            "Average Loading Duration (minutes)": avg_loading_time,
            "Average Volume Utilization": avg_volume_utilization,
            "Average Weight Utilization": avg_weight_utilization,
            "Average Loading Efficiency": avg_loading_efficiency
        },
        "figures": {}
    }
def main():
    print("🚚 Logistics Analytics Dashboard")
    file_path = input("Enter path to your logistics data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252): ")
    if not encoding:
        encoding = 'utf-8'
    df = load_data(file_path, encoding=encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")

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
        "order_delivery_time_estimation_by_shipping_zone",
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
        "inter-warehouse_stock_transfer_analysis",
        "shipment_manifest_and_trip_planning_analysis",
        "last-mile_delivery_confirmation_analysis",
        "order_fulfillment_cycle_time_analysis",
        "truck_loading_efficiency_analysis"
    ]

    print("\nSelect Analysis to Perform:")
    for i, option in enumerate(analysis_options):
        print(f"{i}: {option}")

    choice = input("Enter the option number: ")
    try:
        choice = int(choice)
    except ValueError:
        print("Invalid input. Showing General Insights.")
        choice = -1

    if 0 <= choice < len(analysis_options):
        selected = analysis_options[choice]
    else:
        selected = None

    # Use a mapping between option names and their function objects for scalability:
    analysis_function_map = {
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
        "order_delivery_time_estimation_by_shipping_zone": order_delivery_time_estimation_by_shipping_zone,
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
        "inter-warehouse_stock_transfer_analysis": inter_warehouse_stock_transfer_analysis,
        "shipment_manifest_and_trip_planning_analysis": shipment_manifest_and_trip_planning_analysis,
        "last-mile_delivery_confirmation_analysis": last_mile_delivery_confirmation_analysis,
        "order_fulfillment_cycle_time_analysis": order_fulfillment_cycle_time_analysis,
        "truck_loading_efficiency_analysis": truck_loading_efficiency_analysis
    }

    if selected and selected in analysis_function_map:
        analysis_function_map[selected](df)
    else:
        print(f"Analysis option '{selected}' not recognized or not implemented.")
        show_general_insights(df)


    