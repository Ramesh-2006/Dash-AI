import pandas as pd
import numpy as np
from fuzzywuzzy import process
import warnings
import os
warnings.filterwarnings('ignore')


def fuzzy_match_column(df, target_columns):
    """
    Fuzzy matches target columns to available columns in a DataFrame.
    Returns a dictionary mapping target column names to matched column names.
    If no good match is found (score < 70), the value for that target column is None.
    """
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available)
        matched[target] = match if score >= 70 else None
    return matched

def get_key_metrics(df):
    """
    Calculates and returns key metrics about the dataset.
    """
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    metrics = {
        "Total Records": total_records,
        "Total Features": len(df.columns),
        "Numeric Features": len(numeric_cols),
        "Categorical Features": len(categorical_cols)
    }
    return metrics

def get_general_insights_data(df):
    """
    Prepares data for general insights, including descriptions of numeric and categorical features.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    insights = {
        "key_metrics": get_key_metrics(df),
        "numeric_summaries": {col: df[col].describe().to_dict() for col in numeric_cols},
        "categorical_value_counts": {col: df[col].value_counts().head(10).to_dict() for col in categorical_cols},
        "correlation_matrix": df[numeric_cols].corr().to_dict() if len(numeric_cols) >= 2 else {}
    }
    return insights
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

# ========== ANALYSIS FUNCTIONS (Adapted - No Streamlit or Plotly) ==========

def production_data(df):
    """
    Performs production analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['production_id', 'product_code', 'production_date', 'quantity_produced',
                'quantity_defective', 'production_line', 'operator_id', 'cycle_time']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if 'production_date' in df and not pd.api.types.is_datetime64_any_dtype(df['production_date']):
        df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
    
    df = df.dropna(subset=['quantity_produced', 'quantity_defective'])
    df['quantity_produced'] = pd.to_numeric(df['quantity_produced'], errors='coerce')
    df['quantity_defective'] = pd.to_numeric(df['quantity_defective'], errors='coerce')
    df['cycle_time'] = pd.to_numeric(df['cycle_time'], errors='coerce')
    df.dropna(subset=['quantity_produced', 'quantity_defective'], inplace=True)

    total_production = df['quantity_produced'].sum()
    defect_rate = (df['quantity_defective'].sum() / total_production * 100) if total_production > 0 else 0
    avg_cycle_time = df['cycle_time'].mean() if 'cycle_time' in df and not df['cycle_time'].isnull().all() else None

    metrics = {
        "Total Production": total_production,
        "Defect Rate": defect_rate,
        "Production Runs": len(df),
        "Avg Cycle Time": avg_cycle_time
    }

    production_trend_data = None
    if 'production_date' in df and 'quantity_produced' in df:
        production_trend_data = df.groupby('production_date')['quantity_produced'].sum().reset_index()
        production_trend_data.columns = ['production_date', 'quantity_produced']

    line_performance_data = None
    if 'production_line' in df and 'quantity_produced' in df and 'quantity_defective' in df:
        line_performance = df.groupby('production_line').agg(
            quantity_produced=('quantity_produced', 'sum'),
            quantity_defective=('quantity_defective', 'sum')
        ).reset_index()
        line_performance['defect_rate'] = (line_performance['quantity_defective'] / line_performance['quantity_produced']) * 100
        line_performance_data = line_performance.to_dict('records')

    cycle_time_product_data = None
    if 'cycle_time' in df and 'product_code' in df:
        cycle_time_product_data = df[['product_code', 'cycle_time']].dropna().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "production_trend": production_trend_data.to_dict('records') if production_trend_data is not None else None,
            "line_performance": line_performance_data,
            "cycle_time_by_product": cycle_time_product_data
        }
    }


def quality_control_data(df):
    """
    Performs quality control analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['inspection_id', 'product_code', 'inspection_date', 'defect_type',
                'severity', 'inspector_id', 'batch_number', 'corrective_action']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if 'inspection_date' in df and not pd.api.types.is_datetime64_any_dtype(df['inspection_date']):
        df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
    df.dropna(subset=['defect_type', 'severity'], inplace=True)

    total_inspections = len(df)
    unique_defects = df['defect_type'].nunique()
    avg_severity = df['severity'].mean() if 'severity' in df and not df['severity'].isnull().all() else None

    metrics = {
        "Total Quality Inspections": total_inspections,
        "Unique Defect Types": unique_defects,
        "Avg Defect Severity": avg_severity
    }

    defect_counts_data = None
    if 'defect_type' in df:
        defect_counts = df['defect_type'].value_counts().reset_index()
        defect_counts.columns = ['Defect Type', 'Count']
        defect_counts_data = defect_counts.to_dict('records')

    defects_over_time_data = None
    if 'inspection_date' in df:
        defects_over_time = df.groupby('inspection_date').size().reset_index(name='count')
        defects_over_time_data = defects_over_time.to_dict('records')

    severity_by_defect_data = None
    if 'defect_type' in df and 'severity' in df:
        severity_by_defect_data = df[['defect_type', 'severity']].dropna().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "defect_type_frequency": defect_counts_data,
            "defects_over_time": defects_over_time_data,
            "defect_severity_by_type": severity_by_defect_data
        }
    }


def equipment_data(df):
    """
    Performs equipment analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['equipment_id', 'equipment_type', 'last_maintenance', 'next_maintenance',
                'downtime_hours', 'utilization_rate', 'failure_count', 'status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    date_cols = ['last_maintenance', 'next_maintenance']
    for col in date_cols:
        if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    df['downtime_hours'] = pd.to_numeric(df['downtime_hours'], errors='coerce')
    df['utilization_rate'] = pd.to_numeric(df['utilization_rate'], errors='coerce')
    df.dropna(inplace=True)

    total_equipment = len(df)
    avg_downtime = df['downtime_hours'].mean() if not df['downtime_hours'].isnull().all() else None
    avg_utilization = df['utilization_rate'].mean() if not df['utilization_rate'].isnull().all() else None

    metrics = {
        "Total Equipment": total_equipment,
        "Average Downtime (hours)": avg_downtime,
        "Average Utilization (%)": avg_utilization
    }

    status_distribution_data = None
    if 'status' in df:
        status_distribution_data = df['status'].value_counts().reset_index()
        status_distribution_data.columns = ['status', 'count']
        status_distribution_data = status_distribution_data.to_dict('records')

    downtime_by_type_data = None
    if 'equipment_type' in df and 'downtime_hours' in df:
        downtime_by_type_data = df[['equipment_type', 'downtime_hours']].dropna().to_dict('records')

    maintenance_schedule_data = None
    if 'last_maintenance' in df and 'next_maintenance' in df and 'equipment_id' in df:
        maintenance_df = df[['equipment_id', 'last_maintenance', 'next_maintenance']].melt(
            id_vars='equipment_id',
            var_name='Maintenance Type',
            value_name='Date'
        )
        maintenance_schedule_data = maintenance_df.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "equipment_status_distribution": status_distribution_data,
            "downtime_by_equipment_type": downtime_by_type_data,
            "equipment_maintenance_schedule": maintenance_schedule_data
        }
    }


def inventory_data(df):
    """
    Performs inventory analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['sku', 'product_name', 'current_stock', 'min_stock',
                'max_stock', 'lead_time', 'turnover_rate', 'last_order_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if 'last_order_date' in df and not pd.api.types.is_datetime64_any_dtype(df['last_order_date']):
        df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')

    df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
    df['min_stock'] = pd.to_numeric(df['min_stock'], errors='coerce')
    df['max_stock'] = pd.to_numeric(df['max_stock'], errors='coerce')
    df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
    df.dropna(inplace=True)

    total_skus = len(df)
    stockout_risk = len(df[df['current_stock'] < df['min_stock']])
    avg_turnover = df['turnover_rate'].mean() if not df['turnover_rate'].isnull().all() else None

    metrics = {
        "Total SKUs": total_skus,
        "Items Below Min Stock": stockout_risk,
        "Avg Inventory Turnover": avg_turnover
    }

    inventory_distribution_data = None
    if 'current_stock' in df:
        inventory_distribution_data = df['current_stock'].to_list()

    inventory_status_data = None
    if 'current_stock' in df and 'min_stock' in df and 'product_name' in df and 'max_stock' in df:
        df['stock_status'] = np.where(
            df['current_stock'] < df['min_stock'],
            'Below Minimum',
            np.where(
                df['current_stock'] > df['max_stock'],
                'Above Maximum',
                'Normal'
            )
        )
        inventory_status_data = df[['product_name', 'current_stock', 'stock_status']].to_dict('records')

    abc_analysis_data = None
    if 'current_stock' in df and 'product_name' in df:
        df_sorted = df.sort_values('current_stock', ascending=False)
        df_sorted['cumulative_percent'] = df_sorted['current_stock'].cumsum() / df_sorted['current_stock'].sum() * 100
        abc_analysis_data = df_sorted[['cumulative_percent']].reset_index(drop=True).to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "inventory_distribution": inventory_distribution_data,
            "inventory_status_by_product": inventory_status_data,
            "abc_analysis_cumulative_percent": abc_analysis_data
        }
    }


def oee_data(df):
    """
    Performs OEE analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['machine_id', 'shift_date', 'shift', 'availability',
                'performance', 'quality', 'oee', 'planned_production_time']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if 'shift_date' in df and not pd.api.types.is_datetime64_any_dtype(df['shift_date']):
        df['shift_date'] = pd.to_datetime(df['shift_date'], errors='coerce')

    for col in ['availability', 'performance', 'quality', 'oee']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_oee = df['oee'].mean()
    avg_availability = df['availability'].mean()
    avg_performance = df['performance'].mean()
    avg_quality = df['quality'].mean()

    metrics = {
        "Avg OEE": avg_oee,
        "Avg Availability": avg_availability,
        "Avg Performance": avg_performance,
        "Avg Quality": avg_quality
    }

    oee_trend_data = None
    if 'shift_date' in df and 'oee' in df:
        oee_trend = df.groupby('shift_date')['oee'].mean().reset_index()
        oee_trend_data = oee_trend.to_dict('records')

    oee_components_data = None
    if all(col in df for col in ['availability', 'performance', 'quality']):
        components = df[['availability', 'performance', 'quality']].mean().reset_index()
        components.columns = ['Component', 'Value']
        oee_components_data = components.to_dict('records')

    oee_by_shift_data = None
    if 'shift' in df and 'oee' in df:
        oee_by_shift_data = df[['shift', 'oee']].dropna().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "oee_trend": oee_trend_data,
            "oee_components_breakdown": oee_components_data,
            "oee_by_shift": oee_by_shift_data
        }
    }


def energy_data(df):
    """
    Performs energy analysis and returns metrics and dataframes for visualization.
    """
    df = df.copy()
    expected = ['meter_id', 'timestamp', 'energy_consumption', 'cost',
                'machine_id', 'production_volume', 'energy_per_unit']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if 'timestamp' in df and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    for col in ['energy_consumption', 'cost', 'production_volume', 'energy_per_unit']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    total_energy = df['energy_consumption'].sum()
    total_cost = df['cost'].sum()
    avg_energy_per_unit = df['energy_per_unit'].mean() if 'energy_per_unit' in df and not df['energy_per_unit'].isnull().all() else None

    metrics = {
        "Total Energy (kWh)": total_energy,
        "Total Cost ($)": total_cost,
        "Avg Energy per Unit (kWh/unit)": avg_energy_per_unit
    }

    energy_trend_data = None
    if 'timestamp' in df and 'energy_consumption' in df:
        energy_trend = df.groupby('timestamp')['energy_consumption'].sum().reset_index()
        energy_trend_data = energy_trend.to_dict('records')

    energy_efficiency_machine_data = None
    if 'machine_id' in df and 'energy_per_unit' in df:
        energy_efficiency_machine_data = df[['machine_id', 'energy_per_unit']].dropna().to_dict('records')

    energy_vs_production_data = None
    if 'production_volume' in df and 'energy_consumption' in df:
        energy_vs_production_data = df[['production_volume', 'energy_consumption']].dropna().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "energy_consumption_over_time": energy_trend_data,
            "energy_efficiency_by_machine": energy_efficiency_machine_data,
            "energy_vs_production_volume": energy_vs_production_data
        }
    }


def manufacturing_defect_root_cause_and_cost_data(df):
    """
    Performs manufacturing defect root cause and cost analysis.
    """
    df = df.copy()
    expected = ['defect_type', 'defect_location', 'severity', 'repair_cost']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['repair_cost'] = pd.to_numeric(df['repair_cost'], errors='coerce')
    df.dropna(inplace=True)

    total_repair_cost = df['repair_cost'].sum()
    avg_repair_cost = df['repair_cost'].mean()
    most_common_defect = df['defect_type'].mode()[0] if not df['defect_type'].empty else None

    metrics = {
        "Total Repair Cost": total_repair_cost,
        "Average Repair Cost": avg_repair_cost,
        "Most Common Defect": most_common_defect
    }

    cost_by_defect_data = None
    if 'defect_type' in df and 'repair_cost' in df:
        cost_by_defect = df.groupby('defect_type')['repair_cost'].sum().sort_values(ascending=False).reset_index()
        cost_by_defect_data = cost_by_defect.to_dict('records')

    cost_by_location_data = None
    if 'defect_location' in df and 'repair_cost' in df:
        cost_by_location = df.groupby('defect_location')['repair_cost'].sum().sort_values(ascending=False).reset_index()
        cost_by_location_data = cost_by_location.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "cost_by_defect_type": cost_by_defect_data,
            "cost_by_defect_location": cost_by_location_data
        }
    }


def production_efficiency_and_quality_control_data(df):
    """
    Performs production efficiency and quality control analysis.
    """
    df = df.copy()
    expected = ['date', 'product_type', 'units_produced', 'defects', 'production_time_hours', 'down_time_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['units_produced', 'defects', 'production_time_hours', 'down_time_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

    df['defect_rate'] = (df['defects'] / df['units_produced']) * 100
    df['availability'] = (df['production_time_hours'] - df['down_time_hours']) / df['production_time_hours'] * 100
    
    avg_defect_rate = df['defect_rate'].mean()
    avg_availability = df['availability'].mean()
    total_units = df['units_produced'].sum()

    metrics = {
        "Total Units Produced": total_units,
        "Average Defect Rate": avg_defect_rate,
        "Average Availability": avg_availability
    }

    defect_rate_by_product_data = None
    if 'product_type' in df and 'defect_rate' in df:
        defect_rate_by_product = df.groupby('product_type')['defect_rate'].mean().sort_values().reset_index()
        defect_rate_by_product_data = defect_rate_by_product.to_dict('records')

    daily_prod_data = None
    if 'date' in df and 'units_produced' in df and 'defects' in df:
        daily_prod = df.groupby('date')[['units_produced', 'defects']].sum().reset_index()
        daily_prod_data = daily_prod.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "defect_rate_by_product": defect_rate_by_product_data,
            "daily_production_vs_defects": daily_prod_data
        }
    }


def manufacturing_kpi_data(df):
    """
    Performs manufacturing KPI analysis.
    """
    df = df.copy()
    expected = ['productionvolume', 'productioncost', 'supplierquality', 'deliverydelay', 'defectrate', 'maintenancedurasi', 'downtimepercentage', 'workerproductivity', 'safetyincidents']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_defect_rate = df['defectrate'].mean()
    avg_productivity = df['workerproductivity'].mean()
    total_safety_incidents = df['safetyincidents'].sum()

    metrics = {
        "Average Defect Rate": avg_defect_rate,
        "Average Worker Productivity": avg_productivity,
        "Total Safety Incidents": total_safety_incidents
    }

    correlation_matrix_data = None
    if len(expected) >= 2:
        corr_matrix = df[expected].corr()
        correlation_matrix_data = corr_matrix.to_dict('index')

    defect_vs_productivity_data = None
    if 'workerproductivity' in df and 'defectrate' in df and 'productioncost' in df:
        defect_vs_productivity_data = df[['workerproductivity', 'defectrate', 'productioncost']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "correlation_matrix_kpis": correlation_matrix_data,
            "defect_rate_vs_worker_productivity": defect_vs_productivity_data
        }
    }


def real_time_production_monitoring_and_predictive_maintenance_data(df):
    """
    Performs real-time production monitoring and predictive maintenance analysis.
    """
    df = df.copy()
    expected = ['timestamp', 'machine_id', 'temperature_c', 'vibration_hz', 'power_consumption_kw', 'predictive_maintenance_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['temperature_c', 'vibration_hz', 'power_consumption_kw', 'predictive_maintenance_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('timestamp').dropna()

    latest_temp = df['temperature_c'].iloc[-1] if not df['temperature_c'].empty else None
    avg_vibration = df['vibration_hz'].mean() if not df['vibration_hz'].empty else None
    avg_pred_score = df['predictive_maintenance_score'].mean() if not df['predictive_maintenance_score'].empty else None

    metrics = {
        "Latest Temperature (°C)": latest_temp,
        "Average Vibration (Hz)": avg_vibration,
        "Average Predictive Maintenance Score": avg_pred_score
    }

    machine_options = df['machine_id'].unique().tolist() if 'machine_id' in df else []
    
    sensor_readings_data = {}
    if 'machine_id' in df and 'timestamp' in df and not df.empty:
        for machine_id in machine_options:
            df_machine = df[df['machine_id'] == machine_id].copy()
            if not df_machine.empty:
                sensor_readings_data[machine_id] = df_machine[['timestamp', 'temperature_c', 'vibration_hz', 'power_consumption_kw']].melt(
                    id_vars='timestamp', var_name='variable', value_name='value'
                ).to_dict('records')

    temp_vibration_pred_score_data = None
    if all(col in df for col in ['temperature_c', 'vibration_hz', 'predictive_maintenance_score']):
        temp_vibration_pred_score_data = df[['temperature_c', 'vibration_hz', 'predictive_maintenance_score']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "machine_options": machine_options,
            "sensor_readings_by_machine": sensor_readings_data,
            "temperature_vs_vibration_by_pred_score": temp_vibration_pred_score_data
        }
    }


def garment_factory_productivity_data(df):
    """
    Performs garment factory productivity analysis.
    """
    df = df.copy()
    expected = ['date', 'department', 'team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'actual_productivity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['targeted_productivity', 'actual_productivity', 'smv', 'wip', 'over_time', 'incentive']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(inplace=True)

    avg_actual_prod = df['actual_productivity'].mean()
    avg_targeted_prod = df['targeted_productivity'].mean()
    achievement_rate = (avg_actual_prod / avg_targeted_prod) * 100 if avg_targeted_prod > 0 else 0

    metrics = {
        "Average Actual Productivity": avg_actual_prod,
        "Average Target Productivity": avg_targeted_prod,
        "Overall Achievement Rate": achievement_rate
    }

    productivity_by_department_data = None
    if 'department' in df and 'targeted_productivity' in df and 'actual_productivity' in df:
        prod_by_dept = df.groupby('department')[['targeted_productivity', 'actual_productivity']].mean().reset_index()
        productivity_by_department_data = prod_by_dept.to_dict('records')

    incentive_impact_data = None
    if 'incentive' in df and 'actual_productivity' in df and 'department' in df:
        incentive_impact_data = df[['incentive', 'actual_productivity', 'department']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "productivity_by_department": productivity_by_department_data,
            "impact_of_incentive_on_productivity": incentive_impact_data
        }
    }


def material_fusion_process_quality_prediction_data(df):
    """
    Performs material fusion process quality prediction analysis.
    """
    df = df.copy()
    expected = ['temperature_c', 'pressure_kpa', 'material_fusion_metric', 'material_transformation_metric', 'quality_rating']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_quality = df['quality_rating'].mean()
    temp_corr = df['temperature_c'].corr(df['quality_rating']) if not df[['temperature_c', 'quality_rating']].isnull().any().any() else None
    pressure_corr = df['pressure_kpa'].corr(df['quality_rating']) if not df[['pressure_kpa', 'quality_rating']].isnull().any().any() else None

    metrics = {
        "Average Quality Rating": avg_quality,
        "Temperature/Quality Correlation": temp_corr,
        "Pressure/Quality Correlation": pressure_corr
    }

    fusion_metric_3d_data = None
    if all(col in df for col in ['temperature_c', 'pressure_kpa', 'material_fusion_metric', 'quality_rating']):
        fusion_metric_3d_data = df[['temperature_c', 'pressure_kpa', 'material_fusion_metric', 'quality_rating']].to_dict('records')

    heatmap_quality_data = None
    if all(col in df for col in ["temperature_c", "pressure_kpa", "quality_rating"]):
        heatmap_quality_data = df[["temperature_c", "pressure_kpa", "quality_rating"]].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "fusion_metric_by_temp_pressure": fusion_metric_3d_data,
            "heatmap_avg_quality_by_temp_pressure": heatmap_quality_data
        }
    }


def electric_vehicle_manufacturer_plant_location_data(df):
    """
    Performs electric vehicle manufacturer plant location analysis.
    """
    df = df.copy()
    expected = ['ev_maker', 'place', 'state']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(inplace=True)

    num_makers = df['ev_maker'].nunique()
    num_states = df['state'].nunique()
    top_state = df['state'].mode()[0] if not df['state'].empty else None

    metrics = {
        "Number of EV Makers": num_makers,
        "Number of States with Plants": num_states,
        "State with Most Plants": top_state
    }

    plants_by_state_data = None
    if 'state' in df:
        plants_by_state = df['state'].value_counts().reset_index()
        plants_by_state.columns = ['state', 'count']
        plants_by_state_data = plants_by_state.to_dict('records')

    plants_by_maker_data = None
    if 'ev_maker' in df:
        plants_by_maker = df['ev_maker'].value_counts().reset_index()
        plants_by_maker.columns = ['ev_maker', 'count']
        plants_by_maker_data = plants_by_maker.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "number_of_plants_by_state": plants_by_state_data,
            "market_share_by_number_of_plants": plants_by_maker_data
        }
    }


def macroeconomic_impact_on_industrial_production_data(df):
    """
    Performs macroeconomic impact on industrial production analysis.
    """
    df = df.copy()
    expected = ['month_year', 'cpi', 'interest_rates', 'exchange_rates', 'production']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
    for col in ['cpi', 'interest_rates', 'exchange_rates', 'production']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('month_year').dropna()

    cpi_corr = df['cpi'].corr(df['production']) if not df[['cpi', 'production']].isnull().any().any() else None
    interest_corr = df['interest_rates'].corr(df['production']) if not df[['interest_rates', 'production']].isnull().any().any() else None
    exchange_corr = df['exchange_rates'].corr(df['production']) if not df[['exchange_rates', 'production']].isnull().any().any() else None

    metrics = {
        "CPI/Production Correlation": cpi_corr,
        "Interest Rate/Production Correlation": interest_corr,
        "Exchange Rate/Production Correlation": exchange_corr
    }

    macro_indicators_trend_data = None
    if 'month_year' in df and all(col in df for col in ['production', 'cpi', 'interest_rates']):
        macro_indicators_trend_data = df[['month_year', 'production', 'cpi', 'interest_rates']].melt(
            id_vars='month_year', var_name='variable', value_name='value'
        ).to_dict('records')

    scatter_matrix_data = None
    if all(col in df for col in ['cpi', 'interest_rates', 'exchange_rates', 'production']):
        scatter_matrix_data = df[['cpi', 'interest_rates', 'exchange_rates', 'production']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "macro_indicators_and_production_over_time": macro_indicators_trend_data,
            "relationships_between_macro_variables_and_production": scatter_matrix_data
        }
    }


def temperature_control_system_performance_data(df):
    """
    Performs temperature control system performance analysis.
    """
    df = df.copy()
    expected = ['pid_control_output_perc', 'fuzzy_pid_control_output_perc', 'overshoot_c', 'response_time_s', 'steady_state_error_c']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_overshoot = df['overshoot_c'].mean()
    avg_response_time = df['response_time_s'].mean()
    avg_error = df['steady_state_error_c'].mean()

    metrics = {
        "Average Overshoot (°C)": avg_overshoot,
        "Average Response Time (s)": avg_response_time,
        "Average Steady State Error (°C)": avg_error
    }

    overshoot_vs_response_time_data = None
    if all(col in df for col in ['response_time_s', 'overshoot_c', 'steady_state_error_c']):
        overshoot_vs_response_time_data = df[['response_time_s', 'overshoot_c', 'steady_state_error_c']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "overshoot_vs_response_time_by_error": overshoot_vs_response_time_data
        }
    }


def predictive_maintenance_priority_scoring_data(df):
    """
    Performs predictive maintenance priority scoring analysis.
    """
    df = df.copy()
    expected = ['machine_id', 'temp_c', 'vibration_mm_s', 'pressure_bar', 'failure_prob', 'maintenance_priority']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['temp_c', 'vibration_mm_s', 'pressure_bar', 'failure_prob', 'maintenance_priority']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_priority = df['maintenance_priority'].mean()
    top_priority_machine = df.loc[df['maintenance_priority'].idxmax()]['machine_id'] if not df.empty else None

    metrics = {
        "Average Maintenance Priority": avg_priority,
        "Highest Priority Machine": top_priority_machine
    }

    top_machines_priority_data = None
    if 'machine_id' in df and 'maintenance_priority' in df and 'failure_prob' in df:
        top_machines = df.nlargest(15, 'maintenance_priority')
        top_machines_priority_data = top_machines[['machine_id', 'maintenance_priority', 'failure_prob']].to_dict('records')

    sensor_readings_priority_data = None
    if all(col in df for col in ['temp_c', 'vibration_mm_s', 'maintenance_priority']):
        sensor_readings_priority_data = df[['temp_c', 'vibration_mm_s', 'maintenance_priority']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "top_machines_by_maintenance_priority": top_machines_priority_data,
            "sensor_readings_colored_by_maintenance_priority": sensor_readings_priority_data
        }
    }


def production_order_schedule_adherence_data(df):
    """
    Performs production order schedule adherence analysis.
    """
    df = df.copy()
    expected = ['production_order_id', 'scheduled_start', 'scheduled_end', 'actual_start', 'actual_end', 'status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['scheduled_start', 'scheduled_end', 'actual_start', 'actual_end']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['start_delay'] = (df['actual_start'] - df['scheduled_start']).dt.total_seconds() / 3600
    df['end_delay'] = (df['actual_end'] - df['scheduled_end']).dt.total_seconds() / 3600
    
    avg_start_delay = df['start_delay'].mean()
    avg_end_delay = df['end_delay'].mean()
    on_time_completion_rate = (df['end_delay'] <= 0).mean() * 100

    metrics = {
        "On-Time Completion Rate": on_time_completion_rate,
        "Average Start Delay (hours)": avg_start_delay,
        "Average End Delay (hours)": avg_end_delay
    }

    delay_distribution_data = None
    if 'start_delay' in df and 'end_delay' in df:
        delay_distribution_data = df[['start_delay', 'end_delay']].melt(var_name='delay_type', value_name='delay_hours').to_dict('records')

    status_counts_data = None
    if 'status' in df:
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        status_counts_data = status_counts.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "delay_distribution": delay_distribution_data,
            "production_order_status": status_counts_data
        }
    }


def machine_availability_and_utilization_data(df):
    """
    Performs machine availability and utilization analysis.
    """
    df = df.copy()
    expected = ['machine_id', 'operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours', 'units_produced']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours', 'units_produced']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['total_hours'] = df['operational_hours'] + df['idle_hours'] + df['maintenance_hours'] + df['downtime_hours']
    df['utilization_rate'] = (df['operational_hours'] / df['total_hours']) * 100
    df['availability_rate'] = ((df['total_hours'] - df['downtime_hours']) / df['total_hours']) * 100
    
    avg_utilization = df['utilization_rate'].mean()
    avg_availability = df['availability_rate'].mean()

    metrics = {
        "Average Machine Utilization": avg_utilization,
        "Average Machine Availability": avg_availability
    }

    time_breakdown_by_machine_data = None
    if 'machine_id' in df:
        df_melted = df.melt(id_vars='machine_id', value_vars=['operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours'])
        time_breakdown_by_machine_data = df_melted.to_dict('records')

    units_produced_vs_utilization_data = None
    if 'utilization_rate' in df and 'units_produced' in df and 'machine_id' in df:
        units_produced_vs_utilization_data = df[['utilization_rate', 'units_produced', 'machine_id']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "time_breakdown_by_machine": time_breakdown_by_machine_data,
            "units_produced_vs_utilization_rate": units_produced_vs_utilization_data
        }
    }


def machine_downtime_root_cause_data(df):
    """
    Performs machine downtime root cause analysis.
    """
    df = df.copy()
    expected = ['machine_id', 'start_time', 'end_time', 'downtime_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    df.dropna(inplace=True)
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

    total_downtime = df['duration_minutes'].sum() / 60
    top_reason = df.groupby('downtime_reason')['duration_minutes'].sum().idxmax() if not df['downtime_reason'].empty else None
    worst_machine = df.groupby('machine_id')['duration_minutes'].sum().idxmax() if not df['machine_id'].empty else None

    metrics = {
        "Total Downtime (Hours)": total_downtime,
        "Top Downtime Reason": top_reason,
        "Machine with Most Downtime": worst_machine
    }

    downtime_by_reason_data = None
    if 'downtime_reason' in df and 'duration_minutes' in df:
        downtime_by_reason = df.groupby('downtime_reason')['duration_minutes'].sum().sort_values(ascending=False).reset_index()
        downtime_by_reason.columns = ['downtime_reason', 'total_duration_minutes']
        downtime_by_reason_data = downtime_by_reason.to_dict('records')

    downtime_by_machine_data = None
    if 'machine_id' in df and 'duration_minutes' in df:
        downtime_by_machine = df.groupby('machine_id')['duration_minutes'].sum().sort_values(ascending=False).reset_index()
        downtime_by_machine.columns = ['machine_id', 'total_duration_minutes']
        downtime_by_machine_data = downtime_by_machine.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "downtime_by_reason": downtime_by_reason_data,
            "proportion_of_downtime_by_machine": downtime_by_machine_data
        }
    }


def manufacturing_batch_process_monitoring_data(df):
    """
    Performs manufacturing batch process monitoring analysis.
    """
    df = df.copy()
    expected = ['batch_id', 'material_id', 'quantity', 'temperature', 'pressure']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['quantity', 'temperature', 'pressure']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    num_batches = df['batch_id'].nunique()
    avg_temp = df['temperature'].mean()
    avg_pressure = df['pressure'].mean()

    metrics = {
        "Number of Batches": num_batches,
        "Average Temperature": avg_temp,
        "Average Pressure": avg_pressure
    }

    process_conditions_distribution_data = None
    if 'temperature' in df and 'pressure' in df:
        process_conditions_distribution_data = df[['temperature', 'pressure']].melt(var_name='metric', value_name='value').to_dict('records')

    process_conditions_by_material_data = None
    if all(col in df for col in ['temperature', 'pressure', 'material_id', 'quantity']):
        process_conditions_by_material_data = df[['temperature', 'pressure', 'material_id', 'quantity']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "distribution_of_process_temp_pressure": process_conditions_distribution_data,
            "process_conditions_by_material_id": process_conditions_by_material_data
        }
    }


def shift_based_production_output_and_defect_analysis(df):
    """
    Performs shift-based production output and defect analysis.
    """
    df = df.copy()
    expected = ['workstation_id', 'operator_id', 'shift_type', 'output_count', 'defects_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['output_count', 'defects_count']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['defect_rate'] = (df['defects_count'] / df['output_count']) * 100
    
    total_output = df['output_count'].sum()
    total_defects = df['defects_count'].sum()
    overall_defect_rate = (total_defects / total_output) * 100 if total_output > 0 else 0

    metrics = {
        "Total Production Output": total_output,
        "Total Defects": total_defects,
        "Overall Defect Rate": overall_defect_rate
    }

    output_by_shift_data = None
    if 'shift_type' in df and 'output_count' in df:
        output_by_shift = df.groupby('shift_type')['output_count'].sum().reset_index()
        output_by_shift.columns = ['shift_type', 'total_output']
        output_by_shift_data = output_by_shift.to_dict('records')

    defect_rate_by_workstation_data = None
    if 'workstation_id' in df and 'defect_rate' in df:
        defect_rate_by_workstation = df.groupby('workstation_id')['defect_rate'].mean().reset_index()
        defect_rate_by_workstation.columns = ['workstation_id', 'average_defect_rate']
        defect_rate_by_workstation_data = defect_rate_by_workstation.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "production_output_by_shift": output_by_shift_data,
            "defect_rate_by_workstation": defect_rate_by_workstation_data
        }
    }


def quality_inspection_and_defect_resolution_data(df):
    """
    Performs quality inspection and defect resolution analysis.
    """
    df = df.copy()
    expected = ['inspection_date', 'product_id', 'inspector_id', 'defect_found', 'resolution_status', 'resolution_time_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    df['resolution_time_hours'] = pd.to_numeric(df['resolution_time_hours'], errors='coerce')
    df.dropna(subset=['defect_found', 'resolution_status'], inplace=True)

    total_defects_found = df['defect_found'].sum()
    resolved_defects = df[df['resolution_status'].str.contains('resolved', case=False, na=False)].shape[0]
    resolution_rate = (resolved_defects / total_defects_found) * 100 if total_defects_found > 0 else 0
    avg_resolution_time = df['resolution_time_hours'].mean() if not df['resolution_time_hours'].isnull().all() else None

    metrics = {
        "Total Defects Found": total_defects_found,
        "Resolved Defects": resolved_defects,
        "Defect Resolution Rate": resolution_rate,
        "Average Resolution Time (hours)": avg_resolution_time
    }

    defect_resolution_status_data = None
    if 'resolution_status' in df:
        status_counts = df['resolution_status'].value_counts().reset_index()
        status_counts.columns = ['resolution_status', 'count']
        defect_resolution_status_data = status_counts.to_dict('records')

    resolution_time_by_product_data = None
    if 'product_id' in df and 'resolution_time_hours' in df:
        resolution_time_by_product = df.groupby('product_id')['resolution_time_hours'].mean().reset_index()
        resolution_time_by_product.columns = ['product_id', 'average_resolution_time_hours']
        resolution_time_by_product_data = resolution_time_by_product.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "defect_resolution_status": defect_resolution_status_data,
            "average_resolution_time_by_product": resolution_time_by_product_data
        }
    }


def production_material_cost_analysis(df):
    """
    Performs production material cost analysis.
    """
    df = df.copy()
    expected = ['material_id', 'material_name', 'quantity_used', 'unit_cost', 'product_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['quantity_used'] = pd.to_numeric(df['quantity_used'], errors='coerce')
    df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce')
    df.dropna(inplace=True)

    df['total_material_cost'] = df['quantity_used'] * df['unit_cost']

    total_cost_materials = df['total_material_cost'].sum()
    avg_unit_cost = df['unit_cost'].mean()
    most_expensive_material = df.loc[df['total_material_cost'].idxmax()]['material_name'] if not df.empty else None

    metrics = {
        "Total Material Cost": total_cost_materials,
        "Average Unit Cost": avg_unit_cost,
        "Most Expensive Material": most_expensive_material
    }

    cost_by_material_data = None
    if 'material_name' in df and 'total_material_cost' in df:
        cost_by_material = df.groupby('material_name')['total_material_cost'].sum().sort_values(ascending=False).reset_index()
        cost_by_material.columns = ['material_name', 'total_cost']
        cost_by_material_data = cost_by_material.to_dict('records')

    cost_by_product_data = None
    if 'product_id' in df and 'total_material_cost' in df:
        cost_by_product = df.groupby('product_id')['total_material_cost'].sum().sort_values(ascending=False).reset_index()
        cost_by_product.columns = ['product_id', 'total_cost']
        cost_by_product_data = cost_by_product.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "total_material_cost_by_material": cost_by_material_data,
            "total_material_cost_by_product": cost_by_product_data
        }
    }


def supplier_material_receipt_and_quality_data(df):
    """
    Performs supplier material receipt and quality analysis.
    """
    df = df.copy()
    expected = ['receipt_id', 'supplier_name', 'material_name', 'quantity_received', 'quality_status', 'delivery_date', 'inspection_result']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    df.dropna(subset=['quality_status', 'inspection_result'], inplace=True)

    total_receipts = len(df)
    accepted_receipts = df[df['quality_status'].str.contains('accepted', case=False, na=False)].shape[0]
    rejection_rate = ((total_receipts - accepted_receipts) / total_receipts) * 100 if total_receipts > 0 else 0
    
    metrics = {
        "Total Material Receipts": total_receipts,
        "Accepted Receipts": accepted_receipts,
        "Material Rejection Rate": rejection_rate
    }

    quality_status_by_supplier_data = None
    if 'supplier_name' in df and 'quality_status' in df:
        quality_by_supplier = df.groupby(['supplier_name', 'quality_status']).size().unstack(fill_value=0)
        if 'Accepted' not in quality_by_supplier.columns:
            quality_by_supplier['Accepted'] = 0
        if 'Rejected' not in quality_by_supplier.columns:
            quality_by_supplier['Rejected'] = 0
        quality_by_supplier['Total'] = quality_by_supplier.sum(axis=1)
        quality_by_supplier['Rejection Rate (%)'] = (quality_by_supplier['Rejected'] / quality_by_supplier['Total']) * 100
        quality_status_by_supplier_data = quality_by_supplier.reset_index().to_dict('records')

    inspection_results_distribution_data = None
    if 'inspection_result' in df:
        inspection_results_distribution_data = df['inspection_result'].value_counts().reset_index().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "quality_status_by_supplier": quality_status_by_supplier_data,
            "inspection_results_distribution": inspection_results_distribution_data
        }
    }


def manufacturing_resource_utilization_analysis(df):
    """
    Performs manufacturing resource utilization analysis.
    """
    df = df.copy()
    expected = ['resource_id', 'resource_type', 'total_hours_available', 'hours_utilized', 'downtime_hours', 'idle_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['total_hours_available', 'hours_utilized', 'downtime_hours', 'idle_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['utilization_rate'] = (df['hours_utilized'] / df['total_hours_available']) * 100 if df['total_hours_available'].sum() > 0 else 0
    df['downtime_percentage'] = (df['downtime_hours'] / df['total_hours_available']) * 100 if df['total_hours_available'].sum() > 0 else 0

    avg_utilization = df['utilization_rate'].mean()
    avg_downtime_percentage = df['downtime_percentage'].mean()
    
    metrics = {
        "Average Resource Utilization": avg_utilization,
        "Average Resource Downtime Percentage": avg_downtime_percentage
    }

    utilization_by_resource_type_data = None
    if 'resource_type' in df and 'utilization_rate' in df:
        utilization_by_type = df.groupby('resource_type')['utilization_rate'].mean().reset_index()
        utilization_by_type.columns = ['resource_type', 'average_utilization_rate']
        utilization_by_resource_type_data = utilization_by_type.to_dict('records')

    resource_status_breakdown_data = None
    if 'resource_id' in df and all(col in df for col in ['hours_utilized', 'downtime_hours', 'idle_hours']):
        status_breakdown_df = df.melt(id_vars='resource_id', value_vars=['hours_utilized', 'downtime_hours', 'idle_hours'])
        status_breakdown_df.columns = ['resource_id', 'status_type', 'hours']
        resource_status_breakdown_data = status_breakdown_df.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "utilization_by_resource_type": utilization_by_resource_type_data,
            "resource_status_hours_breakdown": resource_status_breakdown_data
        }
    }


def predictive_maintenance_sensor_data_analysis(df):
    """
    Performs predictive maintenance sensor data analysis.
    """
    df = df.copy()
    expected = ['timestamp', 'machine_id', 'sensor_1_value', 'sensor_2_value', 'sensor_3_value', 'anomaly_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['sensor_1_value', 'sensor_2_value', 'sensor_3_value', 'anomaly_score']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('timestamp').dropna()

    avg_anomaly_score = df['anomaly_score'].mean()
    num_anomalies = df[df['anomaly_score'] > df['anomaly_score'].quantile(0.95)].shape[0] if 'anomaly_score' in df else 0
    
    metrics = {
        "Average Anomaly Score": avg_anomaly_score,
        "Number of High Anomaly Readings": num_anomalies
    }

    sensor_readings_over_time_data = None
    if 'timestamp' in df and all(col in df for col in ['sensor_1_value', 'sensor_2_value', 'sensor_3_value']):
        sensor_readings_over_time_data = df[['timestamp', 'sensor_1_value', 'sensor_2_value', 'sensor_3_value']].melt(
            id_vars='timestamp', var_name='sensor', value_name='value'
        ).to_dict('records')

    anomaly_score_distribution_data = None
    if 'anomaly_score' in df:
        anomaly_score_distribution_data = df['anomaly_score'].to_list()

    return {
        "metrics": metrics,
        "data_for_plots": {
            "sensor_readings_trend": sensor_readings_over_time_data,
            "anomaly_score_distribution": anomaly_score_distribution_data
        }
    }


def energy_consumption_and_production_efficiency_data(df):
    """
    Performs energy consumption and production efficiency analysis.
    """
    df = df.copy()
    expected = ['date', 'production_line', 'energy_consumed_kwh', 'units_produced']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['energy_consumed_kwh', 'units_produced']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['energy_per_unit'] = df['energy_consumed_kwh'] / df['units_produced']
    
    total_energy_consumed = df['energy_consumed_kwh'].sum()
    total_units_produced = df['units_produced'].sum()
    avg_energy_efficiency = df['energy_per_unit'].mean()

    metrics = {
        "Total Energy Consumed (kWh)": total_energy_consumed,
        "Total Units Produced": total_units_produced,
        "Average Energy Efficiency (kWh/unit)": avg_energy_efficiency
    }

    energy_efficiency_by_line_data = None
    if 'production_line' in df and 'energy_per_unit' in df:
        energy_efficiency_by_line = df.groupby('production_line')['energy_per_unit'].mean().reset_index()
        energy_efficiency_by_line.columns = ['production_line', 'average_energy_per_unit']
        energy_efficiency_by_line_data = energy_efficiency_by_line.to_dict('records')

    energy_consumption_over_time_data = None
    if 'date' in df and 'energy_consumed_kwh' in df:
        energy_consumption_over_time = df.groupby('date')['energy_consumed_kwh'].sum().reset_index()
        energy_consumption_over_time.columns = ['date', 'total_energy_consumed_kwh']
        energy_consumption_over_time_data = energy_consumption_over_time.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "energy_efficiency_by_production_line": energy_efficiency_by_line_data,
            "energy_consumption_trend_over_time": energy_consumption_over_time_data
        }
    }


def quality_control_lab_test_result_analysis(df):
    """
    Performs quality control lab test result analysis.
    """
    df = df.copy()
    expected = ['test_id', 'sample_id', 'test_date', 'parameter_1', 'parameter_2', 'parameter_3', 'test_result']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')
    for col in ['parameter_1', 'parameter_2', 'parameter_3']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    pass_tests = df[df['test_result'].str.contains('pass', case=False, na=False)].shape[0]
    total_tests = len(df)
    pass_rate = (pass_tests / total_tests) * 100 if total_tests > 0 else 0
    
    metrics = {
        "Total Tests Conducted": total_tests,
        "Tests Passed": pass_tests,
        "Test Pass Rate": pass_rate
    }

    test_result_distribution_data = None
    if 'test_result' in df:
        test_result_distribution_data = df['test_result'].value_counts().reset_index().to_dict('records')

    parameter_correlation_data = None
    if all(col in df for col in ['parameter_1', 'parameter_2', 'parameter_3', 'test_result']):
        # Assuming 'test_result' can be converted to numeric for correlation (e.g., Pass=1, Fail=0)
        df_corr = df.copy()
        df_corr['test_result_numeric'] = df_corr['test_result'].apply(lambda x: 1 if 'pass' in str(x).lower() else 0)
        corr_matrix = df_corr[['parameter_1', 'parameter_2', 'parameter_3', 'test_result_numeric']].corr()
        parameter_correlation_data = corr_matrix.to_dict('index')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "test_result_distribution": test_result_distribution_data,
            "parameter_correlation_with_test_results": parameter_correlation_data
        }
    }


def equipment_calibration_compliance_and_results_analysis(df):
    """
    Performs equipment calibration compliance and results analysis.
    """
    df = df.copy()
    expected = ['calibration_id', 'equipment_id', 'calibration_date', 'due_date', 'calibration_result', 'deviation']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['calibration_date'] = pd.to_datetime(df['calibration_date'], errors='coerce')
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    df['deviation'] = pd.to_numeric(df['deviation'], errors='coerce')
    df.dropna(inplace=True)

    compliant_calibrations = df[df['calibration_result'].str.contains('pass', case=False, na=False)].shape[0]
    total_calibrations = len(df)
    compliance_rate = (compliant_calibrations / total_calibrations) * 100 if total_calibrations > 0 else 0
    
    metrics = {
        "Total Calibrations": total_calibrations,
        "Compliant Calibrations": compliant_calibrations,
        "Calibration Compliance Rate": compliance_rate
    }

    calibration_result_distribution_data = None
    if 'calibration_result' in df:
        calibration_result_distribution_data = df['calibration_result'].value_counts().reset_index().to_dict('records')

    deviation_by_equipment_data = None
    if 'equipment_id' in df and 'deviation' in df:
        deviation_by_equipment_data = df[['equipment_id', 'deviation']].dropna().to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "calibration_result_distribution": calibration_result_distribution_data,
            "deviation_by_equipment_id": deviation_by_equipment_data
        }
    }


def production_delay_root_cause_analysis(df):
    """
    Performs production delay root cause analysis.
    """
    df = df.copy()
    expected = ['delay_id', 'production_line', 'delay_start_time', 'delay_end_time', 'delay_reason', 'impact_on_production']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['delay_start_time'] = pd.to_datetime(df['delay_start_time'], errors='coerce')
    df['delay_end_time'] = pd.to_datetime(df['delay_end_time'], errors='coerce')
    df['impact_on_production'] = pd.to_numeric(df['impact_on_production'], errors='coerce')
    df.dropna(inplace=True)
    df['delay_duration_hours'] = (df['delay_end_time'] - df['delay_start_time']).dt.total_seconds() / 3600

    total_delay_hours = df['delay_duration_hours'].sum()
    most_common_reason = df['delay_reason'].mode()[0] if not df['delay_reason'].empty else None
    
    metrics = {
        "Total Delay Hours": total_delay_hours,
        "Most Common Delay Reason": most_common_reason
    }

    delay_duration_by_reason_data = None
    if 'delay_reason' in df and 'delay_duration_hours' in df:
        delay_duration_by_reason = df.groupby('delay_reason')['delay_duration_hours'].sum().sort_values(ascending=False).reset_index()
        delay_duration_by_reason.columns = ['delay_reason', 'total_delay_hours']
        delay_duration_by_reason_data = delay_duration_by_reason.to_dict('records')

    delay_impact_by_reason_data = None
    if 'delay_reason' in df and 'impact_on_production' in df:
        delay_impact_by_reason = df.groupby('delay_reason')['impact_on_production'].sum().sort_values(ascending=False).reset_index()
        delay_impact_by_reason.columns = ['delay_reason', 'total_impact_on_production']
        delay_impact_by_reason_data = delay_impact_by_reason.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "delay_duration_by_reason": delay_duration_by_reason_data,
            "delay_impact_by_reason": delay_impact_by_reason_data
        }
    }


def workplace_safety_incident_analysis(df):
    """
    Performs workplace safety incident analysis.
    """
    df = df.copy()
    expected = ['incident_id', 'incident_date', 'incident_type', 'department', 'severity_level', 'lost_work_days']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df['lost_work_days'] = pd.to_numeric(df['lost_work_days'], errors='coerce')
    df.dropna(subset=['incident_type', 'severity_level'], inplace=True)

    total_incidents = len(df)
    total_lost_work_days = df['lost_work_days'].sum()
    most_common_incident_type = df['incident_type'].mode()[0] if not df['incident_type'].empty else None

    metrics = {
        "Total Safety Incidents": total_incidents,
        "Total Lost Work Days": total_lost_work_days,
        "Most Common Incident Type": most_common_incident_type
    }

    incidents_by_type_data = None
    if 'incident_type' in df:
        incidents_by_type_data = df['incident_type'].value_counts().reset_index().to_dict('records')

    incidents_by_severity_data = None
    if 'severity_level' in df:
        incidents_by_severity_data = df['severity_level'].value_counts().reset_index().to_dict('records')

    lost_work_days_by_department_data = None
    if 'department' in df and 'lost_work_days' in df:
        lost_work_days_by_department = df.groupby('department')['lost_work_days'].sum().reset_index()
        lost_work_days_by_department.columns = ['department', 'total_lost_work_days']
        lost_work_days_by_department_data = lost_work_days_by_department.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "incidents_by_type": incidents_by_type_data,
            "incidents_by_severity": incidents_by_severity_data,
            "lost_work_days_by_department": lost_work_days_by_department_data
        }
    }


def equipment_maintenance_cost_and_type_analysis(df):
    """
    Performs equipment maintenance cost and type analysis.
    """
    df = df.copy()
    expected = ['maintenance_id', 'equipment_id', 'maintenance_date', 'maintenance_type', 'cost_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['maintenance_date'] = pd.to_datetime(df['maintenance_date'], errors='coerce')
    df['cost_usd'] = pd.to_numeric(df['cost_usd'], errors='coerce')
    df.dropna(inplace=True)

    total_maintenance_cost = df['cost_usd'].sum()
    avg_maintenance_cost = df['cost_usd'].mean()
    most_common_maintenance_type = df['maintenance_type'].mode()[0] if not df['maintenance_type'].empty else None

    metrics = {
        "Total Maintenance Cost (USD)": total_maintenance_cost,
        "Average Maintenance Cost (USD)": avg_maintenance_cost,
        "Most Common Maintenance Type": most_common_maintenance_type
    }

    cost_by_maintenance_type_data = None
    if 'maintenance_type' in df and 'cost_usd' in df:
        cost_by_type = df.groupby('maintenance_type')['cost_usd'].sum().sort_values(ascending=False).reset_index()
        cost_by_type.columns = ['maintenance_type', 'total_cost_usd']
        cost_by_maintenance_type_data = cost_by_type.to_dict('records')

    cost_by_equipment_data = None
    if 'equipment_id' in df and 'cost_usd' in df:
        cost_by_equipment = df.groupby('equipment_id')['cost_usd'].sum().sort_values(ascending=False).reset_index()
        cost_by_equipment.columns = ['equipment_id', 'total_cost_usd']
        cost_by_equipment_data = cost_by_equipment.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "maintenance_cost_by_type": cost_by_maintenance_type_data,
            "maintenance_cost_by_equipment": cost_by_equipment_data
        }
    }


def tool_lifecycle_and_condition_monitoring_analysis(df):
    """
    Performs tool lifecycle and condition monitoring analysis.
    """
    df = df.copy()
    expected = ['tool_id', 'tool_type', 'purchase_date', 'last_service_date', 'usage_hours', 'condition_score', 'wear_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['purchase_date'] = pd.to_datetime(df['purchase_date'], errors='coerce')
    df['last_service_date'] = pd.to_datetime(df['last_service_date'], errors='coerce')
    for col in ['usage_hours', 'condition_score', 'wear_level']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    avg_usage_hours = df['usage_hours'].mean()
    avg_condition_score = df['condition_score'].mean()
    highest_wear_tool = df.loc[df['wear_level'].idxmax()]['tool_id'] if not df.empty else None

    metrics = {
        "Average Usage Hours": avg_usage_hours,
        "Average Condition Score": avg_condition_score,
        "Tool with Highest Wear": highest_wear_tool
    }

    condition_score_by_tool_type_data = None
    if 'tool_type' in df and 'condition_score' in df:
        condition_score_by_type = df.groupby('tool_type')['condition_score'].mean().reset_index()
        condition_score_by_type.columns = ['tool_type', 'average_condition_score']
        condition_score_by_tool_type_data = condition_score_by_type.to_dict('records')

    wear_level_vs_usage_hours_data = None
    if 'wear_level' in df and 'usage_hours' in df:
        wear_level_vs_usage_hours_data = df[['usage_hours', 'wear_level']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "condition_score_by_tool_type": condition_score_by_tool_type_data,
            "wear_level_vs_usage_hours": wear_level_vs_usage_hours_data
        }
    }


def final_product_quality_grade_analysis(df):
    """
    Performs final product quality grade analysis.
    """
    df = df.copy()
    expected = ['product_id', 'production_batch', 'inspection_date', 'quality_grade', 'defect_category', 'inspector']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
    df.dropna(subset=['quality_grade'], inplace=True)

    total_products = len(df)
    top_grade = df['quality_grade'].mode()[0] if not df['quality_grade'].empty else None
    
    metrics = {
        "Total Products Inspected": total_products,
        "Most Common Quality Grade": top_grade
    }

    quality_grade_distribution_data = None
    if 'quality_grade' in df:
        quality_grade_distribution = df['quality_grade'].value_counts(normalize=True).mul(100).reset_index()
        quality_grade_distribution.columns = ['quality_grade', 'percentage']
        quality_grade_distribution_data = quality_grade_distribution.to_dict('records')

    defect_category_by_grade_data = None
    if 'quality_grade' in df and 'defect_category' in df:
        defect_category_by_grade = df.groupby(['quality_grade', 'defect_category']).size().unstack(fill_value=0).stack().reset_index(name='count')
        defect_category_by_grade.columns = ['quality_grade', 'defect_category', 'count']
        defect_category_by_grade_data = defect_category_by_grade.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "quality_grade_distribution": quality_grade_distribution_data,
            "defect_categories_by_quality_grade": defect_category_by_grade_data
        }
    }


def shift_production_performance_analysis(df):
    """
    Performs shift production performance analysis.
    """
    df = df.copy()
    expected = ['shift_id', 'shift_date', 'production_line', 'units_produced', 'defects', 'target_units']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['shift_date'] = pd.to_datetime(df['shift_date'], errors='coerce')
    for col in ['units_produced', 'defects', 'target_units']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['achievement_rate'] = (df['units_produced'] / df['target_units']) * 100
    df['defect_rate'] = (df['defects'] / df['units_produced']) * 100

    avg_achievement_rate = df['achievement_rate'].mean()
    avg_defect_rate = df['defect_rate'].mean()
    
    metrics = {
        "Average Achievement Rate": avg_achievement_rate,
        "Average Defect Rate": avg_defect_rate
    }

    performance_by_shift_data = None
    if 'shift_id' in df and 'achievement_rate' in df and 'defect_rate' in df:
        performance_by_shift = df.groupby('shift_id').agg(
            avg_achievement_rate=('achievement_rate', 'mean'),
            avg_defect_rate=('defect_rate', 'mean')
        ).reset_index()
        performance_by_shift_data = performance_by_shift.to_dict('records')

    production_trend_by_line_data = None
    if 'shift_date' in df and 'production_line' in df and 'units_produced' in df:
        production_trend_by_line = df.groupby(['shift_date', 'production_line'])['units_produced'].sum().reset_index()
        production_trend_by_line.columns = ['shift_date', 'production_line', 'total_units_produced']
        production_trend_by_line_data = production_trend_by_line.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "performance_by_shift": performance_by_shift_data,
            "production_trend_by_line": production_trend_by_line_data
        }
    }


def inbound_material_quality_and_delivery_analysis(df):
    """
    Performs inbound material quality and delivery analysis.
    """
    df = df.copy()
    expected = ['material_id', 'supplier_id', 'delivery_date', 'quantity_delivered', 'quality_inspection_result', 'delivery_delay_days']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    for col in ['quantity_delivered', 'delivery_delay_days']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['quality_inspection_result'], inplace=True)

    accepted_materials = df[df['quality_inspection_result'].str.contains('pass|accepted', case=False, na=False)].shape[0]
    total_materials = len(df)
    acceptance_rate = (accepted_materials / total_materials) * 100 if total_materials > 0 else 0
    avg_delivery_delay = df['delivery_delay_days'].mean() if not df['delivery_delay_days'].isnull().all() else None

    metrics = {
        "Total Inbound Materials": total_materials,
        "Accepted Materials": accepted_materials,
        "Material Acceptance Rate": acceptance_rate,
        "Average Delivery Delay (Days)": avg_delivery_delay
    }

    quality_by_supplier_data = None
    if 'supplier_id' in df and 'quality_inspection_result' in df:
        quality_by_supplier = df.groupby('supplier_id')['quality_inspection_result'].value_counts().unstack(fill_value=0).reset_index()
        quality_by_supplier_data = quality_by_supplier.to_dict('records')

    delivery_delay_distribution_data = None
    if 'delivery_delay_days' in df:
        delivery_delay_distribution_data = df['delivery_delay_days'].to_list()

    return {
        "metrics": metrics,
        "data_for_plots": {
            "inbound_material_quality_by_supplier": quality_by_supplier_data,
            "delivery_delay_distribution": delivery_delay_distribution_data
        }
    }


def warehouse_inventory_stock_level_analysis(df):
    """
    Performs warehouse inventory stock level analysis.
    """
    df = df.copy()
    expected = ['item_id', 'item_name', 'warehouse_location', 'current_stock_level', 'reorder_point', 'max_stock_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['current_stock_level', 'reorder_point', 'max_stock_level']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    items_below_reorder = df[df['current_stock_level'] < df['reorder_point']].shape[0]
    total_items = len(df)
    avg_stock_level = df['current_stock_level'].mean()

    metrics = {
        "Total Items in Inventory": total_items,
        "Items Below Reorder Point": items_below_reorder,
        "Average Stock Level": avg_stock_level
    }

    stock_level_distribution_data = None
    if 'current_stock_level' in df:
        stock_level_distribution_data = df['current_stock_level'].to_list()

    stock_status_by_location_data = None
    if 'warehouse_location' in df and 'current_stock_level' in df and 'reorder_point' in df:
        df['stock_status'] = np.where(df['current_stock_level'] < df['reorder_point'], 'Below Reorder', 'Above Reorder')
        stock_status_by_location = df.groupby(['warehouse_location', 'stock_status']).size().unstack(fill_value=0).reset_index()
        stock_status_by_location_data = stock_status_by_location.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "inventory_stock_level_distribution": stock_level_distribution_data,
            "stock_status_by_warehouse_location": stock_status_by_location_data
        }
    }


def order_dispatch_and_delivery_status_tracking(df):
    """
    Performs order dispatch and delivery status tracking analysis.
    """
    df = df.copy()
    expected = ['order_id', 'dispatch_date', 'delivery_date', 'delivery_status', 'shipping_carrier', 'delivery_time_days']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    df['delivery_time_days'] = pd.to_numeric(df['delivery_time_days'], errors='coerce')
    df.dropna(subset=['delivery_status'], inplace=True)

    total_orders = len(df)
    delivered_orders = df[df['delivery_status'].str.contains('delivered', case=False, na=False)].shape[0]
    on_time_delivery_rate = (df[df['delivery_delay_days'] <= 0].shape[0] / total_orders) * 100 if 'delivery_delay_days' in df and total_orders > 0 else None
    
    metrics = {
        "Total Orders": total_orders,
        "Delivered Orders": delivered_orders,
        "On-Time Delivery Rate": on_time_delivery_rate
    }

    delivery_status_distribution_data = None
    if 'delivery_status' in df:
        delivery_status_distribution = df['delivery_status'].value_counts().reset_index()
        delivery_status_distribution.columns = ['delivery_status', 'count']
        delivery_status_distribution_data = delivery_status_distribution.to_dict('records')

    avg_delivery_time_by_carrier_data = None
    if 'shipping_carrier' in df and 'delivery_time_days' in df:
        avg_delivery_time_by_carrier = df.groupby('shipping_carrier')['delivery_time_days'].mean().reset_index()
        avg_delivery_time_by_carrier.columns = ['shipping_carrier', 'average_delivery_time_days']
        avg_delivery_time_by_carrier_data = avg_delivery_time_by_carrier.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "delivery_status_distribution": delivery_status_distribution_data,
            "average_delivery_time_by_shipping_carrier": avg_delivery_time_by_carrier_data
        }
    }


def inventory_audit_and_stock_count_analysis(df):
    """
    Performs inventory audit and stock count analysis.
    """
    df = df.copy()
    expected = ['audit_id', 'audit_date', 'item_id', 'recorded_stock', 'actual_stock', 'variance', 'audit_result']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
    for col in ['recorded_stock', 'actual_stock', 'variance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['audit_result'], inplace=True)

    total_audits = len(df)
    accurate_audits = df[df['variance'] == 0].shape[0]
    accuracy_rate = (accurate_audits / total_audits) * 100 if total_audits > 0 else 0
    total_variance = df['variance'].sum()

    metrics = {
        "Total Audits Performed": total_audits,
        "Accurate Audits": accurate_audits,
        "Inventory Accuracy Rate": accuracy_rate,
        "Total Stock Variance": total_variance
    }

    variance_by_item_data = None
    if 'item_id' in df and 'variance' in df:
        variance_by_item = df.groupby('item_id')['variance'].sum().sort_values(ascending=False).reset_index()
        variance_by_item.columns = ['item_id', 'total_variance']
        variance_by_item_data = variance_by_item.to_dict('records')

    audit_result_distribution_data = None
    if 'audit_result' in df:
        audit_result_distribution = df['audit_result'].value_counts().reset_index()
        audit_result_distribution.columns = ['audit_result', 'count']
        audit_result_distribution_data = audit_result_distribution.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "inventory_variance_by_item": variance_by_item_data,
            "audit_result_distribution": audit_result_distribution_data
        }
    }


def product_return_reason_analysis(df):
    """
    Performs product return reason analysis.
    """
    df = df.copy()
    expected = ['return_id', 'product_id', 'return_date', 'return_reason', 'quantity_returned']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
    df['quantity_returned'] = pd.to_numeric(df['quantity_returned'], errors='coerce')
    df.dropna(subset=['return_reason'], inplace=True)

    total_returns = len(df)
    total_quantity_returned = df['quantity_returned'].sum()
    most_common_return_reason = df['return_reason'].mode()[0] if not df['return_reason'].empty else None

    metrics = {
        "Total Returns": total_returns,
        "Total Quantity Returned": total_quantity_returned,
        "Most Common Return Reason": most_common_return_reason
    }

    returns_by_reason_data = None
    if 'return_reason' in df and 'quantity_returned' in df:
        returns_by_reason = df.groupby('return_reason')['quantity_returned'].sum().sort_values(ascending=False).reset_index()
        returns_by_reason.columns = ['return_reason', 'total_quantity_returned']
        returns_by_reason_data = returns_by_reason.to_dict('records')

    returns_over_time_data = None
    if 'return_date' in df:
        returns_over_time = df.groupby(df['return_date'].dt.to_period('M')).size().reset_index(name='count')
        returns_over_time.columns = ['return_month', 'return_count']
        returns_over_time_data = returns_over_time.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "returns_by_reason": returns_by_reason_data,
            "returns_trend_over_time": returns_over_time_data
        }
    }


def factory_environmental_conditions_monitoring(df):
    """
    Performs factory environmental conditions monitoring analysis.
    """
    df = df.copy()
    expected = ['timestamp', 'sensor_location', 'temperature_c', 'humidity_perc', 'air_quality_index']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ['temperature_c', 'humidity_perc', 'air_quality_index']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_values('timestamp').dropna()

    avg_temp = df['temperature_c'].mean()
    avg_humidity = df['humidity_perc'].mean()
    avg_air_quality = df['air_quality_index'].mean()

    metrics = {
        "Average Temperature (°C)": avg_temp,
        "Average Humidity (%)": avg_humidity,
        "Average Air Quality Index": avg_air_quality
    }

    environmental_conditions_trend_data = None
    if 'timestamp' in df and all(col in df for col in ['temperature_c', 'humidity_perc', 'air_quality_index']):
        environmental_conditions_trend = df[['timestamp', 'temperature_c', 'humidity_perc', 'air_quality_index']].melt(
            id_vars='timestamp', var_name='metric', value_name='value'
        )
        environmental_conditions_trend_data = environmental_conditions_trend.to_dict('records')

    conditions_by_location_data = None
    if 'sensor_location' in df and all(col in df for col in ['temperature_c', 'humidity_perc', 'air_quality_index']):
        conditions_by_location = df.groupby('sensor_location').agg(
            avg_temperature=('temperature_c', 'mean'),
            avg_humidity=('humidity_perc', 'mean'),
            avg_air_quality=('air_quality_index', 'mean')
        ).reset_index()
        conditions_by_location_data = conditions_by_location.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "environmental_conditions_over_time": environmental_conditions_trend_data,
            "environmental_conditions_by_sensor_location": conditions_by_location_data
        }
    }


def manufacturing_waste_management_analysis(df):
    """
    Performs manufacturing waste management analysis.
    """
    df = df.copy()
    expected = ['waste_id', 'waste_type', 'generated_date', 'weight_kg', 'disposal_method', 'cost_of_disposal_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['generated_date'] = pd.to_datetime(df['generated_date'], errors='coerce')
    for col in ['weight_kg', 'cost_of_disposal_usd']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['waste_type', 'disposal_method'], inplace=True)

    total_waste_kg = df['weight_kg'].sum()
    total_disposal_cost = df['cost_of_disposal_usd'].sum()
    most_common_waste_type = df['waste_type'].mode()[0] if not df['waste_type'].empty else None

    metrics = {
        "Total Waste Generated (kg)": total_waste_kg,
        "Total Disposal Cost (USD)": total_disposal_cost,
        "Most Common Waste Type": most_common_waste_type
    }

    waste_by_type_data = None
    if 'waste_type' in df and 'weight_kg' in df:
        waste_by_type = df.groupby('waste_type')['weight_kg'].sum().sort_values(ascending=False).reset_index()
        waste_by_type.columns = ['waste_type', 'total_weight_kg']
        waste_by_type_data = waste_by_type.to_dict('records')

    disposal_cost_by_method_data = None
    if 'disposal_method' in df and 'cost_of_disposal_usd' in df:
        disposal_cost_by_method = df.groupby('disposal_method')['cost_of_disposal_usd'].sum().sort_values(ascending=False).reset_index()
        disposal_cost_by_method.columns = ['disposal_method', 'total_cost_usd']
        disposal_cost_by_method_data = disposal_cost_by_method.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "waste_generation_by_type": waste_by_type_data,
            "disposal_cost_by_method": disposal_cost_by_method_data
        }
    }


def product_packaging_process_analysis(df):
    """
    Performs product packaging process analysis.
    """
    df = df.copy()
    expected = ['packaging_run_id', 'product_id', 'packaging_date', 'units_packaged', 'defects_packaging', 'packaging_line', 'packaging_time_minutes']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['packaging_date'] = pd.to_datetime(df['packaging_date'], errors='coerce')
    for col in ['units_packaged', 'defects_packaging', 'packaging_time_minutes']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['packaging_defect_rate'] = (df['defects_packaging'] / df['units_packaged']) * 100
    
    total_units_packaged = df['units_packaged'].sum()
    avg_packaging_defect_rate = df['packaging_defect_rate'].mean()
    avg_packaging_time = df['packaging_time_minutes'].mean()

    metrics = {
        "Total Units Packaged": total_units_packaged,
        "Average Packaging Defect Rate": avg_packaging_defect_rate,
        "Average Packaging Time (minutes)": avg_packaging_time
    }

    packaging_defects_by_line_data = None
    if 'packaging_line' in df and 'defects_packaging' in df:
        packaging_defects_by_line = df.groupby('packaging_line')['defects_packaging'].sum().sort_values(ascending=False).reset_index()
        packaging_defects_by_line.columns = ['packaging_line', 'total_defects']
        packaging_defects_by_line_data = packaging_defects_by_line.to_dict('records')

    packaging_time_by_product_data = None
    if 'product_id' in df and 'packaging_time_minutes' in df:
        packaging_time_by_product = df.groupby('product_id')['packaging_time_minutes'].mean().reset_index()
        packaging_time_by_product.columns = ['product_id', 'average_packaging_time_minutes']
        packaging_time_by_product_data = packaging_time_by_product.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "packaging_defects_by_line": packaging_defects_by_line_data,
            "packaging_time_by_product": packaging_time_by_product_data
        }
    }


def outbound_shipment_tracking_analysis(df):
    """
    Performs outbound shipment tracking analysis.
    """
    df = df.copy()
    expected = ['shipment_id', 'order_id', 'ship_date', 'delivery_date', 'customer_location', 'shipping_cost', 'delivery_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
    df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
    df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce')
    df.dropna(subset=['delivery_status'], inplace=True)

    total_shipments = len(df)
    delivered_shipments = df[df['delivery_status'].str.contains('delivered', case=False, na=False)].shape[0]
    avg_shipping_cost = df['shipping_cost'].mean() if not df['shipping_cost'].isnull().all() else None

    metrics = {
        "Total Shipments": total_shipments,
        "Delivered Shipments": delivered_shipments,
        "Average Shipping Cost": avg_shipping_cost
    }

    delivery_status_breakdown_data = None
    if 'delivery_status' in df:
        delivery_status_breakdown = df['delivery_status'].value_counts().reset_index()
        delivery_status_breakdown.columns = ['delivery_status', 'count']
        delivery_status_breakdown_data = delivery_status_breakdown.to_dict('records')

    shipping_cost_by_location_data = None
    if 'customer_location' in df and 'shipping_cost' in df:
        shipping_cost_by_location = df.groupby('customer_location')['shipping_cost'].mean().reset_index()
        shipping_cost_by_location.columns = ['customer_location', 'average_shipping_cost']
        shipping_cost_by_location_data = shipping_cost_by_location.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "delivery_status_breakdown": delivery_status_breakdown_data,
            "average_shipping_cost_by_customer_location": shipping_cost_by_location_data
        }
    }


def manufacturing_process_step_duration_analysis(df):
    """
    Performs manufacturing process step duration analysis.
    """
    df = df.copy()
    expected = ['process_step_id', 'batch_id', 'start_time', 'end_time', 'process_name']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
    df.dropna(inplace=True)
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

    avg_step_duration = df['duration_minutes'].mean()
    longest_step = df.loc[df['duration_minutes'].idxmax()]['process_name'] if not df.empty else None

    metrics = {
        "Average Process Step Duration (minutes)": avg_step_duration,
        "Longest Process Step": longest_step
    }

    duration_by_process_step_data = None
    if 'process_name' in df and 'duration_minutes' in df:
        duration_by_process_step = df.groupby('process_name')['duration_minutes'].mean().sort_values(ascending=False).reset_index()
        duration_by_process_step.columns = ['process_name', 'average_duration_minutes']
        duration_by_process_step_data = duration_by_process_step.to_dict('records')

    process_flow_data = None
    if 'process_step_id' in df and 'batch_id' in df and 'start_time' in df and 'end_time' in df:
        process_flow_data = df[['process_step_id', 'batch_id', 'start_time', 'end_time']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "average_duration_by_process_step": duration_by_process_step_data,
            "process_flow_for_batches": process_flow_data
        }
    }


def research_and_development_experiment_analysis(df):
    """
    Performs research and development experiment analysis.
    """
    df = df.copy()
    expected = ['experiment_id', 'experiment_date', 'experiment_parameters', 'test_results', 'outcome_metric_1', 'outcome_metric_2', 'conclusion']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['experiment_date'] = pd.to_datetime(df['experiment_date'], errors='coerce')
    for col in ['outcome_metric_1', 'outcome_metric_2']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['test_results', 'conclusion'], inplace=True)

    total_experiments = len(df)
    successful_experiments = df[df['conclusion'].str.contains('success', case=False, na=False)].shape[0]
    success_rate = (successful_experiments / total_experiments) * 100 if total_experiments > 0 else 0

    metrics = {
        "Total Experiments": total_experiments,
        "Successful Experiments": successful_experiments,
        "Experiment Success Rate": success_rate
    }

    outcome_metrics_distribution_data = None
    if 'outcome_metric_1' in df and 'outcome_metric_2' in df:
        outcome_metrics_distribution = df[['outcome_metric_1', 'outcome_metric_2']].melt(var_name='metric', value_name='value')
        outcome_metrics_distribution_data = outcome_metrics_distribution.to_dict('records')

    experiment_results_over_time_data = None
    if 'experiment_date' in df and 'outcome_metric_1' in df:
        experiment_results_over_time = df.groupby('experiment_date')['outcome_metric_1'].mean().reset_index()
        experiment_results_over_time.columns = ['experiment_date', 'average_outcome_metric_1']
        experiment_results_over_time_data = experiment_results_over_time.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "outcome_metrics_distribution": outcome_metrics_distribution_data,
            "experiment_results_trend_over_time": experiment_results_over_time_data
        }
    }


def barcode_based_product_traceability_analysis(df):
    """
    Performs barcode-based product traceability analysis.
    """
    df = df.copy()
    expected = ['product_serial_number', 'batch_number', 'production_date', 'factory_id', 'qc_status', 'shipment_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
    df['shipment_date'] = pd.to_datetime(df['shipment_date'], errors='coerce')
    df.dropna(inplace=True)

    total_traceable_products = len(df)
    products_by_factory = df['factory_id'].nunique()
    
    metrics = {
        "Total Traceable Products": total_traceable_products,
        "Number of Factories Involved": products_by_factory
    }

    qc_status_distribution_data = None
    if 'qc_status' in df:
        qc_status_distribution = df['qc_status'].value_counts().reset_index()
        qc_status_distribution.columns = ['qc_status', 'count']
        qc_status_distribution_data = qc_status_distribution.to_dict('records')

    products_by_production_date_data = None
    if 'production_date' in df:
        products_by_production_date = df.groupby(df['production_date'].dt.to_period('M')).size().reset_index(name='count')
        products_by_production_date.columns = ['production_month', 'product_count']
        products_by_production_date_data = products_by_production_date.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "qc_status_distribution": qc_status_distribution_data,
            "products_by_production_date": products_by_production_date_data
        }
    }


def internal_process_and_compliance_audit_analysis(df):
    """
    Performs internal process and compliance audit analysis.
    """
    df = df.copy()
    expected = ['audit_id', 'audit_date', 'department', 'compliance_area', 'audit_score', 'findings', 'recommendations', 'status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
    df['audit_score'] = pd.to_numeric(df['audit_score'], errors='coerce')
    df.dropna(subset=['compliance_area', 'status'], inplace=True)

    total_audits = len(df)
    avg_audit_score = df['audit_score'].mean()
    non_compliant_audits = df[df['status'].str.contains('non-compliant', case=False, na=False)].shape[0]

    metrics = {
        "Total Audits": total_audits,
        "Average Audit Score": avg_audit_score,
        "Non-Compliant Audits": non_compliant_audits
    }

    audit_score_by_compliance_area_data = None
    if 'compliance_area' in df and 'audit_score' in df:
        audit_score_by_compliance_area = df.groupby('compliance_area')['audit_score'].mean().sort_values(ascending=False).reset_index()
        audit_score_by_compliance_area.columns = ['compliance_area', 'average_audit_score']
        audit_score_by_compliance_area_data = audit_score_by_compliance_area.to_dict('records')

    audit_status_distribution_data = None
    if 'status' in df:
        audit_status_distribution = df['status'].value_counts().reset_index()
        audit_status_distribution.columns = ['status', 'count']
        audit_status_distribution_data = audit_status_distribution.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "audit_score_by_compliance_area": audit_score_by_compliance_area_data,
            "audit_status_distribution": audit_status_distribution_data
        }
    }


def machine_capacity_and_load_analysis(df):
    """
    Performs machine capacity and load analysis.
    """
    df = df.copy()
    expected = ['machine_id', 'capacity_units_per_hour', 'actual_production_units', 'shift_hours_worked', 'downtime_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['capacity_units_per_hour', 'actual_production_units', 'shift_hours_worked', 'downtime_hours']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['theoretical_capacity_units'] = df['capacity_units_per_hour'] * df['shift_hours_worked']
    df['load_percentage'] = (df['actual_production_units'] / df['theoretical_capacity_units']) * 100 if df['theoretical_capacity_units'].sum() > 0 else 0
    df['efficiency'] = (df['actual_production_units'] / (df['capacity_units_per_hour'] * (df['shift_hours_worked'] - df['downtime_hours']))) * 100 if (df['capacity_units_per_hour'] * (df['shift_hours_worked'] - df['downtime_hours'])).sum() > 0 else 0

    avg_load_percentage = df['load_percentage'].mean()
    avg_efficiency = df['efficiency'].mean()

    metrics = {
        "Average Machine Load Percentage": avg_load_percentage,
        "Average Machine Efficiency": avg_efficiency
    }

    load_by_machine_data = None
    if 'machine_id' in df and 'load_percentage' in df:
        load_by_machine = df.groupby('machine_id')['load_percentage'].mean().reset_index()
        load_by_machine.columns = ['machine_id', 'average_load_percentage']
        load_by_machine_data = load_by_machine.to_dict('records')

    efficiency_vs_downtime_data = None
    if 'efficiency' in df and 'downtime_hours' in df:
        efficiency_vs_downtime_data = df[['efficiency', 'downtime_hours']].to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "machine_load_by_machine_id": load_by_machine_data,
            "machine_efficiency_vs_downtime": efficiency_vs_downtime_data
        }
    }


def production_volume_variance_analysis(df):
    """
    Performs production volume variance analysis.
    """
    df = df.copy()
    expected = ['date', 'product_id', 'planned_production', 'actual_production']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ['planned_production', 'actual_production']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    df['variance'] = df['actual_production'] - df['planned_production']
    df['variance_percentage'] = (df['variance'] / df['planned_production']) * 100 if df['planned_production'].sum() > 0 else 0

    total_planned_production = df['planned_production'].sum()
    total_actual_production = df['actual_production'].sum()
    total_variance = df['variance'].sum()
    avg_variance_percentage = df['variance_percentage'].mean()

    metrics = {
        "Total Planned Production": total_planned_production,
        "Total Actual Production": total_actual_production,
        "Total Production Variance": total_variance,
        "Average Variance Percentage": avg_variance_percentage
    }

    variance_by_product_data = None
    if 'product_id' in df and 'variance' in df:
        variance_by_product = df.groupby('product_id')['variance'].sum().sort_values(ascending=False).reset_index()
        variance_by_product.columns = ['product_id', 'total_variance']
        variance_by_product_data = variance_by_product.to_dict('records')

    daily_variance_trend_data = None
    if 'date' in df and 'variance' in df:
        daily_variance_trend = df.groupby('date')['variance'].sum().reset_index()
        daily_variance_trend.columns = ['date', 'daily_variance']
        daily_variance_trend_data = daily_variance_trend.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "production_variance_by_product": variance_by_product_data,
            "daily_production_variance_trend": daily_variance_trend_data
        }
    }


def iot_sensor_data_time_series_analysis(df):
    """
    Performs IoT sensor data time-series analysis.
    """
    df = df.copy()
    expected = ['timestamp', 'sensor_id', 'reading_value', 'unit']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['reading_value'] = pd.to_numeric(df['reading_value'], errors='coerce')
    df = df.sort_values('timestamp').dropna()

    num_sensors = df['sensor_id'].nunique()
    avg_reading = df['reading_value'].mean()
    
    metrics = {
        "Number of Unique Sensors": num_sensors,
        "Average Sensor Reading": avg_reading
    }

    sensor_reading_trend_data = None
    if 'timestamp' in df and 'reading_value' in df and 'sensor_id' in df:
        sensor_reading_trend = df[['timestamp', 'sensor_id', 'reading_value']]
        sensor_reading_trend_data = sensor_reading_trend.to_dict('records')

    reading_distribution_by_sensor_data = None
    if 'sensor_id' in df and 'reading_value' in df:
        reading_distribution_by_sensor = df.groupby('sensor_id')['reading_value'].apply(list).reset_index()
        reading_distribution_by_sensor_data = reading_distribution_by_sensor.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "sensor_reading_trend_over_time": sensor_reading_trend_data,
            "reading_distribution_by_sensor": reading_distribution_by_sensor_data
        }
    }


def cost_center_expense_analysis(df):
    """
    Performs cost center expense analysis.
    """
    df = df.copy()
    expected = ['expense_id', 'date', 'cost_center', 'expense_category', 'amount_usd']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
    df.dropna(inplace=True)

    total_expenses = df['amount_usd'].sum()
    num_cost_centers = df['cost_center'].nunique()
    
    metrics = {
        "Total Expenses (USD)": total_expenses,
        "Number of Cost Centers": num_cost_centers
    }

    expenses_by_cost_center_data = None
    if 'cost_center' in df and 'amount_usd' in df:
        expenses_by_cost_center = df.groupby('cost_center')['amount_usd'].sum().sort_values(ascending=False).reset_index()
        expenses_by_cost_center.columns = ['cost_center', 'total_amount_usd']
        expenses_by_cost_center_data = expenses_by_cost_center.to_dict('records')

    expenses_by_category_data = None
    if 'expense_category' in df and 'amount_usd' in df:
        expenses_by_category = df.groupby('expense_category')['amount_usd'].sum().sort_values(ascending=False).reset_index()
        expenses_by_category.columns = ['expense_category', 'total_amount_usd']
        expenses_by_category_data = expenses_by_category.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "expenses_by_cost_center": expenses_by_cost_center_data,
            "expenses_by_category": expenses_by_category_data
        }
    }


def regulatory_compliance_status_analysis(df):
    """
    Performs regulatory compliance status analysis.
    """
    df = df.copy()
    expected = ['compliance_id', 'audit_date', 'regulation_name', 'compliance_status', 'findings_count', 'severity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {"error": "Missing required columns", "missing_columns": missing, "matched_columns": matched, "general_insights": get_general_insights_data(df)}

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
    df['findings_count'] = pd.to_numeric(df['findings_count'], errors='coerce')
    df.dropna(subset=['compliance_status', 'regulation_name'], inplace=True)

    total_audits = len(df)
    compliant_audits = df[df['compliance_status'].str.contains('compliant', case=False, na=False)].shape[0]
    compliance_rate = (compliant_audits / total_audits) * 100 if total_audits > 0 else 0
    avg_findings_count = df['findings_count'].mean() if not df['findings_count'].isnull().all() else None

    metrics = {
        "Total Compliance Audits": total_audits,
        "Compliant Audits": compliant_audits,
        "Overall Compliance Rate": compliance_rate,
        "Average Findings Count": avg_findings_count
    }

    compliance_status_distribution_data = None
    if 'compliance_status' in df:
        compliance_status_distribution = df['compliance_status'].value_counts().reset_index()
        compliance_status_distribution.columns = ['compliance_status', 'count']
        compliance_status_distribution_data = compliance_status_distribution.to_dict('records')

    findings_by_regulation_data = None
    if 'regulation_name' in df and 'findings_count' in df:
        findings_by_regulation = df.groupby('regulation_name')['findings_count'].sum().sort_values(ascending=False).reset_index()
        findings_by_regulation.columns = ['regulation_name', 'total_findings_count']
        findings_by_regulation_data = findings_by_regulation.to_dict('records')

    return {
        "metrics": metrics,
        "data_for_plots": {
            "compliance_status_distribution": compliance_status_distribution_data,
            "findings_count_by_regulation": findings_by_regulation_data
        }
    }
def main():
    print("🏭 Manufacturing Analytics Dashboard")
    file_path = input("Enter path to your manufacturing data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252), or press Enter for utf-8: ")
    if not encoding:
        encoding = 'utf-8'
    
    df = load_data(file_path, encoding=encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    
    analysis_options = [
        "production_analysis",
        "quality_control",
        "equipment_analysis",
        "inventory_analysis",
        "oee_analysis",
        "energy_analysis",
        "manufacturing_defect_root_cause_and_cost_analysis",
        "production_efficiency_and_quality_control_analysis",
        "manufacturing_key_performance_indicator_(kpi)_analysis",
        "real-time_production_monitoring_and_predictive_maintenance_analysis",
        "garment_factory_productivity_analysis",
        "material_fusion_process_quality_prediction_analysis",
        "electric_vehicle_manufacturer_plant_location_analysis",
        "macroeconomic_impact_on_industrial_production_analysis",
        "temperature_control_system_performance_analysis_(pid_vs._fuzzy)",
        "predictive_maintenance_priority_scoring_analysis",
        "production_order_schedule_adherence_analysis",
        "machine_availability_and_utilization_analysis",
        "manufacturing_batch_process_monitoring_analysis",
        "shift-based_production_output_and_defect_analysis",
        "quality_inspection_and_defect_resolution_analysis",
        "production_material_cost_analysis",
        "supplier_material_receipt_and_quality_analysis",
        "manufacturing_resource_utilization_analysis",
        "predictive_maintenance_sensor_data_analysis",
        "energy_consumption_and_production_efficiency_analysis",
        "quality_control_lab_test_result_analysis",
        "equipment_calibration_compliance_and_results_analysis",
        "production_delay_root_cause_analysis",
        "workplace_safety_incident_analysis",
        "equipment_maintenance_cost_and_type_analysis",
        "tool_lifecycle_and_condition_monitoring_analysis",
        "final_product_quality_grade_analysis",
        "shift_production_performance_analysis",
        "inbound_material_quality_and_delivery_analysis",
        "warehouse_inventory_stock_level_analysis",
        "order_dispatch_and_delivery_status_tracking",
        "inventory_audit_and_stock_count_analysis",
        "product_return_reason_analysis",
        "factory_environmental_conditions_monitoring",
        "manufacturing_waste_management_analysis",
        "product_packaging_process_analysis",
        "outbound_shipment_tracking_analysis",
        "machine_downtime_root_cause_analysis",
        "manufacturing_process_step_duration_analysis",
        "research_and_development_experiment_analysis",
        "barcode-based_product_traceability_analysis",
        "internal_process_and_compliance_audit_analysis",
        "machine_capacity_and_load_analysis",
        "production_volume_variance_analysis",
        "iot_sensor_data_time-series_analysis",
        "cost_center_expense_analysis",
        "regulatory_compliance_status_analysis",
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
        choice = len(analysis_options) - 1  # General Insights

    selected = analysis_options[choice] if 0 <= choice < len(analysis_options) else "General Insights"

    # Execute analysis based on selection using the actual function names from your code
    if selected == "production_analysis":
        result = production_data(df.copy())
        print_analysis_result(result, "Production Analysis")
    elif selected == "quality_control":
        result = quality_control_data(df.copy())
        print_analysis_result(result, "Quality Control Analysis")
    elif selected == "equipment_analysis":
        result = equipment_data(df.copy())
        print_analysis_result(result, "Equipment Analysis")
    elif selected == "inventory_analysis":
        result = inventory_data(df.copy())
        print_analysis_result(result, "Inventory Analysis")
    elif selected == "oee_analysis":
        result = oee_data(df.copy())
        print_analysis_result(result, "OEE Analysis")
    elif selected == "energy_analysis":
        result = energy_data(df.copy())
        print_analysis_result(result, "Energy Analysis")
    elif selected == "manufacturing_defect_root_cause_and_cost_analysis":
        result = manufacturing_defect_root_cause_and_cost_data(df.copy())
        print_analysis_result(result, "Manufacturing Defect Root Cause and Cost Analysis")
    elif selected == "production_efficiency_and_quality_control_analysis":
        result = production_efficiency_and_quality_control_data(df.copy())
        print_analysis_result(result, "Production Efficiency and Quality Control Analysis")
    elif selected == "manufacturing_key_performance_indicator_(kpi)_analysis":
        result = manufacturing_kpi_data(df.copy())
        print_analysis_result(result, "Manufacturing KPI Analysis")
    elif selected == "real-time_production_monitoring_and_predictive_maintenance_analysis":
        result = real_time_production_monitoring_and_predictive_maintenance_data(df.copy())
        print_analysis_result(result, "Real-time Production Monitoring and Predictive Maintenance Analysis")
    elif selected == "garment_factory_productivity_analysis":
        result = garment_factory_productivity_data(df.copy())
        print_analysis_result(result, "Garment Factory Productivity Analysis")
    elif selected == "material_fusion_process_quality_prediction_analysis":
        result = material_fusion_process_quality_prediction_data(df.copy())
        print_analysis_result(result, "Material Fusion Process Quality Prediction Analysis")
    elif selected == "electric_vehicle_manufacturer_plant_location_analysis":
        result = electric_vehicle_manufacturer_plant_location_data(df.copy())
        print_analysis_result(result, "Electric Vehicle Manufacturer Plant Location Analysis")
    elif selected == "macroeconomic_impact_on_industrial_production_analysis":
        result = macroeconomic_impact_on_industrial_production_data(df.copy())
        print_analysis_result(result, "Macroeconomic Impact on Industrial Production Analysis")
    elif selected == "temperature_control_system_performance_analysis_(pid_vs._fuzzy)":
        result = temperature_control_system_performance_data(df.copy())
        print_analysis_result(result, "Temperature Control System Performance Analysis")
    elif selected == "predictive_maintenance_priority_scoring_analysis":
        result = predictive_maintenance_priority_scoring_data(df.copy())
        print_analysis_result(result, "Predictive Maintenance Priority Scoring Analysis")
    elif selected == "production_order_schedule_adherence_analysis":
        result = production_order_schedule_adherence_data(df.copy())
        print_analysis_result(result, "Production Order Schedule Adherence Analysis")
    elif selected == "machine_availability_and_utilization_analysis":
        result = machine_availability_and_utilization_data(df.copy())
        print_analysis_result(result, "Machine Availability and Utilization Analysis")
    elif selected == "manufacturing_batch_process_monitoring_analysis":
        result = manufacturing_batch_process_monitoring_data(df.copy())
        print_analysis_result(result, "Manufacturing Batch Process Monitoring Analysis")
    elif selected == "shift-based_production_output_and_defect_analysis":
        result = shift_based_production_output_and_defect_analysis(df.copy())
        print_analysis_result(result, "Shift-based Production Output and Defect Analysis")
    elif selected == "quality_inspection_and_defect_resolution_analysis":
        result = quality_inspection_and_defect_resolution_data(df.copy())
        print_analysis_result(result, "Quality Inspection and Defect Resolution Analysis")
    elif selected == "production_material_cost_analysis":
        result = production_material_cost_analysis(df.copy())
        print_analysis_result(result, "Production Material Cost Analysis")
    elif selected == "supplier_material_receipt_and_quality_analysis":
        result = supplier_material_receipt_and_quality_data(df.copy())
        print_analysis_result(result, "Supplier Material Receipt and Quality Analysis")
    elif selected == "manufacturing_resource_utilization_analysis":
        result = manufacturing_resource_utilization_analysis(df.copy())
        print_analysis_result(result, "Manufacturing Resource Utilization Analysis")
    elif selected == "predictive_maintenance_sensor_data_analysis":
        result = predictive_maintenance_sensor_data_analysis(df.copy())
        print_analysis_result(result, "Predictive Maintenance Sensor Data Analysis")
    elif selected == "energy_consumption_and_production_efficiency_analysis":
        result = energy_consumption_and_production_efficiency_data(df.copy())
        print_analysis_result(result, "Energy Consumption and Production Efficiency Analysis")
    elif selected == "quality_control_lab_test_result_analysis":
        result = quality_control_lab_test_result_analysis(df.copy())
        print_analysis_result(result, "Quality Control Lab Test Result Analysis")
    elif selected == "equipment_calibration_compliance_and_results_analysis":
        result = equipment_calibration_compliance_and_results_analysis(df.copy())
        print_analysis_result(result, "Equipment Calibration Compliance and Results Analysis")
    elif selected == "production_delay_root_cause_analysis":
        result = production_delay_root_cause_analysis(df.copy())
        print_analysis_result(result, "Production Delay Root Cause Analysis")
    elif selected == "workplace_safety_incident_analysis":
        result = workplace_safety_incident_analysis(df.copy())
        print_analysis_result(result, "Workplace Safety Incident Analysis")
    elif selected == "equipment_maintenance_cost_and_type_analysis":
        result = equipment_maintenance_cost_and_type_analysis(df.copy())
        print_analysis_result(result, "Equipment Maintenance Cost and Type Analysis")
    elif selected == "tool_lifecycle_and_condition_monitoring_analysis":
        result = tool_lifecycle_and_condition_monitoring_analysis(df.copy())
        print_analysis_result(result, "Tool Lifecycle and Condition Monitoring Analysis")
    elif selected == "final_product_quality_grade_analysis":
        result = final_product_quality_grade_analysis(df.copy())
        print_analysis_result(result, "Final Product Quality Grade Analysis")
    elif selected == "shift_production_performance_analysis":
        result = shift_production_performance_analysis(df.copy())
        print_analysis_result(result, "Shift Production Performance Analysis")
    elif selected == "inbound_material_quality_and_delivery_analysis":
        result = inbound_material_quality_and_delivery_analysis(df.copy())
        print_analysis_result(result, "Inbound Material Quality and Delivery Analysis")
    elif selected == "warehouse_inventory_stock_level_analysis":
        result = warehouse_inventory_stock_level_analysis(df.copy())
        print_analysis_result(result, "Warehouse Inventory Stock Level Analysis")
    elif selected == "order_dispatch_and_delivery_status_tracking":
        result = order_dispatch_and_delivery_status_tracking(df.copy())
        print_analysis_result(result, "Order Dispatch and Delivery Status Tracking")
    elif selected == "inventory_audit_and_stock_count_analysis":
        result = inventory_audit_and_stock_count_analysis(df.copy())
        print_analysis_result(result, "Inventory Audit and Stock Count Analysis")
    elif selected == "product_return_reason_analysis":
        result = product_return_reason_analysis(df.copy())
        print_analysis_result(result, "Product Return Reason Analysis")
    elif selected == "factory_environmental_conditions_monitoring":
        result = factory_environmental_conditions_monitoring(df.copy())
        print_analysis_result(result, "Factory Environmental Conditions Monitoring")
    elif selected == "manufacturing_waste_management_analysis":
        result = manufacturing_waste_management_analysis(df.copy())
        print_analysis_result(result, "Manufacturing Waste Management Analysis")
    elif selected == "product_packaging_process_analysis":
        result = product_packaging_process_analysis(df.copy())
        print_analysis_result(result, "Product Packaging Process Analysis")
    elif selected == "outbound_shipment_tracking_analysis":
        result = outbound_shipment_tracking_analysis(df.copy())
        print_analysis_result(result, "Outbound Shipment Tracking Analysis")
    elif selected == "machine_downtime_root_cause_analysis":
        result = machine_downtime_root_cause_data(df.copy())
        print_analysis_result(result, "Machine Downtime Root Cause Analysis")
    elif selected == "manufacturing_process_step_duration_analysis":
        result = manufacturing_process_step_duration_analysis(df.copy())
        print_analysis_result(result, "Manufacturing Process Step Duration Analysis")
    elif selected == "research_and_development_experiment_analysis":
        result = research_and_development_experiment_analysis(df.copy())
        print_analysis_result(result, "Research and Development Experiment Analysis")
    elif selected == "barcode-based_product_traceability_analysis":
        result = barcode_based_product_traceability_analysis(df.copy())
        print_analysis_result(result, "Barcode-based Product Traceability Analysis")
    elif selected == "internal_process_and_compliance_audit_analysis":
        result = internal_process_and_compliance_audit_analysis(df.copy())
        print_analysis_result(result, "Internal Process and Compliance Audit Analysis")
    elif selected == "machine_capacity_and_load_analysis":
        result = machine_capacity_and_load_analysis(df.copy())
        print_analysis_result(result, "Machine Capacity and Load Analysis")
    elif selected == "production_volume_variance_analysis":
        result = production_volume_variance_analysis(df.copy())
        print_analysis_result(result, "Production Volume Variance Analysis")
    elif selected == "iot_sensor_data_time-series_analysis":
        result = iot_sensor_data_time_series_analysis(df.copy())
        print_analysis_result(result, "IoT Sensor Data Time-series Analysis")
    elif selected == "cost_center_expense_analysis":
        result = cost_center_expense_analysis(df.copy())
        print_analysis_result(result, "Cost Center Expense Analysis")
    elif selected == "regulatory_compliance_status_analysis":
        result = regulatory_compliance_status_analysis(df.copy())
        print_analysis_result(result, "Regulatory Compliance Status Analysis")
    elif selected == "General Insights":
        result = get_general_insights_data(df.copy())
        print_general_insights(result)
    else:
        print(f"Analysis option '{selected}' not recognized or not implemented.")
        result = get_general_insights_data(df.copy())
        print_general_insights(result)

def print_analysis_result(result, analysis_name):
    """
    Helper function to print analysis results in a formatted way.
    """
    print(f"\n{'='*60}")
    print(f"📊 {analysis_name} Results")
    print('='*60)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        if "missing_columns" in result:
            print(f"Missing columns: {', '.join(result['missing_columns'])}")
        if "general_insights" in result:
            print("\nShowing general insights instead:")
            print_general_insights(result['general_insights'])
        return
    
    if "metrics" in result:
        print("\n📈 Key Metrics:")
        print("-" * 40)
        for key, value in result["metrics"].items():
            if isinstance(value, (int, float)):
                if isinstance(value, float):
                    print(f"{key}: {value:.2f}")
                else:
                    print(f"{key}: {value:,}")
            else:
                print(f"{key}: {value}")
    
    if "data_for_plots" in result:
        print(f"\n📊 Data available for visualization:")
        print("-" * 40)
        for plot_name, data in result["data_for_plots"].items():
            if data is not None:
                if isinstance(data, list):
                    print(f"✓ {plot_name.replace('_', ' ').title()}: {len(data)} data points")
                else:
                    print(f"✓ {plot_name.replace('_', ' ').title()}: Available")
            else:
                print(f"✗ {plot_name.replace('_', ' ').title()}: No data")

def print_general_insights(insights):
    """
    Helper function to print general insights in a formatted way.
    """
    print("\n📊 General Data Insights")
    print("="*50)
    
    if "key_metrics" in insights:
        print("\n📈 Dataset Overview:")
        print("-" * 30)
        for key, value in insights["key_metrics"].items():
            print(f"{key}: {value:,}")
    
    if "numeric_summaries" in insights and insights["numeric_summaries"]:
        print(f"\n🔢 Numeric Features Summary:")
        print("-" * 30)
        for col, summary in list(insights["numeric_summaries"].items())[:5]:  # Show first 5
            print(f"\n{col}:")
            print(f"  Mean: {summary.get('mean', 0):.2f}")
            print(f"  Std:  {summary.get('std', 0):.2f}")
            print(f"  Min:  {summary.get('min', 0):.2f}")
            print(f"  Max:  {summary.get('max', 0):.2f}")
    
    if "categorical_value_counts" in insights and insights["categorical_value_counts"]:
        print(f"\n📝 Categorical Features (Top Categories):")
        print("-" * 30)
        for col, counts in list(insights["categorical_value_counts"].items())[:3]:  # Show first 3
            print(f"\n{col}:")
            for category, count in list(counts.items())[:3]:  # Show top 3 categories
                print(f"  {category}: {count}")

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

if __name__ == "__main__":
    main()
