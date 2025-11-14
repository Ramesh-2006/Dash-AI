import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# ========== UTILITY FUNCTIONS (Adapted from Example) ==========

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
            # Convert Period to string
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
        # Dump and reload to convert all nested types
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception as e:
        # Fallback for complex unhandled types
        print(f"Warning: Type conversion error - {e}")
        # Return a string representation as a last resort
        return json.loads(json.dumps(str(data)))

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
            # Use a slightly more lenient threshold for manufacturing data
            match, score = process.extractOne(target, available)
            matched[target] = match if score >= 70 else None
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    # Create a rename mapping only for columns that were successfully matched (not None)
    rename_map = {v: k for k, v in matched.items() if v is not None and v in df.columns}
    return df.rename(columns=rename_map)

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
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime', 'datetimetz']).columns.tolist()
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
            except:
                pass # Failsafe
        
        # Categorical columns analysis
        categorical_stats = {}
        if categorical_cols:
            for col in categorical_cols[:5]: # Limit to first 5 for brevity
                try:
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(5).to_dict()
                    categorical_stats[col] = {
                        "unique_count": int(unique_count),
                        "top_values": convert_to_native_types(top_values)
                    }
                except:
                    pass # Failsafe

        # --- Create visualizations ---
        
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
                    missing_df.head(15), 
                    x='column', 
                    y='missing_percentage',
                    title='Top 15 Columns with Missing Values (%)'
                )
                visualizations["missing_values"] = fig_missing.to_json()
            else:
                # Create a placeholder "No Missing Values" plot
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
        
        # 3. Numeric columns distributions (first 2)
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:2]):
                try:
                    fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception:
                    pass
        
        # 4. Categorical columns distributions (first 2)
        if categorical_cols:
            for i, col in enumerate(categorical_cols[:2]):
                try:
                    top_10 = df[col].value_counts().head(10).reset_index()
                    top_10.columns = [col, 'count']
                    fig_bar = px.bar(top_10, x=col, y='count', title=f'Top 10 Categories for {col}')
                    visualizations[f"{col}_distribution"] = fig_bar.to_json()
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
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "datetime_columns_count": len(datetime_cols),
                "other_columns_count": len(other_cols),
                "numeric_columns_list": numeric_cols[:10], # Show first 10
                "categorical_columns_list": categorical_cols[:10] # Show first 10
            },
            "data_quality": {
                "total_missing_values": int(missing_values.sum()),
                "columns_with_missing_count": len(columns_with_missing),
                "complete_columns_count": len(df.columns) - len(columns_with_missing)
            },
            "numeric_column_stats": numeric_stats,
            "categorical_column_stats": categorical_stats
        }
        
        # Generate insights
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns.",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns.",
        ]
        
        if missing_cols and len(missing_cols) > 0:
            insights.append("")
            insights.append("⚠️ REQUIRED COLUMNS NOT FOUND for the requested analysis.")
            insights.append(f"Missing columns: {', '.join(missing_cols)}")
            if matched_cols:
                matches_found = [f"'{k}' (matched to: '{v}')" for k,v in matched_cols.items() if v is not None and k in missing_cols]
                if matches_found:
                    insights.append(f"Fuzzy matching found potential alternatives: {', '.join(matches_found)}.")
            insights.append("Showing General Data Analysis instead.")
            insights.append("")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data).")
        else:
            insights.append("No duplicate rows found. ✅")
        
        if len(columns_with_missing) > 0:
            insights.append(f"{len(columns_with_missing)} columns have missing values. Top affected: {', '.join(columns_with_missing.sort_values(ascending=False).index.tolist()[:3])}")
        else:
            insights.append("No missing values found in the dataset. ✅")
        
        insights.append(f"Generated {len(visualizations)} visualizations for general data exploration.")
        
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
        # Ultra-safe fallback
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "matched_columns": matched_cols or {},
            "visualizations": {},
            "metrics": {"total_rows": len(df), "total_columns": len(df.columns)},
            "insights": [
                f"An error occurred during general analysis: {e}",
                f"Dataset has {len(df)} rows and {len(df.columns)} columns."
            ],
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
    except Exception as fallback_error:
        print(f"General insights fallback also failed: {fallback_error}")
        general_insights_data = {
            "visualizations": {},
            "metrics": {"total_rows": len(df), "total_columns": len(df.columns)},
            "insights": [f"General insights fallback failed: {fallback_error}"]
        }

    # Create the specific error response
    missing_info = {}
    for col in missing_cols:
        match_info = f" (fuzzy matched to: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (No close match found)"
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
    except FileNotFoundError:
        print(f"[ERROR] File not found at path: {file_path}")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading file: {e}")
        return None

# ========== MANUFACTURING ANALYSIS FUNCTIONS ==========

def production_data(df):
    analysis_name = "Production Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['production_id', 'product_code', 'production_date', 'quantity_produced',
                    'quantity_defective', 'production_line', 'operator_id', 'cycle_time']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        if 'production_date' in df and not pd.api.types.is_datetime64_any_dtype(df['production_date']):
            df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
        
        df['quantity_produced'] = pd.to_numeric(df['quantity_produced'], errors='coerce')
        df['quantity_defective'] = pd.to_numeric(df['quantity_defective'], errors='coerce')
        if 'cycle_time' in df:
             df['cycle_time'] = pd.to_numeric(df['cycle_time'], errors='coerce')

        df = df.dropna(subset=['quantity_produced', 'quantity_defective'])

        total_production = df['quantity_produced'].sum()
        defect_rate = (df['quantity_defective'].sum() / total_production * 100) if total_production > 0 else 0
        avg_cycle_time = df['cycle_time'].mean() if 'cycle_time' in df and not df['cycle_time'].isnull().all() else None

        metrics = {
            "Total Production": total_production,
            "Defect Rate": defect_rate,
            "Production Runs": len(df),
            "Avg Cycle Time": avg_cycle_time
        }

        insights.append(f"Total production across {len(df)} runs: {total_production:,.0f} units.")
        insights.append(f"Overall defect rate: {defect_rate:.2f}%.")
        if avg_cycle_time:
            insights.append(f"Average cycle time: {avg_cycle_time:.2f} (units depend on data).")

        # Visualizations
        if 'production_date' in df.columns and not df['production_date'].isnull().all():
            production_trend_data = df.groupby('production_date')['quantity_produced'].sum().reset_index()
            fig1 = px.line(production_trend_data, x='production_date', y='quantity_produced', title="Production Trend Over Time")
            visualizations["production_trend"] = fig1.to_json()

        if 'production_line' in df.columns:
            line_performance = df.groupby('production_line').agg(
                quantity_produced=('quantity_produced', 'sum'),
                quantity_defective=('quantity_defective', 'sum')
            ).reset_index()
            line_performance['defect_rate'] = (line_performance['quantity_defective'] / line_performance['quantity_produced']) * 100
            
            fig2 = px.bar(line_performance, x='production_line', y='quantity_produced', title='Total Production by Line')
            visualizations["production_by_line"] = fig2.to_json()
            
            fig3 = px.bar(line_performance, x='production_line', y='defect_rate', title='Defect Rate (%) by Line')
            visualizations["defect_rate_by_line"] = fig3.to_json()

        if 'cycle_time' in df.columns and 'product_code' in df.columns:
            fig4 = px.box(df, x='product_code', y='cycle_time', title='Cycle Time by Product')
            visualizations["cycle_time_by_product"] = fig4.to_json()

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

def quality_control_data(df):
    analysis_name = "Quality Control Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['inspection_id', 'product_code', 'inspection_date', 'defect_type',
                    'severity', 'inspector_id', 'batch_number', 'corrective_action']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        if 'inspection_date' in df and not pd.api.types.is_datetime64_any_dtype(df['inspection_date']):
            df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
        if 'severity' in df:
            df['severity'] = pd.to_numeric(df['severity'], errors='coerce')
        
        df.dropna(subset=['defect_type'], inplace=True)

        total_inspections = len(df)
        unique_defects = df['defect_type'].nunique()
        avg_severity = df['severity'].mean() if 'severity' in df and not df['severity'].isnull().all() else None

        metrics = {
            "Total Quality Inspections": total_inspections,
            "Unique Defect Types": unique_defects,
            "Avg Defect Severity": avg_severity
        }

        insights.append(f"Analyzed {total_inspections} quality inspections.")
        insights.append(f"Found {unique_defects} unique defect types.")
        if avg_severity:
            insights.append(f"Average defect severity: {avg_severity:.2f}.")

        # Visualizations
        if 'defect_type' in df.columns:
            defect_counts = df['defect_type'].value_counts().reset_index()
            defect_counts.columns = ['Defect Type', 'Count']
            fig1 = px.pie(defect_counts.head(10), names='Defect Type', values='Count', title='Top 10 Defect Type Frequency')
            visualizations["defect_type_frequency"] = fig1.to_json()

        if 'inspection_date' in df.columns and not df['inspection_date'].isnull().all():
            defects_over_time = df.groupby('inspection_date').size().reset_index(name='count')
            fig2 = px.line(defects_over_time, x='inspection_date', y='count', title='Defects Over Time')
            visualizations["defects_over_time"] = fig2.to_json()

        if 'defect_type' in df.columns and 'severity' in df.columns and not df['severity'].isnull().all():
            fig3 = px.box(df, x='defect_type', y='severity', title='Defect Severity by Type')
            visualizations["defect_severity_by_type"] = fig3.to_json()

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

def equipment_data(df):
    analysis_name = "Equipment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['equipment_id', 'equipment_type', 'last_maintenance', 'next_maintenance',
                    'downtime_hours', 'utilization_rate', 'failure_count', 'status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        date_cols = ['last_maintenance', 'next_maintenance']
        for col in date_cols:
            if col in df and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        if 'downtime_hours' in df:
            df['downtime_hours'] = pd.to_numeric(df['downtime_hours'], errors='coerce')
        if 'utilization_rate' in df:
            df['utilization_rate'] = pd.to_numeric(df['utilization_rate'], errors='coerce')
        
        df.dropna(subset=['equipment_id'], inplace=True) # Keep equipment even if some metrics are missing

        total_equipment = len(df)
        avg_downtime = df['downtime_hours'].mean() if 'downtime_hours' in df and not df['downtime_hours'].isnull().all() else None
        avg_utilization = df['utilization_rate'].mean() if 'utilization_rate' in df and not df['utilization_rate'].isnull().all() else None

        metrics = {
            "Total Equipment": total_equipment,
            "Average Downtime (hours)": avg_downtime,
            "Average Utilization (%)": avg_utilization
        }

        insights.append(f"Analyzed {total_equipment} pieces of equipment.")
        if avg_downtime:
            insights.append(f"Average downtime: {avg_downtime:.2f} hours.")
        if avg_utilization:
            insights.append(f"Average utilization: {avg_utilization:.2f}%.")

        # Visualizations
        if 'status' in df.columns:
            status_distribution_data = df['status'].value_counts().reset_index()
            status_distribution_data.columns = ['status', 'count']
            fig1 = px.pie(status_distribution_data, names='status', values='count', title='Equipment Status Distribution')
            visualizations["equipment_status_distribution"] = fig1.to_json()

        if 'equipment_type' in df.columns and 'downtime_hours' in df.columns and not df['downtime_hours'].isnull().all():
            fig2 = px.box(df, x='equipment_type', y='downtime_hours', title='Downtime by Equipment Type')
            visualizations["downtime_by_equipment_type"] = fig2.to_json()

        if 'last_maintenance' in df.columns and 'next_maintenance' in df.columns and 'equipment_id' in df.columns:
            maintenance_df = df[['equipment_id', 'last_maintenance', 'next_maintenance']].melt(
                id_vars='equipment_id',
                var_name='Maintenance Type',
                value_name='Date'
            ).dropna()
            if not maintenance_df.empty:
                # Use a scatter plot for timeline if Gantt fails
                try:
                    fig3 = px.scatter(maintenance_df, x="Date", y="equipment_id", color="Maintenance Type", title="Maintenance Schedule")
                    visualizations["equipment_maintenance_schedule"] = fig3.to_json()
                except Exception as e:
                    insights.append(f"Could not generate maintenance schedule plot: {e}")

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

def inventory_data(df):
    analysis_name = "Inventory Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['sku', 'product_name', 'current_stock', 'min_stock',
                    'max_stock', 'lead_time', 'turnover_rate', 'last_order_date']
        matched = fuzzy_match_column(df, expected)
        # Critical for this analysis
        critical_cols = ['current_stock', 'min_stock', 'product_name']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        if 'last_order_date' in df and not pd.api.types.is_datetime64_any_dtype(df['last_order_date']):
            df['last_order_date'] = pd.to_datetime(df['last_order_date'], errors='coerce')

        df['current_stock'] = pd.to_numeric(df['current_stock'], errors='coerce')
        df['min_stock'] = pd.to_numeric(df['min_stock'], errors='coerce')
        if 'max_stock' in df:
            df['max_stock'] = pd.to_numeric(df['max_stock'], errors='coerce')
        if 'turnover_rate' in df:
            df['turnover_rate'] = pd.to_numeric(df['turnover_rate'], errors='coerce')
        
        df.dropna(subset=['current_stock', 'min_stock', 'product_name'], inplace=True)

        total_skus = len(df)
        stockout_risk = len(df[df['current_stock'] < df['min_stock']])
        stockout_risk_pct = (stockout_risk / total_skus * 100) if total_skus > 0 else 0
        avg_turnover = df['turnover_rate'].mean() if 'turnover_rate' in df and not df['turnover_rate'].isnull().all() else None

        metrics = {
            "Total SKUs": total_skus,
            "Items Below Min Stock": stockout_risk,
            "Items Below Min Stock (%)": stockout_risk_pct,
            "Avg Inventory Turnover": avg_turnover
        }

        insights.append(f"Analyzed {total_skus} SKUs.")
        insights.append(f"{stockout_risk} items ({stockout_risk_pct:.1f}%) are below minimum stock levels.")
        if avg_turnover:
             insights.append(f"Average inventory turnover rate: {avg_turnover:.2f}.")

        # Visualizations
        if 'current_stock' in df.columns:
            fig1 = px.histogram(df, x='current_stock', title='Inventory Stock Level Distribution')
            visualizations["inventory_distribution"] = fig1.to_json()

        if 'current_stock' in df.columns and 'min_stock' in df.columns and 'product_name' in df.columns:
            df['stock_status'] = 'Normal'
            df.loc[df['current_stock'] < df['min_stock'], 'stock_status'] = 'Below Minimum'
            if 'max_stock' in df.columns:
                 df.loc[df['current_stock'] > df['max_stock'], 'stock_status'] = 'Above Maximum'

            status_counts = df['stock_status'].value_counts().reset_index()
            status_counts.columns = ['Stock Status', 'Count']
            fig2 = px.pie(status_counts, names='Stock Status', values='Count', title='Inventory Status')
            visualizations["inventory_status_pie"] = fig2.to_json()

        if 'current_stock' in df.columns and 'product_name' in df.columns:
            df_sorted = df.sort_values('current_stock', ascending=False).copy()
            df_sorted['cumulative_stock'] = df_sorted['current_stock'].cumsum()
            df_sorted['cumulative_percent'] = (df_sorted['cumulative_stock'] / df_sorted['current_stock'].sum()) * 100
            
            fig3 = px.area(df_sorted, y='cumulative_percent', title='ABC Analysis (Cumulative Stock %)')
            fig3.update_layout(xaxis_title="Products (Sorted by Stock Level)", yaxis_title="Cumulative Stock (%)")
            visualizations["abc_analysis"] = fig3.to_json()

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

def oee_data(df):
    analysis_name = "OEE Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['machine_id', 'shift_date', 'shift', 'availability',
                    'performance', 'quality', 'oee', 'planned_production_time']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['availability', 'performance', 'quality', 'oee']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)

        if 'shift_date' in df and not pd.api.types.is_datetime64_any_dtype(df['shift_date']):
            df['shift_date'] = pd.to_datetime(df['shift_date'], errors='coerce')

        for col in ['availability', 'performance', 'quality', 'oee']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['availability', 'performance', 'quality', 'oee'], inplace=True)

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

        insights.append(f"Average OEE: {avg_oee:.2f}%.")
        insights.append(f"OEE is driven by Availability ({avg_availability:.2f}%), Performance ({avg_performance:.2f}%), and Quality ({avg_quality:.2f}%).")

        # Visualizations
        if 'shift_date' in df.columns and not df['shift_date'].isnull().all():
            oee_trend = df.groupby('shift_date')['oee'].mean().reset_index()
            fig1 = px.line(oee_trend, x='shift_date', y='oee', title='OEE Trend Over Time')
            visualizations["oee_trend"] = fig1.to_json()

        components = df[['availability', 'performance', 'quality']].mean().reset_index()
        components.columns = ['Component', 'Value']
        fig2 = px.pie(components, names='Component', values='Value', title='OEE Components Breakdown (Average)', hole=0.4)
        visualizations["oee_components_breakdown"] = fig2.to_json()

        if 'shift' in df.columns:
            fig3 = px.box(df, x='shift', y='oee', title='OEE by Shift')
            visualizations["oee_by_shift"] = fig3.to_json()

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

def energy_data(df):
    analysis_name = "Energy Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['meter_id', 'timestamp', 'energy_consumption', 'cost',
                    'machine_id', 'production_volume', 'energy_per_unit']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['timestamp', 'energy_consumption']
        missing = [col for col in critical_cols if matched[col] is None]
        
        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)

        if 'timestamp' in df and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        df['energy_consumption'] = pd.to_numeric(df['energy_consumption'], errors='coerce')
        if 'cost' in df:
            df['cost'] = pd.to_numeric(df['cost'], errors='coerce')
        if 'production_volume' in df:
            df['production_volume'] = pd.to_numeric(df['production_volume'], errors='coerce')
        if 'energy_per_unit' in df:
            df['energy_per_unit'] = pd.to_numeric(df['energy_per_unit'], errors='coerce')
        
        df.dropna(subset=['timestamp', 'energy_consumption'], inplace=True)

        total_energy = df['energy_consumption'].sum()
        total_cost = df['cost'].sum() if 'cost' in df and not df['cost'].isnull().all() else None
        avg_energy_per_unit = df['energy_per_unit'].mean() if 'energy_per_unit' in df and not df['energy_per_unit'].isnull().all() else None

        metrics = {
            "Total Energy (kWh)": total_energy,
            "Total Cost ($)": total_cost,
            "Avg Energy per Unit (kWh/unit)": avg_energy_per_unit
        }

        insights.append(f"Total energy consumed: {total_energy:,.0f} kWh.")
        if total_cost:
            insights.append(f"Total energy cost: ${total_cost:,.0f}.")
        if avg_energy_per_unit:
            insights.append(f"Average energy efficiency: {avg_energy_per_unit:.2f} kWh per unit.")

        # Visualizations
        if 'timestamp' in df.columns and not df['timestamp'].isnull().all():
            energy_trend = df.groupby('timestamp')['energy_consumption'].sum().reset_index()
            fig1 = px.line(energy_trend, x='timestamp', y='energy_consumption', title='Energy Consumption Over Time')
            visualizations["energy_consumption_over_time"] = fig1.to_json()

        if 'machine_id' in df.columns and 'energy_per_unit' in df.columns and not df['energy_per_unit'].isnull().all():
            fig2 = px.box(df, x='machine_id', y='energy_per_unit', title='Energy Efficiency (kWh/unit) by Machine')
            visualizations["energy_efficiency_by_machine"] = fig2.to_json()

        if 'production_volume' in df.columns and not df['production_volume'].isnull().all():
            fig3 = px.scatter(df, x='production_volume', y='energy_consumption', title='Energy vs. Production Volume', trendline="ols")
            visualizations["energy_vs_production_volume"] = fig3.to_json()

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

def manufacturing_defect_root_cause_and_cost_data(df):
    analysis_name = "Manufacturing Defect Root Cause and Cost Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['defect_type', 'defect_location', 'severity', 'repair_cost']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['defect_type', 'repair_cost']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['repair_cost'] = pd.to_numeric(df['repair_cost'], errors='coerce')
        df.dropna(subset=['defect_type', 'repair_cost'], inplace=True)

        total_repair_cost = df['repair_cost'].sum()
        avg_repair_cost = df['repair_cost'].mean()
        most_common_defect = df['defect_type'].mode()[0] if not df['defect_type'].empty else None

        metrics = {
            "Total Repair Cost": total_repair_cost,
            "Average Repair Cost": avg_repair_cost,
            "Most Common Defect": most_common_defect
        }

        insights.append(f"Total repair cost: ${total_repair_cost:,.0f}.")
        insights.append(f"Average repair cost per defect: ${avg_repair_cost:,.2f}.")
        insights.append(f"The most frequent defect type is: {most_common_defect}.")

        # Visualizations
        cost_by_defect = df.groupby('defect_type')['repair_cost'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(cost_by_defect.head(10), x='defect_type', y='repair_cost', title='Top 10 Total Repair Cost by Defect Type')
        visualizations["cost_by_defect_type"] = fig1.to_json()

        if 'defect_location' in df.columns:
            cost_by_location = df.groupby('defect_location')['repair_cost'].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(cost_by_location.head(10), x='defect_location', y='repair_cost', title='Top 10 Total Repair Cost by Defect Location')
            visualizations["cost_by_defect_location"] = fig2.to_json()

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

def production_efficiency_and_quality_control_data(df):
    analysis_name = "Production Efficiency and Quality Control Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'product_type', 'units_produced', 'defects', 'production_time_hours', 'down_time_hours']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['date', 'units_produced', 'defects', 'production_time_hours', 'down_time_hours']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['units_produced', 'defects', 'production_time_hours', 'down_time_hours']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

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

        insights.append(f"Total units produced: {total_units:,.0f}.")
        insights.append(f"Average defect rate: {avg_defect_rate:.2f}%.")
        insights.append(f"Average production availability: {avg_availability:.2f}%.")

        # Visualizations
        if 'product_type' in df.columns:
            defect_rate_by_product = df.groupby('product_type')['defect_rate'].mean().sort_values().reset_index()
            fig1 = px.bar(defect_rate_by_product, x='product_type', y='defect_rate', title='Average Defect Rate by Product')
            visualizations["defect_rate_by_product"] = fig1.to_json()

        daily_prod = df.groupby('date')[['units_produced', 'defects']].sum().reset_index()
        daily_prod_melted = daily_prod.melt(id_vars='date', var_name='Metric', value_name='Count')
        fig2 = px.line(daily_prod_melted, x='date', y='Count', color='Metric', title='Daily Production vs. Defects')
        visualizations["daily_production_vs_defects"] = fig2.to_json()

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

def manufacturing_kpi_data(df):
    analysis_name = "Manufacturing KPI Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['productionvolume', 'productioncost', 'supplierquality', 'deliverydelay', 'defectrate', 'maintenancedurasi', 'downtimepercentage', 'workerproductivity', 'safetyincidents']
        matched = fuzzy_match_column(df, expected)
        # Check for at least a few KPIs
        kpi_cols = [col for col in expected if matched[col] is not None]
        if len(kpi_cols) < 3:
            return create_fallback_response(analysis_name, expected, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in kpi_cols: # Use only matched columns
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=kpi_cols, inplace=True)

        # Calculate metrics for the KPIs that *are* present
        metrics = {}
        if 'defectrate' in df.columns:
            metrics["Average Defect Rate"] = df['defectrate'].mean()
            insights.append(f"Average Defect Rate: {metrics['Average Defect Rate']:.2f}%.")
        if 'workerproductivity' in df.columns:
            metrics["Average Worker Productivity"] = df['workerproductivity'].mean()
            insights.append(f"Average Worker Productivity: {metrics['Average Worker Productivity']:.2f}.")
        if 'safetyincidents' in df.columns:
            metrics["Total Safety Incidents"] = df['safetyincidents'].sum()
            insights.append(f"Total Safety Incidents: {metrics['Total Safety Incidents']}.")

        # Visualizations
        if len(kpi_cols) >= 2:
            corr_matrix = df[kpi_cols].corr()
            fig1 = px.imshow(corr_matrix, text_auto=True, title='KPI Correlation Matrix')
            visualizations["correlation_matrix_kpis"] = fig1.to_json()

        if 'workerproductivity' in df.columns and 'defectrate' in df.columns:
            color_col = 'productioncost' if 'productioncost' in df.columns else None
            fig2 = px.scatter(df, x='workerproductivity', y='defectrate', color=color_col,
                              title='Defect Rate vs. Worker Productivity', trendline="ols")
            visualizations["defect_rate_vs_worker_productivity"] = fig2.to_json()

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

def real_time_production_monitoring_and_predictive_maintenance_data(df):
    analysis_name = "Real-time Production Monitoring and Predictive Maintenance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['timestamp', 'machine_id', 'temperature_c', 'vibration_hz', 'power_consumption_kw', 'predictive_maintenance_score']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['timestamp', 'machine_id', 'temperature_c', 'vibration_hz', 'predictive_maintenance_score']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['temperature_c', 'vibration_hz', 'power_consumption_kw', 'predictive_maintenance_score']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('timestamp').dropna(subset=critical_cols)

        if df.empty:
            raise ValueError("No valid data remaining after cleaning.")

        latest_temp = df['temperature_c'].iloc[-1]
        avg_vibration = df['vibration_hz'].mean()
        avg_pred_score = df['predictive_maintenance_score'].mean()
        high_risk_machines = df[df['predictive_maintenance_score'] > 0.8]['machine_id'].nunique()

        metrics = {
            "Latest Temperature (°C)": latest_temp,
            "Average Vibration (Hz)": avg_vibration,
            "Average Predictive Maintenance Score": avg_pred_score,
            "Machines with High Maintenance Score (>0.8)": high_risk_machines
        }

        insights.append(f"Latest recorded temperature: {latest_temp:.1f}°C.")
        insights.append(f"Average predictive maintenance score: {avg_pred_score:.2f}.")
        insights.append(f"{high_risk_machines} machines are currently at high risk (score > 0.8).")

        # Visualizations
        sensor_readings_data = df[['timestamp', 'machine_id', 'temperature_c', 'vibration_hz', 'power_consumption_kw']].melt(
            id_vars=['timestamp', 'machine_id'], var_name='Sensor', value_name='Value'
        )
        
        # Plot for the first machine_id as an example
        example_machine = df['machine_id'].iloc[0]
        fig1 = px.line(sensor_readings_data[sensor_readings_data['machine_id'] == example_machine], 
                       x='timestamp', y='Value', color='Sensor', facet_row='Sensor',
                       title=f'Sensor Readings Over Time (Example: Machine {example_machine})')
        fig1.update_yaxes(matches=None) # Unlink y-axes
        visualizations["sensor_readings_by_machine"] = fig1.to_json()
        insights.append(f"Generated sensor trend plot for an example machine: {example_machine}.")

        fig2 = px.scatter(df, x='temperature_c', y='vibration_hz', color='predictive_maintenance_score',
                          title='Temperature vs. Vibration (Colored by Maintenance Score)')
        visualizations["temperature_vs_vibration_by_pred_score"] = fig2.to_json()

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

def garment_factory_productivity_data(df):
    analysis_name = "Garment Factory Productivity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'department', 'team', 'targeted_productivity', 'smv', 'wip', 'over_time', 'incentive', 'actual_productivity']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['department', 'targeted_productivity', 'actual_productivity', 'incentive']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['targeted_productivity', 'actual_productivity', 'smv', 'wip', 'over_time', 'incentive']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
        df.dropna(subset=critical_cols, inplace=True)

        avg_actual_prod = df['actual_productivity'].mean()
        avg_targeted_prod = df['targeted_productivity'].mean()
        achievement_rate = (avg_actual_prod / avg_targeted_prod) * 100 if avg_targeted_prod > 0 else 0
        incentive_prod_corr = df['incentive'].corr(df['actual_productivity'])

        metrics = {
            "Average Actual Productivity": avg_actual_prod,
            "Average Target Productivity": avg_targeted_prod,
            "Overall Achievement Rate": achievement_rate,
            "Incentive/Productivity Correlation": incentive_prod_corr
        }

        insights.append(f"Overall productivity achievement rate: {achievement_rate:.2f}%.")
        insights.append(f"Average actual productivity: {avg_actual_prod:.2f} (vs. target of {avg_targeted_prod:.2f}).")
        insights.append(f"Correlation between incentive and productivity: {incentive_prod_corr:.2f}.")

        # Visualizations
        prod_by_dept = df.groupby('department')[['targeted_productivity', 'actual_productivity']].mean().reset_index()
        prod_by_dept_melted = prod_by_dept.melt(id_vars='department', var_name='Productivity Type', value_name='Value')
        fig1 = px.bar(prod_by_dept_melted, x='department', y='Value', color='Productivity Type', barmode='group',
                      title='Productivity (Actual vs. Target) by Department')
        visualizations["productivity_by_department"] = fig1.to_json()

        fig2 = px.scatter(df, x='incentive', y='actual_productivity', color='department',
                          title='Impact of Incentive on Productivity', trendline="ols")
        visualizations["impact_of_incentive_on_productivity"] = fig2.to_json()

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

def material_fusion_process_quality_prediction_data(df):
    analysis_name = "Material Fusion Process Quality Prediction"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['temperature_c', 'pressure_kpa', 'material_fusion_metric', 'material_transformation_metric', 'quality_rating']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['temperature_c', 'pressure_kpa', 'material_fusion_metric', 'quality_rating']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected: # All expected are numeric
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        avg_quality = df['quality_rating'].mean()
        temp_corr = df['temperature_c'].corr(df['quality_rating'])
        pressure_corr = df['pressure_kpa'].corr(df['quality_rating'])

        metrics = {
            "Average Quality Rating": avg_quality,
            "Temperature/Quality Correlation": temp_corr,
            "Pressure/Quality Correlation": pressure_corr
        }

        insights.append(f"Average quality rating: {avg_quality:.2f}.")
        insights.append(f"Temperature correlation with quality: {temp_corr:.2f}.")
        insights.append(f"Pressure correlation with quality: {pressure_corr:.2f}.")

        # Visualizations
        fig1 = px.scatter_3d(df, x='temperature_c', y='pressure_kpa', z='material_fusion_metric',
                            color='quality_rating', title='Fusion Metric by Temp, Pressure, and Quality')
        visualizations["fusion_metric_by_temp_pressure"] = fig1.to_json()

        fig2 = px.density_heatmap(df, x="temperature_c", y="pressure_kpa", z="quality_rating", histfunc="avg",
                                  title="Heatmap of Average Quality by Temperature and Pressure")
        visualizations["heatmap_avg_quality_by_temp_pressure"] = fig2.to_json()

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

def electric_vehicle_manufacturer_plant_location_data(df):
    analysis_name = "Electric Vehicle Manufacturer Plant Location"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['ev_maker', 'place', 'state']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['ev_maker', 'state']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df.dropna(subset=critical_cols, inplace=True)

        num_makers = df['ev_maker'].nunique()
        num_states = df['state'].nunique()
        top_state = df['state'].mode()[0] if not df['state'].empty else None
        top_state_count = df[df['state'] == top_state].shape[0] if top_state else 0

        metrics = {
            "Number of EV Makers": num_makers,
            "Number of States with Plants": num_states,
            "State with Most Plants": top_state,
            "Plant Count in Top State": top_state_count
        }

        insights.append(f"Analyzed {len(df)} plants from {num_makers} EV makers across {num_states} states.")
        insights.append(f"The state with the most plants is {top_state} with {top_state_count} plants.")

        # Visualizations
        plants_by_state = df['state'].value_counts().reset_index()
        plants_by_state.columns = ['state', 'count']
        fig1 = px.bar(plants_by_state.head(15), x='state', y='count', title='Top 15 States by Number of EV Plants')
        visualizations["number_of_plants_by_state"] = fig1.to_json()

        plants_by_maker = df['ev_maker'].value_counts().reset_index()
        plants_by_maker.columns = ['ev_maker', 'count']
        fig2 = px.pie(plants_by_maker.head(10), names='ev_maker', values='count', title='Top 10 EV Makers by Number of Plants')
        visualizations["market_share_by_number_of_plants"] = fig2.to_json()

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

def macroeconomic_impact_on_industrial_production_data(df):
    analysis_name = "Macroeconomic Impact on Industrial Production"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['month_year', 'cpi', 'interest_rates', 'exchange_rates', 'production']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['month_year', 'cpi', 'interest_rates', 'production']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['month_year'] = pd.to_datetime(df['month_year'], errors='coerce')
        for col in ['cpi', 'interest_rates', 'exchange_rates', 'production']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('month_year').dropna(subset=critical_cols)

        cpi_corr = df['cpi'].corr(df['production'])
        interest_corr = df['interest_rates'].corr(df['production'])
        exchange_corr = df['exchange_rates'].corr(df['production']) if 'exchange_rates' in df.columns else None

        metrics = {
            "CPI/Production Correlation": cpi_corr,
            "Interest Rate/Production Correlation": interest_corr,
            "Exchange Rate/Production Correlation": exchange_corr
        }

        insights.append(f"Correlation between CPI and Production: {cpi_corr:.2f}.")
        insights.append(f"Correlation between Interest Rates and Production: {interest_corr:.2f}.")
        if exchange_corr:
            insights.append(f"Correlation between Exchange Rates and Production: {exchange_corr:.2f}.")

        # Visualizations
        macro_indicators_trend_data = df.melt(id_vars='month_year', value_vars=['production', 'cpi', 'interest_rates'],
                                              var_name='Indicator', value_name='Value')
        fig1 = px.line(macro_indicators_trend_data, x='month_year', y='Value', color='Indicator',
                       title='Macro Indicators and Production Over Time', facet_row='Indicator')
        fig1.update_yaxes(matches=None)
        visualizations["macro_indicators_and_production_over_time"] = fig1.to_json()

        scatter_cols = [col for col in ['cpi', 'interest_rates', 'exchange_rates', 'production'] if col in df.columns]
        if len(scatter_cols) > 1:
            fig2 = px.scatter_matrix(df[scatter_cols], title='Relationships Between Macro Variables and Production')
            visualizations["relationships_between_macro_variables_and_production"] = fig2.to_json()

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

def temperature_control_system_performance_data(df):
    analysis_name = "Temperature Control System Performance (PID vs. Fuzzy)"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['pid_control_output_perc', 'fuzzy_pid_control_output_perc', 'overshoot_c', 'response_time_s', 'steady_state_error_c']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['overshoot_c', 'response_time_s', 'steady_state_error_c']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in expected:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        avg_overshoot = df['overshoot_c'].mean()
        avg_response_time = df['response_time_s'].mean()
        avg_error = df['steady_state_error_c'].mean()

        metrics = {
            "Average Overshoot (°C)": avg_overshoot,
            "Average Response Time (s)": avg_response_time,
            "Average Steady State Error (°C)": avg_error
        }

        insights.append(f"Average overshoot: {avg_overshoot:.2f}°C.")
        insights.append(f"Average response time: {avg_response_time:.2f} s.")
        insights.append(f"Average steady state error: {avg_error:.2f}°C.")

        # Visualizations
        fig1 = px.scatter(df, x='response_time_s', y='overshoot_c', color='steady_state_error_c',
                          title='Overshoot vs. Response Time (Colored by Steady State Error)')
        visualizations["overshoot_vs_response_time_by_error"] = fig1.to_json()
        
        # Compare PID vs Fuzzy if both columns exist
        if 'pid_control_output_perc' in df.columns and 'fuzzy_pid_control_output_perc' in df.columns:
            compare_df = df[['pid_control_output_perc', 'fuzzy_pid_control_output_perc']].melt(var_name='Control Type', value_name='Output (%)')
            fig2 = px.violin(compare_df, x='Control Type', y='Output (%)', box=True, title='Control Output Distribution (PID vs. Fuzzy PID)')
            visualizations["pid_vs_fuzzy_output"] = fig2.to_json()


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

def predictive_maintenance_priority_scoring_data(df):
    analysis_name = "Predictive Maintenance Priority Scoring"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['machine_id', 'temp_c', 'vibration_mm_s', 'pressure_bar', 'failure_prob', 'maintenance_priority']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['machine_id', 'temp_c', 'vibration_mm_s', 'maintenance_priority', 'failure_prob']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['temp_c', 'vibration_mm_s', 'pressure_bar', 'failure_prob', 'maintenance_priority']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        avg_priority = df['maintenance_priority'].mean()
        top_priority_machine = df.loc[df['maintenance_priority'].idxmax()]['machine_id'] if not df.empty else None

        metrics = {
            "Average Maintenance Priority": avg_priority,
            "Highest Priority Machine": top_priority_machine,
            "Machines Analyzed": df['machine_id'].nunique()
        }

        insights.append(f"Analyzed {df['machine_id'].nunique()} machines.")
        insights.append(f"Average maintenance priority score: {avg_priority:.2f}.")
        insights.append(f"Machine '{top_priority_machine}' has the highest priority score.")

        # Visualizations
        top_machines = df.nlargest(15, 'maintenance_priority').sort_values('maintenance_priority', ascending=True)
        fig1 = px.bar(top_machines, y='machine_id', x='maintenance_priority', color='failure_prob',
                      title='Top 15 Machines by Maintenance Priority', orientation='h')
        visualizations["top_machines_by_maintenance_priority"] = fig1.to_json()

        fig2 = px.scatter(df, x='temp_c', y='vibration_mm_s', color='maintenance_priority',
                          title='Sensor Readings Colored by Maintenance Priority')
        visualizations["sensor_readings_colored_by_maintenance_priority"] = fig2.to_json()

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

def production_order_schedule_adherence_data(df):
    analysis_name = "Production Order Schedule Adherence"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['production_order_id', 'scheduled_start', 'scheduled_end', 'actual_start', 'actual_end', 'status']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['scheduled_start', 'scheduled_end', 'actual_start', 'actual_end', 'status']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['scheduled_start', 'scheduled_end', 'actual_start', 'actual_end']:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['start_delay_hours'] = (df['actual_start'] - df['scheduled_start']).dt.total_seconds() / 3600
        df['end_delay_hours'] = (df['actual_end'] - df['scheduled_end']).dt.total_seconds() / 3600
        
        avg_start_delay = df['start_delay_hours'].mean()
        avg_end_delay = df['end_delay_hours'].mean()
        on_time_completion_rate = (df['end_delay_hours'] <= 0).mean() * 100

        metrics = {
            "On-Time Completion Rate": on_time_completion_rate,
            "Average Start Delay (hours)": avg_start_delay,
            "Average End Delay (hours)": avg_end_delay,
            "Total Orders Analyzed": len(df)
        }

        insights.append(f"Analyzed {len(df)} production orders.")
        insights.append(f"On-time completion rate: {on_time_completion_rate:.2f}%.")
        insights.append(f"Orders start on average {avg_start_delay:.2f} hours late.")
        insights.append(f"Orders end on average {avg_end_delay:.2f} hours late.")

        # Visualizations
        delay_distribution_data = df[['start_delay_hours', 'end_delay_hours']].melt(var_name='Delay Type', value_name='Delay (Hours)')
        fig1 = px.violin(delay_distribution_data, x='Delay Type', y='Delay (Hours)', box=True, title='Distribution of Start and End Delays')
        visualizations["delay_distribution"] = fig1.to_json()

        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        fig2 = px.pie(status_counts, names='status', values='count', title='Production Order Status')
        visualizations["production_order_status"] = fig2.to_json()

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

def machine_availability_and_utilization_data(df):
    analysis_name = "Machine Availability and Utilization"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['machine_id', 'operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours', 'units_produced']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['machine_id', 'operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours', 'units_produced']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['total_hours'] = df['operational_hours'] + df['idle_hours'] + df['maintenance_hours'] + df['downtime_hours']
        df['utilization_rate'] = (df['operational_hours'] / df['total_hours']) * 100
        df['availability_rate'] = ((df['total_hours'] - df['downtime_hours']) / df['total_hours']) * 100
        
        avg_utilization = df['utilization_rate'].mean()
        avg_availability = df['availability_rate'].mean()

        metrics = {
            "Average Machine Utilization": avg_utilization,
            "Average Machine Availability": avg_availability,
            "Total Machines Analyzed": df['machine_id'].nunique()
        }

        insights.append(f"Analyzed {df['machine_id'].nunique()} machines.")
        insights.append(f"Average machine utilization: {avg_utilization:.2f}%.")
        insights.append(f"Average machine availability: {avg_availability:.2f}%.")

        # Visualizations
        time_breakdown_by_machine_data = df.melt(id_vars='machine_id', value_vars=['operational_hours', 'idle_hours', 'maintenance_hours', 'downtime_hours'])
        fig1 = px.bar(time_breakdown_by_machine_data, x='machine_id', y='value', color='variable',
                      title='Time Breakdown by Machine')
        visualizations["time_breakdown_by_machine"] = fig1.to_json()

        if 'units_produced' in df.columns and not df['units_produced'].isnull().all():
            fig2 = px.scatter(df, x='utilization_rate', y='units_produced', color='machine_id',
                              title='Units Produced vs. Utilization Rate')
            visualizations["units_produced_vs_utilization_rate"] = fig2.to_json()

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

def machine_downtime_root_cause_data(df):
    analysis_name = "Machine Downtime Root Cause"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['machine_id', 'start_time', 'end_time', 'downtime_reason']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['machine_id', 'start_time', 'end_time', 'downtime_reason']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)
        df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

        total_downtime_hours = df['duration_minutes'].sum() / 60
        top_reason = df.groupby('downtime_reason')['duration_minutes'].sum().idxmax() if not df.empty else None
        worst_machine = df.groupby('machine_id')['duration_minutes'].sum().idxmax() if not df.empty else None

        metrics = {
            "Total Downtime (Hours)": total_downtime_hours,
            "Top Downtime Reason": top_reason,
            "Machine with Most Downtime": worst_machine,
            "Total Downtime Events": len(df)
        }

        insights.append(f"Analyzed {len(df)} downtime events, totaling {total_downtime_hours:.1f} hours.")
        insights.append(f"The top reason for downtime is: {top_reason}.")
        insights.append(f"The machine with the most downtime is: {worst_machine}.")

        # Visualizations
        downtime_by_reason = df.groupby('downtime_reason')['duration_minutes'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(downtime_by_reason.head(10), x='downtime_reason', y='duration_minutes', title='Total Downtime (Minutes) by Reason')
        visualizations["downtime_by_reason"] = fig1.to_json()

        downtime_by_machine = df.groupby('machine_id')['duration_minutes'].sum().sort_values(ascending=False).reset_index()
        fig2 = px.pie(downtime_by_machine.head(10), names='machine_id', values='duration_minutes', title='Proportion of Downtime by Machine (Top 10)')
        visualizations["proportion_of_downtime_by_machine"] = fig2.to_json()

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

def manufacturing_batch_process_monitoring_data(df):
    analysis_name = "Manufacturing Batch Process Monitoring"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['batch_id', 'material_id', 'quantity', 'temperature', 'pressure']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['batch_id', 'material_id', 'temperature', 'pressure']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['quantity', 'temperature', 'pressure']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        num_batches = df['batch_id'].nunique()
        avg_temp = df['temperature'].mean()
        avg_pressure = df['pressure'].mean()

        metrics = {
            "Number of Batches": num_batches,
            "Average Temperature": avg_temp,
            "Average Pressure": avg_pressure
        }

        insights.append(f"Monitored {num_batches} batches.")
        insights.append(f"Average process temperature: {avg_temp:.2f}.")
        insights.append(f"Average process pressure: {avg_pressure:.2f}.")

        # Visualizations
        process_conditions_distribution_data = df[['temperature', 'pressure']].melt(var_name='Metric', value_name='Value')
        fig1 = px.violin(process_conditions_distribution_data, x='Metric', y='Value', box=True, title='Distribution of Process Temperature and Pressure')
        visualizations["distribution_of_process_temp_pressure"] = fig1.to_json()

        size_col = 'quantity' if 'quantity' in df.columns else None
        fig2 = px.scatter(df, x='temperature', y='pressure', color='material_id', size=size_col,
                          title='Process Conditions by Material ID')
        visualizations["process_conditions_by_material_id"] = fig2.to_json()

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

def shift_based_production_output_and_defect_analysis(df):
    analysis_name = "Shift-based Production Output and Defect Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['workstation_id', 'operator_id', 'shift_type', 'output_count', 'defects_count']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['shift_type', 'output_count', 'defects_count']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['output_count', 'defects_count']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['defect_rate'] = (df['defects_count'] / df['output_count']) * 100
        
        total_output = df['output_count'].sum()
        total_defects = df['defects_count'].sum()
        overall_defect_rate = (total_defects / total_output) * 100 if total_output > 0 else 0

        metrics = {
            "Total Production Output": total_output,
            "Total Defects": total_defects,
            "Overall Defect Rate": overall_defect_rate
        }

        insights.append(f"Total output: {total_output:,.0f} units with {total_defects:,.0f} defects.")
        insights.append(f"Overall defect rate: {overall_defect_rate:.2f}%.")

        # Visualizations
        output_by_shift = df.groupby('shift_type')['output_count'].sum().reset_index()
        fig1 = px.pie(output_by_shift, names='shift_type', values='output_count', title='Total Production Output by Shift')
        visualizations["production_output_by_shift"] = fig1.to_json()

        if 'workstation_id' in df.columns:
            defect_rate_by_workstation = df.groupby('workstation_id')['defect_rate'].mean().reset_index()
            fig2 = px.bar(defect_rate_by_workstation, x='workstation_id', y='defect_rate', title='Average Defect Rate by Workstation')
            visualizations["defect_rate_by_workstation"] = fig2.to_json()

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

def quality_inspection_and_defect_resolution_data(df):
    analysis_name = "Quality Inspection and Defect Resolution"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['inspection_date', 'product_id', 'inspector_id', 'defect_found', 'resolution_status', 'resolution_time_hours']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['defect_found', 'resolution_status', 'resolution_time_hours']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'inspection_date' in df.columns:
            df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
        df['resolution_time_hours'] = pd.to_numeric(df['resolution_time_hours'], errors='coerce')
        # defect_found could be boolean or 1/0
        if df['defect_found'].dtype == 'object':
             df['defect_found_flag'] = df['defect_found'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0, 'defect': 1, 'pass': 0})
        else:
             df['defect_found_flag'] = pd.to_numeric(df['defect_found'], errors='coerce')

        df.dropna(subset=['defect_found_flag', 'resolution_status'], inplace=True)

        total_defects_found = df['defect_found_flag'].sum()
        resolved_defects = df[df['resolution_status'].str.contains('resolved', case=False, na=False)].shape[0]
        resolution_rate = (resolved_defects / total_defects_found) * 100 if total_defects_found > 0 else 0
        avg_resolution_time = df[df['resolution_status'].str.contains('resolved', case=False, na=False)]['resolution_time_hours'].mean()

        metrics = {
            "Total Defects Found": total_defects_found,
            "Resolved Defects": resolved_defects,
            "Defect Resolution Rate": resolution_rate,
            "Average Resolution Time (hours)": avg_resolution_time
        }

        insights.append(f"Found {total_defects_found:,.0f} defects.")
        insights.append(f"Resolution rate: {resolution_rate:.2f}%.")
        insights.append(f"Average time to resolve: {avg_resolution_time:.2f} hours.")

        # Visualizations
        status_counts = df['resolution_status'].value_counts().reset_index()
        status_counts.columns = ['resolution_status', 'count']
        fig1 = px.pie(status_counts, names='resolution_status', values='count', title='Defect Resolution Status')
        visualizations["defect_resolution_status"] = fig1.to_json()

        if 'product_id' in df.columns:
            resolution_time_by_product = df.groupby('product_id')['resolution_time_hours'].mean().reset_index()
            fig2 = px.bar(resolution_time_by_product, x='product_id', y='resolution_time_hours', title='Average Resolution Time by Product')
            visualizations["average_resolution_time_by_product"] = fig2.to_json()

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

def production_material_cost_analysis(df):
    analysis_name = "Production Material Cost Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['material_id', 'material_name', 'quantity_used', 'unit_cost', 'product_id']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['material_name', 'quantity_used', 'unit_cost']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['quantity_used'] = pd.to_numeric(df['quantity_used'], errors='coerce')
        df['unit_cost'] = pd.to_numeric(df['unit_cost'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        df['total_material_cost'] = df['quantity_used'] * df['unit_cost']

        total_cost_materials = df['total_material_cost'].sum()
        avg_unit_cost = df['unit_cost'].mean()
        most_expensive_material = df.groupby('material_name')['total_material_cost'].sum().idxmax() if not df.empty else None

        metrics = {
            "Total Material Cost": total_cost_materials,
            "Average Unit Cost (Weighted)": (df['total_material_cost'].sum() / df['quantity_used'].sum()) if df['quantity_used'].sum() > 0 else 0,
            "Material with Highest Total Cost": most_expensive_material
        }

        insights.append(f"Total material cost: ${total_cost_materials:,.0f}.")
        insights.append(f"Material accounting for the highest cost: {most_expensive_material}.")

        # Visualizations
        cost_by_material = df.groupby('material_name')['total_material_cost'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(cost_by_material.head(15), x='material_name', y='total_material_cost', title='Top 15 Total Material Cost by Material')
        visualizations["total_material_cost_by_material"] = fig1.to_json()

        if 'product_id' in df.columns:
            cost_by_product = df.groupby('product_id')['total_material_cost'].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(cost_by_product.head(15), x='product_id', y='total_material_cost', title='Top 15 Total Material Cost by Product')
            visualizations["total_material_cost_by_product"] = fig2.to_json()

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

def supplier_material_receipt_and_quality_data(df):
    analysis_name = "Supplier Material Receipt and Quality"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['receipt_id', 'supplier_name', 'material_name', 'quantity_received', 'quality_status', 'delivery_date', 'inspection_result']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['supplier_name', 'quality_status', 'inspection_result']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'delivery_date' in df.columns:
            df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_receipts = len(df)
        # Standardize quality status
        df['quality_status_std'] = df['quality_status'].str.lower()
        accepted_receipts = df[df['quality_status_std'].str.contains('accepted|pass', case=False, na=False)].shape[0]
        rejection_rate = ((total_receipts - accepted_receipts) / total_receipts) * 100 if total_receipts > 0 else 0
        
        metrics = {
            "Total Material Receipts": total_receipts,
            "Accepted Receipts": accepted_receipts,
            "Material Rejection Rate": rejection_rate
        }

        insights.append(f"Analyzed {total_receipts} material receipts.")
        insights.append(f"Overall material rejection rate: {rejection_rate:.2f}%.")

        # Visualizations
        # Standardize 'quality_status' for grouping
        df['quality_group'] = np.where(df['quality_status_std'].str.contains('accepted|pass', case=False, na=False), 'Accepted', 'Rejected/Other')
        quality_by_supplier = df.groupby(['supplier_name', 'quality_group']).size().unstack(fill_value=0)
        
        if 'Accepted' not in quality_by_supplier: quality_by_supplier['Accepted'] = 0
        if 'Rejected/Other' not in quality_by_supplier: quality_by_supplier['Rejected/Other'] = 0
            
        quality_by_supplier['Total'] = quality_by_supplier.sum(axis=1)
        quality_by_supplier['Rejection Rate (%)'] = (quality_by_supplier['Rejected/Other'] / quality_by_supplier['Total']) * 100
        quality_by_supplier = quality_by_supplier.reset_index()

        fig1 = px.bar(quality_by_supplier.sort_values('Rejection Rate (%)', ascending=False), 
                      x='supplier_name', y='Rejection Rate (%)', title='Rejection Rate by Supplier')
        visualizations["quality_status_by_supplier"] = fig1.to_json()

        inspection_results_distribution = df['inspection_result'].value_counts().reset_index()
        fig2 = px.pie(inspection_results_distribution, names='inspection_result', values='count', title='Inspection Results Distribution')
        visualizations["inspection_results_distribution"] = fig2.to_json()

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

def manufacturing_resource_utilization_analysis(df):
    analysis_name = "Manufacturing Resource Utilization Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['resource_id', 'resource_type', 'total_hours_available', 'hours_utilized', 'downtime_hours', 'idle_hours']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['resource_type', 'total_hours_available', 'hours_utilized', 'downtime_hours', 'idle_hours']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['total_hours_available', 'hours_utilized', 'downtime_hours', 'idle_hours']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['utilization_rate'] = (df['hours_utilized'] / df['total_hours_available']) * 100
        df['downtime_percentage'] = (df['downtime_hours'] / df['total_hours_available']) * 100

        avg_utilization = df['utilization_rate'].mean()
        avg_downtime_percentage = df['downtime_percentage'].mean()
        
        metrics = {
            "Average Resource Utilization": avg_utilization,
            "Average Resource Downtime Percentage": avg_downtime_percentage
        }

        insights.append(f"Average resource utilization: {avg_utilization:.2f}%.")
        insights.append(f"Average resource downtime: {avg_downtime_percentage:.2f}%.")

        # Visualizations
        utilization_by_type = df.groupby('resource_type')['utilization_rate'].mean().reset_index()
        fig1 = px.bar(utilization_by_type, x='resource_type', y='utilization_rate', title='Average Utilization by Resource Type')
        visualizations["utilization_by_resource_type"] = fig1.to_json()

        status_breakdown_df = df.melt(id_vars='resource_type', value_vars=['hours_utilized', 'downtime_hours', 'idle_hours'],
                                      var_name='Status', value_name='Hours')
        status_summary = status_breakdown_df.groupby(['resource_type', 'Status'])['Hours'].sum().reset_index()
        fig2 = px.bar(status_summary, x='resource_type', y='Hours', color='Status', title='Resource Hours Breakdown by Type')
        visualizations["resource_status_hours_breakdown"] = fig2.to_json()

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

def predictive_maintenance_sensor_data_analysis(df):
    analysis_name = "Predictive Maintenance Sensor Data Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['timestamp', 'machine_id', 'sensor_1_value', 'sensor_2_value', 'sensor_3_value', 'anomaly_score']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['timestamp', 'anomaly_score', 'sensor_1_value'] # Need at least one sensor
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        sensor_cols = [col for col in matched.values() if 'sensor_' in col and col in df.columns]
        for col in sensor_cols + ['anomaly_score']:
             df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('timestamp').dropna(subset=['timestamp', 'anomaly_score'] + sensor_cols)

        avg_anomaly_score = df['anomaly_score'].mean()
        anomaly_threshold = df['anomaly_score'].quantile(0.95)
        num_anomalies = df[df['anomaly_score'] > anomaly_threshold].shape[0]
        
        metrics = {
            "Average Anomaly Score": avg_anomaly_score,
            "Number of High Anomaly Readings (Top 5%)": num_anomalies,
            "Anomaly Threshold (95th Percentile)": anomaly_threshold
        }

        insights.append(f"Average anomaly score: {avg_anomaly_score:.3f}.")
        insights.append(f"Found {num_anomalies} readings in the top 5% (above {anomaly_threshold:.3f}), indicating potential issues.")

        # Visualizations
        sensor_readings_over_time_data = df.melt(id_vars='timestamp', value_vars=sensor_cols, 
                                                 var_name='Sensor', value_name='Value')
        fig1 = px.line(sensor_readings_over_time_data, x='timestamp', y='Value', color='Sensor', title='Sensor Readings Over Time')
        visualizations["sensor_readings_trend"] = fig1.to_json()

        fig2 = px.histogram(df, x='anomaly_score', title='Anomaly Score Distribution')
        visualizations["anomaly_score_distribution"] = fig2.to_json()

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

def energy_consumption_and_production_efficiency_data(df):
    analysis_name = "Energy Consumption and Production Efficiency"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'production_line', 'energy_consumed_kwh', 'units_produced']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['date', 'production_line', 'energy_consumed_kwh', 'units_produced']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['energy_consumed_kwh', 'units_produced']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['energy_per_unit'] = df['energy_consumed_kwh'] / df['units_produced']
        
        total_energy_consumed = df['energy_consumed_kwh'].sum()
        total_units_produced = df['units_produced'].sum()
        avg_energy_efficiency = df['energy_per_unit'].mean()

        metrics = {
            "Total Energy Consumed (kWh)": total_energy_consumed,
            "Total Units Produced": total_units_produced,
            "Average Energy Efficiency (kWh/unit)": avg_energy_efficiency,
            "Overall Energy Efficiency (kWh/unit)": total_energy_consumed / total_units_produced
        }

        insights.append(f"Total energy consumed: {total_energy_consumed:,.0f} kWh to produce {total_units_produced:,.0f} units.")
        insights.append(f"Overall efficiency: {metrics['Overall Energy Efficiency (kWh/unit)']:.2f} kWh/unit.")
        insights.append(f"Average daily/batch efficiency: {avg_energy_efficiency:.2f} kWh/unit.")

        # Visualizations
        energy_efficiency_by_line = df.groupby('production_line')['energy_per_unit'].mean().reset_index()
        fig1 = px.bar(energy_efficiency_by_line, x='production_line', y='energy_per_unit', 
                      title='Average Energy Efficiency (kWh/unit) by Production Line')
        visualizations["energy_efficiency_by_production_line"] = fig1.to_json()

        energy_consumption_over_time = df.groupby('date')['energy_consumed_kwh'].sum().reset_index()
        fig2 = px.line(energy_consumption_over_time, x='date', y='energy_consumed_kwh', title='Total Energy Consumption Over Time')
        visualizations["energy_consumption_trend_over_time"] = fig2.to_json()

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

def quality_control_lab_test_result_analysis(df):
    analysis_name = "Quality Control Lab Test Result Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['test_id', 'sample_id', 'test_date', 'parameter_1', 'parameter_2', 'parameter_3', 'test_result']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['test_result', 'parameter_1', 'parameter_2'] # Need at least a few params
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'test_date' in df.columns:
            df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')
        
        param_cols = [col for col in matched.values() if 'parameter_' in col and col in df.columns]
        for col in param_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=['test_result'] + param_cols, inplace=True)

        # Standardize test_result
        df['test_result_std'] = 'Fail'
        df.loc[df['test_result'].str.contains('pass|ok|good', case=False, na=False), 'test_result_std'] = 'Pass'

        pass_tests = (df['test_result_std'] == 'Pass').sum()
        total_tests = len(df)
        pass_rate = (pass_tests / total_tests) * 100 if total_tests > 0 else 0
        
        metrics = {
            "Total Tests Conducted": total_tests,
            "Tests Passed": pass_tests,
            "Test Pass Rate": pass_rate
        }

        insights.append(f"Analyzed {total_tests} lab tests.")
        insights.append(f"Overall pass rate: {pass_rate:.2f}%.")

        # Visualizations
        test_result_distribution = df['test_result_std'].value_counts().reset_index()
        fig1 = px.pie(test_result_distribution, names='test_result_std', values='count', title='Test Result Distribution (Pass/Fail)')
        visualizations["test_result_distribution"] = fig1.to_json()

        df_corr = df[param_cols].copy()
        df_corr['test_result_numeric'] = df['test_result_std'].apply(lambda x: 1 if x == 'Pass' else 0)
        corr_matrix = df_corr.corr()
        
        fig2 = px.imshow(corr_matrix, text_auto=True, title='Parameter Correlation with Test Results')
        visualizations["parameter_correlation_with_test_results"] = fig2.to_json()
        
        # Plot parameter distributions by test result
        param_melt = df.melt(id_vars='test_result_std', value_vars=param_cols, var_name='Parameter', value_name='Value')
        fig3 = px.violin(param_melt, x='Parameter', y='Value', color='test_result_std', box=True,
                         title='Parameter Distributions by Test Result')
        visualizations["parameter_distributions"] = fig3.to_json()


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

def equipment_calibration_compliance_and_results_analysis(df):
    analysis_name = "Equipment Calibration Compliance and Results"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['calibration_id', 'equipment_id', 'calibration_date', 'due_date', 'calibration_result', 'deviation']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['equipment_id', 'calibration_date', 'due_date', 'calibration_result', 'deviation']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['calibration_date'] = pd.to_datetime(df['calibration_date'], errors='coerce')
        df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
        df['deviation'] = pd.to_numeric(df['deviation'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        # Standardize result
        df['result_std'] = 'Fail'
        df.loc[df['calibration_result'].str.contains('pass|compliant', case=False, na=False), 'result_std'] = 'Pass'
        
        compliant_calibrations = (df['result_std'] == 'Pass').sum()
        total_calibrations = len(df)
        compliance_rate = (compliant_calibrations / total_calibrations) * 100 if total_calibrations > 0 else 0
        
        # Check for overdue
        df['overdue'] = (df['calibration_date'] > df['due_date'])
        overdue_calibrations = df['overdue'].sum()
        overdue_rate = (overdue_calibrations / total_calibrations) * 100 if total_calibrations > 0 else 0


        metrics = {
            "Total Calibrations": total_calibrations,
            "Compliant Calibrations (Passed)": compliant_calibrations,
            "Calibration Compliance Rate": compliance_rate,
            "Overdue Calibrations": overdue_calibrations,
            "Overdue Rate": overdue_rate,
            "Average Deviation": df['deviation'].mean()
        }

        insights.append(f"Analyzed {total_calibrations} calibration records.")
        insights.append(f"Overall compliance (pass) rate: {compliance_rate:.2f}%.")
        insights.append(f"{overdue_calibrations} calibrations ({overdue_rate:.1f}%) were performed after their due date.")
        insights.append(f"Average deviation: {df['deviation'].mean():.3f}.")

        # Visualizations
        calibration_result_distribution = df['result_std'].value_counts().reset_index()
        fig1 = px.pie(calibration_result_distribution, names='result_std', values='count', title='Calibration Result Distribution (Pass/Fail)')
        visualizations["calibration_result_distribution"] = fig1.to_json()

        fig2 = px.box(df, x='equipment_id', y='deviation', title='Deviation by Equipment ID')
        visualizations["deviation_by_equipment_id"] = fig2.to_json()
        
        overdue_by_equip = df.groupby('equipment_id')['overdue'].mean().mul(100).sort_values(ascending=False).reset_index()
        fig3 = px.bar(overdue_by_equip, x='equipment_id', y='overdue', title='Overdue Calibration Rate (%) by Equipment')
        visualizations["overdue_by_equipment"] = fig3.to_json()


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

def production_delay_root_cause_analysis(df):
    analysis_name = "Production Delay Root Cause Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['delay_id', 'production_line', 'delay_start_time', 'delay_end_time', 'delay_reason', 'impact_on_production']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['delay_start_time', 'delay_end_time', 'delay_reason']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['delay_start_time'] = pd.to_datetime(df['delay_start_time'], errors='coerce')
        df['delay_end_time'] = pd.to_datetime(df['delay_end_time'], errors='coerce')
        if 'impact_on_production' in df.columns:
            df['impact_on_production'] = pd.to_numeric(df['impact_on_production'], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)
        df['delay_duration_hours'] = (df['delay_end_time'] - df['delay_start_time']).dt.total_seconds() / 3600

        total_delay_hours = df['delay_duration_hours'].sum()
        most_common_reason = df['delay_reason'].mode()[0] if not df.empty else None
        
        metrics = {
            "Total Delay Hours": total_delay_hours,
            "Most Common Delay Reason": most_common_reason,
            "Total Delay Events": len(df)
        }

        insights.append(f"Analyzed {len(df)} delay events, totaling {total_delay_hours:.1f} hours.")
        insights.append(f"The most common reason for delays is: {most_common_reason}.")

        # Visualizations
        delay_duration_by_reason = df.groupby('delay_reason')['delay_duration_hours'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(delay_duration_by_reason.head(10), x='delay_reason', y='delay_duration_hours', title='Total Delay Hours by Reason')
        visualizations["delay_duration_by_reason"] = fig1.to_json()

        if 'impact_on_production' in df.columns and not df['impact_on_production'].isnull().all():
            delay_impact_by_reason = df.groupby('delay_reason')['impact_on_production'].sum().sort_values(ascending=False).reset_index()
            fig2 = px.bar(delay_impact_by_reason.head(10), x='delay_reason', y='impact_on_production', title='Production Impact (Units) by Delay Reason')
            visualizations["delay_impact_by_reason"] = fig2.to_json()

        if 'production_line' in df.columns:
            delay_by_line = df.groupby('production_line')['delay_duration_hours'].sum().reset_index()
            fig3 = px.pie(delay_by_line, names='production_line', values='delay_duration_hours', title='Proportion of Delay Hours by Production Line')
            visualizations["delay_by_line"] = fig3.to_json()

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

def workplace_safety_incident_analysis(df):
    analysis_name = "Workplace Safety Incident Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['incident_id', 'incident_date', 'incident_type', 'department', 'severity_level', 'lost_work_days']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['incident_date', 'incident_type', 'severity_level', 'lost_work_days']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['lost_work_days'] = pd.to_numeric(df['lost_work_days'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_incidents = len(df)
        total_lost_work_days = df['lost_work_days'].sum()
        most_common_incident_type = df['incident_type'].mode()[0] if not df.empty else None

        metrics = {
            "Total Safety Incidents": total_incidents,
            "Total Lost Work Days": total_lost_work_days,
            "Average Lost Work Days per Incident": df['lost_work_days'].mean(),
            "Most Common Incident Type": most_common_incident_type
        }

        insights.append(f"Recorded {total_incidents} safety incidents, leading to {total_lost_work_days} lost work days.")
        insights.append(f"The most common incident type: {most_common_incident_type}.")

        # Visualizations
        incidents_by_type = df['incident_type'].value_counts().reset_index()
        fig1 = px.bar(incidents_by_type, x='incident_type', y='count', title='Incidents by Type')
        visualizations["incidents_by_type"] = fig1.to_json()

        incidents_by_severity = df['severity_level'].value_counts().reset_index()
        fig2 = px.pie(incidents_by_severity, names='severity_level', values='count', title='Incidents by Severity')
        visualizations["incidents_by_severity"] = fig2.to_json()

        if 'department' in df.columns:
            lost_work_days_by_department = df.groupby('department')['lost_work_days'].sum().reset_index()
            fig3 = px.bar(lost_work_days_by_department, x='department', y='lost_work_days', title='Total Lost Work Days by Department')
            visualizations["lost_work_days_by_department"] = fig3.to_json()

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

def equipment_maintenance_cost_and_type_analysis(df):
    analysis_name = "Equipment Maintenance Cost and Type Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['maintenance_id', 'equipment_id', 'maintenance_date', 'maintenance_type', 'cost_usd']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['equipment_id', 'maintenance_type', 'cost_usd']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'maintenance_date' in df.columns:
            df['maintenance_date'] = pd.to_datetime(df['maintenance_date'], errors='coerce')
        df['cost_usd'] = pd.to_numeric(df['cost_usd'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_maintenance_cost = df['cost_usd'].sum()
        avg_maintenance_cost = df['cost_usd'].mean()
        most_common_maintenance_type = df['maintenance_type'].mode()[0] if not df.empty else None
        most_costly_type = df.groupby('maintenance_type')['cost_usd'].sum().idxmax()

        metrics = {
            "Total Maintenance Cost (USD)": total_maintenance_cost,
            "Average Maintenance Cost (USD)": avg_maintenance_cost,
            "Most Common Maintenance Type": most_common_maintenance_type,
            "Most Costly Maintenance Type": most_costly_type
        }

        insights.append(f"Total maintenance cost: ${total_maintenance_cost:,.0f}.")
        insights.append(f"Most common maintenance type: {most_common_maintenance_type}.")
        insights.append(f"Most costly maintenance type (total): {most_costly_type}.")

        # Visualizations
        cost_by_type = df.groupby('maintenance_type')['cost_usd'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(cost_by_type, x='maintenance_type', y='cost_usd', title='Total Maintenance Cost by Type')
        visualizations["maintenance_cost_by_type"] = fig1.to_json()

        cost_by_equipment = df.groupby('equipment_id')['cost_usd'].sum().sort_values(ascending=False).reset_index()
        fig2 = px.bar(cost_by_equipment.head(20), x='equipment_id', y='cost_usd', title='Top 20 Total Maintenance Cost by Equipment')
        visualizations["maintenance_cost_by_equipment"] = fig2.to_json()

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

def tool_lifecycle_and_condition_monitoring_analysis(df):
    analysis_name = "Tool Lifecycle and Condition Monitoring"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['tool_id', 'tool_type', 'purchase_date', 'last_service_date', 'usage_hours', 'condition_score', 'wear_level']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['tool_id', 'tool_type', 'usage_hours', 'condition_score', 'wear_level']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['purchase_date', 'last_service_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        for col in ['usage_hours', 'condition_score', 'wear_level']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        avg_usage_hours = df['usage_hours'].mean()
        avg_condition_score = df['condition_score'].mean()
        highest_wear_tool = df.loc[df['wear_level'].idxmax()]['tool_id'] if not df.empty else None

        metrics = {
            "Total Tools Analyzed": df['tool_id'].nunique(),
            "Average Usage Hours": avg_usage_hours,
            "Average Condition Score": avg_condition_score,
            "Tool with Highest Wear": highest_wear_tool,
            "Highest Wear Level": df['wear_level'].max()
        }

        insights.append(f"Analyzed {df['tool_id'].nunique()} tools.")
        insights.append(f"Average usage: {avg_usage_hours:.1f} hours.")
        insights.append(f"Average condition score: {avg_condition_score:.2f}.")
        insights.append(f"Tool '{highest_wear_tool}' shows the highest wear ({df['wear_level'].max():.2f}).")

        # Visualizations
        condition_score_by_type = df.groupby('tool_type')['condition_score'].mean().reset_index()
        fig1 = px.bar(condition_score_by_type, x='tool_type', y='condition_score', title='Average Condition Score by Tool Type')
        visualizations["condition_score_by_tool_type"] = fig1.to_json()

        fig2 = px.scatter(df, x='usage_hours', y='wear_level', color='condition_score',
                          title='Wear Level vs. Usage Hours', trendline="ols")
        visualizations["wear_level_vs_usage_hours"] = fig2.to_json()

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

def final_product_quality_grade_analysis(df):
    analysis_name = "Final Product Quality Grade Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_id', 'production_batch', 'inspection_date', 'quality_grade', 'defect_category', 'inspector']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['product_id', 'quality_grade', 'defect_category']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'inspection_date' in df.columns:
            df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_products = len(df)
        top_grade = df['quality_grade'].mode()[0] if not df.empty else None
        top_defect_category = df[df['defect_category'] != 'None']['defect_category'].mode()[0] if not df.empty else "N/A"

        metrics = {
            "Total Products Inspected": total_products,
            "Most Common Quality Grade": top_grade,
            "Most Common Defect Category": top_defect_category
        }

        insights.append(f"Inspected {total_products} final products.")
        insights.append(f"The most common quality grade assigned is: {top_grade}.")
        insights.append(f"The most common defect category cited is: {top_defect_category}.")

        # Visualizations
        quality_grade_distribution = df['quality_grade'].value_counts(normalize=True).mul(100).reset_index()
        fig1 = px.pie(quality_grade_distribution, names='quality_grade', values='proportion', title='Quality Grade Distribution')
        visualizations["quality_grade_distribution"] = fig1.to_json()

        # Filter out "Pass" or "None" to see real defects
        defect_df = df[~df['defect_category'].str.contains('none|pass', case=False, na=False)]
        defect_category_by_grade = defect_df.groupby(['quality_grade', 'defect_category']).size().reset_index(name='count')
        fig2 = px.bar(defect_category_by_grade, x='quality_grade', y='count', color='defect_category', 
                      title='Defect Categories by Quality Grade')
        visualizations["defect_categories_by_quality_grade"] = fig2.to_json()

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

def shift_production_performance_analysis(df):
    analysis_name = "Shift Production Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shift_id', 'shift_date', 'production_line', 'units_produced', 'defects', 'target_units']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['shift_id', 'units_produced', 'defects', 'target_units']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'shift_date' in df.columns:
            df['shift_date'] = pd.to_datetime(df['shift_date'], errors='coerce')
        for col in ['units_produced', 'defects', 'target_units']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['achievement_rate'] = (df['units_produced'] / df['target_units']) * 100
        df['defect_rate'] = (df['defects'] / df['units_produced']) * 100
        df['defect_rate'] = df['defect_rate'].replace([np.inf, -np.inf], 0) # Handle division by zero if 0 units produced

        avg_achievement_rate = df['achievement_rate'].mean()
        avg_defect_rate = df['defect_rate'].mean()
        
        metrics = {
            "Average Target Achievement Rate": avg_achievement_rate,
            "Average Defect Rate": avg_defect_rate,
            "Total Shifts Analyzed": len(df)
        }

        insights.append(f"Analyzed {len(df)} shifts.")
        insights.append(f"Average target achievement: {avg_achievement_rate:.2f}%.")
        insights.append(f"Average defect rate: {avg_defect_rate:.2f}%.")

        # Visualizations
        performance_by_shift = df.groupby('shift_id').agg(
            avg_achievement_rate=('achievement_rate', 'mean'),
            avg_defect_rate=('defect_rate', 'mean')
        ).reset_index()
        
        fig1_data = performance_by_shift.melt(id_vars='shift_id', var_name='Metric', value_name='Percentage')
        fig1 = px.bar(fig1_data, x='shift_id', y='Percentage', color='Metric', 
                      barmode='group', title='Performance by Shift ID')
        visualizations["performance_by_shift"] = fig1.to_json()

        if 'production_line' in df.columns and 'shift_date' in df.columns and not df['shift_date'].isnull().all():
            production_trend_by_line = df.groupby(['shift_date', 'production_line'])['units_produced'].sum().reset_index()
            fig2 = px.line(production_trend_by_line, x='shift_date', y='units_produced', color='production_line',
                           title='Production Trend by Line')
            visualizations["production_trend_by_line"] = fig2.to_json()

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

def inbound_material_quality_and_delivery_analysis(df):
    analysis_name = "Inbound Material Quality and Delivery Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['material_id', 'supplier_id', 'delivery_date', 'quantity_delivered', 'quality_inspection_result', 'delivery_delay_days']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['supplier_id', 'quality_inspection_result', 'delivery_delay_days']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'delivery_date' in df.columns:
            df['delivery_date'] = pd.to_datetime(df['delivery_date'], errors='coerce')
        for col in ['quantity_delivered', 'delivery_delay_days']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        # Standardize result
        df['result_std'] = 'Fail'
        df.loc[df['quality_inspection_result'].str.contains('pass|accepted', case=False, na=False), 'result_std'] = 'Pass'

        accepted_materials = (df['result_std'] == 'Pass').sum()
        total_materials = len(df)
        acceptance_rate = (accepted_materials / total_materials) * 100 if total_materials > 0 else 0
        avg_delivery_delay = df['delivery_delay_days'].mean()

        metrics = {
            "Total Inbound Lots": total_materials,
            "Accepted Lots": accepted_materials,
            "Material Acceptance Rate": acceptance_rate,
            "Average Delivery Delay (Days)": avg_delivery_delay
        }

        insights.append(f"Analyzed {total_materials} inbound material lots.")
        insights.append(f"Overall acceptance rate: {acceptance_rate:.2f}%.")
        insights.append(f"Average delivery delay: {avg_delivery_delay:.2f} days.")

        # Visualizations
        quality_by_supplier = df.groupby('supplier_id')['result_std'].value_counts(normalize=True).unstack(fill_value=0)
        if 'Pass' in quality_by_supplier.columns:
            quality_by_supplier = quality_by_supplier.mul(100).reset_index()
            fig1 = px.bar(quality_by_supplier, x='supplier_id', y='Pass', title='Inbound Material Acceptance Rate (%) by Supplier')
            visualizations["inbound_material_quality_by_supplier"] = fig1.to_json()
        
        fig2 = px.histogram(df, x='delivery_delay_days', title='Delivery Delay Distribution (Days)')
        visualizations["delivery_delay_distribution"] = fig2.to_json()

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

def warehouse_inventory_stock_level_analysis(df):
    analysis_name = "Warehouse Inventory Stock Level Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['item_id', 'item_name', 'warehouse_location', 'current_stock_level', 'reorder_point', 'max_stock_level']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['item_id', 'warehouse_location', 'current_stock_level', 'reorder_point']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['current_stock_level', 'reorder_point', 'max_stock_level']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        items_below_reorder = df[df['current_stock_level'] < df['reorder_point']].shape[0]
        total_items = len(df)
        below_reorder_pct = (items_below_reorder / total_items) * 100 if total_items > 0 else 0
        avg_stock_level = df['current_stock_level'].mean()

        metrics = {
            "Total Items in Inventory": total_items,
            "Items Below Reorder Point": items_below_reorder,
            "Below Reorder Point (%)": below_reorder_pct,
            "Average Stock Level": avg_stock_level
        }

        insights.append(f"Analyzed {total_items} items across warehouses.")
        insights.append(f"{items_below_reorder} items ({below_reorder_pct:.1f}%) are below their reorder point.")

        # Visualizations
        fig1 = px.histogram(df, x='current_stock_level', title='Inventory Stock Level Distribution')
        visualizations["inventory_stock_level_distribution"] = fig1.to_json()

        df['stock_status'] = 'OK'
        df.loc[df['current_stock_level'] < df['reorder_point'], 'stock_status'] = 'Below Reorder'
        if 'max_stock_level' in df.columns:
            df.loc[df['current_stock_level'] > df['max_stock_level'], 'stock_status'] = 'Overstocked'
            
        stock_status_by_location = df.groupby(['warehouse_location', 'stock_status']).size().unstack(fill_value=0).reset_index()
        stock_status_melted = stock_status_by_location.melt(id_vars='warehouse_location', var_name='Stock Status', value_name='Count')
        
        fig2 = px.bar(stock_status_melted, x='warehouse_location', y='Count', color='Stock Status', title='Stock Status by Warehouse Location')
        visualizations["stock_status_by_warehouse_location"] = fig2.to_json()

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

def order_dispatch_and_delivery_status_tracking(df):
    analysis_name = "Order Dispatch and Delivery Status Tracking"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['order_id', 'dispatch_date', 'delivery_date', 'delivery_status', 'shipping_carrier', 'delivery_time_days']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['order_id', 'delivery_status', 'shipping_carrier', 'delivery_time_days']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['dispatch_date', 'delivery_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        df['delivery_time_days'] = pd.to_numeric(df['delivery_time_days'], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_orders = len(df)
        delivered_orders = df[df['delivery_status'].str.contains('delivered', case=False, na=False)].shape[0]
        delivery_rate = (delivered_orders / total_orders) * 100 if total_orders > 0 else 0
        avg_delivery_time = df['delivery_time_days'].mean()
        
        metrics = {
            "Total Orders": total_orders,
            "Delivered Orders": delivered_orders,
            "Delivery Rate (%)": delivery_rate,
            "Average Delivery Time (Days)": avg_delivery_time
        }

        insights.append(f"Tracked {total_orders} orders.")
        insights.append(f"{delivered_orders} ({delivery_rate:.1f}%) are confirmed delivered.")
        insights.append(f"Average delivery time: {avg_delivery_time:.2f} days.")

        # Visualizations
        delivery_status_distribution = df['delivery_status'].value_counts().reset_index()
        fig1 = px.pie(delivery_status_distribution, names='delivery_status', values='count', title='Delivery Status Distribution')
        visualizations["delivery_status_distribution"] = fig1.to_json()

        avg_delivery_time_by_carrier = df.groupby('shipping_carrier')['delivery_time_days'].mean().reset_index()
        fig2 = px.bar(avg_delivery_time_by_carrier, x='shipping_carrier', y='delivery_time_days', 
                      title='Average Delivery Time by Shipping Carrier')
        visualizations["average_delivery_time_by_shipping_carrier"] = fig2.to_json()

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

def inventory_audit_and_stock_count_analysis(df):
    analysis_name = "Inventory Audit and Stock Count Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['audit_id', 'audit_date', 'item_id', 'recorded_stock', 'actual_stock', 'variance', 'audit_result']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['item_id', 'recorded_stock', 'actual_stock']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'audit_date' in df.columns:
            df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
        for col in ['recorded_stock', 'actual_stock', 'variance']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        if 'variance' not in df.columns:
             df['variance'] = df['actual_stock'] - df['recorded_stock']
        
        total_audits = len(df)
        accurate_audits = df[df['variance'] == 0].shape[0]
        accuracy_rate = (accurate_audits / total_audits) * 100 if total_audits > 0 else 0
        total_variance = df['variance'].sum() # Net variance
        total_abs_variance = df['variance'].abs().sum() # Gross variance

        metrics = {
            "Total Audits Performed": total_audits,
            "Accurate Audits (Zero Variance)": accurate_audits,
            "Inventory Accuracy Rate": accuracy_rate,
            "Total Net Stock Variance": total_variance,
            "Total Gross Stock Variance": total_abs_variance
        }

        insights.append(f"Performed {total_audits} stock audits.")
        insights.append(f"Inventory accuracy rate (zero variance): {accuracy_rate:.2f}%.")
        insights.append(f"Net stock variance: {total_variance} units.")
        insights.append(f"Gross (absolute) stock variance: {total_abs_variance} units.")

        # Visualizations
        variance_by_item = df.groupby('item_id')['variance'].sum().sort_values(key=abs, ascending=False).reset_index()
        fig1 = px.bar(variance_by_item.head(20), x='item_id', y='variance', title='Top 20 Items by Net Variance')
        visualizations["inventory_variance_by_item"] = fig1.to_json()

        if 'audit_result' in df.columns:
            audit_result_distribution = df['audit_result'].value_counts().reset_index()
            fig2 = px.pie(audit_result_distribution, names='audit_result', values='count', title='Audit Result Distribution')
            visualizations["audit_result_distribution"] = fig2.to_json()
        else:
            # Create one based on variance
            df['audit_result_calc'] = 'Match'
            df.loc[df['variance'] != 0, 'audit_result_calc'] = 'Variance'
            audit_result_distribution = df['audit_result_calc'].value_counts().reset_index()
            fig2 = px.pie(audit_result_distribution, names='audit_result_calc', values='count', title='Audit Result Distribution (Calculated)')
            visualizations["audit_result_distribution"] = fig2.to_json()


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

def product_return_reason_analysis(df):
    analysis_name = "Product Return Reason Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['return_id', 'product_id', 'return_date', 'return_reason', 'quantity_returned']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['return_date', 'return_reason', 'quantity_returned']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['return_date'] = pd.to_datetime(df['return_date'], errors='coerce')
        df['quantity_returned'] = pd.to_numeric(df['quantity_returned'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_returns_events = len(df)
        total_quantity_returned = df['quantity_returned'].sum()
        most_common_return_reason = df['return_reason'].mode()[0] if not df.empty else None

        metrics = {
            "Total Return Events": total_returns_events,
            "Total Quantity Returned": total_quantity_returned,
            "Most Common Return Reason": most_common_return_reason
        }

        insights.append(f"Recorded {total_returns_events} return events, totaling {total_quantity_returned:,.0f} units.")
        insights.append(f"The most common reason for returns is: {most_common_return_reason}.")

        # Visualizations
        returns_by_reason = df.groupby('return_reason')['quantity_returned'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(returns_by_reason.head(10), x='return_reason', y='quantity_returned', title='Top 10 Return Reasons by Quantity')
        visualizations["returns_by_reason"] = fig1.to_json()

        df['return_month'] = df['return_date'].dt.to_period('M').astype(str)
        returns_over_time = df.groupby('return_month')['quantity_returned'].sum().reset_index()
        fig2 = px.line(returns_over_time, x='return_month', y='quantity_returned', title='Returned Quantity Over Time (Monthly)')
        visualizations["returns_trend_over_time"] = fig2.to_json()

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

def factory_environmental_conditions_monitoring(df):
    analysis_name = "Factory Environmental Conditions Monitoring"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['timestamp', 'sensor_location', 'temperature_c', 'humidity_perc', 'air_quality_index']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['timestamp', 'sensor_location', 'temperature_c', 'humidity_perc', 'air_quality_index']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        for col in ['temperature_c', 'humidity_perc', 'air_quality_index']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.sort_values('timestamp').dropna(subset=critical_cols)

        avg_temp = df['temperature_c'].mean()
        avg_humidity = df['humidity_perc'].mean()
        avg_air_quality = df['air_quality_index'].mean()

        metrics = {
            "Average Temperature (°C)": avg_temp,
            "Average Humidity (%)": avg_humidity,
            "Average Air Quality Index": avg_air_quality
        }

        insights.append(f"Average temperature: {avg_temp:.1f}°C.")
        insights.append(f"Average humidity: {avg_humidity:.1f}%.")
        insights.append(f"Average Air Quality Index (AQI): {avg_air_quality:.1f}.")

        # Visualizations
        environmental_conditions_trend = df.melt(id_vars=['timestamp', 'sensor_location'], 
                                                 value_vars=['temperature_c', 'humidity_perc', 'air_quality_index'],
                                                 var_name='Metric', value_name='Value')
        
        fig1 = px.line(environmental_conditions_trend, x='timestamp', y='Value', color='sensor_location',
                       facet_row='Metric', title='Environmental Conditions Over Time by Location')
        fig1.update_yaxes(matches=None)
        visualizations["environmental_conditions_over_time"] = fig1.to_json()

        conditions_by_location = df.groupby('sensor_location').agg(
            avg_temperature=('temperature_c', 'mean'),
            avg_humidity=('humidity_perc', 'mean'),
            avg_air_quality=('air_quality_index', 'mean')
        ).reset_index().melt(id_vars='sensor_location', var_name='Metric', value_name='Average Value')
        
        fig2 = px.bar(conditions_by_location, x='sensor_location', y='Average Value', color='Metric', 
                      barmode='group', title='Average Environmental Conditions by Sensor Location')
        visualizations["environmental_conditions_by_sensor_location"] = fig2.to_json()

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

def manufacturing_waste_management_analysis(df):
    analysis_name = "Manufacturing Waste Management Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['waste_id', 'waste_type', 'generated_date', 'weight_kg', 'disposal_method', 'cost_of_disposal_usd']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['waste_type', 'weight_kg', 'disposal_method', 'cost_of_disposal_usd']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'generated_date' in df.columns:
            df['generated_date'] = pd.to_datetime(df['generated_date'], errors='coerce')
        for col in ['weight_kg', 'cost_of_disposal_usd']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_waste_kg = df['weight_kg'].sum()
        total_disposal_cost = df['cost_of_disposal_usd'].sum()
        most_common_waste_type = df['waste_type'].mode()[0] if not df.empty else None
        most_costly_disposal = df.groupby('disposal_method')['cost_of_disposal_usd'].sum().idxmax()

        metrics = {
            "Total Waste Generated (kg)": total_waste_kg,
            "Total Disposal Cost (USD)": total_disposal_cost,
            "Most Common Waste Type": most_common_waste_type,
            "Most Costly Disposal Method": most_costly_disposal,
            "Avg Cost per kg": total_disposal_cost / total_waste_kg if total_waste_kg > 0 else 0
        }

        insights.append(f"Generated {total_waste_kg:,.0f} kg of waste, costing ${total_disposal_cost:,.0f} to dispose.")
        insights.append(f"Average cost of disposal: ${metrics['Avg Cost per kg']:.2f} per kg.")
        insights.append(f"The most costly disposal method is: {most_costly_disposal}.")

        # Visualizations
        waste_by_type = df.groupby('waste_type')['weight_kg'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(waste_by_type, x='waste_type', y='weight_kg', title='Waste Generation (kg) by Type')
        visualizations["waste_generation_by_type"] = fig1.to_json()

        disposal_cost_by_method = df.groupby('disposal_method')['cost_of_disposal_usd'].sum().sort_values(ascending=False).reset_index()
        fig2 = px.pie(disposal_cost_by_method, names='disposal_method', values='cost_of_disposal_usd', title='Total Disposal Cost by Method')
        visualizations["disposal_cost_by_method"] = fig2.to_json()

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

def product_packaging_process_analysis(df):
    analysis_name = "Product Packaging Process Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['packaging_run_id', 'product_id', 'packaging_date', 'units_packaged', 'defects_packaging', 'packaging_line', 'packaging_time_minutes']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['product_id', 'units_packaged', 'defects_packaging', 'packaging_line', 'packaging_time_minutes']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'packaging_date' in df.columns:
            df['packaging_date'] = pd.to_datetime(df['packaging_date'], errors='coerce')
        for col in ['units_packaged', 'defects_packaging', 'packaging_time_minutes']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['packaging_defect_rate'] = (df['defects_packaging'] / df['units_packaged']) * 100
        
        total_units_packaged = df['units_packaged'].sum()
        avg_packaging_defect_rate = df['packaging_defect_rate'].mean()
        avg_packaging_time = df['packaging_time_minutes'].mean()

        metrics = {
            "Total Units Packaged": total_units_packaged,
            "Average Packaging Defect Rate": avg_packaging_defect_rate,
            "Average Packaging Time (minutes)": avg_packaging_time
        }

        insights.append(f"Packaged {total_units_packaged:,.0f} units.")
        insights.append(f"Average packaging defect rate: {avg_packaging_defect_rate:.2f}%.")
        insights.append(f"Average packaging time: {avg_packaging_time:.2f} minutes.")

        # Visualizations
        packaging_defects_by_line = df.groupby('packaging_line')['defects_packaging'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(packaging_defects_by_line, x='packaging_line', y='defects_packaging', title='Total Packaging Defects by Line')
        visualizations["packaging_defects_by_line"] = fig1.to_json()

        packaging_time_by_product = df.groupby('product_id')['packaging_time_minutes'].mean().reset_index()
        fig2 = px.bar(packaging_time_by_product, x='product_id', y='packaging_time_minutes', title='Average Packaging Time by Product')
        visualizations["packaging_time_by_product"] = fig2.to_json()

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

def outbound_shipment_tracking_analysis(df):
    analysis_name = "Outbound Shipment Tracking Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['shipment_id', 'order_id', 'ship_date', 'delivery_date', 'customer_location', 'shipping_cost', 'delivery_status']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['shipment_id', 'customer_location', 'shipping_cost', 'delivery_status']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in ['ship_date', 'delivery_date']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        df['shipping_cost'] = pd.to_numeric(df['shipping_cost'], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_shipments = len(df)
        delivered_shipments = df[df['delivery_status'].str.contains('delivered', case=False, na=False)].shape[0]
        avg_shipping_cost = df['shipping_cost'].mean()

        metrics = {
            "Total Shipments": total_shipments,
            "Delivered Shipments": delivered_shipments,
            "Delivery Rate": (delivered_shipments / total_shipments * 100) if total_shipments > 0 else 0,
            "Average Shipping Cost": avg_shipping_cost
        }

        insights.append(f"Tracked {total_shipments} shipments.")
        insights.append(f"Delivery rate: {metrics['Delivery Rate']:.1f}%.")
        insights.append(f"Average shipping cost: ${avg_shipping_cost:,.2f}.")

        # Visualizations
        delivery_status_breakdown = df['delivery_status'].value_counts().reset_index()
        fig1 = px.pie(delivery_status_breakdown, names='delivery_status', values='count', title='Delivery Status Breakdown')
        visualizations["delivery_status_breakdown"] = fig1.to_json()

        shipping_cost_by_location = df.groupby('customer_location')['shipping_cost'].mean().reset_index().sort_values('shipping_cost', ascending=False)
        fig2 = px.bar(shipping_cost_by_location.head(20), x='customer_location', y='shipping_cost', title='Top 20 Average Shipping Cost by Customer Location')
        visualizations["average_shipping_cost_by_customer_location"] = fig2.to_json()

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

def manufacturing_process_step_duration_analysis(df):
    analysis_name = "Manufacturing Process Step Duration Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['process_step_id', 'batch_id', 'start_time', 'end_time', 'process_name']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['batch_id', 'start_time', 'end_time', 'process_name']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)
        df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60

        avg_step_duration = df['duration_minutes'].mean()
        longest_step = df.groupby('process_name')['duration_minutes'].mean().idxmax()
        longest_step_duration = df.groupby('process_name')['duration_minutes'].mean().max()

        metrics = {
            "Average Process Step Duration (minutes)": avg_step_duration,
            "Longest Process Step (Avg)": longest_step,
            "Longest Step Avg Duration (minutes)": longest_step_duration,
            "Total Steps Logged": len(df)
        }

        insights.append(f"Logged {len(df)} process steps.")
        insights.append(f"Average step duration: {avg_step_duration:.2f} minutes.")
        insights.append(f"The longest average step is '{longest_step}' at {longest_step_duration:.2f} minutes.")

        # Visualizations
        duration_by_process_step = df.groupby('process_name')['duration_minutes'].mean().sort_values(ascending=False).reset_index()
        fig1 = px.bar(duration_by_process_step, x='process_name', y='duration_minutes', title='Average Duration by Process Step')
        visualizations["average_duration_by_process_step"] = fig1.to_json()

        # Gantt chart for a few example batches
        example_batches = df['batch_id'].unique()[:5] # Get first 5 batches
        gantt_df = df[df['batch_id'].isin(example_batches)]
        fig2 = px.timeline(gantt_df, x_start="start_time", x_end="end_time", y="batch_id", color="process_name",
                           title="Process Flow for Example Batches")
        visualizations["process_flow_for_batches"] = fig2.to_json()

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

def research_and_development_experiment_analysis(df):
    analysis_name = "Research and Development Experiment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['experiment_id', 'experiment_date', 'experiment_parameters', 'test_results', 'outcome_metric_1', 'outcome_metric_2', 'conclusion']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['experiment_date', 'outcome_metric_1', 'outcome_metric_2', 'conclusion']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['experiment_date'] = pd.to_datetime(df['experiment_date'], errors='coerce')
        for col in ['outcome_metric_1', 'outcome_metric_2']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_experiments = len(df)
        successful_experiments = df[df['conclusion'].str.contains('success', case=False, na=False)].shape[0]
        success_rate = (successful_experiments / total_experiments) * 100 if total_experiments > 0 else 0

        metrics = {
            "Total Experiments": total_experiments,
            "Successful Experiments": successful_experiments,
            "Experiment Success Rate": success_rate,
            "Avg Outcome Metric 1": df['outcome_metric_1'].mean(),
            "Avg Outcome Metric 2": df['outcome_metric_2'].mean()
        }

        insights.append(f"Analyzed {total_experiments} experiments.")
        insights.append(f"Overall experiment success rate: {success_rate:.1f}%.")

        # Visualizations
        outcome_metrics_distribution = df[['outcome_metric_1', 'outcome_metric_2']].melt(var_name='Metric', value_name='Value')
        fig1 = px.violin(outcome_metrics_distribution, x='Metric', y='Value', box=True, title='Outcome Metrics Distribution')
        visualizations["outcome_metrics_distribution"] = fig1.to_json()

        experiment_results_over_time = df.melt(id_vars='experiment_date', value_vars=['outcome_metric_1', 'outcome_metric_2'],
                                               var_name='Metric', value_name='Value')
        fig2 = px.scatter(experiment_results_over_time, x='experiment_date', y='Value', color='Metric',
                         title='Experiment Results Trend Over Time', trendline="ols")
        visualizations["experiment_results_trend_over_time"] = fig2.to_json()

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

def barcode_based_product_traceability_analysis(df):
    analysis_name = "Barcode-based Product Traceability Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['product_serial_number', 'batch_number', 'production_date', 'factory_id', 'qc_status', 'shipment_date']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['product_serial_number', 'production_date', 'factory_id', 'qc_status']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['production_date'] = pd.to_datetime(df['production_date'], errors='coerce')
        if 'shipment_date' in df.columns:
            df['shipment_date'] = pd.to_datetime(df['shipment_date'], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        total_traceable_products = df['product_serial_number'].nunique()
        products_by_factory = df['factory_id'].nunique()
        qc_pass_rate = df[df['qc_status'].str.contains('pass', case=False, na=False)].shape[0] / len(df) * 100
        
        metrics = {
            "Total Traceable Products": total_traceable_products,
            "Number of Factories Involved": products_by_factory,
            "Overall QC Pass Rate": qc_pass_rate
        }

        insights.append(f"Tracking {total_traceable_products} unique products from {products_by_factory} factories.")
        insights.append(f"Overall QC pass rate: {qc_pass_rate:.2f}%.")

        # Visualizations
        qc_status_distribution = df['qc_status'].value_counts().reset_index()
        fig1 = px.pie(qc_status_distribution, names='qc_status', values='count', title='QC Status Distribution')
        visualizations["qc_status_distribution"] = fig1.to_json()

        df['production_month'] = df['production_date'].dt.to_period('M').astype(str)
        products_by_production_date = df.groupby('production_month')['product_serial_number'].count().reset_index()
        fig2 = px.bar(products_by_production_date, x='production_month', y='product_serial_number', title='Products by Production Month')
        visualizations["products_by_production_date"] = fig2.to_json()

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

def internal_process_and_compliance_audit_analysis(df):
    analysis_name = "Internal Process and Compliance Audit Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['audit_id', 'audit_date', 'department', 'compliance_area', 'audit_score', 'findings', 'recommendations', 'status']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['audit_date', 'compliance_area', 'audit_score', 'status']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
        df['audit_score'] = pd.to_numeric(df['audit_score'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_audits = len(df)
        avg_audit_score = df['audit_score'].mean()
        non_compliant_audits = df[df['status'].str.contains('non-compliant|fail', case=False, na=False)].shape[0]
        non_compliant_rate = (non_compliant_audits / total_audits) * 100 if total_audits > 0 else 0

        metrics = {
            "Total Audits": total_audits,
            "Average Audit Score": avg_audit_score,
            "Non-Compliant Audits": non_compliant_audits,
            "Non-Compliance Rate (%)": non_compliant_rate
        }

        insights.append(f"Analyzed {total_audits} internal audits.")
        insights.append(f"Average audit score: {avg_audit_score:.1f}.")
        insights.append(f"Non-compliance rate: {non_compliant_rate:.1f}%.")

        # Visualizations
        audit_score_by_compliance_area = df.groupby('compliance_area')['audit_score'].mean().sort_values(ascending=False).reset_index()
        fig1 = px.bar(audit_score_by_compliance_area, x='compliance_area', y='audit_score', title='Average Audit Score by Compliance Area')
        visualizations["audit_score_by_compliance_area"] = fig1.to_json()

        audit_status_distribution = df['status'].value_counts().reset_index()
        fig2 = px.pie(audit_status_distribution, names='status', values='count', title='Audit Status Distribution')
        visualizations["audit_status_distribution"] = fig2.to_json()

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

def machine_capacity_and_load_analysis(df):
    analysis_name = "Machine Capacity and Load Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['machine_id', 'capacity_units_per_hour', 'actual_production_units', 'shift_hours_worked', 'downtime_hours']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['machine_id', 'capacity_units_per_hour', 'actual_production_units', 'shift_hours_worked', 'downtime_hours']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        for col in critical_cols:
            if col != 'machine_id':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['theoretical_capacity_units'] = df['capacity_units_per_hour'] * df['shift_hours_worked']
        df['available_hours'] = df['shift_hours_worked'] - df['downtime_hours']
        df['available_capacity_units'] = df['capacity_units_per_hour'] * df['available_hours']

        # OEE Components
        df['utilization'] = (df['available_hours'] / df['shift_hours_worked']) * 100
        df['performance'] = (df['actual_production_units'] / df['available_capacity_units']) * 100
        df['load_percentage'] = (df['actual_production_units'] / df['theoretical_capacity_units']) * 100
        
        # Clean up inf/-inf from potential division by zero
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        avg_load_percentage = df['load_percentage'].mean()
        avg_utilization = df['utilization'].mean()
        avg_performance = df['performance'].mean()

        metrics = {
            "Average Machine Load Percentage": avg_load_percentage,
            "Average Machine Utilization (OEE Component)": avg_utilization,
            "Average Machine Performance (OEE Component)": avg_performance
        }

        insights.append(f"Average machine load: {avg_load_percentage:.2f}%.")
        insights.append(f"Average machine utilization (availability): {avg_utilization:.2f}%.")
        insights.append(f"Average machine performance (speed): {avg_performance:.2f}%.")

        # Visualizations
        load_by_machine = df.groupby('machine_id')['load_percentage'].mean().reset_index()
        fig1 = px.bar(load_by_machine, x='machine_id', y='load_percentage', title='Average Machine Load by Machine ID')
        visualizations["machine_load_by_machine_id"] = fig1.to_json()

        fig2 = px.scatter(df, x='utilization', y='performance', color='machine_id',
                          title='Machine Efficiency (Performance vs. Utilization)')
        visualizations["machine_efficiency_vs_downtime"] = fig2.to_json() # Renamed for clarity

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

def production_volume_variance_analysis(df):
    analysis_name = "Production Volume Variance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['date', 'product_id', 'planned_production', 'actual_production']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['date', 'product_id', 'planned_production', 'actual_production']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for col in ['planned_production', 'actual_production']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df.dropna(subset=critical_cols, inplace=True)

        df['variance'] = df['actual_production'] - df['planned_production']
        df['variance_percentage'] = (df['variance'] / df['planned_production']) * 100
        df.replace([np.inf, -np.inf], np.nan, inplace=True)


        total_planned_production = df['planned_production'].sum()
        total_actual_production = df['actual_production'].sum()
        total_variance = df['variance'].sum()
        avg_variance_percentage = (total_variance / total_planned_production) * 100 if total_planned_production > 0 else 0

        metrics = {
            "Total Planned Production": total_planned_production,
            "Total Actual Production": total_actual_production,
            "Total Production Variance (Units)": total_variance,
            "Overall Variance Percentage": avg_variance_percentage
        }

        insights.append(f"Planned: {total_planned_production:,.0f} units. Actual: {total_actual_production:,.0f} units.")
        insights.append(f"Total variance: {total_variance:,.0f} units ({avg_variance_percentage:.2f}%).")

        # Visualizations
        variance_by_product = df.groupby('product_id')['variance'].sum().sort_values(key=abs, ascending=False).reset_index()
        fig1 = px.bar(variance_by_product.head(20), x='product_id', y='variance', title='Top 20 Products by Production Variance (Units)')
        visualizations["production_variance_by_product"] = fig1.to_json()

        daily_variance_trend = df.groupby('date')[['planned_production', 'actual_production', 'variance']].sum().reset_index()
        fig2_data = daily_variance_trend.melt(id_vars='date', value_vars=['planned_production', 'actual_production'],
                                              var_name='Production Type', value_name='Volume')
        fig2 = px.line(fig2_data, x='date', y='Volume', color='Production Type', title='Daily Production (Actual vs. Planned)')
        visualizations["daily_production_variance_trend"] = fig2.to_json()

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

def iot_sensor_data_time_series_analysis(df):
    analysis_name = "IoT Sensor Data Time-series Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['timestamp', 'sensor_id', 'reading_value', 'unit']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['timestamp', 'sensor_id', 'reading_value']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['reading_value'] = pd.to_numeric(df['reading_value'], errors='coerce')
        df = df.sort_values('timestamp').dropna(subset=critical_cols)

        num_sensors = df['sensor_id'].nunique()
        avg_reading = df['reading_value'].mean()
        
        metrics = {
            "Number of Unique Sensors": num_sensors,
            "Average Sensor Reading": avg_reading,
            "Total Readings": len(df),
            "Start Time": df['timestamp'].min().isoformat(),
            "End Time": df['timestamp'].max().isoformat()
        }

        insights.append(f"Analyzed {len(df)} readings from {num_sensors} unique sensors.")
        insights.append(f"Data ranges from {metrics['Start Time']} to {metrics['End Time']}.")

        # Visualizations
        fig1 = px.line(df, x='timestamp', y='reading_value', color='sensor_id', title='Sensor Readings Over Time')
        visualizations["sensor_reading_trend_over_time"] = fig1.to_json()

        fig2 = px.violin(df, x='sensor_id', y='reading_value', box=True, title='Reading Distribution by Sensor')
        visualizations["reading_distribution_by_sensor"] = fig2.to_json()

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

def cost_center_expense_analysis(df):
    analysis_name = "Cost Center Expense Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['expense_id', 'date', 'cost_center', 'expense_category', 'amount_usd']
        matched = fuzzy_match_column(df, expected)
        critical_cols = ['cost_center', 'expense_category', 'amount_usd']
        missing = [col for col in critical_cols if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)
            
        df = safe_rename(df, matched)
        
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['amount_usd'] = pd.to_numeric(df['amount_usd'], errors='coerce')
        df.dropna(subset=critical_cols, inplace=True)

        total_expenses = df['amount_usd'].sum()
        num_cost_centers = df['cost_center'].nunique()
        top_cost_center = df.groupby('cost_center')['amount_usd'].sum().idxmax()
        top_expense_category = df.groupby('expense_category')['amount_usd'].sum().idxmax()
        
        metrics = {
            "Total Expenses (USD)": total_expenses,
            "Number of Cost Centers": num_cost_centers,
            "Top Cost Center by Spend": top_cost_center,
            "Top Expense Category by Spend": top_expense_category
        }

        insights.append(f"Total expenses analyzed: ${total_expenses:,.0f}.")
        insights.append(f"Top cost center: {top_cost_center}.")
        insights.append(f"Top expense category: {top_expense_category}.")

        # Visualizations
        expenses_by_cost_center = df.groupby('cost_center')['amount_usd'].sum().sort_values(ascending=False).reset_index()
        fig1 = px.bar(expenses_by_cost_center, x='cost_center', y='amount_usd', title='Total Expenses by Cost Center')
        visualizations["expenses_by_cost_center"] = fig1.to_json()

        expenses_by_category = df.groupby('expense_category')['amount_usd'].sum().sort_values(ascending=False).reset_index()
        fig2 = px.pie(expenses_by_category, names='expense_category', values='amount_usd', title='Expense Distribution by Category')
        visualizations["expenses_by_category"] = fig2.to_json()

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

def regulatory_compliance_status_analysis(df):
    """Performs regulatory compliance status analysis with enhanced visualizations and metrics"""
    analysis_name = "Regulatory Compliance Status Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['compliance_id', 'audit_date', 'regulation_name', 'compliance_status', 'findings_count', 'severity']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        df['audit_date'] = pd.to_datetime(df['audit_date'], errors='coerce')
        df['findings_count'] = pd.to_numeric(df['findings_count'], errors='coerce')
        df.dropna(subset=['compliance_status', 'regulation_name'], inplace=True)

        # Metrics
        total_audits = len(df)
        compliant_audits = df[df['compliance_status'].str.contains('compliant', case=False, na=False)].shape[0]
        compliance_rate = (compliant_audits / total_audits) * 100 if total_audits > 0 else 0
        avg_findings_count = df['findings_count'].mean() if not df['findings_count'].isnull().all() else None

        # Additional metrics
        if 'severity' in df.columns:
            high_severity_findings = df[df['severity'].str.contains('high|critical', case=False, na=False)].shape[0]
            high_severity_percentage = (high_severity_findings / total_audits) * 100 if total_audits > 0 else 0
        else:
            high_severity_findings = None
            high_severity_percentage = None

        metrics = {
            "total_compliance_audits": total_audits,
            "compliant_audits": compliant_audits,
            "overall_compliance_rate_percent": compliance_rate,
            "average_findings_count": avg_findings_count,
            "high_severity_findings_count": high_severity_findings,
            "high_severity_findings_percentage": high_severity_percentage
        }
        
        insights.append(f"Total compliance audits analyzed: {total_audits:,}")
        insights.append(f"Overall compliance rate: {compliance_rate:.1f}%")
        insights.append(f"Average findings per audit: {avg_findings_count:.1f}" if avg_findings_count else "No findings count data available")
        
        if high_severity_findings:
            insights.append(f"High severity findings: {high_severity_findings} ({high_severity_percentage:.1f}% of audits)")

        # Visualizations
        # 1. Compliance Status Distribution
        compliance_status_distribution = df['compliance_status'].value_counts().reset_index()
        compliance_status_distribution.columns = ['compliance_status', 'count']
        fig1 = px.pie(compliance_status_distribution, 
                     values='count', 
                     names='compliance_status',
                     title='Compliance Status Distribution',
                     hole=0.4)
        visualizations["compliance_status_distribution"] = fig1.to_json()

        # 2. Findings by Regulation
        if 'regulation_name' in df.columns and 'findings_count' in df.columns:
            findings_by_regulation = df.groupby('regulation_name')['findings_count'].sum().sort_values(ascending=False).reset_index()
            findings_by_regulation.columns = ['regulation_name', 'total_findings_count']
            fig2 = px.bar(findings_by_regulation.head(10), 
                         x='regulation_name', 
                         y='total_findings_count',
                         title='Top 10 Regulations by Total Findings Count',
                         labels={'total_findings_count': 'Total Findings', 'regulation_name': 'Regulation'})
            visualizations["findings_by_regulation"] = fig2.to_json()

        # 3. Compliance Trend Over Time
        if 'audit_date' in df.columns:
            df_sorted = df.sort_values('audit_date')
            monthly_compliance = df_sorted.groupby(df_sorted['audit_date'].dt.to_period('M')).agg({
                'compliance_status': lambda x: (x.str.contains('compliant', case=False).sum() / len(x)) * 100
            }).reset_index()
            monthly_compliance['audit_date'] = monthly_compliance['audit_date'].astype(str)
            
            fig3 = px.line(monthly_compliance, 
                          x='audit_date', 
                          y='compliance_status',
                          title='Compliance Rate Trend Over Time',
                          labels={'compliance_status': 'Compliance Rate (%)', 'audit_date': 'Month'})
            visualizations["compliance_trend"] = fig3.to_json()

        # 4. Severity Analysis (if available)
        if 'severity' in df.columns:
            severity_distribution = df['severity'].value_counts().reset_index()
            severity_distribution.columns = ['severity', 'count']
            fig4 = px.bar(severity_distribution,
                         x='severity',
                         y='count',
                         title='Finding Severity Distribution',
                         color='severity')
            visualizations["severity_distribution"] = fig4.to_json()

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

# ========== UPDATED MAIN FUNCTION ==========

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
        "regulatory_compliance_status_analysis": regulatory_compliance_status_analysis,
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
    print("🏭 Manufacturing Analytics Dashboard")

    # File path and encoding input
    file_path = input("Enter path to your manufacturing data file (e.g., data.csv or data.xlsx): ")
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
    
    print(f"\n📊 DATASET SHAPE: {df.shape}")
    print(f"🔍 SAMPLE OF FIRST FEW ROWS:")
    print(df.head(3))

    # Analysis selection
    print("\nSelect analysis type:")
    print("1. General Analysis")
    print("2. Specific Analysis")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == "1":
        # General analyses
        general_analyses = [
            "General Insights", 
            "regulatory_compliance_status_analysis"
        ]
        
        print("\nSelect General Analysis:")
        for i, analysis in enumerate(general_analyses):
            print(f"{i+1}. {analysis}")
        
        try:
            analysis_choice = int(input(f"Enter the number of your choice (1-{len(general_analyses)}): ")) - 1
            if 0 <= analysis_choice < len(general_analyses):
                result = main_backend(
                    file_path, 
                    encoding=encoding,
                    category="General",
                    analysis=general_analyses[analysis_choice]
                )
            else:
                print("Invalid choice. Running General Insights.")
                result = main_backend(file_path, encoding=encoding)
        except ValueError:
            print("Invalid input. Running General Insights.")
            result = main_backend(file_path, encoding=encoding)
            
    elif choice == "2":
        # Specific analyses
        specific_analyses = [
            "regulatory_compliance_status_analysis"
        ]
        
        print("\nSelect Specific Analysis:")
        for i, analysis in enumerate(specific_analyses):
            print(f"{i+1}. {analysis}")
        
        try:
            analysis_choice = int(input(f"Enter the number of your choice (1-{len(specific_analyses)}): ")) - 1
            if 0 <= analysis_choice < len(specific_analyses):
                result = main_backend(
                    file_path, 
                    encoding=encoding,
                    category="Specific",
                    specific_analysis_name=specific_analyses[analysis_choice]
                )
            else:
                print("Invalid choice. Running General Insights.")
                result = main_backend(file_path, encoding=encoding)
        except ValueError:
            print("Invalid input. Running General Insights.")
            result = main_backend(file_path, encoding=encoding)
    else:
        print("Invalid choice. Running General Insights.")
        result = main_backend(file_path, encoding=encoding)

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
            print(f"\n📈 Generated Visualizations: {len(visualizations)}")
            for viz_name in visualizations.keys():
                print(f"  - {viz_name}")

        # Save results option
        save_option = input("\n💾 Would you like to save the results to a JSON file? (y/n): ").lower()
        if save_option in ['y', 'yes']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_name_clean = result.get('analysis_type', 'analysis').replace(' ', '_').lower()
            filename = f"manufacturing_analytics_{analysis_name_clean}_{timestamp}.json"
            
            try:
                with open(filename, 'w') as f:
                    json.dump(convert_to_native_types(result), f, indent=2)
                print(f"✅ Results saved to: {filename}")
            except Exception as e:
                print(f"❌ Error saving file: {e}")

        print(f"\n🎉 Analysis completed successfully!")

    else:
        print("❌ No results returned from analysis.")

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