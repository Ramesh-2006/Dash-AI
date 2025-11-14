import pandas as pd
import numpy as np
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# List for choosing analysis from UI, API, etc.
analysis_options = [
    "attrition_analysis",
    "performance_analysis",
    "compensation_analysis",
    "diversity_analysis",
    "training_analysis",
    "engagement_analysis",
    "recruitment_analysis",
    "productivity_analysis",
    "retention_analysis",
    "attendance_analysis",
    "employee_demographic_and_tenure_analysis",
    "employee_profile_and_departmental_analysis",
    "employee_compensation_and_tenure_analysis",
    "employee_attrition_prediction_and_factor_analysis",
    "employee_distribution_and_service_length_analysis",
    "employee_performance_and_compensation_analysis",
    "employee_salary_and_attrition_analysis",
    "employee_salary_hike_and_promotion_factor_analysis",
    "work-life_balance_and_job_satisfaction_impact_on_attrition",
    "commute_distance_and_work_history_impact_on_attrition",
    "employee_performance_and_promotion_cycle_analysis",
    "employee_demographic_and_compensation_profile_analysis",
    "factors_influencing_employee_attrition_analysis",
    "employee_demographics_and_attrition_correlation_analysis",
    "employee_performance_and_tenure_analysis",
    "compensation_promotion_and_career_progression_analysis",
    "employee_profile_and_training_engagement_analysis",
    "attrition_factors_related_to_promotions_and_stock_options",
    "comprehensive_employee_satisfaction_and_attrition_analysis",
    "employee_compensation_structure_and_attrition_analysis",
    "employee_performance_and_career_level_attrition_analysis",
    "employee_salary_structure_analysis_by_department",
    "management_and_its_impact_on_employee_performance_and_attrition",
    "job_involvement_and_training_impact_on_employee_retention",
    "work-life_balance_and_job_satisfactions_effect_on_attrition",
    "employee_lifecycle_and_attrition_trend_analysis",
    "performance_and_workload_impact_on_employee_attrition",
    "training_and_stock_options_effect_on_employee_retention",
    "performance_rating_correlation_with_employee_attrition",
    "job_satisfaction_determinants_for_employee_retention",
    "employee_performance_training_and_attrition_link_analysis",
    "employee_tenure_and_attrition_risk_analysis",
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
        return str(data)

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis - SIMPLIFIED AND ROBUST
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
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            try:
                numeric_stats = df[numeric_cols].describe().to_dict()
            except:
                pass
        
        # Categorical columns analysis
        categorical_stats = {}
        if categorical_cols:
            for col in categorical_cols[:3]:  # Limit to first 3 for brevity
                try:
                    unique_count = df[col].nunique()
                    top_values = df[col].value_counts().head(3).to_dict()
                    categorical_stats[col] = {
                        "unique_count": int(unique_count),
                        "top_values": convert_to_native_types(top_values)
                    }
                except:
                    pass
        
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
            for i, col in enumerate(numeric_cols[:2]):
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
        
        # Generate insights - NOW INCLUDING MISSING COLUMNS WARNINGS
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
                insights.append(f"   - {col}{match_info}")
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
            "Limited analysis due to data compatibility"
        ]
        
        # Add missing columns warning even in error case
        if missing_cols and len(missing_cols) > 0:
            basic_insights.insert(0, "⚠️ REQUIRED COLUMNS NOT FOUND - Showing General Analysis")
            basic_insights.insert(1, f"Missing columns: {', '.join(missing_cols)}")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",  # Still return success for basic info
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
        # Generate the fallback general insights with error handling
        general_insights_data = show_general_insights(
            df, 
            f"General Analysis (Fallback for {analysis_name})",
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
        
        # If general insights also fails, create a basic dataset overview
        if general_insights_data.get('status') == 'error':
            raise Exception(general_insights_data.get('error_message', 'General insights failed'))
            
    except Exception as fallback_error:
        print(f"General insights also failed: {fallback_error}")
        # Create a minimal fallback response with basic dataset info
        general_insights_data = {
            "analysis_type": "General Insights",
            "status": "partial_success",
            "visualizations": {},
            "metrics": {
                "dataset_info": {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist()[:10]  # First 10 columns
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
            match, score = process.extractOne(target, available)
            matched[target] = match if score >= 60 else None
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

# ========== HR ANALYSIS FUNCTIONS (Content) ==========

def attrition_analysis(df):
    analysis_name = "Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['employee_id', 'attrition', 'age', 'department', 'tenure']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})

        # Ensure 'attrition' is numeric (0/1) for calculations
        if df['attrition'].dtype == 'object':
            # Handle potential variations like 'Yes'/'No', 'True'/'False', 1/0 as strings
            attrition_map = {
                'yes': 1, 'true': 1, '1': 1,
                'no': 0, 'false': 0, '0': 0
            }
            df['attrition_flag'] = df['attrition'].str.lower().map(attrition_map).fillna(-1) # -1 for unmapped
            if (df['attrition_flag'] == -1).any():
                insights.append("Warning: Some 'attrition' values were not recognized ('Yes'/'No', 1/0) and were ignored.")
                df = df[df['attrition_flag'] != -1]
        else:
             df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
        
        df.dropna(subset=['attrition_flag', 'tenure', 'age'], inplace=True)

        # Metrics
        total_employees = len(df)
        attrition_rate = df['attrition_flag'].mean() * 100
        avg_tenure_leavers = df[df['attrition_flag'] == 1]['tenure'].mean()
        avg_age_leavers = df[df['attrition_flag'] == 1]['age'].mean()

        metrics = {
            "total_employees": total_employees,
            "attrition_rate_percent": attrition_rate,
            "avg_tenure_leavers_yrs": avg_tenure_leavers,
            "avg_age_leavers": avg_age_leavers
        }
        
        insights.append(f"Total employees analyzed: {total_employees}")
        insights.append(f"Overall attrition rate: {attrition_rate:.1f}%")
        insights.append(f"Average tenure of employees who left: {avg_tenure_leavers:.1f} years")

        # Visualizations
        fig1 = px.histogram(df, x='age', color='attrition', barmode='overlay',
                            title="Age Distribution by Attrition Status")
        visualizations["age_distribution_by_attrition"] = fig1.to_json()

        fig2 = px.box(df, x='attrition', y='tenure',
                      title="Tenure Comparison by Attrition Status")
        visualizations["tenure_comparison_by_attrition"] = fig2.to_json()

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

def performance_analysis(df):
    analysis_name = "Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'performance_rating', 'department', 'manager_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return create_fallback_response(analysis_name, missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['performance_rating'] = pd.to_numeric(df['performance_rating'], errors='coerce')
        df.dropna(subset=['performance_rating'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_rating = df['performance_rating'].mean()
        median_rating = df['performance_rating'].median()
        rating_scale_max = df['performance_rating'].max()
        # Assuming rating >= 4 is top performance on a 5-point scale
        top_performers = (df['performance_rating'] >= 4).sum()
        top_performers_percent = (top_performers / total_employees) * 100 if total_employees > 0 else 0

        metrics = {
            "total_employees": total_employees,
            "avg_rating": avg_rating,
            "median_rating": median_rating,
            "rating_scale_max": rating_scale_max,
            "top_performers_count": top_performers,
            "top_performers_percent": top_performers_percent
        }
        
        insights.append(f"Analyzed {total_employees} employees.")
        insights.append(f"Average performance rating: {avg_rating:.2f} (out of {rating_scale_max})")
        insights.append(f"{top_performers} employees ({top_performers_percent:.1f}%) are top performers (Rating >= 4)")

        # Visualizations
        fig1 = px.histogram(df, x='performance_rating',
                            title="Performance Rating Distribution")
        visualizations["performance_distribution"] = fig1.to_json()

        if 'department' in df.columns:
            fig2 = px.box(df, x='department', y='performance_rating',
                          title="Performance by Department")
            visualizations["performance_by_department"] = fig2.to_json()
            insights.append("Generated plot for performance by department.")
        else:
            insights.append("Skipped 'Performance by Department' plot: 'department' column not found.")

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

def compensation_analysis(df):
    analysis_name = "Compensation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'salary', 'bonus', 'department', 'job_level', 'gender']
        matched = fuzzy_match_column(df, expected)
        # Salary is critical, bonus/job_level/gender are optional for basic analysis
        critical_missing = [col for col in ['employee_id', 'salary', 'department'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['salary'] = pd.to_numeric(df['salary'], errors='coerce')
        df.dropna(subset=['salary'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_salary = df['salary'].mean()
        median_salary = df['salary'].median()
        total_payroll = df['salary'].sum()
        pay_gap_info = "N/A (Gender column not found or insufficient data)"
        
        metrics = {
            "total_employees": total_employees,
            "avg_salary": avg_salary,
            "median_salary": median_salary,
            "total_payroll": total_payroll,
        }

        if 'gender' in df.columns and df['gender'].nunique() > 1:
            gender_salaries = df.groupby('gender')['salary'].mean()
            # Standardize gender names for comparison
            gender_salaries.index = gender_salaries.index.str.lower()
            female_salary = gender_salaries.get('female', 0)
            male_salary = gender_salaries.get('male', 0)
            
            if female_salary > 0 and male_salary > 0:
                pay_gap_abs = abs(female_salary - male_salary)
                pay_gap_pct = (male_salary - female_salary) / male_salary * 100 if male_salary > 0 else 0
                metrics['gender_pay_gap_absolute'] = pay_gap_abs
                metrics['gender_pay_gap_percent_female_vs_male'] = pay_gap_pct
                metrics['avg_salary_female'] = female_salary
                metrics['avg_salary_male'] = male_salary
                pay_gap_info = f"${pay_gap_abs:,.0f} (Females earn {pay_gap_pct:.1f}% of male salary)"

        insights.append(f"Analyzed {total_employees} employees.")
        insights.append(f"Average salary: ${avg_salary:,.0f} (Median: ${median_salary:,.0f})")
        insights.append(f"Total annual payroll (based on 'salary'): ${total_payroll:,.0f}")
        insights.append(f"Gender Pay Gap (Female vs Male): {pay_gap_info}")

        # Visualizations
        fig1 = px.box(df, x='department', y='salary',
                      title="Salary Distribution by Department")
        visualizations["salary_by_department"] = fig1.to_json()

        if 'job_level' in df.columns:
            fig2 = px.scatter(df, x='job_level', y='salary', 
                            color='gender' if 'gender' in df.columns else None,
                            title="Salary by Job Level")
            visualizations["salary_by_job_level"] = fig2.to_json()
            insights.append("Generated 'Salary by Job Level' plot.")
        else:
            insights.append("Skipped 'Salary by Job Level' plot: 'job_level' column not found.")

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

def diversity_analysis(df):
    analysis_name = "Diversity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'gender', 'department', 'age', 'ethnicity']
        matched = fuzzy_match_column(df, expected)
        # Need at least one diversity metric to run
        critical_missing = [col for col in ['employee_id', 'gender', 'ethnicity', 'age'] if matched[col] is None]

        if len(critical_missing) == 4: # If all are missing
             return create_fallback_response(analysis_name, ['gender', 'ethnicity', 'age'], matched, df)
        
        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        total_employees = len(df)
        metrics['total_employees'] = total_employees
        
        # Gender Analysis
        if 'gender' in df.columns:
            gender_dist = df['gender'].value_counts()
            gender_dist_pct = df['gender'].value_counts(normalize=True) * 100
            metrics['gender_distribution_count'] = gender_dist.to_dict()
            metrics['gender_distribution_percent'] = gender_dist_pct.to_dict()
            insights.append(f"Gender breakdown: {', '.join([f'{k}: {v:.1f}%' for k,v in gender_dist_pct.items()])}")
            
            fig1 = px.pie(df, names='gender', title="Gender Distribution")
            visualizations["gender_distribution"] = fig1.to_json()
            
            if 'department' in df.columns:
                fig2 = px.bar(df.groupby(['department','gender']).size().reset_index(name='count'),
                                x='department', y='count', color='gender',
                                title="Departmental Gender Distribution")
                visualizations["departmental_gender_distribution"] = fig2.to_json()
                insights.append("Generated 'Departmental Gender Distribution' plot.")
            else:
                insights.append("Skipped 'Departmental Gender Distribution' plot: 'department' column not found.")
        else:
            insights.append("Skipping gender analysis: 'gender' column not found.")

        # Ethnicity Analysis
        if 'ethnicity' in df.columns:
            ethnicity_count = df['ethnicity'].nunique()
            ethnicity_dist = df['ethnicity'].value_counts()
            ethnicity_dist_pct = df['ethnicity'].value_counts(normalize=True) * 100
            metrics['ethnicity_unique_count'] = ethnicity_count
            metrics['ethnicity_distribution_count'] = ethnicity_dist.to_dict()
            metrics['ethnicity_distribution_percent'] = ethnicity_dist_pct.to_dict()
            insights.append(f"Found {ethnicity_count} unique ethnicities.")
            
            fig3 = px.pie(df, names='ethnicity', title="Ethnicity Distribution")
            visualizations["ethnicity_distribution"] = fig3.to_json()
        else:
            insights.append("Skipping ethnicity analysis: 'ethnicity' column not found.")

        # Age Analysis
        if 'age' in df.columns:
            df['age'] = pd.to_numeric(df['age'], errors='coerce')
            df.dropna(subset=['age'], inplace=True)
            metrics['average_age'] = df['age'].mean()
            metrics['median_age'] = df['age'].median()
            insights.append(f"Average employee age: {df['age'].mean():.1f} years")
            
            fig4 = px.histogram(df, x='age', title="Age Distribution")
            visualizations["age_distribution"] = fig4.to_json()
        else:
            insights.append("Skipping age analysis: 'age' column not found.")

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

def training_analysis(df):
    analysis_name = "Training Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['employee_id', 'training_hours', 'training_completed', 'skill_gain', 'department']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['employee_id', 'training_hours', 'training_completed'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['training_hours'] = pd.to_numeric(df['training_hours'], errors='coerce')
        # Handle 'training_completed' (assuming 1/0 or True/False)
        if df['training_completed'].dtype == 'object':
             df['training_completed_flag'] = df['training_completed'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
             df['training_completed_flag'] = pd.to_numeric(df['training_completed'], errors='coerce')
             
        df.dropna(subset=['training_hours', 'training_completed_flag'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_hours = df['training_hours'].mean()
        total_hours = df['training_hours'].sum()
        completion_rate = df['training_completed_flag'].mean() * 100

        metrics = {
            "employees_with_training_data": total_employees,
            "avg_training_hours": avg_hours,
            "total_training_hours": total_hours,
            "completion_rate_percent": completion_rate
        }
        
        insights.append(f"Analyzed {total_employees} employees with training data.")
        insights.append(f"Average training hours per employee: {avg_hours:.1f}")
        insights.append(f"Overall training completion rate: {completion_rate:.1f}%")

        # Visualizations
        fig1 = px.histogram(df, x='training_hours',
                            title="Training Hours Distribution")
        visualizations["training_hours_distribution"] = fig1.to_json()

        if 'skill_gain' in df.columns:
            df['skill_gain'] = pd.to_numeric(df['skill_gain'], errors='coerce')
            if not df['skill_gain'].isnull().all():
                metrics['avg_skill_gain'] = df['skill_gain'].mean()
                insights.append(f"Average skill gain: {metrics['avg_skill_gain']:.2f}")
                fig2 = px.scatter(df, x='training_hours', y='skill_gain',
                                  title="Training Impact on Skills",
                                  trendline='ols', trendline_scope='overall')
                visualizations["training_impact_on_skills"] = fig2.to_json()
            else:
                 insights.append("Skipped 'Training Impact on Skills' plot: 'skill_gain' column has no valid data.")
        else:
            insights.append("Skipped 'Training Impact on Skills' plot: 'skill_gain' column not found.")

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

def engagement_analysis(df):
    analysis_name = "Engagement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'engagement_score', 'department', 'survey_date']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['employee_id', 'engagement_score'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce')
        df.dropna(subset=['engagement_score'], inplace=True)
        
        # Metrics
        total_responses = len(df)
        avg_engagement = df['engagement_score'].mean()
        max_score = df['engagement_score'].max()
        # Assuming score on a scale of 1-5 where <3 is low
        low_engagement_threshold = 3
        low_engagement_count = (df['engagement_score'] < low_engagement_threshold).sum()
        low_engagement_percent = (low_engagement_count / total_responses) * 100 if total_responses > 0 else 0

        metrics = {
            "survey_responses": total_responses,
            "avg_engagement_score": avg_engagement,
            "assumed_max_score": max_score,
            "low_engagement_count": low_engagement_count,
            "low_engagement_percent": low_engagement_percent,
            "low_engagement_threshold": low_engagement_threshold
        }
        
        insights.append(f"Analyzed {total_responses} survey responses.")
        insights.append(f"Average engagement score: {avg_engagement:.2f} (out of {max_score})")
        insights.append(f"{low_engagement_count} employees ({low_engagement_percent:.1f}%) have low engagement (< {low_engagement_threshold})")

        # Visualizations
        fig1 = px.histogram(df, x='engagement_score',
                            title="Engagement Score Distribution")
        visualizations["engagement_score_distribution"] = fig1.to_json()

        if 'department' in df.columns:
            fig2 = px.box(df, x='department', y='engagement_score',
                          title="Engagement by Department")
            visualizations["engagement_by_department"] = fig2.to_json()
            insights.append("Generated 'Engagement by Department' plot.")
        else:
            insights.append("Skipped 'Engagement by Department' plot: 'department' column not found.")
            
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

def recruitment_analysis(df):
    analysis_name = "Recruitment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['hire_date', 'time_to_hire', 'source', 'department']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['hire_date', 'time_to_hire', 'source'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})

        # Convert dates and numeric
        if not pd.api.types.is_datetime64_any_dtype(df['hire_date']):
            df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
        df['time_to_hire'] = pd.to_numeric(df['time_to_hire'], errors='coerce')
        df.dropna(subset=['hire_date', 'time_to_hire', 'source'], inplace=True)

        # Metrics
        total_hires = len(df)
        avg_time_to_hire = df['time_to_hire'].mean()
        median_time_to_hire = df['time_to_hire'].median()
        top_source = df['source'].mode()[0] if not df['source'].empty else "N/A"
        top_source_count = df[df['source'] == top_source].shape[0] if top_source != "N/A" else 0

        metrics = {
            "total_hires_analyzed": total_hires,
            "avg_time_to_hire_days": avg_time_to_hire,
            "median_time_to_hire_days": median_time_to_hire,
            "top_source_of_hires": top_source,
            "top_source_hires_count": top_source_count
        }
        
        insights.append(f"Analyzed {total_hires} new hires.")
        insights.append(f"Average time to hire: {avg_time_to_hire:.1f} days (Median: {median_time_to_hire:.1f})")
        insights.append(f"Top source of hires: {top_source} (with {top_source_count} hires)")

        # Visualizations
        fig1 = px.histogram(df, x='time_to_hire',
                            title="Time to Hire Distribution (Days)")
        visualizations["time_to_hire_distribution"] = fig1.to_json()

        if 'source' in df.columns:
            source_counts = df['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Count']
            fig2 = px.bar(source_counts.head(10), x='Source', y='Count',
                          title="Top 10 Hires by Source")
            visualizations["hires_by_source"] = fig2.to_json()
            
        if 'hire_date' in df.columns:
            hires_over_time = df.set_index('hire_date').resample('M').size().reset_index(name='hire_count')
            fig3 = px.line(hires_over_time, x='hire_date', y='hire_count', title='Hires Over Time (Monthly)')
            visualizations["hires_over_time"] = fig3.to_json()
            insights.append("Generated 'Hires Over Time' plot.")

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

def productivity_analysis(df):
    analysis_name = "Productivity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'projects_completed', 'productivity_score', 'department']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['employee_id', 'productivity_score'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['productivity_score'] = pd.to_numeric(df['productivity_score'], errors='coerce')
        df.dropna(subset=['productivity_score'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_productivity = df['productivity_score'].mean()
        max_score = df['productivity_score'].max()
        # Assuming 1-5 scale where 4-5 is high
        top_performers_threshold = 4
        top_performers = (df['productivity_score'] >= top_performers_threshold).sum()
        top_performers_percent = (top_performers / total_employees) * 100 if total_employees > 0 else 0

        metrics = {
            "total_employees": total_employees,
            "avg_productivity_score": avg_productivity,
            "assumed_max_score": max_score,
            "top_performers_count": top_performers,
            "top_performers_percent": top_performers_percent,
            "top_performers_threshold": top_performers_threshold
        }
        
        insights.append(f"Analyzed {total_employees} employees.")
        insights.append(f"Average productivity score: {avg_productivity:.2f} (out of {max_score})")
        insights.append(f"{top_performers} employees ({top_performers_percent:.1f}%) are top performers (Score >= {top_performers_threshold})")

        # Visualizations
        fig1 = px.histogram(df, x='productivity_score',
                            title="Productivity Score Distribution")
        visualizations["productivity_distribution"] = fig1.to_json()

        if 'projects_completed' in df.columns:
            df['projects_completed'] = pd.to_numeric(df['projects_completed'], errors='coerce')
            if not df['projects_completed'].isnull().all():
                fig2 = px.scatter(df, x='projects_completed', y='productivity_score',
                                  title="Projects Completed vs Productivity Score",
                                  trendline='ols', trendline_scope='overall')
                visualizations["projects_vs_productivity"] = fig2.to_json()
                insights.append("Generated 'Projects Completed vs Productivity Score' plot.")
            else:
                insights.append("Skipped 'Projects vs Productivity' plot: 'projects_completed' has no valid data.")
        else:
            insights.append("Skipped 'Projects vs Productivity' plot: 'projects_completed' column not found.")
            
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

def retention_analysis(df):
    analysis_name = "Retention Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'tenure', 'retention_risk', 'department', 'attrition']
        matched = fuzzy_match_column(df, expected)
        
        # Need 'tenure'. 'retention_risk' or 'attrition' is also needed for meaningful analysis.
        if matched['tenure'] is None:
             return create_fallback_response(analysis_name, ['tenure'], matched, df)
        if matched['retention_risk'] is None and matched['attrition'] is None:
            return create_fallback_response(analysis_name, ['retention_risk', 'attrition'], matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
        df.dropna(subset=['tenure'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_tenure = df['tenure'].mean()
        median_tenure = df['tenure'].median()
        
        metrics = {
            "total_employees": total_employees,
            "avg_tenure_yrs": avg_tenure,
            "median_tenure_yrs": median_tenure
        }
        
        insights.append(f"Analyzed {total_employees} employees.")
        insights.append(f"Average tenure: {avg_tenure:.1f} years (Median: {median_tenure:.1f} years)")
        
        # Visualizations
        fig1 = px.histogram(df, x='tenure',
                            title="Employee Tenure Distribution (Years)")
        visualizations["tenure_distribution"] = fig1.to_json()

        # Prioritize 'retention_risk' if available
        if 'retention_risk' in df.columns:
            df['retention_risk'] = pd.to_numeric(df['retention_risk'], errors='coerce')
            df.dropna(subset=['retention_risk'], inplace=True)
            
            high_risk_threshold = 0.7 # Assuming risk score 0-1
            high_risk_employees = (df['retention_risk'] >= high_risk_threshold).sum()
            high_risk_percent = (high_risk_employees / total_employees) * 100 if total_employees > 0 else 0
            
            metrics['high_risk_employees_count'] = high_risk_employees
            metrics['high_risk_employees_percent'] = high_risk_percent
            metrics['high_risk_threshold'] = high_risk_threshold
            
            insights.append(f"{high_risk_employees} employees ({high_risk_percent:.1f}%) are high retention risk (Score >= {high_risk_threshold})")
            
            if 'department' in df.columns:
                fig2 = px.box(df, x='department', y='retention_risk',
                              title="Retention Risk by Department")
                visualizations["retention_risk_by_department"] = fig2.to_json()
                insights.append("Generated 'Retention Risk by Department' plot.")
            else:
                insights.append("Skipped 'Retention Risk by Department' plot: 'department' column not found.")
        
        # If no risk score, use 'attrition' for retention insights
        elif 'attrition' in df.columns:
            if df['attrition'].dtype == 'object':
                df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0}).fillna(0)
            else:
                df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce').fillna(0)
            
            retention_rate = (1 - df['attrition_flag'].mean()) * 100
            metrics['retention_rate_percent'] = retention_rate
            insights.append(f"Overall retention rate (based on 'attrition' column): {retention_rate:.1f}%")
            
            if 'department' in df.columns:
                retention_by_dept = (1 - df.groupby('department')['attrition_flag'].mean()) * 100
                retention_by_dept = retention_by_dept.reset_index(name='retention_rate')
                fig3 = px.bar(retention_by_dept, x='department', y='retention_rate', title='Retention Rate by Department')
                visualizations["retention_rate_by_department"] = fig3.to_json()
                insights.append("Generated 'Retention Rate by Department' plot.")

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

def attendance_analysis(df):
    analysis_name = "Attendance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['employee_id', 'absent_days', 'late_arrivals', 'department']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['employee_id', 'absent_days'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v:k for k,v in matched.items() if v})
        
        df['absent_days'] = pd.to_numeric(df['absent_days'], errors='coerce')
        df.dropna(subset=['absent_days'], inplace=True)

        # Metrics
        total_employees = len(df)
        avg_absent = df['absent_days'].mean()
        total_absent_days = df['absent_days'].sum()
        problem_threshold = 5 # Employees with > 5 absent days
        problem_employees = (df['absent_days'] > problem_threshold).sum()
        problem_employees_percent = (problem_employees / total_employees) * 100 if total_employees > 0 else 0

        metrics = {
            "total_employees": total_employees,
            "avg_absent_days": avg_absent,
            "total_absent_days": total_absent_days,
            "problem_employees_count": problem_employees,
            "problem_employees_percent": problem_employees_percent,
            "problem_absenteeism_threshold_days": problem_threshold
        }
        
        insights.append(f"Analyzed {total_employees} employees.")
        insights.append(f"Average absent days: {avg_absent:.1f}")
        insights.append(f"{problem_employees} employees ({problem_employees_percent:.1f}%) had high absenteeism (> {problem_threshold} days)")

        # Visualizations
        fig1 = px.histogram(df, x='absent_days',
                            title="Absent Days Distribution")
        visualizations["absent_days_distribution"] = fig1.to_json()

        if 'late_arrivals' in df.columns:
            df['late_arrivals'] = pd.to_numeric(df['late_arrivals'], errors='coerce')
            if not df['late_arrivals'].isnull().all():
                metrics['avg_late_arrivals'] = df['late_arrivals'].mean()
                insights.append(f"Average late arrivals: {metrics['avg_late_arrivals']:.1f}")
                
                fig2 = px.scatter(df, x='absent_days', y='late_arrivals',
                                  title="Absenteeism vs Late Arrivals",
                                  trendline='ols', trendline_scope='overall')
                visualizations["absenteeism_vs_late_arrivals"] = fig2.to_json()
            else:
                insights.append("Skipped 'Absenteeism vs Late Arrivals' plot: 'late_arrivals' has no valid data.")
        else:
            insights.append("Skipped 'Absenteeism vs Late Arrivals' plot: 'late_arrivals' column not found.")
            
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

def employee_demographic_and_tenure_analysis(df):
    analysis_name = "Employee Demographic and Tenure Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['birthdate', 'hiredate', 'jobrole', 'gender', 'ethnicity']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['birthdate', 'hiredate', 'jobrole'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)

        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
        df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
        df.dropna(subset=['birthdate', 'hiredate'], inplace=True)

        current_date = datetime.now()
        df['age'] = (current_date - df['birthdate']).dt.days / 365.25
        df['tenure_years'] = (current_date - df['hiredate']).dt.days / 365.25
        df.dropna(subset=['age', 'tenure_years'], inplace=True)

        # Metrics
        avg_age = df['age'].mean()
        avg_tenure = df['tenure_years'].mean()
        most_common_role = df['jobrole'].mode()[0] if not df['jobrole'].empty else "N/A"

        metrics = {
            "avg_employee_age": avg_age,
            "avg_tenure_years": avg_tenure,
            "most_common_job_role": most_common_role,
            "total_employees": len(df)
        }
        
        insights.append(f"Average employee age: {avg_age:.1f} years.")
        insights.append(f"Average employee tenure: {avg_tenure:.1f} years.")
        insights.append(f"Most common job role: {most_common_role}")

        # Visualizations
        fig1 = px.histogram(df, x='age', nbins=30, title="Distribution of Employee Ages")
        visualizations["age_distribution"] = fig1.to_json()

        fig2 = px.histogram(df, x='tenure_years', nbins=30, title="Distribution of Employee Tenure (Years)")
        visualizations["tenure_distribution"] = fig2.to_json()

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

def employee_profile_and_departmental_analysis(df):
    analysis_name = "Employee Profile and Departmental Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}
    
    try:
        expected = ['workdept', 'job', 'edlevel', 'gender', 'ethnicity']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['workdept', 'job'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})

        # Metrics
        num_departments = df['workdept'].nunique()
        num_jobs = df['job'].nunique()
        
        metrics = {
            "num_departments": num_departments,
            "num_unique_jobs": num_jobs,
            "total_employees": len(df)
        }
        
        insights.append(f"Company has {len(df)} employees across {num_departments} departments and {num_jobs} unique jobs.")

        if 'edlevel' in df.columns:
            df['edlevel'] = pd.to_numeric(df['edlevel'], errors='coerce')
            if not df['edlevel'].isnull().all():
                metrics['avg_education_level'] = df['edlevel'].mean()
                insights.append(f"Average education level: {metrics['avg_education_level']:.2f}")
            else:
                 insights.append("Could not calculate average education level (no valid data).")
        else:
            insights.append("Skipping education level analysis: 'edlevel' column not found.")


        # Visualizations
        dept_counts = df['workdept'].value_counts().reset_index()
        dept_counts.columns = ['Department', 'Count']
        fig1 = px.pie(dept_counts, names='Department', values='Count', title="Employee Distribution by Department", hole=0.4)
        visualizations["department_distribution"] = fig1.to_json()

        if 'edlevel' in df.columns and not df['edlevel'].isnull().all():
            fig2 = px.box(df, x='workdept', y='edlevel', title="Education Level Distribution by Department")
            visualizations["education_by_department"] = fig2.to_json()
            insights.append("Generated 'Education Level by Department' plot.")
        else:
            insights.append("Skipped 'Education Level by Department' plot: 'edlevel' column not found or empty.")

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

def employee_compensation_and_tenure_analysis(df):
    analysis_name = "Employee Compensation and Tenure Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['position', 'dateofhire', 'yearsatcompany', 'monthlyincome', 'overtime']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['yearsatcompany', 'monthlyincome'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in ['yearsatcompany', 'monthlyincome']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['yearsatcompany', 'monthlyincome'], inplace=True)

        # Metrics
        avg_income = df['monthlyincome'].mean()
        avg_tenure = df['yearsatcompany'].mean()
        income_tenure_corr = df['yearsatcompany'].corr(df['monthlyincome'])

        metrics = {
            "avg_monthly_income": avg_income,
            "avg_tenure_years": avg_tenure,
            "income_tenure_correlation": income_tenure_corr,
            "total_employees": len(df)
        }
        
        insights.append(f"Average monthly income: ${avg_income:,.0f}")
        insights.append(f"Average tenure: {avg_tenure:.1f} years.")
        insights.append(f"Correlation between income and tenure: {income_tenure_corr:.2f}")

        # Visualizations
        fig1 = px.scatter(df, x='yearsatcompany', y='monthlyincome', 
                          color='overtime' if 'overtime' in df.columns else None,
                          title="Monthly Income vs. Years at Company",
                          labels={'yearsatcompany': 'Years at Company', 'monthlyincome': 'Monthly Income'},
                          trendline='ols', trendline_scope='overall')
        visualizations["income_vs_tenure_scatter"] = fig1.to_json()

        if 'position' in df.columns:
            fig2 = px.box(df, x='position', y='monthlyincome', title="Monthly Income by Position")
            visualizations["income_by_position"] = fig2.to_json()
            insights.append("Generated 'Monthly Income by Position' plot.")
        else:
            insights.append("Skipped 'Monthly Income by Position' plot: 'position' column not found.")
            
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

def employee_attrition_prediction_and_factor_analysis(df):
    analysis_name = "Employee Attrition Factor Analysis" # Renamed as per user comments
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['maritalstatus', 'jobrole', 'monthlyincome', 'jobsatisfaction', 'environmentsatisfaction', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['attrition', 'jobrole', 'monthlyincome'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})

        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['monthlyincome', 'jobsatisfaction', 'environmentsatisfaction']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag'], inplace=True)

        # Metrics
        attrition_rate = df['attrition_flag'].mean() * 100
        metrics['overall_attrition_rate_percent'] = attrition_rate
        insights.append(f"Overall attrition rate: {attrition_rate:.2f}%")

        if 'jobsatisfaction' in df.columns and not df['jobsatisfaction'].isnull().all():
            avg_satisfaction_leavers = df[df['attrition_flag'] == 1]['jobsatisfaction'].mean()
            avg_satisfaction_stayers = df[df['attrition_flag'] == 0]['jobsatisfaction'].mean()
            metrics['avg_job_satisfaction_leavers'] = avg_satisfaction_leavers
            metrics['avg_job_satisfaction_stayers'] = avg_satisfaction_stayers
            insights.append(f"Avg. Job Satisfaction (Leavers): {avg_satisfaction_leavers:.2f} vs. (Stayers): {avg_satisfaction_stayers:.2f}")

        # Visualizations
        if 'jobrole' in df.columns:
            attrition_by_role = df.groupby('jobrole')['attrition_flag'].mean().mul(100).sort_values().reset_index()
            fig1 = px.bar(attrition_by_role, x='attrition_flag', y='jobrole', orientation='h', 
                          title="Attrition Rate by Job Role", labels={'attrition_flag': 'Attrition Rate (%)'})
            visualizations["attrition_by_job_role"] = fig1.to_json()
        else:
            insights.append("Skipped 'Attrition Rate by Job Role' plot: 'jobrole' column not found.")

        if 'monthlyincome' in df.columns and 'maritalstatus' in df.columns:
            fig2 = px.box(df, x='attrition', y='monthlyincome', color='maritalstatus', title="Monthly Income by Attrition and Marital Status")
            visualizations["income_by_attrition_marital_status"] = fig2.to_json()
        else:
            insights.append("Skipped 'Monthly Income by Attrition and Marital Status' plot: Missing 'monthlyincome' or 'maritalstatus'.")
            
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

def employee_distribution_and_service_length_analysis(df):
    analysis_name = "Employee Distribution and Service Length Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobtitle', 'storelocation', 'businessunit', 'division', 'lengthofservice']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['division', 'lengthofservice'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['lengthofservice'] = pd.to_numeric(df['lengthofservice'], errors='coerce')
        df.dropna(subset=['lengthofservice', 'division'], inplace=True)

        # Metrics
        avg_service_length = df['lengthofservice'].mean()
        top_division = df['division'].mode()[0] if not df['division'].empty else "N/A"
        top_division_count = df[df['division'] == top_division].shape[0] if top_division != "N/A" else 0
        
        metrics = {
            "avg_length_of_service_yrs": avg_service_length,
            "largest_division": top_division,
            "largest_division_employee_count": top_division_count
        }

        if 'storelocation' in df.columns:
            top_location = df['storelocation'].mode()[0] if not df['storelocation'].empty else "N/A"
            metrics['top_store_location'] = top_location
            insights.append(f"Top Store Location: {top_location}")

        insights.append(f"Avg. Length of Service: {avg_service_length:.1f} years")
        insights.append(f"Largest Division: {top_division} (with {top_division_count} employees)")
        
        # Visualizations
        if 'businessunit' in df.columns and 'storelocation' in df.columns:
            fig1 = px.treemap(df, path=[px.Constant("All Employees"), 'division', 'businessunit', 'storelocation'],
                              values='lengthofservice', color='lengthofservice',
                              title="Hierarchical View of Employee Distribution by Service Length")
            visualizations["distribution_treemap"] = fig1.to_json()
            insights.append("Generated hierarchical treemap.")
        else:
            insights.append("Skipped treemap: Missing 'businessunit' or 'storelocation'.")

        fig2 = px.violin(df, x='division', y='lengthofservice', box=True, title="Length of Service Distribution by Division")
        visualizations["service_length_by_division"] = fig2.to_json()

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

def employee_performance_and_compensation_analysis(df):
    analysis_name = "Employee Performance and Compensation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'monthlyincome', 'stockoptionlevel', 'performancerating', 'yearsatcompany', 'yearsincurrentrole']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['monthlyincome', 'performancerating'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in expected:
            if col != 'jobrole' and col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['monthlyincome', 'performancerating'], inplace=True)

        # Metrics
        avg_rating = df['performancerating'].mean()
        avg_income = df['monthlyincome'].mean()
        income_performance_corr = df['monthlyincome'].corr(df['performancerating'])
        
        metrics = {
            "avg_performance_rating": avg_rating,
            "avg_monthly_income": avg_income,
            "income_performance_correlation": income_performance_corr
        }
        
        insights.append(f"Average Performance Rating: {avg_rating:.2f}")
        insights.append(f"Average Monthly Income: ${avg_income:,.0f}")
        insights.append(f"Correlation between income and performance: {income_performance_corr:.2f}")

        # Visualizations
        fig1 = px.box(df, x='performancerating', y='monthlyincome', title="Monthly Income by Performance Rating")
        visualizations["income_by_performance_rating"] = fig1.to_json()

        if 'yearsatcompany' in df.columns and 'stockoptionlevel' in df.columns:
            fig2 = px.scatter(df, x='yearsatcompany', y='monthlyincome', color='performancerating',
                              size='stockoptionlevel', title="Income vs. Tenure (Colored by Performance, Sized by Stock Options)")
            visualizations["income_vs_tenure_performance_stock"] = fig2.to_json()
            insights.append("Generated complex scatter plot for income vs. tenure.")
        else:
            insights.append("Skipped 'Income vs. Tenure' plot: Missing 'yearsatcompany' or 'stockoptionlevel'.")
            
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

def employee_salary_and_attrition_analysis(df):
    analysis_name = "Employee Salary and Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobtitle', 'annualsalary', 'stockoptionlevel', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['annualsalary', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        df['annualsalary'] = pd.to_numeric(df['annualsalary'], errors='coerce')
        df.dropna(subset=['attrition_flag', 'annualsalary'], inplace=True)

        # Metrics
        attrition_rate = df['attrition_flag'].mean() * 100
        avg_salary_leavers = df[df['attrition_flag']==1]['annualsalary'].mean()
        avg_salary_stayers = df[df['attrition_flag']==0]['annualsalary'].mean()
        salary_diff = avg_salary_stayers - avg_salary_leavers

        metrics = {
            "attrition_rate_percent": attrition_rate,
            "avg_salary_leavers": avg_salary_leavers,
            "avg_salary_stayers": avg_salary_stayers,
            "avg_salary_difference_stayers_vs_leavers": salary_diff
        }
        
        insights.append(f"Attrition Rate: {attrition_rate:.2f}%")
        insights.append(f"Avg. Salary (Leavers): ${avg_salary_leavers:,.0f}")
        insights.append(f"Avg. Salary (Stayers): ${avg_salary_stayers:,.0f} (Stayers earn ${salary_diff:,.0f} more on average)")

        # Visualizations
        fig1 = px.box(df, x='attrition', y='annualsalary', title="Annual Salary by Attrition Status")
        visualizations["salary_by_attrition_box"] = fig1.to_json()

        fig2 = px.histogram(df, x='annualsalary', color='attrition', barmode='overlay', title="Salary Distribution by Attrition Status")
        visualizations["salary_distribution_by_attrition"] = fig2.to_json()
        
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

def employee_salary_hike_and_promotion_factor_analysis(df):
    analysis_name = "Employee Salary Hike and Promotion Factor Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['education', 'jobrole', 'monthlyincome', 'percentsalaryhike', 'joblevel']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['monthlyincome', 'percentsalaryhike', 'joblevel'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in expected:
            if col not in ['jobrole', 'education'] and col in df.columns: # 'education' might be categorical
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['monthlyincome', 'percentsalaryhike', 'joblevel'], inplace=True)

        # Metrics
        avg_hike = df['percentsalaryhike'].mean()
        avg_job_level = df['joblevel'].mean()
        hike_income_corr = df['monthlyincome'].corr(df['percentsalaryhike'])

        metrics = {
            "avg_salary_hike_percent": avg_hike,
            "avg_job_level": avg_job_level,
            "hike_income_correlation": hike_income_corr
        }
        
        insights.append(f"Average Salary Hike: {avg_hike:.2f}%")
        insights.append(f"Average Job Level: {avg_job_level:.2f}")
        insights.append(f"Correlation between hike % and income: {hike_income_corr:.2f}")

        # Visualizations
        fig1 = px.box(df, x='joblevel', y='percentsalaryhike', title="Salary Hike Percentage by Job Level")
        visualizations["hike_by_job_level"] = fig1.to_json()

        fig2 = px.scatter(df, x='monthlyincome', y='percentsalaryhike', color='joblevel',
                          title="Salary Hike vs. Monthly Income")
        visualizations["hike_vs_income_scatter"] = fig2.to_json()
        
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

def work_life_balance_and_job_satisfaction_impact_on_attrition(df):
    analysis_name = "Work-Life Balance and Job Satisfaction Impact on Attrition"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobsatisfaction', 'worklifebalance', 'totalworkingyears', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['jobsatisfaction', 'worklifebalance', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['jobsatisfaction', 'worklifebalance']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['attrition_flag', 'jobsatisfaction', 'worklifebalance'], inplace=True)

        # Metrics
        attrition_rate = df['attrition_flag'].mean() * 100
        avg_wlb_leavers = df[df['attrition_flag'] == 1]['worklifebalance'].mean()
        avg_wlb_stayers = df[df['attrition_flag'] == 0]['worklifebalance'].mean()
        avg_js_leavers = df[df['attrition_flag'] == 1]['jobsatisfaction'].mean()
        avg_js_stayers = df[df['attrition_flag'] == 0]['jobsatisfaction'].mean()

        metrics = {
            "attrition_rate_percent": attrition_rate,
            "avg_wlb_leavers": avg_wlb_leavers,
            "avg_wlb_stayers": avg_wlb_stayers,
            "avg_job_satisfaction_leavers": avg_js_leavers,
            "avg_job_satisfaction_stayers": avg_js_stayers
        }
        
        insights.append(f"Attrition Rate: {attrition_rate:.2f}%")
        insights.append(f"Avg. Work-Life Balance (Leavers): {avg_wlb_leavers:.2f} vs. (Stayers): {avg_wlb_stayers:.2f}")
        insights.append(f"Avg. Job Satisfaction (Leavers): {avg_js_leavers:.2f} vs. (Stayers): {avg_js_stayers:.2f}")

        # Visualizations
        fig1 = px.box(df, x='attrition', y='worklifebalance', title="Work-Life Balance Score by Attrition Status")
        visualizations["wlb_by_attrition"] = fig1.to_json()

        fig2 = px.box(df, x='attrition', y='jobsatisfaction', title="Job Satisfaction Score by Attrition Status")
        visualizations["js_by_attrition"] = fig2.to_json()

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

def commute_distance_and_work_history_impact_on_attrition(df):
    analysis_name = "Commute Distance and Work History Impact on Attrition"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['distancefromhome', 'numcompaniesworked', 'totalworkingyears', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['distancefromhome', 'numcompaniesworked', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['distancefromhome', 'numcompaniesworked', 'totalworkingyears']:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'distancefromhome', 'numcompaniesworked'], inplace=True)

        # Metrics
        avg_dist_leavers = df[df['attrition_flag'] == 1]['distancefromhome'].mean()
        avg_dist_stayers = df[df['attrition_flag'] == 0]['distancefromhome'].mean()
        avg_num_comp_leavers = df[df['attrition_flag'] == 1]['numcompaniesworked'].mean()
        avg_num_comp_stayers = df[df['attrition_flag'] == 0]['numcompaniesworked'].mean()

        metrics = {
            "avg_distance_leavers": avg_dist_leavers,
            "avg_distance_stayers": avg_dist_stayers,
            "avg_num_companies_leavers": avg_num_comp_leavers,
            "avg_num_companies_stayers": avg_num_comp_stayers
        }
        
        insights.append(f"Avg. Commute Distance (Leavers): {avg_dist_leavers:.2f} vs. (Stayers): {avg_dist_stayers:.2f}")
        insights.append(f"Avg. Companies Worked (Leavers): {avg_num_comp_leavers:.2f} vs. (Stayers): {avg_num_comp_stayers:.2f}")

        # Visualizations
        fig1 = px.box(df, x='attrition', y='distancefromhome', title="Distance From Home by Attrition Status")
        visualizations["distance_by_attrition"] = fig1.to_json()

        if 'totalworkingyears' in df.columns and not df['totalworkingyears'].isnull().all():
            fig2 = px.violin(df, x='numcompaniesworked', y='totalworkingyears', color='attrition',
                            title="Work History by Attrition Status")
            visualizations["work_history_by_attrition"] = fig2.to_json()
            insights.append("Generated 'Work History by Attrition' plot.")
        else:
            insights.append("Skipped 'Work History by Attrition' plot: 'totalworkingyears' column not found or empty.")
            
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

def employee_performance_and_promotion_cycle_analysis(df):
    analysis_name = "Employee Performance and Promotion Cycle Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobinvolvement', 'performancerating', 'yearssincelastpromotion', 'yearsatcompany']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['performancerating', 'yearssincelastpromotion', 'yearsatcompany'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in expected:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['performancerating', 'yearssincelastpromotion', 'yearsatcompany'], inplace=True)

        # Metrics
        avg_years_since_promo = df['yearssincelastpromotion'].mean()
        avg_perf_rating = df['performancerating'].mean()
        promo_perf_corr = df['yearssincelastpromotion'].corr(df['performancerating'])

        metrics = {
            "avg_years_since_last_promotion": avg_years_since_promo,
            "avg_performance_rating": avg_perf_rating,
            "promotion_performance_correlation": promo_perf_corr
        }
        
        insights.append(f"Avg. Years Since Last Promotion: {avg_years_since_promo:.2f}")
        insights.append(f"Avg. Performance Rating: {avg_perf_rating:.2f}")
        insights.append(f"Correlation between performance and years since promotion: {promo_perf_corr:.2f}")

        # Visualizations
        fig1 = px.scatter(df, x='yearssincelastpromotion', y='performancerating',
                          title="Performance Rating vs. Years Since Last Promotion",
                          trendline='ols', trendline_scope='overall')
        visualizations["performance_vs_years_since_promotion"] = fig1.to_json()

        fig2 = px.density_heatmap(df, x="yearsatcompany", y="yearssincelastpromotion",
                                  title="Heatmap of Years at Company vs. Years Since Promotion")
        visualizations["tenure_vs_promotion_heatmap"] = fig2.to_json()

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

def employee_demographic_and_compensation_profile_analysis(df):
    analysis_name = "Employee Demographic and Compensation Profile Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['dob', 'maritalstatus', 'education', 'joblevel', 'jobrole', 'monthlyincome', 'gender']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['dob', 'maritalstatus', 'jobrole', 'monthlyincome'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (datetime.now() - df['dob']).dt.days / 365.25
        
        for col in ['education', 'joblevel', 'monthlyincome']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['age', 'maritalstatus', 'jobrole', 'monthlyincome'], inplace=True)

        # Metrics
        avg_age = df['age'].mean()
        avg_income = df['monthlyincome'].mean()
        age_income_corr = df['age'].corr(df['monthlyincome'])
        
        metrics = {
            "avg_age": avg_age,
            "avg_monthly_income": avg_income,
            "age_income_correlation": age_income_corr
        }
        
        insights.append(f"Average employee age: {avg_age:.1f}")
        insights.append(f"Average monthly income: ${avg_income:,.0f}")
        insights.append(f"Correlation between age and income: {age_income_corr:.2f}")

        # Visualizations
        if 'joblevel' in df.columns:
            fig1 = px.box(df, x='maritalstatus', y='monthlyincome', 
                          color='joblevel',
                          title="Monthly Income by Marital Status and Job Level")
            visualizations["income_by_marital_status_job_level"] = fig1.to_json()
        else:
            fig1 = px.box(df, x='maritalstatus', y='monthlyincome', 
                          title="Monthly Income by Marital Status")
            visualizations["income_by_marital_status"] = fig1.to_json()
            insights.append("Skipped job level coloring on 'Income by Marital Status' plot: 'joblevel' not found.")


        fig2 = px.scatter(df, x='age', y='monthlyincome', color='jobrole',
                          title="Age vs. Monthly Income by Job Role")
        visualizations["age_vs_income_by_job_role"] = fig2.to_json()
        
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

def factors_influencing_employee_attrition_analysis(df):
    analysis_name = "Factors Influencing Employee Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'attrition', 'monthlyincome', 'distancefromhome', 'numcompaniesworked', 'education', 'gender']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['attrition', 'monthlyincome', 'distancefromhome'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['monthlyincome', 'distancefromhome', 'education']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'monthlyincome', 'distancefromhome'], inplace=True)
        
        # Metrics
        attrition_rate = df['attrition_flag'].mean() * 100
        metrics['overall_attrition_rate_percent'] = attrition_rate
        insights.append(f"Overall Attrition Rate: {attrition_rate:.2f}%")

        # Visualizations
        fig1 = px.density_heatmap(df, x='distancefromhome', y='monthlyincome', z='attrition_flag', histfunc='avg',
                                  title="Attrition Rate Heatmap by Distance from Home and Monthly Income",
                                  labels={'z': 'Attrition Rate'})
        visualizations["attrition_heatmap_distance_income"] = fig1.to_json()

        if 'gender' in df.columns and 'education' in df.columns and not df['education'].isnull().all():
            attrition_by_demographics = df.groupby(['gender', 'education'])['attrition_flag'].mean().mul(100).reset_index()
            fig2 = px.bar(attrition_by_demographics, x='gender', y='attrition_flag', color='education',
                          barmode='group', title="Attrition Rate by Gender and Education Level",
                          labels={'attrition_flag': 'Attrition Rate (%)'})
            visualizations["attrition_by_gender_education"] = fig2.to_json()
            insights.append("Generated 'Attrition by Gender and Education' plot.")
        else:
            insights.append("Skipped 'Attrition by Gender and Education' plot: Missing 'gender' or 'education' columns.")
            
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

def employee_demographics_and_attrition_correlation_analysis(df):
    analysis_name = "Employee Demographics and Attrition Correlation Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['dob', 'hiredate', 'jobrole', 'monthlyincome', 'yearsatcompany', 'attrition', 'gender', 'ethnicity']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['dob', 'jobrole', 'monthlyincome', 'yearsatcompany', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})

        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')

        df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
        df['age'] = (datetime.now() - df['dob']).dt.days / 365.25

        for col in ['monthlyincome', 'yearsatcompany']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['attrition_flag', 'age', 'jobrole', 'monthlyincome', 'yearsatcompany'], inplace=True)

        # Metrics
        avg_age_leavers = df[df['attrition_flag'] == 1]['age'].mean()
        avg_age_stayers = df[df['attrition_flag'] == 0]['age'].mean()
        avg_income_leavers = df[df['attrition_flag'] == 1]['monthlyincome'].mean()
        avg_income_stayers = df[df['attrition_flag'] == 0]['monthlyincome'].mean()
        
        metrics = {
            "avg_age_leavers": avg_age_leavers,
            "avg_age_stayers": avg_age_stayers,
            "avg_income_leavers": avg_income_leavers,
            "avg_income_stayers": avg_income_stayers
        }
        
        insights.append(f"Avg. Age (Leavers): {avg_age_leavers:.1f} vs. (Stayers): {avg_age_stayers:.1f}")
        insights.append(f"Avg. Income (Leavers): ${avg_income_leavers:,.0f} vs. (Stayers): ${avg_income_stayers:,.0f}")

        # Visualizations
        fig1 = px.violin(df, x='jobrole', y='age', color='attrition', box=True,
                         title="Age Distribution by Job Role and Attrition Status")
        visualizations["age_by_job_role_attrition"] = fig1.to_json()

        fig2 = px.scatter(df, x='yearsatcompany', y='monthlyincome', color='attrition',
                          title="Income vs. Tenure by Attrition Status",
                          trendline='ols', trendline_scope='overall')
        visualizations["income_vs_tenure_by_attrition"] = fig2.to_json()
        
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

def employee_performance_and_tenure_analysis(df):
    analysis_name = "Employee Performance and Tenure Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['position', 'startdate', 'manager', 'yearswithcompany', 'performancerating']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['yearswithcompany', 'performancerating', 'position'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in ['yearswithcompany', 'performancerating']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['yearswithcompany', 'performancerating', 'position'], inplace=True)

        # Metrics
        avg_rating = df['performancerating'].mean()
        avg_tenure = df['yearswithcompany'].mean()
        corr = df['yearswithcompany'].corr(df['performancerating'])
        
        metrics = {
            "avg_performance_rating": avg_rating,
            "avg_tenure_years": avg_tenure,
            "tenure_performance_correlation": corr
        }
        
        insights.append(f"Average Performance Rating: {avg_rating:.2f}")
        insights.append(f"Average Tenure: {avg_tenure:.2f} years")
        insights.append(f"Tenure/Performance Correlation: {corr:.2f}")

        # Visualizations
        fig1 = px.scatter(df, x='yearswithcompany', y='performancerating', trendline='ols',
                          title="Performance Rating vs. Years with Company")
        visualizations["performance_vs_tenure_scatter"] = fig1.to_json()

        perf_by_position = df.groupby('position')['performancerating'].mean().reset_index().sort_values(by='performancerating', ascending=False)
        fig2 = px.bar(perf_by_position.head(15), x='position', y='performancerating', title="Top 15 Avg Performance Rating by Position")
        visualizations["avg_performance_by_position"] = fig2.to_json()

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

def compensation_promotion_and_career_progression_analysis(df):
    analysis_name = "Compensation, Promotion, and Career Progression Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['educationfield', 'joblevel', 'jobrole', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['joblevel', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in ['joblevel', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['joblevel', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears'], inplace=True)

        # Metrics
        income_joblevel_corr = df['joblevel'].corr(df['monthlyincome'])
        income_twy_corr = df['totalworkingyears'].corr(df['monthlyincome'])
        avg_hike = df['percentsalaryhike'].mean()
        
        metrics = {
            "income_joblevel_correlation": income_joblevel_corr,
            "income_total_working_years_correlation": income_twy_corr,
            "avg_salary_hike_percent": avg_hike
        }
        
        insights.append(f"Correlation between Job Level and Monthly Income: {income_joblevel_corr:.2f}")
        insights.append(f"Correlation between Total Working Years and Monthly Income: {income_twy_corr:.2f}")
        insights.append(f"Average salary hike: {avg_hike:.1f}%")

        # Visualizations
        fig1 = px.scatter(df, x='totalworkingyears', y='monthlyincome', color='joblevel',
                          title="Monthly Income vs. Total Working Years by Job Level")
        visualizations["income_vs_working_years_by_job_level"] = fig1.to_json()

        hike_by_level = df.groupby('joblevel')['percentsalaryhike'].mean().reset_index()
        fig2 = px.bar(hike_by_level, x='joblevel', y='percentsalaryhike', title="Average Salary Hike % by Job Level")
        visualizations["avg_hike_by_job_level"] = fig2.to_json()
        
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

def employee_profile_and_training_engagement_analysis(df):
    analysis_name = "Employee Profile and Training Engagement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['maritalstatus', 'jobrole', 'monthlyincome', 'distancefromhome', 'trainingtimeslastyear']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['jobrole', 'monthlyincome', 'trainingtimeslastyear'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        for col in ['monthlyincome', 'distancefromhome', 'trainingtimeslastyear']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['jobrole', 'monthlyincome', 'trainingtimeslastyear'], inplace=True)

        # Metrics
        avg_trainings = df['trainingtimeslastyear'].mean()
        training_income_corr = df['monthlyincome'].corr(df['trainingtimeslastyear'])
        
        metrics = {
            "avg_trainings_last_year": avg_trainings,
            "training_income_correlation": training_income_corr
        }
        
        insights.append(f"Average trainings last year: {avg_trainings:.2f}")
        insights.append(f"Correlation between training times and income: {training_income_corr:.2f}")

        # Visualizations
        training_by_role = df.groupby('jobrole')['trainingtimeslastyear'].mean().reset_index().sort_values(by='trainingtimeslastyear', ascending=False)
        fig1 = px.bar(training_by_role, x='jobrole', y='trainingtimeslastyear', title="Average Trainings by Job Role")
        visualizations["avg_training_by_job_role"] = fig1.to_json()

        if 'maritalstatus' in df.columns:
            fig2 = px.scatter(df, x='monthlyincome', y='trainingtimeslastyear', color='maritalstatus',
                              title="Training Frequency vs. Monthly Income by Marital Status")
            visualizations["training_vs_income_by_marital_status"] = fig2.to_json()
        else:
            insights.append("Skipped 'Training vs. Income by Marital Status' plot: 'maritalstatus' column not found.")
            
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

def attrition_factors_related_to_promotions_and_stock_options(df):
    analysis_name = "Attrition Factors: Promotions and Stock Options"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'yearssincelastpromotion', 'yearsatcompany', 'stockoptionlevel', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['yearssincelastpromotion', 'stockoptionlevel', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['yearssincelastpromotion', 'yearsatcompany', 'stockoptionlevel']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'yearssincelastpromotion', 'stockoptionlevel'], inplace=True)

        # Metrics
        avg_promo_wait_leavers = df[df['attrition_flag'] == 1]['yearssincelastpromotion'].mean()
        avg_promo_wait_stayers = df[df['attrition_flag'] == 0]['yearssincelastpromotion'].mean()
        
        metrics = {
            "avg_years_since_promo_leavers": avg_promo_wait_leavers,
            "avg_years_since_promo_stayers": avg_promo_wait_stayers
        }
        
        insights.append(f"Avg. Years Since Promotion (Leavers): {avg_promo_wait_leavers:.2f} vs. (Stayers): {avg_promo_wait_stayers:.2f}")

        # Visualizations
        fig1 = px.box(df, x='attrition', y='yearssincelastpromotion', title="Years Since Last Promotion by Attrition Status")
        visualizations["promo_wait_by_attrition"] = fig1.to_json()

        attrition_by_stock = df.groupby('stockoptionlevel')['attrition_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(attrition_by_stock, x='stockoptionlevel', y='attrition_flag', 
                      title="Attrition Rate by Stock Option Level",
                      labels={'stockoptionlevel': 'Stock Option Level', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_stock_option"] = fig2.to_json()

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

def comprehensive_employee_satisfaction_and_attrition_analysis(df):
    analysis_name = "Comprehensive Employee Satisfaction and Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'monthlyincome', 'totalworkingyears', 'yearsatcompany', 'worklifebalance', 'jobsatisfaction', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['worklifebalance', 'jobsatisfaction', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['monthlyincome', 'totalworkingyears', 'yearsatcompany', 'worklifebalance', 'jobsatisfaction']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'worklifebalance', 'jobsatisfaction'], inplace=True)

        # Create a composite satisfaction score
        df['satisfaction_score'] = df[['worklifebalance', 'jobsatisfaction']].mean(axis=1)
        
        # Metrics
        avg_score_leavers = df[df['attrition_flag'] == 1]['satisfaction_score'].mean()
        avg_score_stayers = df[df['attrition_flag'] == 0]['satisfaction_score'].mean()
        
        metrics = {
            "avg_composite_satisfaction_leavers": avg_score_leavers,
            "avg_composite_satisfaction_stayers": avg_score_stayers
        }
        
        insights.append(f"Avg. Composite Satisfaction (WLB+JS) (Leavers): {avg_score_leavers:.2f} vs. (Stayers): {avg_score_stayers:.2f}")

        # Visualizations
        fig1 = px.density_heatmap(df, x="jobsatisfaction", y="worklifebalance", z="attrition_flag", histfunc="avg",
                                  title="Heatmap of Attrition Rate by Job Satisfaction and Work-Life Balance",
                                  labels={'z': 'Attrition Rate'})
        visualizations["attrition_heatmap_js_wlb"] = fig1.to_json()

        if 'totalworkingyears' in df.columns:
            fig2 = px.scatter(df, x='totalworkingyears', y='satisfaction_score', color='attrition',
                              title="Composite Satisfaction vs. Total Working Years by Attrition")
            visualizations["satisfaction_vs_working_years"] = fig2.to_json()
            insights.append("Generated 'Satisfaction vs. Total Working Years' plot.")
        else:
            insights.append("Skipped 'Satisfaction vs. Total Working Years' plot: 'totalworkingyears' column not found.")
            
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

def employee_compensation_structure_and_attrition_analysis(df):
    analysis_name = "Employee Compensation Structure and Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'hourlyrate', 'monthlyincome', 'overtime', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['jobrole', 'monthlyincome', 'overtime', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['hourlyrate', 'monthlyincome']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'jobrole', 'monthlyincome', 'overtime'], inplace=True)

        # Metrics
        attrition_by_overtime = df.groupby('overtime')['attrition_flag'].mean().mul(100)
        
        metrics = {
            "attrition_rate_by_overtime_percent": attrition_by_overtime.to_dict()
        }
        
        insights.append(f"Attrition Rate (Overtime): {attrition_by_overtime.get('Yes', 0):.2f}%")
        insights.append(f"Attrition Rate (No Overtime): {attrition_by_overtime.get('No', 0):.2f}%")

        # Visualizations
        attrition_by_overtime_df = attrition_by_overtime.reset_index()
        fig1 = px.pie(attrition_by_overtime_df, names='overtime', values='attrition_flag', hole=0.4,
                      title="Attrition Rate by Overtime Status")
        visualizations["attrition_by_overtime"] = fig1.to_json()

        fig2 = px.box(df, x='jobrole', y='monthlyincome', color='attrition',
                      title="Monthly Income by Job Role and Attrition Status")
        visualizations["income_by_job_role_attrition"] = fig2.to_json()
        
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

def employee_performance_and_career_level_attrition_analysis(df):
    analysis_name = "Employee Performance and Career Level Attrition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['joblevel', 'performancerating', 'stockoptionlevel', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['joblevel', 'performancerating', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['joblevel', 'performancerating', 'stockoptionlevel']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['attrition_flag', 'joblevel', 'performancerating'], inplace=True)

        # Metrics
        attrition_by_level_perf = df.groupby(['joblevel', 'performancerating'])['attrition_flag'].mean().mul(100)
        highest_risk_group = attrition_by_level_perf.idxmax()
        highest_risk_rate = attrition_by_level_perf.max()
        
        metrics = {
            "attrition_rate_by_level_performance": attrition_by_level_perf.reset_index().to_dict('records'),
            "highest_risk_group_level_perf": highest_risk_group,
            "highest_risk_group_rate": highest_risk_rate
        }
        
        insights.append(f"Highest attrition risk group: Job Level {highest_risk_group[0]} & Performance {highest_risk_group[1]} ({highest_risk_rate:.2f}% rate)")

        # Visualizations
        attrition_by_level_perf_df = attrition_by_level_perf.reset_index()
        fig1 = px.density_heatmap(attrition_by_level_perf_df, x='joblevel', y='performancerating', z='attrition_flag',
                                  title="Attrition Rate by Job Level and Performance Rating",
                                  labels={'joblevel': 'Job Level', 'performancerating': 'Performance Rating', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_heatmap_level_performance"] = fig1.to_json()

        if 'stockoptionlevel' in df.columns and not df['stockoptionlevel'].isnull().all():
            attrition_by_stock_level = df.groupby(['stockoptionlevel', 'joblevel'])['attrition_flag'].mean().mul(100).reset_index()
            fig2 = px.bar(attrition_by_stock_level, x='joblevel', y='attrition_flag', color='stockoptionlevel',
                          barmode='group', title="Attrition Rate by Job Level and Stock Option Level",
                          labels={'joblevel': 'Job Level', 'attrition_flag': 'Attrition Rate (%)', 'stockoptionlevel': 'Stock Option Level'})
            visualizations["attrition_by_job_level_stock"] = fig2.to_json()
            insights.append("Generated 'Attrition by Job Level and Stock Option' plot.")
        else:
            insights.append("Skipped 'Attrition by Job Level and Stock Option' plot: 'stockoptionlevel' column not found or empty.")
            
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

def employee_salary_structure_analysis_by_department(df):
    analysis_name = "Employee Salary Structure Analysis by Department"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobrole', 'departmentid', 'annualsalary', 'gender']
        matched = fuzzy_match_column(df, expected)
        # Use 'departmentid' as the department column
        expected_renamed = {'departmentid': 'department'}
        matched.update(fuzzy_match_column(df, expected_renamed))
        if matched.get('departmentid') and not matched.get('department'):
            matched['department'] = matched['departmentid']
        
        critical_missing = [col for col in ['jobrole', 'department', 'annualsalary'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['annualsalary'] = pd.to_numeric(df['annualsalary'], errors='coerce')
        df.dropna(subset=['jobrole', 'department', 'annualsalary'], inplace=True)

        # Metrics
        avg_salary = df['annualsalary'].mean()
        dept_salaries = df.groupby('department')['annualsalary'].mean()
        highest_paid_dept = dept_salaries.idxmax()
        highest_paid_dept_avg = dept_salaries.max()
        
        metrics = {
            "avg_annual_salary": avg_salary,
            "avg_salary_by_department": dept_salaries.to_dict(),
            "highest_paying_department": highest_paid_dept,
            "highest_paying_department_avg_salary": highest_paid_dept_avg
        }
        
        insights.append(f"Average Annual Salary: ${avg_salary:,.0f}")
        insights.append(f"Highest Paying Department: {highest_paid_dept} (Avg. Salary: ${highest_paid_dept_avg:,.0f})")

        # Visualizations
        fig1 = px.box(df, x='department', y='annualsalary', color='jobrole',
                      title="Annual Salary Distribution by Department and Job Role")
        visualizations["salary_by_dept_job_role"] = fig1.to_json()

        if 'gender' in df.columns:
            gender_salary_by_dept = df.groupby(['department', 'gender'])['annualsalary'].mean().unstack(fill_value=0)
            fig2 = px.bar(gender_salary_by_dept, x=gender_salary_by_dept.index, y=gender_salary_by_dept.columns,
                          barmode='group', title="Average Annual Salary by Department and Gender")
            visualizations["avg_salary_by_dept_gender"] = fig2.to_json()
            insights.append("Generated 'Avg Salary by Dept and Gender' plot.")
        else:
            insights.append("Skipped 'Avg Salary by Dept and Gender' plot: 'gender' column not found.")
            
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

def management_and_its_impact_on_employee_performance_and_attrition(df):
    analysis_name = "Management Impact on Performance and Attrition"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['manager', 'jobrole', 'performancerating', 'attrition', 'employee_id']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['manager', 'performancerating', 'attrition', 'employee_id'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        df['performancerating'] = pd.to_numeric(df['performancerating'], errors='coerce')
        df.dropna(subset=['manager', 'attrition_flag', 'performancerating'], inplace=True)

        # Analysis
        manager_kpis = df.groupby('manager').agg(
            team_size=('employee_id', 'count'),
            avg_performance=('performancerating', 'mean'),
            attrition_rate=('attrition_flag', 'mean')
        ).reset_index()
        manager_kpis['attrition_rate'] *= 100
        
        manager_with_highest_attrition = manager_kpis.sort_values(by='attrition_rate', ascending=False).iloc[0]
        manager_with_lowest_performance = manager_kpis.sort_values(by='avg_performance', ascending=True).iloc[0]

        metrics = {
            "manager_kpis": manager_kpis.round(2).to_dict('records'),
            "manager_highest_attrition": manager_with_highest_attrition.to_dict(),
            "manager_lowest_performance": manager_with_lowest_performance.to_dict()
        }
        
        insights.append(f"Manager with highest attrition: {manager_with_highest_attrition['manager']} ({manager_with_highest_attrition['attrition_rate']:.1f}%)")
        insights.append(f"Manager with lowest avg. performance: {manager_with_lowest_performance['manager']} ({manager_with_lowest_performance['avg_performance']:.2f})")

        # Visualizations
        fig1 = px.scatter(manager_kpis, x='avg_performance', y='attrition_rate', size='team_size',
                          hover_name='manager', title="Team Attrition Rate vs. Average Performance by Manager",
                          labels={'avg_performance': 'Avg. Team Performance', 'attrition_rate': 'Team Attrition Rate (%)'})
        visualizations["manager_kpi_scatter"] = fig1.to_json()

        if 'jobrole' in df.columns:
            fig2 = px.box(df, x='jobrole', y='performancerating', color='attrition',
                          title="Performance Rating by Job Role and Attrition Status")
            visualizations["performance_by_job_role_attrition"] = fig2.to_json()
            insights.append("Generated 'Performance by Job Role and Attrition' plot.")
        else:
            insights.append("Skipped 'Performance by Job Role and Attrition' plot: 'jobrole' column not found.")
            
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

def job_involvement_and_training_impact_on_employee_retention(df):
    analysis_name = "Job Involvement and Training Impact on Retention"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobinvolvement', 'trainingtimeslastyear', 'yearsatcompany', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['jobinvolvement', 'trainingtimeslastyear', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['retention_flag'] = df['attrition'].str.lower().map({'yes': 0, 'true': 0, '1': 0, 'no': 1, 'false': 1, '0': 1})
        else:
            df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['jobinvolvement', 'trainingtimeslastyear', 'yearsatcompany']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['retention_flag', 'jobinvolvement', 'trainingtimeslastyear'], inplace=True)

        # Metrics
        retention_by_involvement = df.groupby('jobinvolvement')['retention_flag'].mean().mul(100)
        retention_by_training = df.groupby('trainingtimeslastyear')['retention_flag'].mean().mul(100)
        
        metrics = {
            "retention_rate_by_job_involvement": retention_by_involvement.to_dict(),
            "retention_rate_by_training_times": retention_by_training.to_dict()
        }
        
        insights.append(f"Highest retention rate by involvement: {retention_by_involvement.max():.1f}% at level {retention_by_involvement.idxmax()}")
        insights.append(f"Highest retention rate by training: {retention_by_training.max():.1f}% for {retention_by_training.idxmax()} trainings")

        # Visualizations
        fig1 = px.bar(retention_by_involvement.reset_index(), x='jobinvolvement', y='retention_flag', 
                      title="Retention Rate by Job Involvement Level",
                      labels={'jobinvolvement': 'Job Involvement Level', 'retention_flag': 'Retention Rate (%)'})
        visualizations["retention_by_job_involvement"] = fig1.to_json()

        fig2 = px.bar(retention_by_training.reset_index(), x='trainingtimeslastyear', y='retention_flag', 
                      title="Retention Rate by Number of Trainings Last Year",
                      labels={'trainingtimeslastyear': 'Trainings Last Year', 'retention_flag': 'Retention Rate (%)'})
        visualizations["retention_by_training_times"] = fig2.to_json()
        
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

def employee_lifecycle_and_attrition_trend_analysis(df):
    analysis_name = "Employee Lifecycle and Attrition Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['hiredate', 'education', 'monthlyincome', 'yearsatcompany', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['hiredate', 'yearsatcompany', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['monthlyincome', 'yearsatcompany']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['hiredate', 'attrition_flag', 'yearsatcompany'], inplace=True)

        # Analysis
        df['hire_year'] = df['hiredate'].dt.year
        attrition_by_hire_year = df.groupby('hire_year')['attrition_flag'].mean().mul(100).reset_index()
        
        # Metrics
        tenure_leavers = df[df['attrition_flag'] == 1]['yearsatcompany'].mean()
        tenure_stayers = df[df['attrition_flag'] == 0]['yearsatcompany'].mean()
        
        metrics = {
            "attrition_rate_by_hire_year_cohort": attrition_by_hire_year.to_dict('records'),
            "avg_tenure_leavers": tenure_leavers,
            "avg_tenure_stayers": tenure_stayers
        }
        
        insights.append(f"Avg. Tenure (Leavers): {tenure_leavers:.2f} years vs. (Stayers): {tenure_stayers:.2f} years")
        if not attrition_by_hire_year.empty:
            insights.append(f"Highest attrition cohort: {attrition_by_hire_year.loc[attrition_by_hire_year['attrition_flag'].idxmax()]['hire_year']} ({attrition_by_hire_year['attrition_flag'].max():.1f}%)")

        # Visualization
        fig1 = px.line(attrition_by_hire_year, x='hire_year', y='attrition_flag', 
                       title="Attrition Rate by Hire Year Cohort",
                       labels={'hire_year': 'Hire Year', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_hire_cohort"] = fig1.to_json()

        fig2 = px.box(df, x='attrition', y='yearsatcompany', title="Tenure at Company by Attrition Status")
        visualizations["tenure_by_attrition"] = fig2.to_json()
        
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

def performance_and_workload_impact_on_employee_attrition(df):
    analysis_name = "Performance and Workload Impact on Attrition"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['overtime', 'totalworkingyears', 'performancerating', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['overtime', 'performancerating', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['totalworkingyears', 'performancerating']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['overtime', 'attrition_flag', 'performancerating'], inplace=True)

        # Metrics
        attrition_by_overtime = df.groupby('overtime')['attrition_flag'].mean().mul(100)
        attrition_by_performance = df.groupby('performancerating')['attrition_flag'].mean().mul(100)
        
        metrics = {
            "attrition_rate_by_overtime_percent": attrition_by_overtime.to_dict(),
            "attrition_rate_by_performance_rating_percent": attrition_by_performance.to_dict()
        }
        
        insights.append(f"Attrition Rate (Overtime): {attrition_by_overtime.get('Yes', 0):.2f}% vs. (No Overtime): {attrition_by_overtime.get('No', 0):.2f}%")
        insights.append(f"Attrition seems highest for performance rating: {attrition_by_performance.idxmax()} ({attrition_by_performance.max():.1f}%)")
        
        # Visualizations
        fig1 = px.bar(attrition_by_overtime.reset_index(), x='overtime', y='attrition_flag',
                      title="Attrition Rate by Overtime Status",
                      labels={'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_overtime"] = fig1.to_json()

        fig2 = px.bar(attrition_by_performance.reset_index(),
                      x='performancerating', y='attrition_flag', title="Attrition Rate by Performance Rating",
                      labels={'performancerating': 'Performance Rating', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_performance"] = fig2.to_json()
        
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

def training_and_stock_options_effect_on_employee_retention(df):
    analysis_name = "Training and Stock Options' Effect on Retention"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['stockoptionlevel', 'trainingtimeslastyear', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['stockoptionlevel', 'trainingtimeslastyear', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['retention_flag'] = df['attrition'].str.lower().map({'yes': 0, 'true': 0, '1': 0, 'no': 1, 'false': 1, '0': 1})
        else:
            df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['stockoptionlevel', 'trainingtimeslastyear']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['retention_flag', 'stockoptionlevel', 'trainingtimeslastyear'], inplace=True)

        # Metrics
        retention_by_training = df.groupby('trainingtimeslastyear')['retention_flag'].mean().mul(100)
        retention_by_stock = df.groupby('stockoptionlevel')['retention_flag'].mean().mul(100)
        
        metrics = {
            "retention_rate_by_training_times": retention_by_training.to_dict(),
            "retention_rate_by_stock_option_level": retention_by_stock.to_dict()
        }
        
        insights.append(f"Highest retention by training: {retention_by_training.max():.1f}% for {retention_by_training.idxmax()} trainings.")
        insights.append(f"Highest retention by stock level: {retention_by_stock.max():.1f}% for level {retention_by_stock.idxmax()}.")

        # Visualizations
        fig1 = px.bar(retention_by_training.reset_index(),
                      x='trainingtimeslastyear', y='retention_flag', title="Retention Rate by Number of Trainings Last Year",
                      labels={'trainingtimeslastyear': 'Trainings Last Year', 'retention_flag': 'Retention Rate (%)'})
        visualizations["retention_by_training"] = fig1.to_json()

        fig2 = px.bar(retention_by_stock.reset_index(),
                      x='stockoptionlevel', y='retention_flag', title="Retention Rate by Stock Option Level",
                      labels={'stockoptionlevel': 'Stock Option Level', 'retention_flag': 'Retention Rate (%)'})
        visualizations["retention_by_stock_level"] = fig2.to_json()
        
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

def performance_rating_correlation_with_employee_attrition(df):
    analysis_name = "Performance Rating Correlation with Employee Attrition"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['performancerating', 'yearsatcompany', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['performancerating', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        df['performancerating'] = pd.to_numeric(df['performancerating'], errors='coerce')
        df.dropna(subset=['attrition_flag', 'performancerating'], inplace=True)

        # Analysis
        attrition_by_rating = df.groupby('performancerating')['attrition_flag'].mean().mul(100).reset_index()
        
        metrics = {
            "attrition_rate_by_performance_rating": attrition_by_rating.to_dict('records')
        }
        
        insights.append(f"Attrition rate for top performers (Rating {attrition_by_rating['performancerating'].max()}): {attrition_by_rating['attrition_flag'].iloc[-1]:.1f}%")
        insights.append(f"Attrition rate for low performers (Rating {attrition_by_rating['performancerating'].min()}): {attrition_by_rating['attrition_flag'].iloc[0]:.1f}%")

        # Visualization
        fig = px.bar(attrition_by_rating, x='performancerating', y='attrition_flag',
                       title="Attrition Rate vs. Performance Rating",
                       labels={'performancerating': 'Performance Rating', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_performance_rating"] = fig.to_json()
        
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

            
def job_satisfaction_determinants_for_employee_retention(df):
    analysis_name = "Job Satisfaction Determinants for Employee Retention"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['jobsatisfaction', 'monthlyincome', 'totalworkingyears', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['jobsatisfaction', 'monthlyincome', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['retention_flag'] = df['attrition'].str.lower().map({'yes': 0, 'true': 0, '1': 0, 'no': 1, 'false': 1, '0': 1})
        else:
            df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['jobsatisfaction', 'monthlyincome', 'totalworkingyears']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        df.dropna(subset=['retention_flag', 'jobsatisfaction', 'monthlyincome'], inplace=True)

        # Analysis
        retained_df = df[df['retention_flag'] == 1].copy()
        income_by_satisfaction = retained_df.groupby('jobsatisfaction')['monthlyincome'].mean().reset_index()
        
        metrics = {
            "avg_income_by_satisfaction_retained_employees": income_by_satisfaction.to_dict('records')
        }
        
        insights.append("Analysis of retained employees shows average income by job satisfaction level.")
        if not income_by_satisfaction.empty:
            insights.append(f"Retained employees with highest satisfaction ({income_by_satisfaction['jobsatisfaction'].max()}) earn avg: ${income_by_satisfaction['monthlyincome'].iloc[-1]:,.0f}")

        # Visualization
        fig = px.bar(income_by_satisfaction, x='jobsatisfaction', y='monthlyincome',
                      title="Average Income by Job Satisfaction (Retained Employees)",
                      labels={'jobsatisfaction': 'Job Satisfaction Level', 'monthlyincome': 'Average Monthly Income'})
        visualizations["income_by_satisfaction_retained"] = fig.to_json()
        
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

def employee_performance_training_and_attrition_link_analysis(df):
    analysis_name = "Employee Performance, Training, and Attrition Link Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['performancerating', 'trainingtimeslastyear', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['performancerating', 'trainingtimeslastyear', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        for col in ['performancerating', 'trainingtimeslastyear']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df.dropna(subset=['attrition_flag', 'performancerating', 'trainingtimeslastyear'], inplace=True)

        # Analysis
        attrition_rates = df.groupby(['performancerating', 'trainingtimeslastyear'])['attrition_flag'].mean().mul(100).reset_index()
        
        highest_risk = attrition_rates.sort_values(by='attrition_flag', ascending=False).iloc[0]
        
        metrics = {
            "attrition_rates_by_performance_training": attrition_rates.to_dict('records'),
            "highest_attrition_group": highest_risk.to_dict()
        }
        
        insights.append(f"Highest attrition risk group: Performance {highest_risk['performancerating']} & {highest_risk['trainingtimeslastyear']} trainings ({highest_risk['attrition_flag']:.1f}%)")

        # Visualization
        fig = px.density_heatmap(attrition_rates, x='performancerating', y='trainingtimeslastyear', z='attrition_flag',
                                 title="Attrition Rate by Performance and Training Frequency",
                                 labels={'performancerating': 'Performance Rating', 'trainingtimeslastyear': 'Trainings Last Year', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_heatmap_performance_training"] = fig.to_json()
        
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

def employee_tenure_and_attrition_risk_analysis(df):
    analysis_name = "Employee Tenure and Attrition Risk Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['dob', 'hiredate', 'attrition']
        matched = fuzzy_match_column(df, expected)
        critical_missing = [col for col in ['hiredate', 'attrition'] if matched[col] is None]

        if critical_missing:
            return create_fallback_response(analysis_name, critical_missing, matched, df)
            
        df = df.rename(columns={v: k for k, v in matched.items() if v})
        
        df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
        
        if df['attrition'].dtype == 'object':
            df['attrition_flag'] = df['attrition'].str.lower().map({'yes': 1, 'true': 1, '1': 1, 'no': 0, 'false': 0, '0': 0})
        else:
            df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
            
        df.dropna(subset=['hiredate', 'attrition_flag'], inplace=True)

        df['tenure_years'] = (datetime.now() - df['hiredate']).dt.days / 365.25
        df.dropna(subset=['tenure_years'], inplace=True)

        # Create tenure bins to analyze risk
        bins = [0, 1, 3, 5, 10, 20, df['tenure_years'].max() + 1]
        labels = ['0-1 Yr', '1-3 Yrs', '3-5 Yrs', '5-10 Yrs', '10-20 Yrs', '20+ Yrs']
        df['tenure_bin'] = pd.cut(df['tenure_years'], bins=bins, labels=labels, right=False)
        
        attrition_by_tenure = df.groupby('tenure_bin', observed=True)['attrition_flag'].mean().mul(100).reset_index()
        
        metrics = {
            "attrition_rate_by_tenure_bin": attrition_by_tenure.to_dict('records')
        }
        
        if not attrition_by_tenure.empty:
            highest_risk = attrition_by_tenure.loc[attrition_by_tenure['attrition_flag'].idxmax()]
            insights.append(f"Highest attrition risk is in the '{highest_risk['tenure_bin']}' tenure group ({highest_risk['attrition_flag']:.1f}%)")

        # Visualization
        fig = px.histogram(df, x='tenure_years', color='attrition', barmode='overlay',
                           title="Distribution of Employee Tenure by Attrition Status")
        visualizations["tenure_distribution_by_attrition"] = fig.to_json()

        fig2 = px.bar(attrition_by_tenure, x='tenure_bin', y='attrition_flag',
                      title="Attrition Risk by Tenure Group",
                      labels={'tenure_bin': 'Tenure Group', 'attrition_flag': 'Attrition Rate (%)'})
        visualizations["attrition_by_tenure_group"] = fig2.to_json()
        
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
    Main function to run employee data analysis
    
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
    
    # Mapping of all analysis functions
    analysis_functions = {
        # General analyses
        "General Insights": show_general_insights,
        "Attrition Analysis": attrition_analysis,
        "Performance Analysis": performance_analysis,
        "Compensation Analysis": compensation_analysis,
        "Diversity Analysis": diversity_analysis,
        "Training Analysis": training_analysis,
        "Engagement Analysis": engagement_analysis,
        "Recruitment Analysis": recruitment_analysis,
        "Productivity Analysis": productivity_analysis,
        "Retention Analysis": retention_analysis,
        "Attendance Analysis": attendance_analysis,

        # Specific analyses
        "Employee Demographic and Tenure Analysis": employee_demographic_and_tenure_analysis,
        "Employee Profile and Departmental Analysis": employee_profile_and_departmental_analysis,
        "Employee Compensation and Tenure Analysis": employee_compensation_and_tenure_analysis,
        "Employee Attrition Prediction and Factor Analysis": employee_attrition_prediction_and_factor_analysis,
        "Employee Distribution and Service Length Analysis": employee_distribution_and_service_length_analysis,
        "Employee Performance and Compensation Analysis": employee_performance_and_compensation_analysis,
        "Employee Salary and Attrition Analysis": employee_salary_and_attrition_analysis,
        "Employee Salary Hike and Promotion Factor Analysis": employee_salary_hike_and_promotion_factor_analysis,
        "Work-Life Balance and Job Satisfaction Impact on Attrition": work_life_balance_and_job_satisfaction_impact_on_attrition,
        "Commute Distance and Work History Impact on Attrition": commute_distance_and_work_history_impact_on_attrition,
        "Employee Performance and Promotion Cycle Analysis": employee_performance_and_promotion_cycle_analysis,
        "Employee Demographic and Compensation Profile Analysis": employee_demographic_and_compensation_profile_analysis,
        "Factors Influencing Employee Attrition Analysis": factors_influencing_employee_attrition_analysis,
        "Employee Demographics and Attrition Correlation Analysis": employee_demographics_and_attrition_correlation_analysis,
        "Employee Performance and Tenure Analysis": employee_performance_and_tenure_analysis,
        "Compensation, Promotion, and Career Progression Analysis": compensation_promotion_and_career_progression_analysis,
        "Employee Profile and Training Engagement Analysis": employee_profile_and_training_engagement_analysis,
        "Attrition Factors related to Promotions and Stock Options": attrition_factors_related_to_promotions_and_stock_options,
        "Comprehensive Employee Satisfaction and Attrition Analysis": comprehensive_employee_satisfaction_and_attrition_analysis,
        "Employee Compensation Structure and Attrition Analysis": employee_compensation_structure_and_attrition_analysis,
        "Employee Performance and Career Level Attrition Analysis": employee_performance_and_career_level_attrition_analysis,
        "Employee Salary Structure Analysis by Department": employee_salary_structure_analysis_by_department,
        "Management and its Impact on Employee Performance and Attrition": management_and_its_impact_on_employee_performance_and_attrition,
        "Job Involvement and Training Impact on Employee Retention": job_involvement_and_training_impact_on_employee_retention,
        "Work-Life Balance and Job Satisfaction's Effect on Attrition": work_life_balance_and_job_satisfaction_impact_on_attrition,
        "Employee Lifecycle and Attrition Trend Analysis": employee_lifecycle_and_attrition_trend_analysis,
        "Performance and Workload Impact on Employee Attrition": performance_and_workload_impact_on_employee_attrition,
        "Training and Stock Options' Effect on Employee Retention": training_and_stock_options_effect_on_employee_retention,
        "Performance Rating Correlation with Employee Attrition": performance_rating_correlation_with_employee_attrition,
        "Job Satisfaction Determinants for Employee Retention": job_satisfaction_determinants_for_employee_retention,
        "Employee Performance, Training, and Attrition Link Analysis": employee_performance_training_and_attrition_link_analysis,
        "Employee Tenure and Attrition Risk Analysis": employee_tenure_and_attrition_risk_analysis,
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
    print("👥 Employee Analytics Script")

    # File path and encoding input
    file_path = input("Enter path to your employee data file (e.g., data.csv or data.xlsx): ")
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
            "General Insights", "Attrition Analysis", "Performance Analysis", 
            "Compensation Analysis", "Diversity Analysis", "Training Analysis",
            "Engagement Analysis", "Recruitment Analysis", "Productivity Analysis",
            "Retention Analysis", "Attendance Analysis"
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
        # Specific analyses (all other analyses)
        specific_analyses = [
            "Employee Demographic and Tenure Analysis",
            "Employee Profile and Departmental Analysis",
            "Employee Compensation and Tenure Analysis",
            "Employee Attrition Prediction and Factor Analysis",
            "Employee Distribution and Service Length Analysis",
            "Employee Performance and Compensation Analysis",
            "Employee Salary and Attrition Analysis",
            "Employee Salary Hike and Promotion Factor Analysis",
            "Work-Life Balance and Job Satisfaction Impact on Attrition",
            "Commute Distance and Work History Impact on Attrition",
            "Employee Performance and Promotion Cycle Analysis",
            "Employee Demographic and Compensation Profile Analysis",
            "Factors Influencing Employee Attrition Analysis",
            "Employee Demographics and Attrition Correlation Analysis",
            "Employee Performance and Tenure Analysis",
            "Compensation, Promotion, and Career Progression Analysis",
            "Employee Profile and Training Engagement Analysis",
            "Attrition Factors related to Promotions and Stock Options",
            "Comprehensive Employee Satisfaction and Attrition Analysis",
            "Employee Compensation Structure and Attrition Analysis",
            "Employee Performance and Career Level Attrition Analysis",
            "Employee Salary Structure Analysis by Department",
            "Management and its Impact on Employee Performance and Attrition",
            "Job Involvement and Training Impact on Employee Retention",
            "Employee Lifecycle and Attrition Trend Analysis",
            "Performance and Workload Impact on Employee Attrition",
            "Training and Stock Options' Effect on Employee Retention",
            "Performance Rating Correlation with Employee Attrition",
            "Job Satisfaction Determinants for Employee Retention",
            "Employee Performance, Training, and Attrition Link Analysis",
            "Employee Tenure and Attrition Risk Analysis"
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
            filename = f"hr_analytics_{analysis_name_clean}_{timestamp}.json"
            
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
    file_path = "sample_employee_data.csv"  # Replace with your actual file path
    
    # Run general insights
    result = main_backend(file_path)
    print("General Insights:", result.keys() if isinstance(result, dict) else "No result")
    
    # Run specific analysis
    result = main_backend(
        file_path, 
        category="Specific", 
        specific_analysis_name="Attrition Analysis"
    )
    print("Attrition Analysis completed:", "status" in result if isinstance(result, dict) else "No result")