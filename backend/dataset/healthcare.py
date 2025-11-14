import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
import json
import io
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


healthcare_analysis_options = [
    "clinical_trial_adverse_event_and_efficacy_analysis",
    "clinical_trial_recruitment_and_dropout_rate_analysis",
    "disease_outbreak_correlation_with_population_density_analysis",
    "disease_specific_cost_of_care_analysis",
    "electronic_health_record_ehr_system_performance_analysis",
    "emergency_department_triage_and_patient_outcome_analysis",
    "emergency_room_wait_time_and_patient_flow_analysis",
    "emergency_services_and_response_time_analysis",
    "geospatial_mortality_rate_and_public_health_analysis",
    "general_insights",
    "healthcare_facility_distribution_and_service_area_analysis",
    "hospital_acquired_infection_rate_analysis",
    "hospital_financial_performance_and_profitability_analysis",
    "hospital_quality_metrics_and_resource_ratio_analysis",
    "hospital_staffing_and_turnover_rate_analysis",
    "hospital_supply_chain_and_vendor_cost_analysis",
    "insurance_plan_enrollment_and_market_share_analysis",
    "insurance_claim_and_reimbursement_analysis",
    "laboratory_test_turnaround_time_analysis",
    "laboratory_test_volume_and_abnormality_rate_analysis",
    "medical_billing_code_and_charge_amount_analysis",
    "medical_device_performance_and_safety_analysis",
    "medical_equipment_inventory_and_maintenance_cost_analysis",
    "medication_analysis",
    "medication_side_effect_and_adherence_analysis",
    "mental_health_therapy_utilization_and_outcome_analysis",
    "nutritional_intervention_and_health_outcome_analysis",
    "patient_appointment_scheduling_and_cancellation_analysis",
    "patient_chronic_condition_and_comorbidity_analysis",
    "patient_demographics",
    "patient_demographics_and_disease_prevalence_analysis",
    "patient_experience_and_satisfaction_survey_analysis",
    "patient_insurance_coverage_and_claim_denials_analysis",
    "patient_length_of_stay_analysis",
    "patient_readmission_risk_and_predictive_analysis",
    "patient_satisfaction",
    "patient_transport_and_logistics_analysis",
    "population_health_risk_assessment_analysis",
    "prescription_drug_utilization_analysis",
    "primary_care_visit_and_wait_time_analysis",
    "provider_practice_demographics_analysis",
    "provider_specialization_and_patient_load_analysis",
    "provider_staffing_and_patient_load_analysis",
    "readmission_analysis",
    "socioeconomic_factors_in_healthcare_access_analysis",
    "surgical_and_clinical_procedure_cost_analysis",
    "treatment_effectiveness_and_patient_outcome_analysis",
    "treatment_outcomes",
    "treatment_trends_by_patient_age_group_analysis",
]


# --- JSON Serialization Helpers ---

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
        # Dump to string and reload to force type conversion
        return json.loads(json.dumps(data, cls=NumpyJSONEncoder))
    except Exception as e:
        print(f"Error in type conversion: {e}")
        # Fallback for complex un-serializable objects
        return str(data)

# --- Helper Functions ---

def fuzzy_match_column(df, target_columns):
    """Finds the best fuzzy match for target columns in the dataframe."""
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available) if available else (None, 0)
        matched[target] = match if score >= 70 else None
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    return df.rename(columns={v: k for k, v in matched.items() if v is not None})

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """Provides comprehensive general insights with visualizations and metrics, including warnings for missing columns"""
    analysis_type = "General Insights"
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        # Data types analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns.tolist()
        other_cols = [col for col in df.columns if col not in numeric_cols + categorical_cols + datetime_cols]
        
        # Memory usage
        memory_usage_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        # Missing values analysis
        missing_values = df.isnull().sum()
        missing_percentage = (missing_values / total_rows) * 100
        columns_with_missing = missing_values[missing_values > 0]
        
        # Duplicate analysis
        duplicate_rows = df.duplicated().sum()
        duplicate_percentage = (duplicate_rows / total_rows) * 100 if total_rows > 0 else 0
        
        # Basic statistics for numeric columns
        numeric_stats = {}
        if numeric_cols:
            numeric_stats = df[numeric_cols].describe().to_dict()
        
        # Categorical columns analysis
        categorical_stats = {}
        if categorical_cols:
            for col in categorical_cols[:5]:  # Limit to first 5 for brevity
                unique_count = df[col].nunique()
                top_values = df[col].value_counts().head(5).to_dict()
                categorical_stats[col] = {
                    "unique_count": int(unique_count),
                    "top_values": convert_to_native_types(top_values)
                }
        
        # Create visualizations
        visualizations = {}
        
        # 1. Data types distribution
        dtype_counts = {
            'Numeric': len(numeric_cols),
            'Categorical': len(categorical_cols),
            'Datetime': len(datetime_cols),
            'Other': len(other_cols)
        }
        fig_dtypes = px.pie(
            values=list(dtype_counts.values()), 
            names=list(dtype_counts.keys()),
            title='Data Types Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        visualizations["data_types_distribution"] = fig_dtypes.to_json()
        
        # 2. Missing values visualization
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
                title='Top 10 Columns with Missing Values (%)',
                labels={'missing_percentage': 'Missing %', 'column': 'Column'},
                color='missing_percentage',
                color_continuous_scale='reds'
            )
            visualizations["missing_values"] = fig_missing.to_json()
        else:
            # Create a simple message visualization when no missing values
            fig_no_missing = go.Figure()
            fig_no_missing.add_annotation(
                text="No Missing Values Found!",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=20, color="green")
            )
            fig_no_missing.update_layout(
                title="Missing Values Analysis",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            visualizations["missing_values"] = fig_no_missing.to_json()
        
        # 3. Numeric columns distributions (first 3 columns)
        if numeric_cols:
            for i, col in enumerate(numeric_cols[:3]):
                try:
                    fig_hist = px.histogram(
                        df, 
                        x=col, 
                        title=f'Distribution of {col}',
                        marginal='box',
                        color_discrete_sequence=['#6366f1']
                    )
                    visualizations[f"{col}_distribution"] = fig_hist.to_json()
                except Exception as e:
                    print(f"Could not create histogram for {col}: {e}")
            
            # Correlation heatmap if multiple numeric columns
            if len(numeric_cols) > 1:
                try:
                    corr_matrix = df[numeric_cols].corr(numeric_only=True).round(2)
                    fig_corr = px.imshow(
                        corr_matrix, 
                        title='Correlation Matrix of Numeric Columns',
                        color_continuous_scale='RdBu_r',
                        aspect="auto",
                        text_auto=True
                    )
                    visualizations["correlation_matrix"] = fig_corr.to_json()
                except Exception as e:
                    print(f"Could not create correlation matrix: {e}")
        
        # 4. Categorical columns (first 2 columns)
        if categorical_cols:
            for i, col in enumerate(categorical_cols[:2]):
                try:
                    value_counts = df[col].value_counts().head(10)  # Top 10 values
                    if len(value_counts) > 0:
                        fig_bar = px.bar(
                            x=value_counts.index.astype(str), 
                            y=value_counts.values,
                            title=f'Top 10 Values in {col}',
                            labels={'x': col, 'y': 'Count'},
                            color=value_counts.values,
                            color_continuous_scale='blues'
                        )
                        visualizations[f"{col}_top_values"] = fig_bar.to_json()
                except Exception as e:
                    print(f"Could not create bar chart for {col}: {e}")
        
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
            },
            "sample_columns": {
                "numeric_columns_sample": numeric_cols[:5],
                "categorical_columns_sample": categorical_cols[:5]
            }
        }
        
        # Add numeric statistics if available
        if numeric_stats:
            metrics["numeric_statistics_sample"] = {k: numeric_stats[k] for k in list(numeric_stats.keys())[:3]}
        
        # Add categorical statistics if available
        if categorical_stats:
            metrics["categorical_statistics_sample"] = categorical_stats
        
        # Generate insights - NOW INCLUDING MISSING COLUMNS WARNINGS
        insights = [
            f"Dataset contains {total_rows:,} rows and {total_columns} columns ({memory_usage_mb:.1f} MB)",
            f"Found {len(numeric_cols)} numeric, {len(categorical_cols)} categorical, and {len(datetime_cols)} datetime columns",
        ]
        
        # Add missing columns warning if provided (for fallback case)
        if missing_cols and len(missing_cols) > 0:
            insights.append("Showing General Analysis instead of the requested specific analysis.")
        
        if duplicate_rows > 0:
            insights.append(f"Found {duplicate_rows:,} duplicate rows ({duplicate_percentage:.1f}% of data)")
        else:
            insights.append("No duplicate rows found")
        
        if len(columns_with_missing) > 0:
            top_missing_col = columns_with_missing.idxmax()
            top_missing_pct = missing_percentage[top_missing_col]
            insights.append(f"{len(columns_with_missing)} columns have missing values (max: {top_missing_col} with {top_missing_pct:.1f}% missing)")
        else:
            insights.append("No missing values found in the dataset")
        
        if numeric_cols:
            sample_numeric = ', '.join(numeric_cols[:3])
            insights.append(f"Numeric columns include: {sample_numeric}{'...' if len(numeric_cols) > 3 else ''}")
        
        if categorical_cols:
            sample_categorical = ', '.join(categorical_cols[:3])
            insights.append(f"Categorical columns include: {sample_categorical}{'...' if len(categorical_cols) > 3 else ''}")
        
        insights.append(f"Generated {len(visualizations)} visualizations for comprehensive data exploration")
        
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
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred during General Insights analysis: {e}"],
            "matched_columns": matched_cols or {},
            "missing_columns": missing_cols or []
        }

def get_fallback_analysis(df, analysis_type, missing_columns, matched_cols={}):
    """
    Helper function to run the general insights analysis as a fallback
    and prepend a warning message.
    """
    general_result = show_general_insights(df, analysis_type, missing_columns, matched_cols)
    
    # Prepend a clear fallback warning to the insights from general_result
    warning_insights = [
        f"⚠️ REQUIRED COLUMNS NOT FOUND for '{analysis_type}'",
        "Showing General Analysis instead."
    ]
    
    # Add details about missing columns
    for col in missing_columns:
        match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
        warning_insights.append(f" - Missing: {col}{match_info}")
    
    # Combine insights
    general_result["insights"] = warning_insights + ["---"] + general_result.get("insights", [])
    
    # Set the status and original analysis type
    general_result["status"] = "fallback"
    general_result["analysis_type"] = analysis_type # Overwrite "General Insights" with the requested type
    general_result["missing_columns"] = missing_columns
    general_result["matched_columns"] = matched_cols
    
    return general_result


# --- Analysis Functions (Refactored) ---

def treatment_effectiveness_and_patient_outcome_analysis(df):
    analysis_type = "Treatment Effectiveness and Patient Outcome Analysis"
    try:
        expected = ['patient_id', 'treatment_type', 'outcome_status', 'pre_treatment_score', 'post_treatment_score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'outcome_status', 'treatment_type'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        # Outcome status distribution by treatment type
        outcome_by_treatment = df.groupby(['treatment_type', 'outcome_status']).size().unstack(fill_value=0)
        fig_outcome_by_treatment = px.bar(outcome_by_treatment, barmode='stack', title='Outcome Status by Treatment Type')
        visualizations['outcome_by_treatment_type'] = fig_outcome_by_treatment.to_json()
        insights.append("Generated plot for outcome status by treatment type.")

        # Change in score (pre vs post) by treatment type
        if 'pre_treatment_score' in df.columns and 'post_treatment_score' in df.columns:
            df['pre_treatment_score'] = pd.to_numeric(df['pre_treatment_score'], errors='coerce')
            df['post_treatment_score'] = pd.to_numeric(df['post_treatment_score'], errors='coerce')
            df['score_improvement'] = df['post_treatment_score'] - df['pre_treatment_score']
            avg_score_improvement_by_treatment = df.groupby('treatment_type')['score_improvement'].mean().reset_index()
            fig_score_improvement = px.bar(avg_score_improvement_by_treatment, x='treatment_type', y='score_improvement', title='Average Score Improvement by Treatment Type')
            visualizations['average_score_improvement_by_treatment'] = fig_score_improvement.to_json()
            metrics['avg_score_improvement'] = avg_score_improvement_by_treatment.set_index('treatment_type')['score_improvement'].to_dict()
            insights.append("Calculated and plotted average score improvement by treatment.")
        else:
            insights.append("Pre/Post treatment scores not available for improvement analysis.")


        metrics["total_treatment_records"] = len(df)
        metrics["num_unique_treatments"] = df['treatment_type'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_staffing_and_turnover_rate_analysis(df):
    analysis_type = "Hospital Staffing and Turnover Rate Analysis"
    try:
        expected = ['staff_id', 'department', 'role', 'hire_date', 'termination_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
        df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
        df.dropna(subset=['staff_id', 'department', 'role'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Staff count by department
        staff_by_department = df.groupby('department').size().reset_index(name='staff_count')
        fig_staff_by_department = px.bar(staff_by_department, x='department', y='staff_count', title='Staff Count by Department')
        visualizations['staff_by_department'] = fig_staff_by_department.to_json()
        insights.append("Generated plot for staff count by department.")

        # Turnover rate by department
        df['is_terminated'] = df['termination_date'].notna()
        turnover_data = df.groupby('department')['is_terminated'].sum().reset_index(name='terminated_count')
        total_staff_dept = df.groupby('department').size().reset_index(name='total_staff')
        turnover_merged = pd.merge(turnover_data, total_staff_dept, on='department', how='left')
        turnover_merged['turnover_rate'] = (turnover_merged['terminated_count'] / turnover_merged['total_staff']) * 100
        turnover_merged.fillna(0, inplace=True)

        fig_turnover_rate = px.bar(turnover_merged, x='department', y='turnover_rate', title='Estimated Turnover Rate by Department (%)')
        visualizations['turnover_rate_by_department'] = fig_turnover_rate.to_json()
        insights.append("Calculated and plotted estimated turnover rate by department.")

        metrics["total_staff_records"] = len(df)
        metrics["num_unique_departments"] = df['department'].nunique()
        metrics["turnover_rates_by_dept"] = turnover_merged.set_index('department')['turnover_rate'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def prescription_drug_utilization_analysis(df):
    analysis_type = "Prescription Drug Utilization Analysis"
    try:
        expected = ['medication_name', 'patient_id', 'prescription_date', 'quantity_prescribed']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        
        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['prescription_date'] = pd.to_datetime(df['prescription_date'], errors='coerce')
        df['quantity_prescribed'] = pd.to_numeric(df['quantity_prescribed'], errors='coerce')
        df.dropna(subset=['medication_name', 'quantity_prescribed', 'prescription_date'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Top 10 most utilized drugs by total quantity prescribed
        drug_utilization = df.groupby('medication_name')['quantity_prescribed'].sum().nlargest(10).reset_index()
        fig_drug_utilization = px.bar(drug_utilization, x='medication_name', y='quantity_prescribed', title='Top 10 Drugs by Quantity Prescribed')
        visualizations['top_drug_utilization'] = fig_drug_utilization.to_json()
        insights.append("Generated plot for top 10 drug utilization.")

        # Monthly trend of prescription volume
        monthly_prescriptions = df.groupby(df['prescription_date'].dt.to_period('M').dt.start_time).size().reset_index(name='num_prescriptions')
        monthly_prescriptions.columns = ['month_year', 'num_prescriptions']
        monthly_prescriptions = monthly_prescriptions.sort_values('month_year')
        
        fig_monthly_prescriptions = px.line(monthly_prescriptions, x='month_year', y='num_prescriptions', title='Monthly Prescription Volume Trend')
        visualizations['monthly_prescription_trend'] = fig_monthly_prescriptions.to_json()
        insights.append("Generated plot for monthly prescription volume.")

        metrics["total_prescriptions"] = len(df)
        metrics["total_quantity_prescribed"] = df['quantity_prescribed'].sum()
        metrics["num_unique_drugs"] = df['medication_name'].nunique()
        metrics["top_10_drugs"] = drug_utilization.to_dict('records')

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_appointment_scheduling_and_cancellation_analysis(df):
    analysis_type = "Patient Appointment Scheduling and Cancellation Analysis"
    try:
        expected = ['appointment_id', 'patient_id', 'appointment_date', 'appointment_status', 'cancellation_reason']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce')
        df.dropna(subset=['appointment_id', 'appointment_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Appointment status distribution
        status_distribution = df['appointment_status'].value_counts(normalize=True).reset_index()
        status_distribution.columns = ['status', 'proportion']
        fig_appointment_status = px.pie(status_distribution, names='status', values='proportion', title='Appointment Status Distribution')
        visualizations['appointment_status_distribution'] = fig_appointment_status.to_json()
        insights.append("Generated pie chart for appointment status distribution.")

        # Top 10 cancellation reasons
        if 'cancellation_reason' in df.columns:
            cancellation_reasons = df[df['appointment_status'].astype(str).str.lower() == 'cancelled']['cancellation_reason'].value_counts().nlargest(10).reset_index()
            cancellation_reasons.columns = ['reason', 'count']
            fig_cancellation_reasons = px.bar(cancellation_reasons, x='reason', y='count', title='Top 10 Appointment Cancellation Reasons')
            visualizations['top_cancellation_reasons'] = fig_cancellation_reasons.to_json()
            metrics['top_cancellation_reasons'] = cancellation_reasons.to_dict('records')
            insights.append("Generated plot for top 10 cancellation reasons.")
        else:
            insights.append("Cancellation reason data not available.")

        cancellation_rate = status_distribution[status_distribution['status'].astype(str).str.lower() == 'cancelled']['proportion'].sum() * 100 if 'cancelled' in status_distribution['status'].astype(str).str.lower().values else 0
        metrics["total_appointments"] = len(df)
        metrics["cancellation_rate_percent"] = cancellation_rate
        metrics["status_distribution"] = status_distribution.set_index('status')['proportion'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def geospatial_mortality_rate_and_public_health_analysis(df):
    analysis_type = "Geospatial Mortality Rate and Public Health Analysis"
    try:
        expected = ['patient_id', 'patient_latitude', 'patient_longitude', 'death_status', 'disease_name', 'population_density']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['patient_latitude'] = pd.to_numeric(df['patient_latitude'], errors='coerce')
        df['patient_longitude'] = pd.to_numeric(df['patient_longitude'], errors='coerce')
        df.dropna(subset=['patient_id', 'patient_latitude', 'patient_longitude', 'death_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Map of mortality events
        deceased_patients = df[df['death_status'].astype(str).str.lower() == 'deceased']
        if not deceased_patients.empty:
            fig_mortality_map = px.scatter_mapbox(deceased_patients, lat="patient_latitude", lon="patient_longitude",
                                                  hover_name="patient_id", color_discrete_sequence=["fuchsia"], zoom=3, height=400,
                                                  title='Geospatial Distribution of Mortality Events')
            fig_mortality_map.update_layout(mapbox_style="open-street-map")
            visualizations['mortality_event_map'] = fig_mortality_map.to_json()
            insights.append("Generated map of mortality events.")
        else:
            insights.append("No deceased patient data or location data for mortality map.")

        # Disease prevalence by location
        if 'disease_name' in df.columns:
            df['lat_lon_group'] = df['patient_latitude'].round(1).astype(str) + ',' + df['patient_longitude'].round(1).astype(str)
            disease_prevalence_by_location = df.groupby(['lat_lon_group', 'disease_name']).size().reset_index(name='count')
            plots_disease_location_data = []
            for loc_group in disease_prevalence_by_location['lat_lon_group'].unique():
                subset = disease_prevalence_by_location[disease_prevalence_by_location['lat_lon_group'] == loc_group]
                plots_disease_location_data.append(subset.nlargest(5, 'count'))

            if plots_disease_location_data:
                top_diseases_in_loc = pd.concat(plots_disease_location_data)
                fig_disease_prevalence_location = px.bar(top_diseases_in_loc, x='disease_name', y='count', color='lat_lon_group',
                                                        title='Top Disease Prevalence by Geospatial Cluster (Simplified)')
                visualizations['disease_prevalence_by_location'] = fig_disease_prevalence_location.to_json()
                insights.append("Generated plot for disease prevalence by location.")
            else:
                insights.append("Not enough data for disease prevalence by location clusters.")
        else:
            insights.append("Disease name data missing for disease prevalence by location.")

        metrics["total_patients"] = df['patient_id'].nunique()
        metrics["total_deaths"] = deceased_patients['patient_id'].nunique() if not deceased_patients.empty else 0

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def surgical_and_clinical_procedure_cost_analysis(df):
    analysis_type = "Surgical and Clinical Procedure Cost Analysis"
    try:
        expected = ['procedure_id', 'procedure_name', 'total_cost', 'patient_id', 'procedure_date', 'cpt_code']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['total_cost'] = pd.to_numeric(df['total_cost'], errors='coerce')
        df.dropna(subset=['procedure_id', 'total_cost', 'procedure_name'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Average cost per procedure
        avg_cost_per_procedure = df.groupby('procedure_name')['total_cost'].mean().nlargest(10).reset_index()
        fig_avg_cost_procedure = px.bar(avg_cost_per_procedure, x='procedure_name', y='total_cost', title='Top 10 Most Expensive Procedures (Average Cost)')
        visualizations['avg_cost_per_procedure'] = fig_avg_cost_procedure.to_json()
        insights.append("Generated plot for top 10 most expensive procedures by average cost.")

        # Total cost distribution by procedure type
        total_cost_by_procedure = df.groupby('procedure_name')['total_cost'].sum().nlargest(10).reset_index()
        fig_total_cost_procedure = px.pie(total_cost_by_procedure, names='procedure_name', values='total_cost', title='Total Cost Distribution for Top 10 Procedures')
        visualizations['total_cost_by_procedure_type'] = fig_total_cost_procedure.to_json()
        insights.append("Generated pie chart for total cost distribution by procedure.")
        
        metrics["total_procedure_cost"] = df['total_cost'].sum()
        metrics["num_unique_procedures"] = df['procedure_name'].nunique()
        metrics["avg_cost_top_10"] = avg_cost_per_procedure.to_dict('records')

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def electronic_health_record_ehr_system_performance_analysis(df):
    analysis_type = "Electronic Health Record (EHR) System Performance Analysis"
    try:
        expected = ['record_id', 'user_id', 'action_type', 'response_time_ms', 'error_status', 'timestamp']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df['response_time_ms'] = pd.to_numeric(df['response_time_ms'], errors='coerce')
        df.dropna(subset=['record_id', 'response_time_ms', 'timestamp'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Average response time over time
        daily_avg_response_time = df.groupby(df['timestamp'].dt.date)['response_time_ms'].mean().reset_index()
        daily_avg_response_time.columns = ['date', 'avg_response_time_ms']
        fig_daily_response_time = px.line(daily_avg_response_time, x='date', y='avg_response_time_ms', title='Daily Average EHR System Response Time')
        visualizations['daily_response_time_trend'] = fig_daily_response_time.to_json()
        insights.append("Generated plot for daily average EHR response time.")

        # Distribution of error statuses
        error_rate = 'N/A'
        if 'error_status' in df.columns:
            error_status_counts = df['error_status'].value_counts(normalize=True).reset_index()
            error_status_counts.columns = ['status', 'proportion']
            fig_error_status = px.pie(error_status_counts, names='status', values='proportion', title='Distribution of EHR Error Statuses')
            visualizations['error_status_distribution'] = fig_error_status.to_json()
            error_rate = (df[df['error_status'].astype(str).str.lower() != 'success'].shape[0] / len(df)) * 100 if len(df) > 0 else 0
            metrics['error_status_distribution'] = error_status_counts.set_index('status')['proportion'].to_dict()
            insights.append("Generated pie chart for EHR error status distribution.")
        else:
            insights.append("Error status data not available.")

        metrics["overall_avg_response_time_ms"] = df['response_time_ms'].mean()
        metrics["total_ehr_actions"] = len(df)
        metrics["error_rate_percent"] = error_rate

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_experience_and_satisfaction_survey_analysis(df):
    analysis_type = "Patient Experience and Satisfaction Survey Analysis"
    try:
        expected = ['survey_id', 'patient_id', 'overall_satisfaction_score', 'question_category', 'score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['overall_satisfaction_score'] = pd.to_numeric(df['overall_satisfaction_score'], errors='coerce')
        df.dropna(subset=['survey_id', 'overall_satisfaction_score'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Overall satisfaction score distribution
        overall_satisfaction_dist = df['overall_satisfaction_score'].value_counts(normalize=True).sort_index().reset_index()
        overall_satisfaction_dist.columns = ['score', 'proportion']
        fig_overall_satisfaction = px.bar(overall_satisfaction_dist, x='score', y='proportion', title='Distribution of Overall Patient Satisfaction Scores')
        visualizations['overall_satisfaction_distribution'] = fig_overall_satisfaction.to_json()
        insights.append("Generated plot for overall satisfaction score distribution.")

        # Average score by question category
        if 'question_category' in df.columns and 'score' in df.columns:
            df['score'] = pd.to_numeric(df['score'], errors='coerce')
            avg_score_by_category = df.groupby('question_category')['score'].mean().reset_index()
            fig_avg_score_category = px.bar(avg_score_by_category, x='question_category', y='score', title='Average Score by Survey Question Category')
            visualizations['avg_score_by_question_category'] = fig_avg_score_category.to_json()
            metrics['avg_score_by_category'] = avg_score_by_category.set_index('question_category')['score'].to_dict()
            insights.append("Generated plot for average score by question category.")
        else:
            insights.append("Question category or individual score data not available.")

        metrics["avg_overall_satisfaction_score"] = df['overall_satisfaction_score'].mean()
        metrics["total_surveys_completed"] = len(df)

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def emergency_room_wait_time_and_patient_flow_analysis(df):
    analysis_type = "Emergency Room Wait Time and Patient Flow Analysis"
    try:
        expected = ['er_visit_id', 'arrival_time', 'triage_time', 'discharge_time', 'patient_id', 'reason_for_visit']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
        df['triage_time'] = pd.to_datetime(df['triage_time'], errors='coerce')
        df['discharge_time'] = pd.to_datetime(df['discharge_time'], errors='coerce')
        df.dropna(subset=['er_visit_id', 'arrival_time'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_wait_time = 'N/A'
        avg_stay_time = 'N/A'

        if 'triage_time' in df.columns:
            df['wait_time_to_triage_minutes'] = (df['triage_time'] - df['arrival_time']).dt.total_seconds() / 60
            fig_wait_time_dist = px.histogram(df.dropna(subset=['wait_time_to_triage_minutes']), x='wait_time_to_triage_minutes', nbins=50, title='Distribution of ER Wait Time to Triage (Minutes)')
            visualizations['wait_time_to_triage_distribution'] = fig_wait_time_dist.to_json()
            avg_wait_time = df['wait_time_to_triage_minutes'].mean()
            insights.append("Generated plot for wait time to triage distribution.")
        else:
            insights.append("Triage time data not available for wait time analysis.")

        if 'discharge_time' in df.columns:
            df['total_er_stay_minutes'] = (df['discharge_time'] - df['arrival_time']).dt.total_seconds() / 60
            avg_stay_time = df['total_er_stay_minutes'].mean()
            
            if 'reason_for_visit' in df.columns:
                avg_er_stay_by_reason = df.groupby('reason_for_visit')['total_er_stay_minutes'].mean().nlargest(10).reset_index()
                fig_avg_er_stay = px.bar(avg_er_stay_by_reason, x='reason_for_visit', y='total_er_stay_minutes', title='Average Total ER Stay by Reason for Visit (Top 10)')
                visualizations['avg_er_stay_by_reason_for_visit'] = fig_avg_er_stay.to_json()
                insights.append("Generated plot for average ER stay by reason for visit.")
            else:
                insights.append("Reason for visit data not available for ER stay analysis.")
        else:
            insights.append("Discharge time data not available for total ER stay analysis.")

        metrics["total_er_visits"] = len(df)
        metrics["avg_wait_time_to_triage_minutes"] = avg_wait_time
        metrics["avg_total_er_stay_minutes"] = avg_stay_time

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def vaccination_coverage_and_compliance_analysis(df):
    analysis_type = "Vaccination Coverage and Compliance Analysis"
    try:
        expected = ['patient_id', 'vaccine_type', 'vaccination_date', 'compliance_status', 'age_group']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'vaccine_type', 'compliance_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Vaccination coverage by vaccine type
        total_patients = df['patient_id'].nunique()
        vaccine_coverage = df.groupby('vaccine_type')['patient_id'].nunique().reset_index(name='vaccinated_patients')
        vaccine_coverage['coverage_percent'] = (vaccine_coverage['vaccinated_patients'] / total_patients) * 100 if total_patients > 0 else 0

        fig_vaccine_coverage = px.bar(vaccine_coverage, x='vaccine_type', y='coverage_percent', title='Vaccination Coverage by Vaccine Type (%)')
        visualizations['vaccination_coverage'] = fig_vaccine_coverage.to_json()
        insights.append("Generated plot for vaccination coverage by vaccine type.")

        # Compliance status distribution
        compliance_dist = df['compliance_status'].value_counts(normalize=True).reset_index()
        compliance_dist.columns = ['status', 'proportion']
        fig_compliance_status = px.pie(compliance_dist, names='status', values='proportion', title='Vaccination Compliance Status Distribution')
        visualizations['compliance_status_distribution'] = fig_compliance_status.to_json()
        insights.append("Generated pie chart for compliance status distribution.")

        overall_compliance = compliance_dist[compliance_dist['status'].astype(str).str.lower() == 'compliant']['proportion'].sum() * 100 if 'compliant' in compliance_dist['status'].astype(str).str.lower().values else 0
        metrics["total_patients_in_data"] = total_patients
        metrics["overall_compliance_rate_percent"] = overall_compliance
        metrics["coverage_by_vaccine"] = vaccine_coverage.set_index('vaccine_type')['coverage_percent'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def laboratory_test_volume_and_abnormality_rate_analysis(df):
    analysis_type = "Laboratory Test Volume and Abnormality Rate Analysis"
    try:
        expected = ['test_id', 'patient_id', 'test_name', 'test_date', 'result_value', 'normal_range_low', 'normal_range_high']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')
        df['result_value'] = pd.to_numeric(df['result_value'], errors='coerce')
        df.dropna(subset=['test_id', 'test_name', 'result_value', 'test_date'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Test volume over time
        monthly_test_volume = df.groupby(df['test_date'].dt.to_period('M').dt.start_time).size().reset_index(name='test_count')
        monthly_test_volume.columns = ['month_year', 'test_count']
        monthly_test_volume = monthly_test_volume.sort_values('month_year')
        
        fig_monthly_test_volume = px.line(monthly_test_volume, x='month_year', y='test_count', title='Monthly Laboratory Test Volume Trend')
        visualizations['monthly_test_volume_trend'] = fig_monthly_test_volume.to_json()
        insights.append("Generated plot for monthly lab test volume.")

        # Abnormality rate by test type
        if 'normal_range_low' in df.columns and 'normal_range_high' in df.columns:
            df['normal_range_low'] = pd.to_numeric(df['normal_range_low'], errors='coerce')
            df['normal_range_high'] = pd.to_numeric(df['normal_range_high'], errors='coerce')
            df['is_abnormal'] = (df['result_value'] < df['normal_range_low']) | (df['result_value'] > df['normal_range_high'])
            
            abnormality_counts = df.groupby('test_name')['is_abnormal'].sum().reset_index(name='abnormal_count')
            total_test_counts = df.groupby('test_name').size().reset_index(name='total_count')
            abnormality_rates = pd.merge(abnormality_counts, total_test_counts, on='test_name', how='left')
            abnormality_rates['abnormality_rate_percent'] = (abnormality_rates['abnormal_count'] / abnormality_rates['total_count']) * 100
            abnormality_rates.fillna(0, inplace=True)
            
            top_abnormal_tests = abnormality_rates.nlargest(10, 'abnormality_rate_percent')
            fig_abnormality_rates = px.bar(top_abnormal_tests, x='test_name', y='abnormality_rate_percent', title='Top 10 Tests by Abnormality Rate (%)')
            visualizations['abnormality_rates_by_test_type'] = fig_abnormality_rates.to_json()
            metrics['top_10_abnormal_tests'] = top_abnormal_tests.to_dict('records')
            insights.append("Calculated and plotted abnormality rates by test type.")
        else:
            insights.append("Normal range data not available for abnormality rate calculation.")

        metrics["total_tests_performed"] = len(df)
        metrics["num_unique_test_types"] = df['test_name'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_financial_performance_and_profitability_analysis(df):
    analysis_type = "Hospital Financial Performance and Profitability Analysis"
    try:
        expected = ['financial_record_id', 'revenue_amount', 'expense_amount', 'date', 'department']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['revenue_amount'] = pd.to_numeric(df['revenue_amount'], errors='coerce')
        df['expense_amount'] = pd.to_numeric(df['expense_amount'], errors='coerce')
        df.dropna(subset=['financial_record_id', 'revenue_amount', 'expense_amount', 'date'], inplace=True)

        df['profit'] = df['revenue_amount'] - df['expense_amount']

        metrics = {}
        visualizations = {}
        insights = []

        # Monthly revenue and expense trend
        df['month_year'] = df['date'].dt.to_period('M').dt.start_time
        monthly_financials = df.groupby('month_year').agg(
            total_revenue=('revenue_amount', 'sum'),
            total_expense=('expense_amount', 'sum'),
            total_profit=('profit', 'sum')
        ).reset_index().sort_values('month_year')

        fig_monthly_financials = px.line(monthly_financials, x='month_year', y=['total_revenue', 'total_expense', 'total_profit'],
                                         title='Monthly Hospital Revenue, Expense, and Profit Trend')
        visualizations['monthly_financial_trend'] = fig_monthly_financials.to_json()
        insights.append("Generated plot for monthly financial trends.")

        # Profitability by department
        if 'department' in df.columns:
            profit_by_department = df.groupby('department')['profit'].sum().reset_index()
            fig_profit_by_department = px.bar(profit_by_department, x='department', y='profit', title='Profitability by Department')
            visualizations['profit_by_department'] = fig_profit_by_department.to_json()
            metrics['profit_by_department'] = profit_by_department.set_index('department')['profit'].to_dict()
            insights.append("Generated plot for profitability by department.")
        else:
            insights.append("Department data not available for profitability by department.")

        total_revenue = df['revenue_amount'].sum()
        total_expense = df['expense_amount'].sum()
        total_profit = df['profit'].sum()
        profit_margin = (total_profit / total_revenue) * 100 if total_revenue > 0 else 0

        metrics["total_hospital_revenue"] = total_revenue
        metrics["total_hospital_expense"] = total_expense
        metrics["total_hospital_profit"] = total_profit
        metrics["overall_profit_margin_percent"] = profit_margin

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_readmission_risk_and_predictive_analysis(df):
    analysis_type = "Patient Readmission Risk and Predictive Analysis"
    try:
        expected = ['patient_id', 'age', 'gender', 'diagnosis', 'num_prior_admissions', 'readmission_status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'readmission_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Readmission status distribution
        readmission_status_counts = df['readmission_status'].value_counts(normalize=True).reset_index()
        readmission_status_counts.columns = ['status', 'proportion']
        fig_readmission_status_pie = px.pie(readmission_status_counts, names='status', values='proportion', title='Overall Readmission Status Distribution')
        visualizations['readmission_status_pie'] = fig_readmission_status_pie.to_json()
        insights.append("Generated pie chart for readmission status.")

        # Readmission rate by number of prior admissions
        if 'num_prior_admissions' in df.columns:
            df['num_prior_admissions'] = pd.to_numeric(df['num_prior_admissions'], errors='coerce')
            readmission_by_prior_admissions = df.groupby('num_prior_admissions')['readmission_status'].apply(
                lambda x: (x.astype(str).str.lower() == 'readmitted').mean() * 100
            ).reset_index(name='readmission_rate_percent')
            fig_readmission_by_prior_admissions = px.bar(readmission_by_prior_admissions, x='num_prior_admissions', y='readmission_rate_percent', title='Readmission Rate by Number of Prior Admissions (%)')
            visualizations['readmission_rate_by_prior_admissions'] = fig_readmission_by_prior_admissions.to_json()
            metrics['readmission_by_prior_admissions'] = readmission_by_prior_admissions.set_index('num_prior_admissions')['readmission_rate_percent'].to_dict()
            insights.append("Generated plot for readmission rate by prior admissions.")
        else:
            insights.append("Number of prior admissions data not available.")

        overall_readmission_rate = readmission_status_counts[readmission_status_counts['status'].astype(str).str.lower() == 'readmitted']['proportion'].sum() * 100 if 'readmitted' in readmission_status_counts['status'].astype(str).str.lower().values else 0
        metrics["total_patient_records"] = len(df)
        metrics["overall_readmission_rate_percent"] = overall_readmission_rate

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def provider_specialization_and_patient_load_analysis(df):
    analysis_type = "Provider Specialization and Patient Load Analysis"
    try:
        expected = ['provider_id', 'specialization', 'patient_id', 'visit_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['provider_id', 'specialization', 'patient_id'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Number of patients by specialization
        patients_by_specialization = df.groupby('specialization')['patient_id'].nunique().reset_index(name='num_patients')
        fig_patients_by_specialization = px.bar(patients_by_specialization, x='specialization', y='num_patients', title='Number of Unique Patients by Specialization')
        visualizations['patients_by_specialization'] = fig_patients_by_specialization.to_json()
        insights.append("Generated plot for unique patients by specialization.")

        # Average patient load per provider by specialization
        provider_patient_counts = df.groupby(['provider_id', 'specialization'])['patient_id'].nunique().reset_index(name='patient_load')
        avg_patient_load_by_specialization = provider_patient_counts.groupby('specialization')['patient_load'].mean().reset_index()
        fig_avg_patient_load_specialization = px.bar(avg_patient_load_by_specialization, x='specialization', y='patient_load', title='Average Patient Load per Provider by Specialization')
        visualizations['average_patient_load_by_specialization'] = fig_avg_patient_load_specialization.to_json()
        insights.append("Generated plot for average patient load by specialization.")

        metrics["total_providers"] = df['provider_id'].nunique()
        metrics["num_unique_specializations"] = df['specialization'].nunique()
        metrics["total_patients_seen"] = df['patient_id'].nunique()
        metrics["avg_patient_load_by_specialization"] = avg_patient_load_by_specialization.set_index('specialization')['patient_load'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def clinical_trial_adverse_event_and_efficacy_analysis(df):
    analysis_type = "Clinical Trial Adverse Event and Efficacy Analysis"
    try:
        expected = ['trial_id', 'patient_id', 'treatment_group', 'adverse_event_type', 'event_severity', 'outcome_measure_value']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['trial_id', 'patient_id', 'treatment_group'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Adverse event rate by treatment group
        if 'adverse_event_type' in df.columns:
            total_patients_per_group = df.groupby('treatment_group')['patient_id'].nunique().reset_index(name='total_patients')
            adverse_event_patients_per_group = df[df['adverse_event_type'].notna()].groupby('treatment_group')['patient_id'].nunique().reset_index(name='patients_with_ae')

            ae_rate_merged = pd.merge(total_patients_per_group, adverse_event_patients_per_group, on='treatment_group', how='left').fillna(0)
            ae_rate_merged['ae_rate_percent'] = (ae_rate_merged['patients_with_ae'] / ae_rate_merged['total_patients']) * 100
            
            fig_ae_rate = px.bar(ae_rate_merged, x='treatment_group', y='ae_rate_percent', title='Adverse Event Rate by Treatment Group (%)')
            visualizations['adverse_event_rate_by_treatment_group'] = fig_ae_rate.to_json()
            metrics['adverse_event_rates'] = ae_rate_merged.set_index('treatment_group')['ae_rate_percent'].to_dict()
            insights.append("Generated plot for adverse event rate by treatment group.")
        else:
            insights.append("Adverse event data not available.")

        # Efficacy measure distribution
        if 'outcome_measure_value' in df.columns:
            df['outcome_measure_value'] = pd.to_numeric(df['outcome_measure_value'], errors='coerce')
            fig_efficacy_dist = px.box(df.dropna(subset=['outcome_measure_value']), x='treatment_group', y='outcome_measure_value', title='Outcome Measure Distribution by Treatment Group')
            visualizations['efficacy_measure_distribution_by_treatment_group'] = fig_efficacy_dist.to_json()
            insights.append("Generated box plot for efficacy measure distribution.")
        else:
            insights.append("Outcome measure data not available for efficacy distribution.")

        metrics["total_trial_participants"] = df['patient_id'].nunique()
        metrics["num_adverse_events"] = df['adverse_event_type'].count() if 'adverse_event_type' in df.columns else 0

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_chronic_condition_and_comorbidity_analysis(df):
    analysis_type = "Patient Chronic Condition and Comorbidity Analysis"
    try:
        expected = ['patient_id', 'condition_name', 'diagnosis_date', 'number_of_comorbidities']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'condition_name'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Top 10 chronic conditions
        chronic_condition_counts = df['condition_name'].value_counts().nlargest(10).reset_index()
        chronic_condition_counts.columns = ['condition_name', 'count']
        fig_chronic_conditions = px.bar(chronic_condition_counts, x='condition_name', y='count', title='Top 10 Most Common Chronic Conditions')
        visualizations['top_chronic_conditions'] = fig_chronic_conditions.to_json()
        metrics['top_10_conditions'] = chronic_condition_counts.to_dict('records')
        insights.append("Generated plot for top 10 chronic conditions.")

        # Distribution of number of comorbidities
        avg_comorbidities = 'N/A'
        if 'number_of_comorbidities' in df.columns:
            df['number_of_comorbidities'] = pd.to_numeric(df['number_of_comorbidities'], errors='coerce')
            comorbidity_dist = df['number_of_comorbidities'].value_counts().sort_index().reset_index()
            comorbidity_dist.columns = ['num_comorbidities', 'count']
            fig_comorbidity_dist = px.bar(comorbidity_dist, x='num_comorbidities', y='count', title='Distribution of Number of Comorbidities per Patient')
            visualizations['comorbidity_distribution'] = fig_comorbidity_dist.to_json()
            avg_comorbidities = df['number_of_comorbidities'].mean()
            insights.append("Generated plot for comorbidity distribution.")
        else:
            insights.append("Comorbidity count data not available.")

        metrics["total_diagnosed_conditions"] = len(df)
        metrics["num_unique_chronic_conditions"] = df['condition_name'].nunique()
        metrics["avg_comorbidities_per_patient"] = avg_comorbidities

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def medical_device_performance_and_safety_analysis(df):
    analysis_type = "Medical Device Performance and Safety Analysis"
    try:
        expected = ['device_id', 'device_type', 'manufacturer', 'malfunction_rate', 'adverse_event_count', 'usage_hours']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['device_id', 'device_type'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        total_adverse_events = 'N/A'

        # Malfunction rates by device type
        if 'malfunction_rate' in df.columns:
            df['malfunction_rate'] = pd.to_numeric(df['malfunction_rate'], errors='coerce')
            avg_malfunction_rate = df.groupby('device_type')['malfunction_rate'].mean().reset_index()
            fig_malfunction_rate = px.bar(avg_malfunction_rate, x='device_type', y='malfunction_rate', title='Average Malfunction Rate by Device Type (%)')
            visualizations['malfunction_rate_by_device_type'] = fig_malfunction_rate.to_json()
            insights.append("Generated plot for malfunction rate by device type.")
        else:
            insights.append("Malfunction rate data not available.")

        # Adverse event counts by device type
        if 'adverse_event_count' in df.columns:
            df['adverse_event_count'] = pd.to_numeric(df['adverse_event_count'], errors='coerce')
            adverse_events_by_type = df.groupby('device_type')['adverse_event_count'].sum().reset_index()
            top_adverse_events = adverse_events_by_type.nlargest(10, 'adverse_event_count')
            fig_adverse_events = px.bar(top_adverse_events, x='device_type', y='adverse_event_count', title='Top 10 Device Types by Total Adverse Events')
            visualizations['adverse_events_by_device_type'] = fig_adverse_events.to_json()
            total_adverse_events = df['adverse_event_count'].sum()
            insights.append("Generated plot for adverse events by device type.")
        else:
            insights.append("Adverse event count data not available.")

        metrics["total_devices"] = len(df)
        metrics["num_unique_device_types"] = df['device_type'].nunique()
        metrics["total_adverse_events_recorded"] = total_adverse_events

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_quality_metrics_and_resource_ratio_analysis(df):
    analysis_type = "Hospital Quality Metrics and Resource Ratio Analysis"
    try:
        expected = ['metric_name', 'metric_value', 'hospital_id', 'staff_patient_ratio', 'bed_occupancy_rate']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
        df.dropna(subset=['metric_name', 'metric_value'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Average metric values by metric name
        avg_metric_values = df.groupby('metric_name')['metric_value'].mean().reset_index()
        fig_avg_metric_values = px.bar(avg_metric_values, x='metric_name', y='metric_value', title='Average Values for Key Quality Metrics')
        visualizations['average_quality_metric_values'] = fig_avg_metric_values.to_json()
        metrics['avg_metric_values'] = avg_metric_values.set_index('metric_name')['metric_value'].to_dict()
        insights.append("Generated plot for average quality metric values.")

        # Hospital performance across key ratios
        ratio_data = []
        if 'hospital_id' in df.columns:
            if 'staff_patient_ratio' in df.columns:
                df['staff_patient_ratio'] = pd.to_numeric(df['staff_patient_ratio'], errors='coerce')
                ratio_data.append(go.Bar(name='Staff-Patient Ratio', x=df['hospital_id'], y=df['staff_patient_ratio']))
            if 'bed_occupancy_rate' in df.columns:
                df['bed_occupancy_rate'] = pd.to_numeric(df['bed_occupancy_rate'], errors='coerce')
                ratio_data.append(go.Bar(name='Bed Occupancy Rate', x=df['hospital_id'], y=df['bed_occupancy_rate']))

        if ratio_data:
            fig_hospital_ratios = go.Figure(data=ratio_data)
            fig_hospital_ratios.update_layout(barmode='group', title='Hospital Performance Across Key Ratios')
            visualizations['hospital_resource_ratios'] = fig_hospital_ratios.to_json()
            insights.append("Generated plot for hospital resource ratios.")
        else:
            insights.append("Ratio data (staff-patient or bed occupancy) or hospital_id not available.")

        metrics["num_unique_metrics"] = df['metric_name'].nunique()
        metrics["num_unique_hospitals"] = df['hospital_id'].nunique() if 'hospital_id' in df.columns else 'N/A'

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def insurance_plan_enrollment_and_market_share_analysis(df):
    analysis_type = "Insurance Plan Enrollment and Market Share Analysis"
    try:
        expected = ['enrollment_id', 'insurance_plan_name', 'enrollment_date', 'patient_id']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['enrollment_id', 'insurance_plan_name'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Market share by insurance plan
        plan_market_share = df['insurance_plan_name'].value_counts(normalize=True).reset_index()
        plan_market_share.columns = ['plan_name', 'market_share_percent']
        plan_market_share['market_share_percent'] = plan_market_share['market_share_percent'] * 100
        
        fig_market_share = px.pie(plan_market_share.head(10), names='plan_name', values='market_share_percent', title='Insurance Plan Market Share by Enrollments (%) (Top 10)')
        visualizations['insurance_plan_market_share'] = fig_market_share.to_json()
        metrics['market_share_top_10'] = plan_market_share.head(10).set_index('plan_name')['market_share_percent'].to_dict()
        insights.append("Generated pie chart for insurance plan market share.")

        # Monthly enrollment trend for top 5 plans
        if 'enrollment_date' in df.columns:
            df['enrollment_date'] = pd.to_datetime(df['enrollment_date'], errors='coerce')
            df.dropna(subset=['enrollment_date'], inplace=True)
            monthly_enrollment = df.groupby([df['enrollment_date'].dt.to_period('M').dt.start_time, 'insurance_plan_name']).size().reset_index(name='num_enrollments')
            monthly_enrollment.columns = ['month_year', 'insurance_plan_name', 'num_enrollments']

            top_5_plans = plan_market_share['plan_name'].nlargest(5).tolist()
            monthly_enrollment_top5 = monthly_enrollment[monthly_enrollment['insurance_plan_name'].isin(top_5_plans)].sort_values('month_year')

            fig_monthly_enrollment = px.line(monthly_enrollment_top5, x='month_year', y='num_enrollments', color='insurance_plan_name',
                                             title='Monthly Enrollment Trend for Top 5 Insurance Plans')
            visualizations['monthly_enrollment_trend_top5_plans'] = fig_monthly_enrollment.to_json()
            insights.append("Generated plot for monthly enrollment trends.")
        else:
            insights.append("Enrollment date data not available for trend analysis.")

        metrics["total_enrollments"] = len(df)
        metrics["num_unique_plans"] = df['insurance_plan_name'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def mental_health_therapy_utilization_and_outcome_analysis(df):
    analysis_type = "Mental Health Therapy Utilization and Outcome Analysis"
    try:
        expected = ['patient_id', 'therapy_type', 'session_count', 'outcome_status', 'pre_therapy_score', 'post_therapy_score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['session_count'] = pd.to_numeric(df['session_count'], errors='coerce')
        df.dropna(subset=['patient_id', 'therapy_type', 'session_count'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Utilization by therapy type
        utilization_by_therapy = df.groupby('therapy_type')['session_count'].sum().reset_index()
        fig_therapy_utilization = px.bar(utilization_by_therapy, x='therapy_type', y='session_count', title='Total Sessions by Therapy Type')
        visualizations['therapy_utilization'] = fig_therapy_utilization.to_json()
        insights.append("Generated plot for therapy session utilization by type.")

        # Average outcome score improvement
        if 'pre_therapy_score' in df.columns and 'post_therapy_score' in df.columns:
            df['pre_therapy_score'] = pd.to_numeric(df['pre_therapy_score'], errors='coerce')
            df['post_therapy_score'] = pd.to_numeric(df['post_therapy_score'], errors='coerce')
            df['score_improvement'] = df['post_therapy_score'] - df['pre_therapy_score']
            avg_improvement_by_therapy = df.groupby('therapy_type')['score_improvement'].mean().reset_index()
            fig_avg_improvement = px.bar(avg_improvement_by_therapy, x='therapy_type', y='score_improvement', title='Average Score Improvement by Therapy Type')
            visualizations['average_score_improvement_by_therapy'] = fig_avg_improvement.to_json()
            metrics['avg_score_improvement'] = avg_improvement_by_therapy.set_index('therapy_type')['score_improvement'].to_dict()
            insights.append("Calculated and plotted average score improvement by therapy type.")
        else:
            insights.append("Pre/Post therapy scores not available for improvement analysis.")

        metrics["total_therapy_sessions"] = df['session_count'].sum()
        metrics["num_unique_therapy_types"] = df['therapy_type'].nunique()
        metrics["num_patients_in_therapy"] = df['patient_id'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def medical_billing_code_and_charge_amount_analysis(df):
    analysis_type = "Medical Billing Code and Charge Amount Analysis"
    try:
        expected = ['bill_id', 'cpt_code', 'charge_amount', 'patient_id', 'service_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['charge_amount'] = pd.to_numeric(df['charge_amount'], errors='coerce')
        df.dropna(subset=['bill_id', 'cpt_code', 'charge_amount'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Top 10 CPT codes by total charge amount
        top_cpt_codes_charges = df.groupby('cpt_code')['charge_amount'].sum().nlargest(10).reset_index()
        fig_top_cpt_codes = px.bar(top_cpt_codes_charges, x='cpt_code', y='charge_amount', title='Top 10 CPT Codes by Total Charge Amount')
        visualizations['top_cpt_codes_by_charge'] = fig_top_cpt_codes.to_json()
        metrics['top_10_cpt_codes'] = top_cpt_codes_charges.to_dict('records')
        insights.append("Generated plot for top 10 CPT codes by charge amount.")

        # Distribution of charge amounts
        fig_charge_amount_dist = px.histogram(df, x='charge_amount', nbins=50, title='Distribution of Charge Amounts')
        visualizations['charge_amount_distribution'] = fig_charge_amount_dist.to_json()
        insights.append("Generated histogram for charge amount distribution.")

        metrics["total_billed_charges"] = df['charge_amount'].sum()
        metrics["num_unique_cpt_codes"] = df['cpt_code'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_acquired_infection_rate_analysis(df):
    analysis_type = "Hospital-Acquired Infection Rate Analysis"
    try:
        expected = ['patient_id', 'infection_type', 'hospital_acquired', 'admission_date', 'discharge_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'hospital_acquired'], inplace=True)
        
        # Ensure 'hospital_acquired' is boolean
        df['hospital_acquired'] = df['hospital_acquired'].apply(lambda x: True if str(x).lower() in ['true', '1', 'yes'] else False)

        metrics = {}
        visualizations = {}
        insights = []

        # Overall hospital-acquired infection rate
        total_admissions = df['patient_id'].nunique()
        total_hai = df[df['hospital_acquired'] == True]['patient_id'].nunique()
        hai_rate_percent = (total_hai / total_admissions) * 100 if total_admissions > 0 else 0
        
        metrics["overall_hai_rate_percent"] = hai_rate_percent
        metrics["total_hai_cases"] = total_hai
        metrics["total_patients_in_sample"] = total_admissions

        # Pie chart for HAI vs. Non-HAI
        hai_counts = df['hospital_acquired'].value_counts(normalize=True).reset_index()
        hai_counts.columns = ['status', 'proportion']
        hai_counts['status'] = hai_counts['status'].map({True: 'Hospital-Acquired', False: 'Non-Hospital-Acquired'})
        fig_hai_overall = px.pie(hai_counts, names='status', values='proportion', title='Overall Hospital-Acquired Infection Status')
        visualizations['overall_hai_status'] = fig_hai_overall.to_json()
        insights.append("Generated pie chart for overall HAI status.")

        # Top 10 hospital-acquired infection types
        if 'infection_type' in df.columns:
            hai_types = df[df['hospital_acquired'] == True]['infection_type'].value_counts().nlargest(10).reset_index()
            hai_types.columns = ['infection_type', 'count']
            fig_hai_types = px.bar(hai_types, x='infection_type', y='count', title='Top 10 Hospital-Acquired Infection Types')
            visualizations['top_hai_types'] = fig_hai_types.to_json()
            metrics['top_10_hai_types'] = hai_types.to_dict('records')
            insights.append("Generated plot for top 10 HAI types.")
        else:
            insights.append("Infection type data not available for HAI types.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_transport_and_logistics_analysis(df):
    analysis_type = "Patient Transport and Logistics Analysis"
    try:
        expected = ['transport_id', 'patient_id', 'transport_date', 'transport_mode', 'transport_duration_minutes', 'origin_location', 'destination_location']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['transport_date'] = pd.to_datetime(df['transport_date'], errors='coerce')
        df.dropna(subset=['transport_id', 'transport_mode'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_duration = 'N/A'

        # Number of transports by mode
        transports_by_mode = df['transport_mode'].value_counts().reset_index()
        transports_by_mode.columns = ['transport_mode', 'count']
        fig_transports_by_mode = px.pie(transports_by_mode, names='transport_mode', values='count', title='Patient Transports by Mode')
        visualizations['transports_by_mode'] = fig_transports_by_mode.to_json()
        insights.append("Generated pie chart for transports by mode.")

        # Average transport duration by mode
        if 'transport_duration_minutes' in df.columns:
            df['transport_duration_minutes'] = pd.to_numeric(df['transport_duration_minutes'], errors='coerce')
            avg_duration_by_mode = df.groupby('transport_mode')['transport_duration_minutes'].mean().reset_index()
            fig_avg_duration_mode = px.bar(avg_duration_by_mode, x='transport_mode', y='transport_duration_minutes', title='Average Transport Duration by Mode (Minutes)')
            visualizations['average_duration_by_mode'] = fig_avg_duration_mode.to_json()
            avg_duration = df['transport_duration_minutes'].mean()
            insights.append("Generated plot for average transport duration.")
        else:
            insights.append("Transport duration data not available.")

        metrics["total_transports"] = len(df)
        metrics["avg_transport_duration_minutes"] = avg_duration
        metrics["transports_by_mode_count"] = transports_by_mode.set_index('transport_mode')['count'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def disease_outbreak_correlation_with_population_density_analysis(df):
    analysis_type = "Disease Outbreak Correlation with Population Density Analysis"
    try:
        expected = ['location_id', 'disease_case_count', 'population_density', 'outbreak_date', 'disease_name']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['outbreak_date'] = pd.to_datetime(df['outbreak_date'], errors='coerce')
        df['disease_case_count'] = pd.to_numeric(df['disease_case_count'], errors='coerce')
        df['population_density'] = pd.to_numeric(df['population_density'], errors='coerce')
        df.dropna(subset=['location_id', 'disease_case_count', 'population_density'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Scatter plot: Disease case count vs. population density
        fig_cases_vs_density = px.scatter(df, x='population_density', y='disease_case_count',
                                          title='Disease Case Count vs. Population Density',
                                          hover_name='location_id', color='disease_name' if 'disease_name' in df.columns else None)
        visualizations['disease_cases_vs_population_density'] = fig_cases_vs_density.to_json()
        insights.append("Generated scatter plot for case count vs. population density.")

        # Top 10 locations by disease case count
        top_outbreak_locations = df.groupby('location_id')['disease_case_count'].sum().nlargest(10).reset_index()
        fig_top_outbreak_locations = px.bar(top_outbreak_locations, x='location_id', y='disease_case_count', title='Top 10 Locations by Total Disease Cases')
        visualizations['top_outbreak_locations'] = fig_top_outbreak_locations.to_json()
        insights.append("Generated plot for top 10 outbreak locations.")
        
        # Correlation
        correlation = df['population_density'].corr(df['disease_case_count'])
        metrics["correlation_cases_vs_density"] = correlation
        insights.append(f"Correlation between population density and case count: {correlation:.2f}")

        metrics["total_disease_cases"] = df['disease_case_count'].sum()
        metrics["num_unique_locations"] = df['location_id'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def nutritional_intervention_and_health_outcome_analysis(df):
    analysis_type = "Nutritional Intervention and Health Outcome Analysis"
    try:
        expected = ['patient_id', 'nutritional_intervention_type', 'health_outcome_status', 'weight_change_kg', 'blood_pressure_change']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'nutritional_intervention_type', 'health_outcome_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Health outcome status by nutritional intervention type
        outcome_by_intervention = df.groupby(['nutritional_intervention_type', 'health_outcome_status']).size().unstack(fill_value=0)
        fig_outcome_by_intervention = px.bar(outcome_by_intervention, barmode='stack', title='Health Outcome Status by Nutritional Intervention Type')
        visualizations['health_outcome_by_intervention'] = fig_outcome_by_intervention.to_json()
        insights.append("Generated plot for health outcomes by intervention type.")

        # Average weight change by intervention type
        if 'weight_change_kg' in df.columns:
            df['weight_change_kg'] = pd.to_numeric(df['weight_change_kg'], errors='coerce')
            avg_weight_change = df.groupby('nutritional_intervention_type')['weight_change_kg'].mean().reset_index()
            fig_avg_weight_change = px.bar(avg_weight_change, x='nutritional_intervention_type', y='weight_change_kg', title='Average Weight Change (kg) by Intervention Type')
            visualizations['average_weight_change_by_intervention'] = fig_avg_weight_change.to_json()
            metrics['avg_weight_change'] = avg_weight_change.set_index('nutritional_intervention_type')['weight_change_kg'].to_dict()
            insights.append("Generated plot for average weight change by intervention.")
        else:
            insights.append("Weight change data not available.")

        metrics["total_interventions"] = len(df)
        metrics["num_unique_intervention_types"] = df['nutritional_intervention_type'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def medical_equipment_inventory_and_maintenance_cost_analysis(df):
    analysis_type = "Medical Equipment Inventory and Maintenance Cost Analysis"
    try:
        expected = ['equipment_id', 'equipment_type', 'maintenance_cost', 'purchase_cost', 'last_maintenance_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['equipment_id', 'equipment_type'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        total_maint_cost = 'N/A'

        # Total maintenance cost by equipment type
        if 'maintenance_cost' in df.columns:
            df['maintenance_cost'] = pd.to_numeric(df['maintenance_cost'], errors='coerce')
            maintenance_cost_by_type = df.groupby('equipment_type')['maintenance_cost'].sum().reset_index()
            fig_maintenance_cost = px.pie(maintenance_cost_by_type, names='equipment_type', values='maintenance_cost', title='Total Maintenance Cost by Equipment Type')
            visualizations['total_maintenance_cost_by_type'] = fig_maintenance_cost.to_json()
            total_maint_cost = df['maintenance_cost'].sum()
            insights.append("Generated pie chart for maintenance cost by equipment type.")
        else:
            insights.append("Maintenance cost data not available.")

        # Distribution of purchase costs
        if 'purchase_cost' in df.columns:
            df['purchase_cost'] = pd.to_numeric(df['purchase_cost'], errors='coerce')
            fig_purchase_cost_dist = px.histogram(df.dropna(subset=['purchase_cost']), x='purchase_cost', nbins=50, title='Distribution of Equipment Purchase Costs')
            visualizations['purchase_cost_distribution'] = fig_purchase_cost_dist.to_json()
            insights.append("Generated histogram for purchase cost distribution.")
        else:
            insights.append("Purchase cost data not available.")

        metrics["total_equipment_items"] = len(df)
        metrics["num_unique_equipment_types"] = df['equipment_type'].nunique()
        metrics["total_maintenance_cost_overall"] = total_maint_cost

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def primary_care_visit_and_wait_time_analysis(df):
    analysis_type = "Primary Care Visit and Wait Time Analysis"
    try:
        expected = ['visit_id', 'patient_id', 'visit_date', 'clinic_name', 'wait_time_minutes', 'consultation_duration_minutes']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
        df.dropna(subset=['visit_id', 'clinic_name', 'visit_date'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_wait_time = 'N/A'

        # Average wait time by clinic
        if 'wait_time_minutes' in df.columns:
            df['wait_time_minutes'] = pd.to_numeric(df['wait_time_minutes'], errors='coerce')
            avg_wait_time_by_clinic = df.groupby('clinic_name')['wait_time_minutes'].mean().reset_index()
            fig_avg_wait_time = px.bar(avg_wait_time_by_clinic, x='clinic_name', y='wait_time_minutes', title='Average Wait Time by Clinic (Minutes)')
            visualizations['average_wait_time_by_clinic'] = fig_avg_wait_time.to_json()
            avg_wait_time = df['wait_time_minutes'].mean()
            insights.append("Generated plot for average wait time by clinic.")
        else:
            insights.append("Wait time data not available.")

        # Monthly visit volume trend
        monthly_visits = df.groupby(df['visit_date'].dt.to_period('M').dt.start_time).size().reset_index(name='num_visits')
        monthly_visits.columns = ['month_year', 'num_visits']
        monthly_visits = monthly_visits.sort_values('month_year')

        fig_monthly_visits = px.line(monthly_visits, x='month_year', y='num_visits', title='Monthly Primary Care Visit Volume Trend')
        visualizations['monthly_visit_volume_trend'] = fig_monthly_visits.to_json()
        insights.append("Generated plot for monthly visit volume.")

        metrics["total_visits"] = len(df)
        metrics["avg_overall_wait_time_minutes"] = avg_wait_time

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def medication_side_effect_and_adherence_analysis(df):
    analysis_type = "Medication Side Effect and Adherence Analysis"
    try:
        expected = ['patient_id', 'medication_name', 'side_effect_reported', 'adherence_score', 'prescription_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'medication_name'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_adherence = 'N/A'

        # Top 10 reported side effects
        if 'side_effect_reported' in df.columns:
            side_effect_counts = df[df['side_effect_reported'].notna()]['side_effect_reported'].value_counts().nlargest(10).reset_index()
            side_effect_counts.columns = ['side_effect', 'count']
            fig_side_effects = px.bar(side_effect_counts, x='side_effect', y='count', title='Top 10 Reported Medication Side Effects')
            visualizations['top_reported_side_effects'] = fig_side_effects.to_json()
            metrics['top_10_side_effects'] = side_effect_counts.to_dict('records')
            insights.append("Generated plot for top 10 reported side effects.")
        else:
            insights.append("Side effect data not available.")

        # Adherence score distribution
        if 'adherence_score' in df.columns:
            df['adherence_score'] = pd.to_numeric(df['adherence_score'], errors='coerce')
            fig_adherence_dist = px.histogram(df.dropna(subset=['adherence_score']), x='adherence_score', nbins=50, title='Distribution of Medication Adherence Scores')
            visualizations['adherence_score_distribution'] = fig_adherence_dist.to_json()
            avg_adherence = df['adherence_score'].mean()
            insights.append("Generated histogram for adherence score distribution.")
        else:
            insights.append("Adherence score data not available.")

        metrics["total_medication_records"] = len(df)
        metrics["num_unique_medications"] = df['medication_name'].nunique()
        metrics["avg_adherence_score"] = avg_adherence

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def provider_practice_demographics_analysis(df):
    analysis_type = "Provider Practice Demographics Analysis"
    try:
        expected = ['provider_id', 'specialty', 'location', 'years_in_practice', 'patient_load']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['provider_id', 'specialty', 'location'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Number of providers by specialty
        providers_by_specialty = df['specialty'].value_counts().reset_index()
        providers_by_specialty.columns = ['specialty', 'count']
        fig_providers_by_specialty = px.pie(providers_by_specialty, names='specialty', values='count', title='Number of Providers by Specialty')
        visualizations['providers_by_specialty'] = fig_providers_by_specialty.to_json()
        metrics['providers_by_specialty'] = providers_by_specialty.set_index('specialty')['count'].to_dict()
        insights.append("Generated pie chart for provider distribution by specialty.")

        # Average years in practice by specialty
        if 'years_in_practice' in df.columns:
            df['years_in_practice'] = pd.to_numeric(df['years_in_practice'], errors='coerce')
            avg_years_in_practice = df.groupby('specialty')['years_in_practice'].mean().reset_index()
            fig_avg_years_practice = px.bar(avg_years_in_practice, x='specialty', y='years_in_practice', title='Average Years in Practice by Specialty')
            visualizations['average_years_in_practice_by_specialty'] = fig_avg_years_practice.to_json()
            insights.append("Generated plot for average years in practice by specialty.")
        else:
            insights.append("Years in practice data not available.")

        metrics["total_providers"] = len(df)
        metrics["num_unique_specialties"] = df['specialty'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def laboratory_test_turnaround_time_analysis(df):
    analysis_type = "Laboratory Test Turnaround Time Analysis"
    try:
        expected = ['test_id', 'test_name', 'sample_collection_time', 'result_release_time', 'test_type']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['sample_collection_time'] = pd.to_datetime(df['sample_collection_time'], errors='coerce')
        df['result_release_time'] = pd.to_datetime(df['result_release_time'], errors='coerce')
        df.dropna(subset=['test_id', 'sample_collection_time', 'result_release_time'], inplace=True)

        df['turnaround_time_hours'] = (df['result_release_time'] - df['sample_collection_time']).dt.total_seconds() / 3600
        df = df[df['turnaround_time_hours'] >= 0] # Filter out negative times

        metrics = {}
        visualizations = {}
        insights = []
        
        if df.empty:
            insights.append("No valid turnaround time data found after processing.")
            return {
                "analysis_type": analysis_type, "status": "error", "error_message": "No valid data after processing.",
                "visualizations": {}, "metrics": {}, "insights": insights, "matched_columns": matched
            }

        # Distribution of turnaround times
        fig_tat_distribution = px.histogram(df, x='turnaround_time_hours', nbins=50, title='Distribution of Laboratory Test Turnaround Times (Hours)')
        visualizations['turnaround_time_distribution'] = fig_tat_distribution.to_json()
        insights.append("Generated histogram for turnaround time distribution.")

        # Average turnaround time by test type
        if 'test_type' in df.columns:
            avg_tat_by_type = df.groupby('test_type')['turnaround_time_hours'].mean().reset_index()
            fig_avg_tat_by_type = px.bar(avg_tat_by_type, x='test_type', y='turnaround_time_hours', title='Average Turnaround Time by Test Type (Hours)')
            visualizations['average_turnaround_time_by_test_type'] = fig_avg_tat_by_type.to_json()
            metrics['avg_tat_by_type'] = avg_tat_by_type.set_index('test_type')['turnaround_time_hours'].to_dict()
            insights.append("Generated plot for average TAT by test type.")
        else:
            insights.append("Test type data not available for average TAT by type.")

        metrics["overall_avg_turnaround_time_hours"] = df['turnaround_time_hours'].mean()
        metrics["total_tests_with_tat"] = len(df)

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def emergency_services_and_response_time_analysis(df):
    analysis_type = "Emergency Services and Response Time Analysis"
    try:
        expected = ['incident_id', 'response_time_minutes', 'incident_type', 'incident_date', 'patient_outcome_status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
        df['response_time_minutes'] = pd.to_numeric(df['response_time_minutes'], errors='coerce')
        df.dropna(subset=['incident_id', 'response_time_minutes'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Distribution of response times
        fig_response_time_dist = px.histogram(df, x='response_time_minutes', nbins=50, title='Distribution of Emergency Response Times (Minutes)')
        visualizations['response_time_distribution'] = fig_response_time_dist.to_json()
        insights.append("Generated histogram for response time distribution.")

        # Average response time by incident type
        if 'incident_type' in df.columns:
            avg_response_by_type = df.groupby('incident_type')['response_time_minutes'].mean().reset_index()
            fig_avg_response_type = px.bar(avg_response_by_type, x='incident_type', y='response_time_minutes', title='Average Response Time by Incident Type (Minutes)')
            visualizations['average_response_time_by_incident_type'] = fig_avg_response_type.to_json()
            metrics['avg_response_by_type'] = avg_response_by_type.set_index('incident_type')['response_time_minutes'].to_dict()
            insights.append("Generated plot for average response time by incident type.")
        else:
            insights.append("Incident type data not available.")

        metrics["total_incidents"] = len(df)
        metrics["overall_avg_response_time_minutes"] = df['response_time_minutes'].mean()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_insurance_coverage_and_claim_denials_analysis(df):
    analysis_type = "Patient Insurance Coverage and Claim Denials Analysis"
    try:
        expected = ['patient_id', 'insurance_provider', 'claim_id', 'claim_status', 'denial_reason']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id', 'insurance_provider', 'claim_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Claim status distribution
        claim_status_counts = df['claim_status'].value_counts(normalize=True).reset_index()
        claim_status_counts.columns = ['status', 'proportion']
        fig_claim_status = px.pie(claim_status_counts, names='status', values='proportion', title='Claim Status Distribution')
        visualizations['claim_status_distribution'] = fig_claim_status.to_json()
        insights.append("Generated pie chart for claim status distribution.")

        # Top 10 claim denial reasons
        if 'denial_reason' in df.columns:
            denied_claims = df[df['claim_status'].astype(str).str.lower() == 'denied']
            denial_reasons = denied_claims['denial_reason'].value_counts().nlargest(10).reset_index()
            denial_reasons.columns = ['reason', 'count']
            fig_denial_reasons = px.bar(denial_reasons, x='reason', y='count', title='Top 10 Claim Denial Reasons')
            visualizations['top_denial_reasons'] = fig_denial_reasons.to_json()
            metrics['top_10_denial_reasons'] = denial_reasons.to_dict('records')
            insights.append("Generated plot for top 10 denial reasons.")
        else:
            insights.append("Denial reason data not available.")

        denial_rate = claim_status_counts[claim_status_counts['status'].astype(str).str.lower() == 'denied']['proportion'].sum() * 100 if 'denied' in claim_status_counts['status'].astype(str).str.lower().values else 0
        metrics["total_claims"] = len(df)
        metrics["denial_rate_percent"] = denial_rate
        metrics["claim_status_distribution"] = claim_status_counts.set_index('status')['proportion'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def provider_staffing_and_patient_load_analysis(df):
    analysis_type = "Provider Staffing and Patient Load Analysis"
    try:
        expected = ['provider_id', 'department', 'staff_hours_worked', 'num_patients_seen']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['provider_id', 'department'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Total staff hours by department
        if 'staff_hours_worked' in df.columns:
            df['staff_hours_worked'] = pd.to_numeric(df['staff_hours_worked'], errors='coerce')
            staff_hours_by_dept = df.groupby('department')['staff_hours_worked'].sum().reset_index()
            fig_staff_hours = px.bar(staff_hours_by_dept, x='department', y='staff_hours_worked', title='Total Staff Hours Worked by Department')
            visualizations['total_staff_hours_by_department'] = fig_staff_hours.to_json()
            metrics['staff_hours_by_dept'] = staff_hours_by_dept.set_index('department')['staff_hours_worked'].to_dict()
            insights.append("Generated plot for staff hours by department.")
        else:
            insights.append("Staff hours worked data not available.")

        # Average patients seen per provider by department
        if 'num_patients_seen' in df.columns:
            df['num_patients_seen'] = pd.to_numeric(df['num_patients_seen'], errors='coerce')
            patients_seen_per_provider = df.groupby(['provider_id', 'department'])['num_patients_seen'].sum().reset_index()
            avg_patients_per_provider_by_dept = patients_seen_per_provider.groupby('department')['num_patients_seen'].mean().reset_index()
            fig_avg_patients_seen = px.bar(avg_patients_per_provider_by_dept, x='department', y='num_patients_seen', title='Average Patients Seen per Provider by Department')
            visualizations['average_patients_seen_by_department'] = fig_avg_patients_seen.to_json()
            metrics['avg_patients_by_dept'] = avg_patients_per_provider_by_dept.set_index('department')['num_patients_seen'].to_dict()
            insights.append("Generated plot for average patients seen by department.")
        else:
            insights.append("Number of patients seen data not available.")

        metrics["total_providers"] = df['provider_id'].nunique()
        metrics["num_unique_departments"] = df['department'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def healthcare_facility_distribution_and_service_area_analysis(df):
    analysis_type = "Healthcare Facility Distribution and Service Area Analysis"
    try:
        expected = ['facility_id', 'facility_type', 'latitude', 'longitude', 'patient_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
        df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
        df.dropna(subset=['facility_id', 'latitude', 'longitude', 'facility_type'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Map of healthcare facilities by type
        fig_facility_map = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="facility_type",
                                             hover_name="facility_id", zoom=3, height=400,
                                             title='Healthcare Facility Distribution by Type')
        fig_facility_map.update_layout(mapbox_style="open-street-map")
        visualizations['healthcare_facility_map'] = fig_facility_map.to_json()
        insights.append("Generated map of healthcare facilities.")

        # Number of patients served by facility type
        if 'patient_count' in df.columns:
            df['patient_count'] = pd.to_numeric(df['patient_count'], errors='coerce')
            patients_by_facility_type = df.groupby('facility_type')['patient_count'].sum().reset_index()
            fig_patients_by_facility_type = px.bar(patients_by_facility_type, x='facility_type', y='patient_count', title='Total Patients Served by Facility Type')
            visualizations['patients_served_by_facility_type'] = fig_patients_by_facility_type.to_json()
            metrics['patients_by_facility_type'] = patients_by_facility_type.set_index('facility_type')['patient_count'].to_dict()
            insights.append("Generated plot for patients served by facility type.")
        else:
            insights.append("Patient count data for facilities not available.")

        metrics["total_facilities"] = len(df)
        metrics["num_unique_facility_types"] = df['facility_type'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def clinical_trial_recruitment_and_dropout_rate_analysis(df):
    analysis_type = "Clinical Trial Recruitment and Dropout Rate Analysis"
    try:
        expected = ['trial_id', 'patient_id', 'recruitment_status', 'dropout_reason', 'enrollment_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['trial_id', 'patient_id', 'recruitment_status'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Recruitment status distribution
        recruitment_status_dist = df['recruitment_status'].value_counts(normalize=True).reset_index()
        recruitment_status_dist.columns = ['status', 'proportion']
        fig_recruitment_status = px.pie(recruitment_status_dist, names='status', values='proportion', title='Clinical Trial Recruitment Status Distribution')
        visualizations['recruitment_status_distribution'] = fig_recruitment_status.to_json()
        insights.append("Generated pie chart for recruitment status.")

        # Top 10 dropout reasons
        if 'dropout_reason' in df.columns:
            dropout_reasons = df[df['recruitment_status'].astype(str).str.lower() == 'dropout']['dropout_reason'].value_counts().nlargest(10).reset_index()
            dropout_reasons.columns = ['reason', 'count']
            fig_dropout_reasons = px.bar(dropout_reasons, x='reason', y='count', title='Top 10 Clinical Trial Dropout Reasons')
            visualizations['top_dropout_reasons'] = fig_dropout_reasons.to_json()
            metrics['top_10_dropout_reasons'] = dropout_reasons.to_dict('records')
            insights.append("Generated plot for top 10 dropout reasons.")
        else:
            insights.append("Dropout reason data not available.")

        dropout_rate = recruitment_status_dist[recruitment_status_dist['status'].astype(str).str.lower() == 'dropout']['proportion'].sum() * 100 if 'dropout' in recruitment_status_dist['status'].astype(str).str.lower().values else 0
        metrics["total_records"] = len(df)
        metrics["dropout_rate_percent"] = dropout_rate
        metrics["recruitment_status_distribution"] = recruitment_status_dist.set_index('status')['proportion'].to_dict()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def socioeconomic_factors_in_healthcare_access_analysis(df):
    analysis_type = "Socioeconomic Factors in Healthcare Access Analysis"
    try:
        expected = ['patient_id', 'income_level', 'education_level', 'healthcare_access_score', 'zip_code']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['patient_id'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_access_score = 'N/A'

        # Healthcare access score by income level
        if 'income_level' in df.columns and 'healthcare_access_score' in df.columns:
            df['healthcare_access_score'] = pd.to_numeric(df['healthcare_access_score'], errors='coerce')
            avg_access_by_income = df.groupby('income_level')['healthcare_access_score'].mean().reset_index()
            fig_access_by_income = px.bar(avg_access_by_income, x='income_level', y='healthcare_access_score', title='Average Healthcare Access Score by Income Level')
            visualizations['healthcare_access_by_income_level'] = fig_access_by_income.to_json()
            avg_access_score = df['healthcare_access_score'].mean()
            insights.append("Generated plot for access score by income level.")
        else:
            insights.append("Income level or healthcare access score data not available.")

        # Distribution of education levels
        if 'education_level' in df.columns:
            education_dist = df['education_level'].value_counts(normalize=True).reset_index()
            education_dist.columns = ['education_level', 'proportion']
            fig_education_dist = px.pie(education_dist, names='education_level', values='proportion', title='Patient Education Level Distribution')
            visualizations['patient_education_level_distribution'] = fig_education_dist.to_json()
            insights.append("Generated pie chart for education level distribution.")
        else:
            insights.append("Education level data not available.")

        metrics["total_patients"] = len(df)
        metrics["avg_healthcare_access_score"] = avg_access_score

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_supply_chain_and_vendor_cost_analysis(df):
    analysis_type = "Hospital Supply Chain and Vendor Cost Analysis"
    try:
        expected = ['item_id', 'vendor_name', 'purchase_price', 'quantity_purchased', 'supply_category', 'purchase_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['purchase_price'] = pd.to_numeric(df['purchase_price'], errors='coerce')
        df['quantity_purchased'] = pd.to_numeric(df['quantity_purchased'], errors='coerce')
        df.dropna(subset=['item_id', 'purchase_price', 'quantity_purchased', 'vendor_name'], inplace=True)

        df['total_cost'] = df['purchase_price'] * df['quantity_purchased']

        metrics = {}
        visualizations = {}
        insights = []

        # Total cost by vendor
        total_cost_by_vendor = df.groupby('vendor_name')['total_cost'].sum().nlargest(10).reset_index()
        fig_cost_by_vendor = px.bar(total_cost_by_vendor, x='vendor_name', y='total_cost', title='Top 10 Vendors by Total Supply Cost')
        visualizations['total_cost_by_vendor'] = fig_cost_by_vendor.to_json()
        metrics['top_10_vendors'] = total_cost_by_vendor.to_dict('records')
        insights.append("Generated plot for top 10 vendors by cost.")

        # Total cost by supply category
        if 'supply_category' in df.columns:
            total_cost_by_category = df.groupby('supply_category')['total_cost'].sum().reset_index()
            fig_cost_by_category = px.pie(total_cost_by_category, names='supply_category', values='total_cost', title='Total Supply Cost by Category')
            visualizations['total_cost_by_supply_category'] = fig_cost_by_category.to_json()
            metrics['cost_by_category'] = total_cost_by_category.set_index('supply_category')['total_cost'].to_dict()
            insights.append("Generated pie chart for supply cost by category.")
        else:
            insights.append("Supply category data not available.")

        metrics["total_supply_chain_cost"] = df['total_cost'].sum()
        metrics["num_unique_vendors"] = df['vendor_name'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def disease_specific_cost_of_care_analysis(df):
    analysis_type = "Disease-Specific Cost of Care Analysis"
    try:
        expected = ['patient_id', 'disease_name', 'total_cost_of_care', 'treatment_type']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['total_cost_of_care'] = pd.to_numeric(df['total_cost_of_care'], errors='coerce')
        df.dropna(subset=['patient_id', 'disease_name', 'total_cost_of_care'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Average cost of care by disease
        avg_cost_by_disease = df.groupby('disease_name')['total_cost_of_care'].mean().nlargest(10).reset_index()
        fig_avg_cost_by_disease = px.bar(avg_cost_by_disease, x='disease_name', y='total_cost_of_care', title='Average Cost of Care by Disease (Top 10)')
        visualizations['average_cost_by_disease'] = fig_avg_cost_by_disease.to_json()
        metrics['avg_cost_top_10_diseases'] = avg_cost_by_disease.to_dict('records')
        insights.append("Generated plot for average cost by disease.")

        # Total cost of care distribution by disease
        total_cost_by_disease = df.groupby('disease_name')['total_cost_of_care'].sum().nlargest(10).reset_index()
        fig_total_cost_by_disease = px.pie(total_cost_by_disease, names='disease_name', values='total_cost_of_care', title='Total Cost of Care Distribution by Disease (Top 10)')
        visualizations['total_cost_distribution_by_disease'] = fig_total_cost_by_disease.to_json()
        insights.append("Generated pie chart for total cost distribution by disease.")

        metrics["overall_total_cost_of_care"] = df['total_cost_of_care'].sum()
        metrics["num_unique_diseases"] = df['disease_name'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def treatment_trends_by_patient_age_group_analysis(df):
    analysis_type = "Treatment Trends by Patient Age Group Analysis"
    try:
        expected = ['patient_id', 'age', 'treatment_type', 'diagnosis_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df.dropna(subset=['patient_id', 'age', 'treatment_type'], inplace=True)

        age_bins = [0, 18, 45, 65, 85, df['age'].max() + 1]
        age_labels = ['0-17', '18-44', '45-64', '65-84', '85+']
        df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)
        
        df = df.dropna(subset=['age_group']) # Drop rows where age didn't fit bins

        metrics = {}
        visualizations = {}
        insights = []

        # Treatment type prevalence by age group
        treatment_by_age_group = df.groupby(['age_group', 'treatment_type']).size().unstack(fill_value=0)
        fig_treatment_by_age = px.bar(treatment_by_age_group, barmode='stack', title='Treatment Type Prevalence by Age Group')
        visualizations['treatment_prevalence_by_age_group'] = fig_treatment_by_age.to_json()
        insights.append("Generated plot for treatment prevalence by age group.")

        # Time trend for top treatment
        top_5_treatments = df['treatment_type'].value_counts().nlargest(5).index.tolist()
        if 'diagnosis_date' in df.columns:
            df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
            df.dropna(subset=['diagnosis_date'], inplace=True)
            df['month_year'] = df['diagnosis_date'].dt.to_period('M').dt.start_time

            if top_5_treatments:
                trend_data = df[df['treatment_type'] == top_5_treatments[0]].groupby('month_year').size().reset_index(name='count')
                trend_data = trend_data.sort_values('month_year')
                fig_treatment_trend = px.line(trend_data, x='month_year', y='count', title=f'Monthly Trend for {top_5_treatments[0]} Treatment')
                visualizations['monthly_treatment_trend'] = fig_treatment_trend.to_json()
                insights.append(f"Generated trend plot for top treatment: {top_5_treatments[0]}.")
            else:
                insights.append("No top treatments found for trend analysis.")
        else:
            insights.append("Diagnosis date data not available for treatment trend analysis.")

        metrics["total_patients"] = df['patient_id'].nunique()
        metrics["num_unique_treatment_types"] = df['treatment_type'].nunique()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def emergency_department_triage_and_patient_outcome_analysis(df):
    analysis_type = "Emergency Department Triage and Patient Outcome Analysis"
    try:
        expected = ['er_visit_id', 'triage_level', 'discharge_disposition', 'er_wait_time_minutes', 'patient_outcome_status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df.dropna(subset=['er_visit_id', 'triage_level'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []
        
        avg_wait_time = 'N/A'
        if 'er_wait_time_minutes' in df.columns:
            df['er_wait_time_minutes'] = pd.to_numeric(df['er_wait_time_minutes'], errors='coerce')
            avg_wait_time = df['er_wait_time_minutes'].mean()

        # Distribution of triage levels
        triage_level_dist = df['triage_level'].value_counts(normalize=True).reset_index()
        triage_level_dist.columns = ['triage_level', 'proportion']
        fig_triage_level = px.pie(triage_level_dist, names='triage_level', values='proportion', title='Emergency Department Triage Level Distribution')
        visualizations['triage_level_distribution'] = fig_triage_level.to_json()
        metrics['triage_level_distribution'] = triage_level_dist.set_index('triage_level')['proportion'].to_dict()
        insights.append("Generated pie chart for triage level distribution.")

        # Patient outcome status by triage level
        if 'patient_outcome_status' in df.columns:
            outcome_by_triage = df.groupby(['triage_level', 'patient_outcome_status']).size().unstack(fill_value=0)
            fig_outcome_by_triage = px.bar(outcome_by_triage, barmode='stack', title='Patient Outcome Status by Triage Level')
            visualizations['patient_outcome_by_triage_level'] = fig_outcome_by_triage.to_json()
            insights.append("Generated plot for patient outcome by triage level.")
        else:
            insights.append("Patient outcome status data not available.")

        metrics["total_er_visits"] = len(df)
        metrics["avg_wait_time_minutes"] = avg_wait_time

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def population_health_risk_assessment_analysis(df):
    analysis_type = "Population Health Risk Assessment Analysis"
    try:
        expected = ['patient_id', 'risk_score', 'risk_category', 'age', 'chronic_disease_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
        df.dropna(subset=['patient_id', 'risk_score'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        # Distribution of risk scores
        fig_risk_score_dist = px.histogram(df, x='risk_score', nbins=50, title='Distribution of Population Health Risk Scores')
        visualizations['risk_score_distribution'] = fig_risk_score_dist.to_json()
        insights.append("Generated histogram for risk score distribution.")

        # Number of patients by risk category
        if 'risk_category' in df.columns:
            risk_category_counts = df['risk_category'].value_counts().reset_index()
            risk_category_counts.columns = ['risk_category', 'count']
            fig_risk_category = px.bar(risk_category_counts, x='risk_category', y='count', title='Number of Patients by Risk Category')
            visualizations['patients_by_risk_category'] = fig_risk_category.to_json()
            metrics['patients_by_risk_category'] = risk_category_counts.set_index('risk_category')['count'].to_dict()
            insights.append("Generated plot for patient counts by risk category.")
        else:
            insights.append("Risk category data not available.")

        metrics["total_patients_assessed"] = len(df)
        metrics["avg_risk_score"] = df['risk_score'].mean()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


# --- Refactored General Analysis Functions ---

def analyze_sales_performance(df):
    analysis_type = "Sales Performance"
    try:
        expected = ['transaction_id', 'sales_amount', 'salesperson', 'date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['sales_amount'] = pd.to_numeric(df['sales_amount'], errors='coerce')
        df.dropna(subset=['sales_amount', 'salesperson'], inplace=True)
        
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        metrics = {}
        visualizations = {}
        insights = []

        total_sales = df['sales_amount'].sum()
        avg_sale = df['sales_amount'].mean()
        top_salesperson = df.groupby('salesperson')['sales_amount'].sum().idxmax()

        metrics["total_sales"] = total_sales
        metrics["avg_sale"] = avg_sale
        metrics["top_salesperson"] = top_salesperson
        insights.append(f"Total Sales: ${total_sales:,.2f}")
        insights.append(f"Average Sale: ${avg_sale:,.2f}")
        insights.append(f"Top Salesperson: {top_salesperson}")

        hist = px.histogram(df, x='sales_amount', nbins=50, title='Distribution of Sales Amount')
        visualizations["sales_amount_distribution"] = hist.to_json()

        top_10_salespeople = df.groupby('salesperson')['sales_amount'].sum().sort_values(ascending=False).head(10).reset_index()
        bar = px.bar(top_10_salespeople, x='salesperson', y='sales_amount', title='Top 10 Salespeople by Revenue')
        visualizations["top_salespeople"] = bar.to_json()
        insights.append("Generated plots for sales distribution and top performers.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def analyze_customer(df):
    analysis_type = "Customer Analysis"
    try:
        expected = ['customer_id', 'purchase_amount', 'purchase_date', 'customer_segment']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['purchase_amount'] = pd.to_numeric(df['purchase_amount'], errors='coerce')
        df.dropna(subset=['customer_id', 'purchase_amount', 'customer_segment'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total_customers = df['customer_id'].nunique()
        avg_purchase = df['purchase_amount'].mean()
        top_segment = df['customer_segment'].mode()[0]

        metrics["total_customers"] = total_customers
        metrics["avg_purchase"] = avg_purchase
        metrics["top_segment"] = top_segment
        insights.append(f"Total Unique Customers: {total_customers}")
        insights.append(f"Average Purchase Amount: ${avg_purchase:,.2f}")
        insights.append(f"Most Common Segment: {top_segment}")

        segment_dist = df['customer_segment'].value_counts(normalize=True).reset_index()
        segment_dist.columns = ['customer_segment', 'proportion']
        pie = px.pie(segment_dist, names='customer_segment', values='proportion', title='Customer Segment Distribution')
        visualizations["segment_distribution"] = pie.to_json()

        hist = px.histogram(df, x='purchase_amount', color='customer_segment', barmode='overlay')
        visualizations["purchase_amount_histogram"] = hist.to_json()
        insights.append("Generated plots for customer segmentation and purchase behavior.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def readmission_analysis(df):
    analysis_type = "Readmission Analysis"
    try:
        expected = ['patient_id', 'readmission_status', 'risk_score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['readmitted'] = df['readmission_status'].str.lower().str.contains('readmitted', na=False)
        df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
        df.dropna(subset=['patient_id', 'readmitted', 'risk_score'], inplace=True)

        metrics = {}
        visualizations = {}
        insights = []

        total_patients = df['patient_id'].nunique()
        readmission_rate = df['readmitted'].mean() * 100
        avg_risk_score = df['risk_score'].mean()
        
        metrics["total_patients"] = total_patients
        metrics["readmission_rate"] = readmission_rate
        metrics["avg_risk_score"] = avg_risk_score
        insights.append(f"Total Patients: {total_patients}, Readmission Rate: {readmission_rate:.2f}%, Avg. Risk Score: {avg_risk_score:.2f}")

        pie_fig = px.pie(df, names='readmitted', title="Readmission Status Distribution")
        visualizations["readmission_pie"] = pie_fig.to_json()

        hist_fig = px.histogram(df, x='risk_score', color='readmitted', barmode='overlay', title="Risk Score by Readmission")
        visualizations["risk_hist"] = hist_fig.to_json()
        insights.append("Generated plots for readmission status and risk scores.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def treatment_outcomes(df):
    analysis_type = "Treatment Outcomes"
    try:
        expected = ['treatment_id', 'treatment_type', 'outcome_score']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['outcome_score'] = pd.to_numeric(df['outcome_score'], errors='coerce')
        df.dropna(subset=['treatment_id', 'treatment_type', 'outcome_score'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['treatment_id'].nunique()
        avg_score = df['outcome_score'].mean()
        best_type = df.groupby('treatment_type')['outcome_score'].mean().idxmax()
        
        metrics["total_treatments"] = total
        metrics["avg_outcome_score"] = avg_score
        metrics["best_treatment_type"] = best_type
        insights.append(f"Total Treatments: {total}, Avg. Outcome Score: {avg_score:.2f}, Best Type: {best_type}")

        box_fig = px.box(df, x='treatment_type', y='outcome_score', title="Outcomes by Treatment Type")
        visualizations["outcome_box"] = box_fig.to_json()

        avg_by_type = df.groupby('treatment_type')['outcome_score'].mean().reset_index()
        bar_fig = px.bar(avg_by_type, x='treatment_type', y='outcome_score', title="Avg Outcome Score per Treatment Type")
        visualizations["avg_by_type"] = bar_fig.to_json()
        insights.append("Generated plots for treatment outcomes.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def cost_analysis(df):
    analysis_type = "Cost Analysis"
    try:
        expected = ['patient_id', 'diagnosis_code', 'avg_cost_of_care']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)

        df = safe_rename(df, matched)
        df['avg_cost_of_care'] = pd.to_numeric(df['avg_cost_of_care'], errors='coerce')
        df.dropna(subset=['diagnosis_code', 'avg_cost_of_care'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []
        
        avg_cost = df['avg_cost_of_care'].mean()
        most_expensive = df.groupby('diagnosis_code')['avg_cost_of_care'].mean().idxmax()
        
        metrics["avg_cost"] = avg_cost
        metrics["most_expensive_diagnosis"] = most_expensive
        insights.append(f"Average Cost of Care: ${avg_cost:,.2f}")
        insights.append(f"Most Expensive Diagnosis (on avg): {most_expensive}")

        top_10_expensive = df.groupby('diagnosis_code')['avg_cost_of_care'].mean().nlargest(10).reset_index()
        bar_fig = px.bar(top_10_expensive, x='diagnosis_code', y='avg_cost_of_care', title="Top 10 Most Expensive Diagnoses")
        visualizations["expensive_bar"] = bar_fig.to_json()

        box_fig = px.box(df[df['diagnosis_code'].isin(top_10_expensive['diagnosis_code'])], 
                         x='diagnosis_code', y='avg_cost_of_care', title="Cost Distribution by Diagnosis (Top 10)")
        visualizations["cost_box"] = box_fig.to_json()
        insights.append("Generated plots for cost analysis.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def chronic_disease_analysis(df):
    analysis_type = "Chronic Disease Analysis"
    try:
        expected = ['patient_id', 'age_group', 'bmi_status', 'chronic_condition_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['chronic_condition_count'] = pd.to_numeric(df['chronic_condition_count'], errors='coerce')
        df.dropna(subset=['patient_id', 'age_group', 'bmi_status', 'chronic_condition_count'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['patient_id'].nunique()
        avg_chronic = df['chronic_condition_count'].mean()
        common_bmi = df['bmi_status'].mode()[0]
        
        metrics["total_patients"] = total
        metrics["avg_chronic_conditions"] = avg_chronic
        metrics["most_common_bmi"] = common_bmi
        insights.append(f"Total Patients: {total}, Avg. Chronic Conditions: {avg_chronic:.2f}, Most Common BMI: {common_bmi}")

        box_fig = px.box(df, x='age_group', y='chronic_condition_count', title="Chronic Conditions by Age Group")
        visualizations["chronic_box"] = box_fig.to_json()

        hist_fig = px.histogram(df, x='chronic_condition_count', color='bmi_status', title="Chronic Conditions by BMI", barmode='overlay')
        visualizations["chronic_hist"] = hist_fig.to_json()
        insights.append("Generated plots for chronic disease analysis.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def emergency_cases(df):
    analysis_type = "Emergency Cases"
    try:
        expected = ['hospital_id', 'emergency_type', 'discharge_status']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df.dropna(subset=['emergency_type', 'discharge_status'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['emergency_type'].count()
        home = df[df['discharge_status'].str.contains('Discharged Home', case=False, na=False)].shape[0]
        rate = (home / total) * 100 if total else 0
        
        metrics["total_visits"] = total
        metrics["discharged_home"] = home
        metrics["discharge_rate"] = rate
        insights.append(f"Total Visits: {total}, Discharged Home: {home} ({rate:.2f}%)")

        pie_fig = px.pie(df, names='discharge_status', title="Discharge Status Distribution")
        visualizations["status_pie"] = pie_fig.to_json()

        bar_fig = px.bar(df.groupby('emergency_type')['discharge_status'].value_counts().reset_index(name='count'),
                         x='emergency_type', y='count', color='discharge_status',
                         title="Discharge Status by Emergency Type")
        visualizations["by_type_bar"] = bar_fig.to_json()
        insights.append("Generated plots for emergency case discharge status.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def medication_analysis(df):
    analysis_type = "Medication Analysis"
    try:
        expected = ['medication_id', 'patient_id', 'adherence_score', 'side_effect_type']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['adherence_score'] = pd.to_numeric(df['adherence_score'], errors='coerce')
        df.dropna(subset=['patient_id', 'adherence_score', 'side_effect_type'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['patient_id'].nunique()
        avg_adh = df['adherence_score'].mean()
        common_side = df['side_effect_type'].mode()[0]
        
        metrics["total_patients"] = total
        metrics["avg_adherence"] = avg_adh
        metrics["most_common_side_effect"] = common_side
        insights.append(f"Total Patients: {total}, Avg. Adherence: {avg_adh:.2f}, Common Side Effect: {common_side}")

        hist_fig = px.histogram(df, x='adherence_score', color='side_effect_type', title="Adherence by Side Effect Type", barmode='overlay')
        visualizations["adherence_hist"] = hist_fig.to_json()

        pie_fig = px.pie(df, names='side_effect_type', title="Side Effect Type Distribution")
        visualizations["sideeffect_pie"] = pie_fig.to_json()
        insights.append("Generated plots for medication adherence and side effects.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def hospital_resources(df):
    analysis_type = "Hospital Resources"
    try:
        expected = ['hospital_id', 'city', 'state', 'bed_count']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['bed_count'] = pd.to_numeric(df['bed_count'], errors='coerce')
        df.dropna(subset=['hospital_id', 'state', 'bed_count'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total_hosp = df['hospital_id'].nunique()
        total_beds = df['bed_count'].sum()
        avg_beds = df['bed_count'].mean()
        
        metrics["total_hospitals"] = total_hosp
        metrics["total_beds"] = total_beds
        metrics["avg_beds_per_hospital"] = avg_beds
        insights.append(f"Total Hospitals: {total_hosp}, Total Beds: {total_beds:,}, Avg. Beds: {avg_beds:.1f}")

        top_10_states = df.groupby('state')['bed_count'].sum().sort_values(ascending=False).head(10).reset_index()
        bar_fig = px.bar(top_10_states, x='state', y='bed_count', title="Top 10 States by Bed Count")
        visualizations["top_states_bar"] = bar_fig.to_json()

        hist_fig = px.histogram(df, x='bed_count', title="Distribution of Hospital Bed Counts")
        visualizations["bed_hist"] = hist_fig.to_json()
        insights.append("Generated plots for hospital bed resources.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_satisfaction(df):
    analysis_type = "Patient Satisfaction"
    try:
        expected = ['survey_id', 'patient_id', 'satisfaction_score', 'feedback_category']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['satisfaction_score'] = pd.to_numeric(df['satisfaction_score'], errors='coerce')
        df.dropna(subset=['survey_id', 'satisfaction_score', 'feedback_category'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['survey_id'].nunique()
        avg_score = df['satisfaction_score'].mean()
        best_cat = df.groupby('feedback_category')['satisfaction_score'].mean().idxmax()
        
        metrics["total_surveys"] = total
        metrics["avg_satisfaction"] = avg_score
        metrics["top_feedback_category"] = best_cat
        insights.append(f"Total Surveys: {total}, Avg. Score: {avg_score:.2f}, Best Category: {best_cat}")

        hist_fig = px.histogram(df, x='satisfaction_score', title="Distribution of Satisfaction Scores")
        visualizations["satisfaction_hist"] = hist_fig.to_json()

        by_category = df.groupby('feedback_category')['satisfaction_score'].mean().reset_index()
        bar_fig = px.bar(by_category, x='feedback_category', y='satisfaction_score', title="Satisfaction by Feedback Category")
        visualizations["by_category_bar"] = bar_fig.to_json()
        insights.append("Generated plots for patient satisfaction scores.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def infection_control(df):
    analysis_type = "Infection Control"
    try:
        expected = ['hospital_id', 'infection_rate', 'hospital_type']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['infection_rate'] = pd.to_numeric(df['infection_rate'], errors='coerce')
        df.dropna(subset=['hospital_id', 'infection_rate', 'hospital_type'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        avg_rate = df['infection_rate'].mean()
        low_hosp = df.loc[df['infection_rate'].idxmin(), 'hospital_id']
        high_hosp = df.loc[df['infection_rate'].idxmax(), 'hospital_id']
        
        metrics["avg_infection_rate"] = avg_rate
        metrics["lowest_rate_hospital"] = low_hosp
        metrics["highest_rate_hospital"] = high_hosp
        insights.append(f"Avg. Infection Rate: {avg_rate:.2f}%")
        insights.append(f"Lowest Rate: {low_hosp} (Hospital ID)")
        insights.append(f"Highest Rate: {high_hosp} (Hospital ID)")

        hist_fig = px.histogram(df, x='infection_rate', color='hospital_type', barmode='overlay', title="Infection Rate by Hospital Type")
        visualizations["infection_hist"] = hist_fig.to_json()

        by_type = df.groupby('hospital_type')['infection_rate'].mean().reset_index()
        bar_fig = px.bar(by_type, x='hospital_type', y='infection_rate', title="Average Infection Rate by Hospital Type")
        visualizations["hospitaltype_bar"] = bar_fig.to_json()
        insights.append("Generated plots for infection control analysis.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_demographics_and_disease_prevalence_analysis(df):
    analysis_type = "Patient Demographics and Disease Prevalence Analysis"
    try:
        expected = ['patient_id', 'age', 'gender', 'diagnosis_code']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df.dropna(subset=['patient_id', 'age', 'gender', 'diagnosis_code'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total = df['patient_id'].nunique()
        avg_age = df['age'].mean()
        common_diag = df['diagnosis_code'].mode()[0]
        
        metrics["total_patients"] = total
        metrics["avg_age"] = avg_age
        metrics["most_common_diagnosis"] = common_diag
        insights.append(f"Total Patients: {total}, Avg. Age: {avg_age:.1f}, Most Common Diagnosis: {common_diag}")

        hist_fig = px.histogram(df, x='age', color='gender', barmode='overlay', title="Age Distribution by Gender")
        visualizations["age_hist"] = hist_fig.to_json()

        diag_counts = df['diagnosis_code'].value_counts().nlargest(10).reset_index()
        diag_counts.columns = ['diagnosis_code', 'count']
        bar_fig = px.bar(diag_counts, x='diagnosis_code', y='count', title="Top 10 Diagnoses")
        visualizations["diag_bar"] = bar_fig.to_json()
        insights.append("Generated plots for patient demographics and diagnoses.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_length_of_stay_analysis(df):
    analysis_type = "Patient Length of Stay Analysis"
    try:
        expected = ['patient_id', 'admission_date', 'discharge_date']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
        df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
        df.dropna(subset=['patient_id', 'admission_date', 'discharge_date'], inplace=True)
        
        df['length_of_stay'] = (df['discharge_date'] - df['admission_date']).dt.days
        df = df[df['length_of_stay'] >= 0] # Remove errors

        if df.empty:
            raise ValueError("No valid Length of Stay data found after processing.")

        metrics = {}
        visualizations = {}
        insights = []

        total = df['patient_id'].nunique()
        avg_stay = df['length_of_stay'].mean()
        max_stay = df['length_of_stay'].max()
        
        metrics["total_patients"] = total
        metrics["avg_length_of_stay"] = avg_stay
        metrics["max_length_of_stay"] = max_stay
        insights.append(f"Total Patients: {total}, Avg. Stay: {avg_stay:.1f} days, Max Stay: {max_stay} days")

        hist_fig = px.histogram(df, x='length_of_stay', nbins=30, title="Distribution of Length of Stay")
        visualizations["stay_hist"] = hist_fig.to_json()

        los_by_month = df.groupby(df['admission_date'].dt.to_period('M'))['length_of_stay'].mean().reset_index()
        los_by_month['admission_date'] = los_by_month['admission_date'].astype(str)
        line_fig = px.line(los_by_month, x='admission_date', y='length_of_stay', title="Avg LOS by Month")
        visualizations["los_line"] = line_fig.to_json()
        insights.append("Generated plots for length of stay (LOS).")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def insurance_claim_and_reimbursement_analysis(df):
    analysis_type = "Insurance Claim and Reimbursement Analysis"
    try:
        expected = ['claim_id', 'patient_id', 'payer_type', 'billed_amount', 'reimbursement_amount']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['billed_amount'] = pd.to_numeric(df['billed_amount'], errors='coerce')
        df['reimbursement_amount'] = pd.to_numeric(df['reimbursement_amount'], errors='coerce')
        df.dropna(subset=['claim_id', 'payer_type', 'billed_amount', 'reimbursement_amount'], inplace=True)
        
        metrics = {}
        visualizations = {}
        insights = []

        total_claims = df['claim_id'].nunique()
        total_billed = df['billed_amount'].sum()
        total_reimbursed = df['reimbursement_amount'].sum()
        reimb_rate = (total_reimbursed / total_billed) * 100 if total_billed else 0
        
        metrics["total_claims"] = total_claims
        metrics["total_billed"] = total_billed
        metrics["total_reimbursed"] = total_reimbursed
        metrics["reimbursement_rate"] = reimb_rate
        insights.append(f"Total Claims: {total_claims}, Total Billed: ${total_billed:,.2f}, Total Reimbursed: ${total_reimbursed:,.2f} ({reimb_rate:.2f}%)")

        payer_bar_data = df.groupby('payer_type')['reimbursement_amount'].sum().reset_index()
        payer_bar = px.bar(payer_bar_data, x='payer_type', y='reimbursement_amount', title="Reimbursement by Payer Type")
        visualizations["payer_bar"] = payer_bar.to_json()

        df['reimbursement_rate'] = df['reimbursement_amount'] / df['billed_amount']
        box_fig = px.box(df[df['reimbursement_rate'].between(0, 1)], x='payer_type', y='reimbursement_rate', title="Reimbursement Rate by Payer")
        visualizations["rate_box"] = box_fig.to_json()
        insights.append("Generated plots for claim reimbursement.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


def patient_demographics(df):
    analysis_type = "Patient Demographics"
    try:
        expected = ['patient_id', 'age', 'gender']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched[col] is None]

        if missing:
            return get_fallback_analysis(df, analysis_type, missing, matched)
            
        df = safe_rename(df, matched)
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df_clean = df.dropna(subset=['age', 'gender'])
        
        metrics = {}
        visualizations = {}
        insights = []

        total_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else len(df)
        avg_age = df_clean['age'].mean() if not df_clean.empty else None
        gender_counts = df_clean['gender'].value_counts()
        most_common_gender = gender_counts.idxmax() if not gender_counts.empty else None
        
        metrics["total_patients"] = total_patients
        metrics["average_age"] = avg_age
        metrics["most_common_gender"] = most_common_gender
        insights.append(f"Total Patients: {total_patients}, Avg. Age: {avg_age:.1f}, Most Common Gender: {most_common_gender}")

        age_hist = px.histogram(df_clean, x='age', nbins=20, title="Age Distribution of Patients")
        visualizations["age_hist"] = age_hist.to_json()

        gender_pie = px.pie(df_clean, names='gender', title='Gender Distribution of Patients')
        visualizations["gender_pie"] = gender_pie.to_json()
        insights.append("Generated plots for patient age and gender distribution.")
        
        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    except Exception as e:
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": [f"An error occurred: {e}"],
            "matched_columns": matched if 'matched' in locals() else {}
        }


# --- Main API/Backend Functions ---
# Dictionary mapping all analysis names to their refactored functions
specific_healthcare_function_mapping = {
    "general_insights": show_general_insights,
    "treatment_effectiveness_and_patient_outcome_analysis": treatment_effectiveness_and_patient_outcome_analysis,
    "hospital_staffing_and_turnover_rate_analysis": hospital_staffing_and_turnover_rate_analysis,
    "prescription_drug_utilization_analysis": prescription_drug_utilization_analysis,
    "patient_appointment_scheduling_and_cancellation_analysis": patient_appointment_scheduling_and_cancellation_analysis,
    "geospatial_mortality_rate_and_public_health_analysis": geospatial_mortality_rate_and_public_health_analysis,
    "surgical_and_clinical_procedure_cost_analysis": surgical_and_clinical_procedure_cost_analysis,
    "electronic_health_record_ehr_system_performance_analysis": electronic_health_record_ehr_system_performance_analysis,
    "patient_experience_and_satisfaction_survey_analysis": patient_experience_and_satisfaction_survey_analysis,
    "emergency_room_wait_time_and_patient_flow_analysis": emergency_room_wait_time_and_patient_flow_analysis,
    "vaccination_coverage_and_compliance_analysis": vaccination_coverage_and_compliance_analysis,
    "laboratory_test_volume_and_abnormality_rate_analysis": laboratory_test_volume_and_abnormality_rate_analysis,
    "hospital_financial_performance_and_profitability_analysis": hospital_financial_performance_and_profitability_analysis,
    "patient_readmission_risk_and_predictive_analysis": patient_readmission_risk_and_predictive_analysis,
    "provider_specialization_and_patient_load_analysis": provider_specialization_and_patient_load_analysis,
    "clinical_trial_adverse_event_and_efficacy_analysis": clinical_trial_adverse_event_and_efficacy_analysis,
    "patient_chronic_condition_and_comorbidity_analysis": patient_chronic_condition_and_comorbidity_analysis,
    "medical_device_performance_and_safety_analysis": medical_device_performance_and_safety_analysis,
    "hospital_quality_metrics_and_resource_ratio_analysis": hospital_quality_metrics_and_resource_ratio_analysis,
    "insurance_plan_enrollment_and_market_share_analysis": insurance_plan_enrollment_and_market_share_analysis,
    "mental_health_therapy_utilization_and_outcome_analysis": mental_health_therapy_utilization_and_outcome_analysis,
    "medical_billing_code_and_charge_amount_analysis": medical_billing_code_and_charge_amount_analysis,
    "hospital_acquired_infection_rate_analysis": hospital_acquired_infection_rate_analysis,
    "patient_transport_and_logistics_analysis": patient_transport_and_logistics_analysis,
    "disease_outbreak_correlation_with_population_density_analysis": disease_outbreak_correlation_with_population_density_analysis,
    "nutritional_intervention_and_health_outcome_analysis": nutritional_intervention_and_health_outcome_analysis,
    "medical_equipment_inventory_and_maintenance_cost_analysis": medical_equipment_inventory_and_maintenance_cost_analysis,
    "primary_care_visit_and_wait_time_analysis": primary_care_visit_and_wait_time_analysis,
    "medication_side_effect_and_adherence_analysis": medication_side_effect_and_adherence_analysis,
    "provider_practice_demographics_analysis": provider_practice_demographics_analysis,
    "laboratory_test_turnaround_time_analysis": laboratory_test_turnaround_time_analysis,
    "emergency_services_and_response_time_analysis": emergency_services_and_response_time_analysis,
    "patient_insurance_coverage_and_claim_denials_analysis": patient_insurance_coverage_and_claim_denials_analysis,
    "provider_staffing_and_patient_load_analysis": provider_staffing_and_patient_load_analysis,
    "healthcare_facility_distribution_and_service_area_analysis": healthcare_facility_distribution_and_service_area_analysis,
    "clinical_trial_recruitment_and_dropout_rate_analysis": clinical_trial_recruitment_and_dropout_rate_analysis,
    "socioeconomic_factors_in_healthcare_access_analysis": socioeconomic_factors_in_healthcare_access_analysis,
    "hospital_supply_chain_and_vendor_cost_analysis": hospital_supply_chain_and_vendor_cost_analysis,
    "disease_specific_cost_of_care_analysis": disease_specific_cost_of_care_analysis,
    "treatment_trends_by_patient_age_group_analysis": treatment_trends_by_patient_age_group_analysis,
    "emergency_department_triage_and_patient_outcome_analysis": emergency_department_triage_and_patient_outcome_analysis,
    "population_health_risk_assessment_analysis": population_health_risk_assessment_analysis,
    "sales_performance": analyze_sales_performance,
    "customer_analysis": analyze_customer,
    "readmission_analysis": readmission_analysis,
    "treatment_outcomes": treatment_outcomes,
    "cost_analysis": cost_analysis,
    "chronic_disease_analysis": chronic_disease_analysis,
    "emergency_cases": emergency_cases,
    "medication_analysis": medication_analysis,
    "hospital_resources": hospital_resources,
    "patient_satisfaction": patient_satisfaction,
    "infection_control": infection_control,
    "patient_demographics_and_disease_prevalence_analysis": patient_demographics_and_disease_prevalence_analysis,
    "hospital_resource_and_capacity_analysis": hospital_resources, # Alias
    "patient_length_of_stay_analysis": patient_length_of_stay_analysis,
    "insurance_claim_and_reimbursement_analysis": insurance_claim_and_reimbursement_analysis,
    "patient_demographics": patient_demographics,
}

# List of available analyses for the frontend
analysis_options = list(specific_healthcare_function_mapping.keys())

def load_data(file_path_or_buffer, encoding='utf-8'):
    """Load data from CSV or Excel file path or buffer"""
    try:
        # Check if it's a file path (string) or a buffer
        if isinstance(file_path_or_buffer, str):
            file_name = file_path_or_buffer.lower()
            if file_name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer, encoding=encoding)
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                return pd.read_excel(file_path_or_buffer)
            else:
                raise ValueError("Unsupported file type. Please provide CSV or Excel file.")
        
        # If it's a buffer (like from a file upload)
        else:
            # We need to figure out the file type, often from the 'name' attribute
            file_name = getattr(file_path_or_buffer, 'name', '').lower()
            if file_name.endswith('.csv'):
                return pd.read_csv(file_path_or_buffer, encoding=encoding)
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                # read_excel can handle buffers directly
                return pd.read_excel(file_path_or_buffer)
            else:
                # Fallback: Try reading as CSV, then Excel
                try:
                    # Reset buffer position
                    file_path_or_buffer.seek(0)
                    return pd.read_csv(file_path_or_buffer, encoding=encoding)
                except Exception:
                    try:
                        # Reset buffer position
                        file_path_or_buffer.seek(0)
                        return pd.read_excel(file_path_or_buffer)
                    except Exception:
                        raise ValueError("Could not determine file type from buffer. Please provide a .csv or .xlsx file.")
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def run_analysis(df, analysis_name):
    """Main function to run any analysis by name"""
    func = specific_healthcare_function_mapping.get(analysis_name)
    if func is None:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": f"No analysis function found for '{analysis_name}'",
            "visualizations": {},
            "metrics": {},
            "insights": []
        }
    
    try:
        # Pass df directly. If func is show_general_insights, it will handle it.
        if analysis_name == "general_insights":
             return func(df)
        return func(df)
    except Exception as e:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": str(e),
            "visualizations": {},
            "metrics": {},
            "insights": []
        }

def main_backend(file_path_or_buffer, analysis_name, encoding='utf-8'):
    """Main backend function to load data and run analysis"""
    # Load data
    df = load_data(file_path_or_buffer, encoding)
    if df is None:
        return {
            "analysis_type": analysis_name,
            "status": "error",
            "error_message": "Failed to load data from file",
            "visualizations": {},
            "metrics": {},
            "insights": []
        }
    
    # Run the requested analysis
    return run_analysis(df, analysis_name)

# Example usage
if __name__ == "__main__":
    file_path = "sample_healthcare_data.csv"
    result = main_backend(file_path)
    print("General Insights:", result.keys() if isinstance(result, dict) else "No result")
    
    # Run specific analysis
    result = main_backend(
        file_path, 
        category="Specific", 
        specific_analysis_name="Attrition Analysis"
    )
    print("Attrition Analysis completed:", "status" in result if isinstance(result, dict) else "No result")