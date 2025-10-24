import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process

import warnings
warnings.filterwarnings('ignore')

# --- Utility Functions (as provided earlier, crucial for the analysis functions) ---
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

def show_missing_columns_warning(missing_columns, matched_columns):
    warning_message = f"Warning: The following expected columns are missing or could not be matched: {', '.join(missing_columns)}. "
    if matched_columns:
        warning_message += f"Successfully matched: {', '.join([f'{k}:{v}' for k, v in matched_columns.items() if v])}."
    return warning_message

def show_general_insights(df, analysis_name):
    return f"General insights for {analysis_name}: Dataset has {len(df)} rows and {len(df.columns)} columns."

def get_key_metrics(df):
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    return {
        "total_records": total_records,
        "num_columns": len(df.columns),
        "numeric_features": list(numeric_cols),
        "categorical_features": list(categorical_cols),
        "num_numeric_features": len(numeric_cols),
        "num_categorical_features": len(categorical_cols)
    }

# --- Placeholder for load_data function ---
def load_data(file_path, encoding='utf-8'):
    """
    Placeholder function to load data from a file.
    Replace this with your actual data loading logic (e.g., pd.read_csv, pd.read_excel).
    """
    try:
        # Assuming CSV for example, adjust as needed for your file type
        df = pd.read_csv(file_path, encoding=encoding)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# --- Analysis Functions (Continuing from where we left off) ---

def treatment_effectiveness_and_patient_outcome_analysis(df):
    expected = ['patient_id', 'treatment_type', 'outcome_status', 'pre_treatment_score', 'post_treatment_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Treatment Effectiveness and Patient Outcome Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'outcome_status', 'treatment_type'], inplace=True)

    # Outcome status distribution by treatment type
    outcome_by_treatment = df.groupby(['treatment_type', 'outcome_status']).size().unstack(fill_value=0)
    fig_outcome_by_treatment = px.bar(outcome_by_treatment, barmode='stack', title='Outcome Status by Treatment Type')

    # Change in score (pre vs post) by treatment type
    if 'pre_treatment_score' in df.columns and 'post_treatment_score' in df.columns:
        df['score_improvement'] = df['post_treatment_score'] - df['pre_treatment_score']
        avg_score_improvement_by_treatment = df.groupby('treatment_type')['score_improvement'].mean().reset_index()
        fig_score_improvement = px.bar(avg_score_improvement_by_treatment, x='treatment_type', y='score_improvement', title='Average Score Improvement by Treatment Type')
    else:
        fig_score_improvement = go.Figure().add_annotation(text="Pre/Post treatment scores not available for improvement analysis.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'outcome_by_treatment_type': fig_outcome_by_treatment,
        'average_score_improvement_by_treatment': fig_score_improvement
    }

    metrics = {
        "total_treatment_records": len(df),
        "num_unique_treatments": df['treatment_type'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def hospital_staffing_and_turnover_rate_analysis(df):
    expected = ['staff_id', 'department', 'role', 'hire_date', 'termination_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Hospital Staffing and Turnover Rate Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['hire_date'] = pd.to_datetime(df['hire_date'], errors='coerce')
    df['termination_date'] = pd.to_datetime(df['termination_date'], errors='coerce')
    df.dropna(subset=['staff_id', 'department', 'role'], inplace=True)

    # Staff count by department
    staff_by_department = df.groupby('department').size().reset_index(name='staff_count')

    # Turnover rate by department (simplified: count terminated staff / total staff per department)
    # This requires a snapshot of total staff at a given time or careful date handling.
    # For this example, we'll assume termination_date implies turnover.
    df['is_terminated'] = df['termination_date'].notna()
    turnover_data = df.groupby('department')['is_terminated'].sum().reset_index(name='terminated_count')
    total_staff_dept = df.groupby('department').size().reset_index(name='total_staff')
    turnover_merged = pd.merge(turnover_data, total_staff_dept, on='department', how='left')
    turnover_merged['turnover_rate'] = (turnover_merged['terminated_count'] / turnover_merged['total_staff']) * 100
    turnover_merged.fillna(0, inplace=True) # Handle departments with no terminations

    fig_staff_by_department = px.bar(staff_by_department, x='department', y='staff_count', title='Staff Count by Department')
    fig_turnover_rate = px.bar(turnover_merged, x='department', y='turnover_rate', title='Estimated Turnover Rate by Department (%)')

    plots = {
        'staff_by_department': fig_staff_by_department,
        'turnover_rate_by_department': fig_turnover_rate
    }

    metrics = {
        "total_staff_records": len(df),
        "num_unique_departments": df['department'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def prescription_drug_utilization_analysis(df):
    expected = ['medication_name', 'patient_id', 'prescription_date', 'quantity_prescribed']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Prescription Drug Utilization Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['prescription_date'] = pd.to_datetime(df['prescription_date'], errors='coerce')
    df.dropna(subset=['medication_name', 'quantity_prescribed'], inplace=True)

    # Top 10 most utilized drugs by total quantity prescribed
    drug_utilization = df.groupby('medication_name')['quantity_prescribed'].sum().nlargest(10).reset_index()

    # Monthly trend of prescription volume
    monthly_prescriptions = df.groupby(df['prescription_date'].dt.to_period('M').dt.start_time).size().reset_index(name='num_prescriptions')
    monthly_prescriptions.columns = ['month_year', 'num_prescriptions']
    monthly_prescriptions = monthly_prescriptions.sort_values('month_year')

    fig_drug_utilization = px.bar(drug_utilization, x='medication_name', y='quantity_prescribed', title='Top 10 Drugs by Quantity Prescribed')
    fig_monthly_prescriptions = px.line(monthly_prescriptions, x='month_year', y='num_prescriptions', title='Monthly Prescription Volume Trend')

    plots = {
        'top_drug_utilization': fig_drug_utilization,
        'monthly_prescription_trend': fig_monthly_prescriptions
    }

    metrics = {
        "total_prescriptions": len(df),
        "total_quantity_prescribed": df['quantity_prescribed'].sum(),
        "num_unique_drugs": df['medication_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def patient_appointment_scheduling_and_cancellation_analysis(df):
    expected = ['appointment_id', 'patient_id', 'appointment_date', 'appointment_status', 'cancellation_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Appointment Scheduling and Cancellation Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['appointment_date'] = pd.to_datetime(df['appointment_date'], errors='coerce')
    df.dropna(subset=['appointment_id', 'appointment_status'], inplace=True)

    # Appointment status distribution (e.g., Scheduled, Completed, Cancelled)
    status_distribution = df['appointment_status'].value_counts(normalize=True).reset_index()
    status_distribution.columns = ['status', 'proportion']

    # Top 10 cancellation reasons (if available)
    if 'cancellation_reason' in df.columns:
        cancellation_reasons = df[df['appointment_status'].astype(str).str.lower() == 'cancelled']['cancellation_reason'].value_counts().nlargest(10).reset_index()
        cancellation_reasons.columns = ['reason', 'count']
        fig_cancellation_reasons = px.bar(cancellation_reasons, x='reason', y='count', title='Top 10 Appointment Cancellation Reasons')
    else:
        fig_cancellation_reasons = go.Figure().add_annotation(text="Cancellation reason data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_appointment_status = px.pie(status_distribution, names='status', values='proportion', title='Appointment Status Distribution')

    plots = {
        'appointment_status_distribution': fig_appointment_status,
        'top_cancellation_reasons': fig_cancellation_reasons
    }

    metrics = {
        "total_appointments": len(df),
        "cancellation_rate_percent": status_distribution[status_distribution['status'].astype(str).str.lower() == 'cancelled']['proportion'].sum() * 100 if 'cancelled' in status_distribution['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def geospatial_mortality_rate_and_public_health_analysis(df):
    expected = ['patient_id', 'patient_latitude', 'patient_longitude', 'death_status', 'disease_name', 'population_density']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Geospatial Mortality Rate and Public Health Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'patient_latitude', 'patient_longitude', 'death_status'], inplace=True)

    # Map of mortality events (if death_status indicates deceased)
    deceased_patients = df[df['death_status'].astype(str).str.lower() == 'deceased']
    if not deceased_patients.empty:
        fig_mortality_map = px.scatter_mapbox(deceased_patients, lat="patient_latitude", lon="patient_longitude",
                                              hover_name="patient_id", color_discrete_sequence=["fuchsia"], zoom=3, height=400,
                                              title='Geospatial Distribution of Mortality Events')
        fig_mortality_map.update_layout(mapbox_style="open-street-map")
    else:
        fig_mortality_map = go.Figure().add_annotation(text="No deceased patient data or location data for mortality map.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Disease prevalence by region/location clusters (if enough data and clustering needed, simplified here)
    if 'disease_name' in df.columns and 'patient_latitude' in df.columns and 'patient_longitude' in df.columns:
        # Group by approximate location for simplified visualization
        df['lat_lon_group'] = df['patient_latitude'].round(1).astype(str) + ',' + df['patient_longitude'].round(1).astype(str)
        disease_prevalence_by_location = df.groupby(['lat_lon_group', 'disease_name']).size().reset_index(name='count')
        # Only show top 5 diseases for clarity per location if many
        plots_disease_location_data = []
        for loc_group in disease_prevalence_by_location['lat_lon_group'].unique():
            subset = disease_prevalence_by_location[disease_prevalence_by_location['lat_lon_group'] == loc_group]
            plots_disease_location_data.append(subset.nlargest(5, 'count'))

        if plots_disease_location_data:
            top_diseases_in_loc = pd.concat(plots_disease_location_data)
            fig_disease_prevalence_location = px.bar(top_diseases_in_loc, x='disease_name', y='count', color='lat_lon_group',
                                                    title='Top Disease Prevalence by Geospatial Cluster (Simplified)')
        else:
            fig_disease_prevalence_location = go.Figure().add_annotation(text="Not enough data for disease prevalence by location clusters.",
                                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    else:
        fig_disease_prevalence_location = go.Figure().add_annotation(text="Disease name or location data missing for disease prevalence by location.",
                                                                     xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'mortality_event_map': fig_mortality_map,
        'disease_prevalence_by_location': fig_disease_prevalence_location
    }

    metrics = {
        "total_patients": df['patient_id'].nunique(),
        "total_deaths": deceased_patients['patient_id'].nunique() if not deceased_patients.empty else 0
    }

    return {"metrics": metrics, "plots": plots}

def surgical_and_clinical_procedure_cost_analysis(df):
    expected = ['procedure_id', 'procedure_name', 'total_cost', 'patient_id', 'procedure_date', 'cpt_code']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Surgical and Clinical Procedure Cost Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['procedure_id', 'total_cost'], inplace=True)

    # Average cost per procedure (top 10 most expensive procedures)
    avg_cost_per_procedure = df.groupby('procedure_name')['total_cost'].mean().nlargest(10).reset_index()

    # Total cost distribution by procedure type (if procedure_name provides types)
    total_cost_by_procedure = df.groupby('procedure_name')['total_cost'].sum().nlargest(10).reset_index()

    fig_avg_cost_procedure = px.bar(avg_cost_per_procedure, x='procedure_name', y='total_cost', title='Top 10 Most Expensive Procedures (Average Cost)')
    fig_total_cost_procedure = px.pie(total_cost_by_procedure, names='procedure_name', values='total_cost', title='Total Cost Distribution for Top 10 Procedures')

    plots = {
        'avg_cost_per_procedure': fig_avg_cost_procedure,
        'total_cost_by_procedure_type': fig_total_cost_procedure
    }

    metrics = {
        "total_procedure_cost": df['total_cost'].sum(),
        "num_unique_procedures": df['procedure_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def electronic_health_record_ehr_system_performance_analysis(df):
    expected = ['record_id', 'user_id', 'action_type', 'response_time_ms', 'error_status', 'timestamp']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Electronic Health Record (EHR) System Performance Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df.dropna(subset=['record_id', 'response_time_ms'], inplace=True)

    # Average response time over time (e.g., daily)
    daily_avg_response_time = df.groupby(df['timestamp'].dt.date)['response_time_ms'].mean().reset_index()
    daily_avg_response_time.columns = ['date', 'avg_response_time_ms']

    # Distribution of error statuses
    if 'error_status' in df.columns:
        error_status_counts = df['error_status'].value_counts(normalize=True).reset_index()
        error_status_counts.columns = ['status', 'proportion']
        fig_error_status = px.pie(error_status_counts, names='status', values='proportion', title='Distribution of EHR Error Statuses')
    else:
        fig_error_status = go.Figure().add_annotation(text="Error status data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_daily_response_time = px.line(daily_avg_response_time, x='date', y='avg_response_time_ms', title='Daily Average EHR System Response Time')

    plots = {
        'daily_response_time_trend': fig_daily_response_time,
        'error_status_distribution': fig_error_status
    }

    metrics = {
        "overall_avg_response_time_ms": df['response_time_ms'].mean(),
        "total_ehr_actions": len(df),
        "error_rate_percent": (df[df['error_status'].astype(str).str.lower() != 'success'].shape[0] / len(df)) * 100 if 'error_status' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def patient_experience_and_satisfaction_survey_analysis(df):
    expected = ['survey_id', 'patient_id', 'overall_satisfaction_score', 'question_category', 'score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Experience and Satisfaction Survey Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['survey_id', 'overall_satisfaction_score'], inplace=True)

    # Overall satisfaction score distribution
    overall_satisfaction_dist = df['overall_satisfaction_score'].value_counts(normalize=True).sort_index().reset_index()
    overall_satisfaction_dist.columns = ['score', 'proportion']

    # Average score by question category (if available)
    if 'question_category' in df.columns and 'score' in df.columns:
        avg_score_by_category = df.groupby('question_category')['score'].mean().reset_index()
        fig_avg_score_category = px.bar(avg_score_by_category, x='question_category', y='score', title='Average Score by Survey Question Category')
    else:
        fig_avg_score_category = go.Figure().add_annotation(text="Question category or individual score data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_overall_satisfaction = px.bar(overall_satisfaction_dist, x='score', y='proportion', title='Distribution of Overall Patient Satisfaction Scores')

    plots = {
        'overall_satisfaction_distribution': fig_overall_satisfaction,
        'avg_score_by_question_category': fig_avg_score_category
    }

    metrics = {
        "avg_overall_satisfaction_score": df['overall_satisfaction_score'].mean(),
        "total_surveys_completed": len(df)
    }

    return {"metrics": metrics, "plots": plots}

def emergency_room_wait_time_and_patient_flow_analysis(df):
    expected = ['er_visit_id', 'arrival_time', 'triage_time', 'discharge_time', 'patient_id', 'reason_for_visit']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Emergency Room Wait Time and Patient Flow Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['arrival_time'] = pd.to_datetime(df['arrival_time'], errors='coerce')
    df['triage_time'] = pd.to_datetime(df['triage_time'], errors='coerce')
    df['discharge_time'] = pd.to_datetime(df['discharge_time'], errors='coerce')
    df.dropna(subset=['er_visit_id', 'arrival_time'], inplace=True)

    if 'triage_time' in df.columns:
        df['wait_time_to_triage_minutes'] = (df['triage_time'] - df['arrival_time']).dt.total_seconds() / 60
    if 'discharge_time' in df.columns:
        df['total_er_stay_minutes'] = (df['discharge_time'] - df['arrival_time']).dt.total_seconds() / 60

    # Distribution of wait times to triage
    if 'wait_time_to_triage_minutes' in df.columns:
        fig_wait_time_dist = px.histogram(df, x='wait_time_to_triage_minutes', nbins=50, title='Distribution of ER Wait Time to Triage (Minutes)')
    else:
        fig_wait_time_dist = go.Figure().add_annotation(text="Wait time to triage data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average total ER stay by reason for visit (top 10 reasons)
    if 'reason_for_visit' in df.columns and 'total_er_stay_minutes' in df.columns:
        avg_er_stay_by_reason = df.groupby('reason_for_visit')['total_er_stay_minutes'].mean().nlargest(10).reset_index()
        fig_avg_er_stay = px.bar(avg_er_stay_by_reason, x='reason_for_visit', y='total_er_stay_minutes', title='Average Total ER Stay by Reason for Visit (Top 10)')
    else:
        fig_avg_er_stay = go.Figure().add_annotation(text="Reason for visit or total ER stay data not available.",
                                                     xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'wait_time_to_triage_distribution': fig_wait_time_dist,
        'avg_er_stay_by_reason_for_visit': fig_avg_er_stay
    }

    metrics = {
        "total_er_visits": len(df),
        "avg_wait_time_to_triage_minutes": df['wait_time_to_triage_minutes'].mean() if 'wait_time_to_triage_minutes' in df.columns else 'N/A',
        "avg_total_er_stay_minutes": df['total_er_stay_minutes'].mean() if 'total_er_stay_minutes' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def vaccination_coverage_and_compliance_analysis(df):
    expected = ['patient_id', 'vaccine_type', 'vaccination_date', 'compliance_status', 'age_group']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Vaccination Coverage and Compliance Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'vaccine_type', 'compliance_status'], inplace=True)

    # Vaccination coverage by vaccine type (percentage of patients covered by each vaccine)
    total_patients = df['patient_id'].nunique()
    vaccine_coverage = df.groupby('vaccine_type')['patient_id'].nunique().reset_index(name='vaccinated_patients')
    vaccine_coverage['coverage_percent'] = (vaccine_coverage['vaccinated_patients'] / total_patients) * 100

    # Compliance status distribution
    compliance_dist = df['compliance_status'].value_counts(normalize=True).reset_index()
    compliance_dist.columns = ['status', 'proportion']

    fig_vaccine_coverage = px.bar(vaccine_coverage, x='vaccine_type', y='coverage_percent', title='Vaccination Coverage by Vaccine Type (%)')
    fig_compliance_status = px.pie(compliance_dist, names='status', values='proportion', title='Vaccination Compliance Status Distribution')

    plots = {
        'vaccination_coverage': fig_vaccine_coverage,
        'compliance_status_distribution': fig_compliance_status
    }

    metrics = {
        "total_patients_in_data": total_patients,
        "overall_compliance_rate_percent": compliance_dist[compliance_dist['status'].astype(str).str.lower() == 'compliant']['proportion'].sum() * 100 if 'compliant' in compliance_dist['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def laboratory_test_volume_and_abnormality_rate_analysis(df):
    expected = ['test_id', 'patient_id', 'test_name', 'test_date', 'result_value', 'normal_range_low', 'normal_range_high']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Laboratory Test Volume and Abnormality Rate Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['test_date'] = pd.to_datetime(df['test_date'], errors='coerce')
    df.dropna(subset=['test_id', 'test_name', 'result_value'], inplace=True)

    # Test volume over time (e.g., monthly)
    monthly_test_volume = df.groupby(df['test_date'].dt.to_period('M').dt.start_time).size().reset_index(name='test_count')
    monthly_test_volume.columns = ['month_year', 'test_count']
    monthly_test_volume = monthly_test_volume.sort_values('month_year')

    # Abnormality rate by test type
    df['is_abnormal'] = False
    if 'normal_range_low' in df.columns and 'normal_range_high' in df.columns:
        df['is_abnormal'] = (df['result_value'] < df['normal_range_low']) | (df['result_value'] > df['normal_range_high'])
        abnormality_counts = df.groupby('test_name')['is_abnormal'].sum().reset_index(name='abnormal_count')
        total_test_counts = df.groupby('test_name').size().reset_index(name='total_count')
        abnormality_rates = pd.merge(abnormality_counts, total_test_counts, on='test_name', how='left')
        abnormality_rates['abnormality_rate_percent'] = (abnormality_rates['abnormal_count'] / abnormality_rates['total_count']) * 100
        abnormality_rates.fillna(0, inplace=True)
        fig_abnormality_rates = px.bar(abnormality_rates.nlargest(10, 'abnormality_rate_percent'), x='test_name', y='abnormality_rate_percent', title='Top 10 Tests by Abnormality Rate (%)')
    else:
        fig_abnormality_rates = go.Figure().add_annotation(text="Normal range data not available for abnormality rate calculation.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_monthly_test_volume = px.line(monthly_test_volume, x='month_year', y='test_count', title='Monthly Laboratory Test Volume Trend')

    plots = {
        'monthly_test_volume_trend': fig_monthly_test_volume,
        'abnormality_rates_by_test_type': fig_abnormality_rates
    }

    metrics = {
        "total_tests_performed": len(df),
        "num_unique_test_types": df['test_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def hospital_financial_performance_and_profitability_analysis(df):
    expected = ['financial_record_id', 'revenue_amount', 'expense_amount', 'date', 'department']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Hospital Financial Performance and Profitability Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['financial_record_id', 'revenue_amount', 'expense_amount'], inplace=True)

    df['profit'] = df['revenue_amount'] - df['expense_amount']

    # Monthly revenue and expense trend
    df['month_year'] = df['date'].dt.to_period('M').dt.start_time
    monthly_financials = df.groupby('month_year').agg(
        total_revenue=('revenue_amount', 'sum'),
        total_expense=('expense_amount', 'sum'),
        total_profit=('profit', 'sum')
    ).reset_index().sort_values('month_year')

    # Profitability by department (if department column exists)
    if 'department' in df.columns:
        profit_by_department = df.groupby('department')['profit'].sum().reset_index()
        fig_profit_by_department = px.bar(profit_by_department, x='department', y='profit', title='Profitability by Department')
    else:
        fig_profit_by_department = go.Figure().add_annotation(text="Department data not available for profitability by department.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_monthly_financials = px.line(monthly_financials, x='month_year', y=['total_revenue', 'total_expense', 'total_profit'],
                                     title='Monthly Hospital Revenue, Expense, and Profit Trend')

    plots = {
        'monthly_financial_trend': fig_monthly_financials,
        'profit_by_department': fig_profit_by_department
    }

    metrics = {
        "total_hospital_revenue": df['revenue_amount'].sum(),
        "total_hospital_expense": df['expense_amount'].sum(),
        "total_hospital_profit": df['profit'].sum(),
        "overall_profit_margin_percent": (df['profit'].sum() / df['revenue_amount'].sum()) * 100 if df['revenue_amount'].sum() > 0 else 0
    }

    return {"metrics": metrics, "plots": plots}

def patient_readmission_risk_and_predictive_analysis(df):
    expected = ['patient_id', 'age', 'gender', 'diagnosis', 'num_prior_admissions', 'readmission_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Readmission Risk and Predictive Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'readmission_status'], inplace=True)

    # Readmission status distribution
    readmission_status_counts = df['readmission_status'].value_counts(normalize=True).reset_index()
    readmission_status_counts.columns = ['status', 'proportion']

    # Readmission rate by number of prior admissions (if available)
    if 'num_prior_admissions' in df.columns:
        readmission_by_prior_admissions = df.groupby('num_prior_admissions')['readmission_status'].apply(
            lambda x: (x.astype(str).str.lower() == 'readmitted').mean() * 100
        ).reset_index(name='readmission_rate_percent')
        fig_readmission_by_prior_admissions = px.bar(readmission_by_prior_admissions, x='num_prior_admissions', y='readmission_rate_percent', title='Readmission Rate by Number of Prior Admissions (%)')
    else:
        fig_readmission_by_prior_admissions = go.Figure().add_annotation(text="Number of prior admissions data not available.",
                                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_readmission_status_pie = px.pie(readmission_status_counts, names='status', values='proportion', title='Overall Readmission Status Distribution')

    plots = {
        'readmission_status_pie': fig_readmission_status_pie,
        'readmission_rate_by_prior_admissions': fig_readmission_by_prior_admissions
    }

    metrics = {
        "total_patient_records": len(df),
        "overall_readmission_rate_percent": readmission_status_counts[readmission_status_counts['status'].astype(str).str.lower() == 'readmitted']['proportion'].sum() * 100 if 'readmitted' in readmission_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def provider_specialization_and_patient_load_analysis(df):
    expected = ['provider_id', 'specialization', 'patient_id', 'visit_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Provider Specialization and Patient Load Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['provider_id', 'specialization', 'patient_id'], inplace=True)

    # Number of patients by specialization
    patients_by_specialization = df.groupby('specialization')['patient_id'].nunique().reset_index(name='num_patients')

    # Average patient load per provider by specialization
    provider_patient_counts = df.groupby(['provider_id', 'specialization'])['patient_id'].nunique().reset_index(name='patient_load')
    avg_patient_load_by_specialization = provider_patient_counts.groupby('specialization')['patient_load'].mean().reset_index()

    fig_patients_by_specialization = px.bar(patients_by_specialization, x='specialization', y='num_patients', title='Number of Unique Patients by Specialization')
    fig_avg_patient_load_specialization = px.bar(avg_patient_load_by_specialization, x='specialization', y='patient_load', title='Average Patient Load per Provider by Specialization')

    plots = {
        'patients_by_specialization': fig_patients_by_specialization,
        'average_patient_load_by_specialization': fig_avg_patient_load_specialization
    }

    metrics = {
        "total_providers": df['provider_id'].nunique(),
        "num_unique_specializations": df['specialization'].nunique(),
        "total_patients_seen": df['patient_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def clinical_trial_adverse_event_and_efficacy_analysis(df):
    expected = ['trial_id', 'patient_id', 'treatment_group', 'adverse_event_type', 'event_severity', 'outcome_measure_value']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Clinical Trial Adverse Event and Efficacy Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['trial_id', 'patient_id', 'treatment_group'], inplace=True)

    # Adverse event rate by treatment group
    if 'adverse_event_type' in df.columns:
        total_patients_per_group = df.groupby('treatment_group')['patient_id'].nunique().reset_index(name='total_patients')
        adverse_event_patients_per_group = df[df['adverse_event_type'].notna()].groupby('treatment_group')['patient_id'].nunique().reset_index(name='patients_with_ae')

        ae_rate_merged = pd.merge(total_patients_per_group, adverse_event_patients_per_group, on='treatment_group', how='left').fillna(0)
        ae_rate_merged['ae_rate_percent'] = (ae_rate_merged['patients_with_ae'] / ae_rate_merged['total_patients']) * 100
        fig_ae_rate = px.bar(ae_rate_merged, x='treatment_group', y='ae_rate_percent', title='Adverse Event Rate by Treatment Group (%)')
    else:
        fig_ae_rate = go.Figure().add_annotation(text="Adverse event data not available.",
                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Efficacy measure distribution by treatment group (if outcome_measure_value exists)
    if 'outcome_measure_value' in df.columns:
        fig_efficacy_dist = px.box(df, x='treatment_group', y='outcome_measure_value', title='Outcome Measure Distribution by Treatment Group')
    else:
        fig_efficacy_dist = go.Figure().add_annotation(text="Outcome measure data not available for efficacy distribution.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'adverse_event_rate_by_treatment_group': fig_ae_rate,
        'efficacy_measure_distribution_by_treatment_group': fig_efficacy_dist
    }

    metrics = {
        "total_trial_participants": df['patient_id'].nunique(),
        "num_adverse_events": df['adverse_event_type'].count() if 'adverse_event_type' in df.columns else 0
    }

    return {"metrics": metrics, "plots": plots}

def patient_chronic_condition_and_comorbidity_analysis(df):
    expected = ['patient_id', 'condition_name', 'diagnosis_date', 'number_of_comorbidities']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Chronic Condition and Comorbidity Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'condition_name'], inplace=True)

    # Top 10 chronic conditions
    chronic_condition_counts = df['condition_name'].value_counts().nlargest(10).reset_index()
    chronic_condition_counts.columns = ['condition_name', 'count']

    # Distribution of number of comorbidities (if available)
    if 'number_of_comorbidities' in df.columns:
        comorbidity_dist = df['number_of_comorbidities'].value_counts().sort_index().reset_index()
        comorbidity_dist.columns = ['num_comorbidities', 'count']
        fig_comorbidity_dist = px.bar(comorbidity_dist, x='num_comorbidities', y='count', title='Distribution of Number of Comorbidities per Patient')
    else:
        fig_comorbidity_dist = go.Figure().add_annotation(text="Comorbidity count data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_chronic_conditions = px.bar(chronic_condition_counts, x='condition_name', y='count', title='Top 10 Most Common Chronic Conditions')

    plots = {
        'top_chronic_conditions': fig_chronic_conditions,
        'comorbidity_distribution': fig_comorbidity_dist
    }

    metrics = {
        "total_diagnosed_conditions": len(df),
        "num_unique_chronic_conditions": df['condition_name'].nunique(),
        "avg_comorbidities_per_patient": df['number_of_comorbidities'].mean() if 'number_of_comorbidities' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def medical_device_performance_and_safety_analysis(df):
    expected = ['device_id', 'device_type', 'manufacturer', 'malfunction_rate', 'adverse_event_count', 'usage_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Medical Device Performance and Safety Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['device_id', 'device_type'], inplace=True)

    # Malfunction rates by device type (if available)
    if 'malfunction_rate' in df.columns:
        fig_malfunction_rate = px.bar(df, x='device_type', y='malfunction_rate', title='Malfunction Rate by Device Type (%)')
    else:
        fig_malfunction_rate = go.Figure().add_annotation(text="Malfunction rate data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Adverse event counts by device type (if available)
    if 'adverse_event_count' in df.columns:
        adverse_events_by_type = df.groupby('device_type')['adverse_event_count'].sum().reset_index()
        fig_adverse_events = px.bar(adverse_events_by_type.nlargest(10, 'adverse_event_count'), x='device_type', y='adverse_event_count', title='Top 10 Device Types by Total Adverse Events')
    else:
        fig_adverse_events = go.Figure().add_annotation(text="Adverse event count data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'malfunction_rate_by_device_type': fig_malfunction_rate,
        'adverse_events_by_device_type': fig_adverse_events
    }

    metrics = {
        "total_devices": len(df),
        "num_unique_device_types": df['device_type'].nunique(),
        "total_adverse_events_recorded": df['adverse_event_count'].sum() if 'adverse_event_count' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def hospital_quality_metrics_and_resource_ratio_analysis(df):
    expected = ['metric_name', 'metric_value', 'hospital_id', 'staff_patient_ratio', 'bed_occupancy_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Hospital Quality Metrics and Resource Ratio Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['metric_name', 'metric_value'], inplace=True)

    # Average metric values by metric name
    avg_metric_values = df.groupby('metric_name')['metric_value'].mean().reset_index()

    # Hospital performance across key ratios (if available)
    ratio_data = []
    if 'staff_patient_ratio' in df.columns:
        ratio_data.append(go.Bar(name='Staff-Patient Ratio', x=df['hospital_id'], y=df['staff_patient_ratio']))
    if 'bed_occupancy_rate' in df.columns:
        ratio_data.append(go.Bar(name='Bed Occupancy Rate', x=df['hospital_id'], y=df['bed_occupancy_rate']))

    if ratio_data:
        fig_hospital_ratios = go.Figure(data=ratio_data)
        fig_hospital_ratios.update_layout(barmode='group', title='Hospital Performance Across Key Ratios')
    else:
        fig_hospital_ratios = go.Figure().add_annotation(text="Ratio data (staff-patient or bed occupancy) not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_avg_metric_values = px.bar(avg_metric_values, x='metric_name', y='metric_value', title='Average Values for Key Quality Metrics')

    plots = {
        'average_quality_metric_values': fig_avg_metric_values,
        'hospital_resource_ratios': fig_hospital_ratios
    }

    metrics = {
        "num_unique_metrics": df['metric_name'].nunique(),
        "num_unique_hospitals": df['hospital_id'].nunique() if 'hospital_id' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def insurance_plan_enrollment_and_market_share_analysis(df):
    expected = ['enrollment_id', 'insurance_plan_name', 'enrollment_date', 'patient_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Insurance Plan Enrollment and Market Share Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['enrollment_id', 'insurance_plan_name'], inplace=True)

    # Market share by insurance plan (based on number of enrollments)
    plan_market_share = df['insurance_plan_name'].value_counts(normalize=True).reset_index()
    plan_market_share.columns = ['plan_name', 'market_share_percent']
    plan_market_share['market_share_percent'] = plan_market_share['market_share_percent'] * 100

    # Monthly enrollment trend for top 5 plans
    if 'enrollment_date' in df.columns:
        df['enrollment_date'] = pd.to_datetime(df['enrollment_date'], errors='coerce')
        monthly_enrollment = df.groupby([df['enrollment_date'].dt.to_period('M').dt.start_time, 'insurance_plan_name']).size().reset_index(name='num_enrollments')
        monthly_enrollment.columns = ['month_year', 'insurance_plan_name', 'num_enrollments']

        top_5_plans = plan_market_share['plan_name'].nlargest(5).tolist()
        monthly_enrollment_top5 = monthly_enrollment[monthly_enrollment['insurance_plan_name'].isin(top_5_plans)].sort_values('month_year')

        fig_monthly_enrollment = px.line(monthly_enrollment_top5, x='month_year', y='num_enrollments', color='insurance_plan_name',
                                         title='Monthly Enrollment Trend for Top 5 Insurance Plans')
    else:
        fig_monthly_enrollment = go.Figure().add_annotation(text="Enrollment date data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_market_share = px.pie(plan_market_share, names='plan_name', values='market_share_percent', title='Insurance Plan Market Share by Enrollments (%)')

    plots = {
        'insurance_plan_market_share': fig_market_share,
        'monthly_enrollment_trend_top5_plans': fig_monthly_enrollment
    }

    metrics = {
        "total_enrollments": len(df),
        "num_unique_plans": df['insurance_plan_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def mental_health_therapy_utilization_and_outcome_analysis(df):
    expected = ['patient_id', 'therapy_type', 'session_count', 'outcome_status', 'pre_therapy_score', 'post_therapy_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Mental Health Therapy Utilization and Outcome Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'therapy_type'], inplace=True)

    # Utilization by therapy type (number of sessions or unique patients)
    utilization_by_therapy = df.groupby('therapy_type')['session_count'].sum().reset_index()

    # Average outcome score improvement by therapy type
    if 'pre_therapy_score' in df.columns and 'post_therapy_score' in df.columns:
        df['score_improvement'] = df['post_therapy_score'] - df['pre_therapy_score']
        avg_improvement_by_therapy = df.groupby('therapy_type')['score_improvement'].mean().reset_index()
        fig_avg_improvement = px.bar(avg_improvement_by_therapy, x='therapy_type', y='score_improvement', title='Average Score Improvement by Therapy Type')
    else:
        fig_avg_improvement = go.Figure().add_annotation(text="Pre/Post therapy scores not available for improvement analysis.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_therapy_utilization = px.bar(utilization_by_therapy, x='therapy_type', y='session_count', title='Total Sessions by Therapy Type')

    plots = {
        'therapy_utilization': fig_therapy_utilization,
        'average_score_improvement_by_therapy': fig_avg_improvement
    }

    metrics = {
        "total_therapy_sessions": df['session_count'].sum(),
        "num_unique_therapy_types": df['therapy_type'].nunique(),
        "num_patients_in_therapy": df['patient_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def medical_billing_code_and_charge_amount_analysis(df):
    expected = ['bill_id', 'cpt_code', 'charge_amount', 'patient_id', 'service_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Medical Billing Code and Charge Amount Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['bill_id', 'cpt_code', 'charge_amount'], inplace=True)

    # Top 10 CPT codes by total charge amount
    top_cpt_codes_charges = df.groupby('cpt_code')['charge_amount'].sum().nlargest(10).reset_index()

    # Distribution of charge amounts
    fig_charge_amount_dist = px.histogram(df, x='charge_amount', nbins=50, title='Distribution of Charge Amounts')

    fig_top_cpt_codes = px.bar(top_cpt_codes_charges, x='cpt_code', y='charge_amount', title='Top 10 CPT Codes by Total Charge Amount')

    plots = {
        'top_cpt_codes_by_charge': fig_top_cpt_codes,
        'charge_amount_distribution': fig_charge_amount_dist
    }

    metrics = {
        "total_billed_charges": df['charge_amount'].sum(),
        "num_unique_cpt_codes": df['cpt_code'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def hospital_acquired_infection_rate_analysis(df):
    expected = ['patient_id', 'infection_type', 'hospital_acquired', 'admission_date', 'discharge_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Hospital-Acquired Infection Rate Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'hospital_acquired'], inplace=True)

    # Ensure 'hospital_acquired' is boolean
    df['hospital_acquired'] = df['hospital_acquired'].astype(bool)

    # Overall hospital-acquired infection rate (total HAI / total admissions or patient-days if available)
    total_admissions = df['patient_id'].nunique() # Simplified, might need more robust calculation
    total_hai = df[df['hospital_acquired'] == True]['patient_id'].nunique()
    hai_rate_percent = (total_hai / total_admissions) * 100 if total_admissions > 0 else 0

    # Top 10 hospital-acquired infection types
    if 'infection_type' in df.columns:
        hai_types = df[df['hospital_acquired'] == True]['infection_type'].value_counts().nlargest(10).reset_index()
        hai_types.columns = ['infection_type', 'count']
        fig_hai_types = px.bar(hai_types, x='infection_type', y='count', title='Top 10 Hospital-Acquired Infection Types')
    else:
        fig_hai_types = go.Figure().add_annotation(text="Infection type data not available for HAI types.",
                                                    xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Pie chart for HAI vs. Non-HAI (overall)
    hai_counts = df['hospital_acquired'].value_counts(normalize=True).reset_index()
    hai_counts.columns = ['status', 'proportion']
    hai_counts['status'] = hai_counts['status'].map({True: 'Hospital-Acquired', False: 'Non-Hospital-Acquired'})
    fig_hai_overall = px.pie(hai_counts, names='status', values='proportion', title='Overall Hospital-Acquired Infection Status')

    plots = {
        'overall_hai_status': fig_hai_overall,
        'top_hai_types': fig_hai_types
    }

    metrics = {
        "overall_hai_rate_percent": hai_rate_percent,
        "total_hai_cases": total_hai
    }

    return {"metrics": metrics, "plots": plots}

def patient_transport_and_logistics_analysis(df):
    expected = ['transport_id', 'patient_id', 'transport_date', 'transport_mode', 'transport_duration_minutes', 'origin_location', 'destination_location']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Transport and Logistics Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['transport_date'] = pd.to_datetime(df['transport_date'], errors='coerce')
    df.dropna(subset=['transport_id', 'transport_mode'], inplace=True)

    # Number of transports by mode
    transports_by_mode = df['transport_mode'].value_counts().reset_index()
    transports_by_mode.columns = ['transport_mode', 'count']

    # Average transport duration by mode (if available)
    if 'transport_duration_minutes' in df.columns:
        avg_duration_by_mode = df.groupby('transport_mode')['transport_duration_minutes'].mean().reset_index()
        fig_avg_duration_mode = px.bar(avg_duration_by_mode, x='transport_mode', y='transport_duration_minutes', title='Average Transport Duration by Mode (Minutes)')
    else:
        fig_avg_duration_mode = go.Figure().add_annotation(text="Transport duration data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_transports_by_mode = px.pie(transports_by_mode, names='transport_mode', values='count', title='Patient Transports by Mode')

    plots = {
        'transports_by_mode': fig_transports_by_mode,
        'average_duration_by_mode': fig_avg_duration_mode
    }

    metrics = {
        "total_transports": len(df),
        "avg_transport_duration_minutes": df['transport_duration_minutes'].mean() if 'transport_duration_minutes' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def disease_outbreak_correlation_with_population_density_analysis(df):
    expected = ['location_id', 'disease_case_count', 'population_density', 'outbreak_date', 'disease_name']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Disease Outbreak Correlation with Population Density Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['outbreak_date'] = pd.to_datetime(df['outbreak_date'], errors='coerce')
    df.dropna(subset=['location_id', 'disease_case_count', 'population_density'], inplace=True)

    # Scatter plot: Disease case count vs. population density
    fig_cases_vs_density = px.scatter(df, x='population_density', y='disease_case_count',
                                      title='Disease Case Count vs. Population Density',
                                      hover_name='location_id', color='disease_name' if 'disease_name' in df.columns else None)

    # Top 10 locations by disease case count
    top_outbreak_locations = df.groupby('location_id')['disease_case_count'].sum().nlargest(10).reset_index()

    fig_top_outbreak_locations = px.bar(top_outbreak_locations, x='location_id', y='disease_case_count', title='Top 10 Locations by Total Disease Cases')

    plots = {
        'disease_cases_vs_population_density': fig_cases_vs_density,
        'top_outbreak_locations': fig_top_outbreak_locations
    }

    metrics = {
        "total_disease_cases": df['disease_case_count'].sum(),
        "num_unique_locations": df['location_id'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def nutritional_intervention_and_health_outcome_analysis(df):
    expected = ['patient_id', 'nutritional_intervention_type', 'health_outcome_status', 'weight_change_kg', 'blood_pressure_change']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Nutritional Intervention and Health Outcome Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'nutritional_intervention_type', 'health_outcome_status'], inplace=True)

    # Health outcome status by nutritional intervention type
    outcome_by_intervention = df.groupby(['nutritional_intervention_type', 'health_outcome_status']).size().unstack(fill_value=0)
    fig_outcome_by_intervention = px.bar(outcome_by_intervention, barmode='stack', title='Health Outcome Status by Nutritional Intervention Type')

    # Average weight change by intervention type (if available)
    if 'weight_change_kg' in df.columns:
        avg_weight_change = df.groupby('nutritional_intervention_type')['weight_change_kg'].mean().reset_index()
        fig_avg_weight_change = px.bar(avg_weight_change, x='nutritional_intervention_type', y='weight_change_kg', title='Average Weight Change (kg) by Intervention Type')
    else:
        fig_avg_weight_change = go.Figure().add_annotation(text="Weight change data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'health_outcome_by_intervention': fig_outcome_by_intervention,
        'average_weight_change_by_intervention': fig_avg_weight_change
    }

    metrics = {
        "total_interventions": len(df),
        "num_unique_intervention_types": df['nutritional_intervention_type'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def medical_equipment_inventory_and_maintenance_cost_analysis(df):
    expected = ['equipment_id', 'equipment_type', 'maintenance_cost', 'purchase_cost', 'last_maintenance_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Medical Equipment Inventory and Maintenance Cost Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['equipment_id', 'equipment_type'], inplace=True)

    # Total maintenance cost by equipment type
    if 'maintenance_cost' in df.columns:
        maintenance_cost_by_type = df.groupby('equipment_type')['maintenance_cost'].sum().reset_index()
        fig_maintenance_cost = px.pie(maintenance_cost_by_type, names='equipment_type', values='maintenance_cost', title='Total Maintenance Cost by Equipment Type')
    else:
        fig_maintenance_cost = go.Figure().add_annotation(text="Maintenance cost data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of purchase costs (if available)
    if 'purchase_cost' in df.columns:
        fig_purchase_cost_dist = px.histogram(df, x='purchase_cost', nbins=50, title='Distribution of Equipment Purchase Costs')
    else:
        fig_purchase_cost_dist = go.Figure().add_annotation(text="Purchase cost data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'total_maintenance_cost_by_type': fig_maintenance_cost,
        'purchase_cost_distribution': fig_purchase_cost_dist
    }

    metrics = {
        "total_equipment_items": len(df),
        "num_unique_equipment_types": df['equipment_type'].nunique(),
        "total_maintenance_cost_overall": df['maintenance_cost'].sum() if 'maintenance_cost' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def primary_care_visit_and_wait_time_analysis(df):
    expected = ['visit_id', 'patient_id', 'visit_date', 'clinic_name', 'wait_time_minutes', 'consultation_duration_minutes']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Primary Care Visit and Wait Time Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['visit_date'] = pd.to_datetime(df['visit_date'], errors='coerce')
    df.dropna(subset=['visit_id', 'clinic_name'], inplace=True)

    # Average wait time by clinic
    if 'wait_time_minutes' in df.columns:
        avg_wait_time_by_clinic = df.groupby('clinic_name')['wait_time_minutes'].mean().reset_index()
        fig_avg_wait_time = px.bar(avg_wait_time_by_clinic, x='clinic_name', y='wait_time_minutes', title='Average Wait Time by Clinic (Minutes)')
    else:
        fig_avg_wait_time = go.Figure().add_annotation(text="Wait time data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Monthly visit volume trend
    monthly_visits = df.groupby(df['visit_date'].dt.to_period('M').dt.start_time).size().reset_index(name='num_visits')
    monthly_visits.columns = ['month_year', 'num_visits']
    monthly_visits = monthly_visits.sort_values('month_year')

    fig_monthly_visits = px.line(monthly_visits, x='month_year', y='num_visits', title='Monthly Primary Care Visit Volume Trend')

    plots = {
        'average_wait_time_by_clinic': fig_avg_wait_time,
        'monthly_visit_volume_trend': fig_monthly_visits
    }

    metrics = {
        "total_visits": len(df),
        "avg_overall_wait_time_minutes": df['wait_time_minutes'].mean() if 'wait_time_minutes' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def medication_side_effect_and_adherence_analysis(df):
    expected = ['patient_id', 'medication_name', 'side_effect_reported', 'adherence_score', 'prescription_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Medication Side Effect and Adherence Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'medication_name'], inplace=True)

    # Top 10 reported side effects
    if 'side_effect_reported' in df.columns:
        side_effect_counts = df[df['side_effect_reported'].notna()]['side_effect_reported'].value_counts().nlargest(10).reset_index()
        side_effect_counts.columns = ['side_effect', 'count']
        fig_side_effects = px.bar(side_effect_counts, x='side_effect', y='count', title='Top 10 Reported Medication Side Effects')
    else:
        fig_side_effects = go.Figure().add_annotation(text="Side effect data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Adherence score distribution
    if 'adherence_score' in df.columns:
        fig_adherence_dist = px.histogram(df, x='adherence_score', nbins=50, title='Distribution of Medication Adherence Scores')
    else:
        fig_adherence_dist = go.Figure().add_annotation(text="Adherence score data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'top_reported_side_effects': fig_side_effects,
        'adherence_score_distribution': fig_adherence_dist
    }

    metrics = {
        "total_medication_records": len(df),
        "num_unique_medications": df['medication_name'].nunique(),
        "avg_adherence_score": df['adherence_score'].mean() if 'adherence_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def provider_practice_demographics_analysis(df):
    expected = ['provider_id', 'specialty', 'location', 'years_in_practice', 'patient_load']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Provider Practice Demographics Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['provider_id', 'specialty', 'location'], inplace=True)

    # Number of providers by specialty
    providers_by_specialty = df['specialty'].value_counts().reset_index()
    providers_by_specialty.columns = ['specialty', 'count']

    # Average years in practice by specialty (if available)
    if 'years_in_practice' in df.columns:
        avg_years_in_practice = df.groupby('specialty')['years_in_practice'].mean().reset_index()
        fig_avg_years_practice = px.bar(avg_years_in_practice, x='specialty', y='years_in_practice', title='Average Years in Practice by Specialty')
    else:
        fig_avg_years_practice = go.Figure().add_annotation(text="Years in practice data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_providers_by_specialty = px.pie(providers_by_specialty, names='specialty', values='count', title='Number of Providers by Specialty')

    plots = {
        'providers_by_specialty': fig_providers_by_specialty,
        'average_years_in_practice_by_specialty': fig_avg_years_practice
    }

    metrics = {
        "total_providers": len(df),
        "num_unique_specialties": df['specialty'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def laboratory_test_turnaround_time_analysis(df):
    expected = ['test_id', 'test_name', 'sample_collection_time', 'result_release_time', 'test_type']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Laboratory Test Turnaround Time Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['sample_collection_time'] = pd.to_datetime(df['sample_collection_time'], errors='coerce')
    df['result_release_time'] = pd.to_datetime(df['result_release_time'], errors='coerce')
    df.dropna(subset=['test_id', 'sample_collection_time', 'result_release_time'], inplace=True)

    df['turnaround_time_hours'] = (df['result_release_time'] - df['sample_collection_time']).dt.total_seconds() / 3600

    # Distribution of turnaround times
    fig_tat_distribution = px.histogram(df, x='turnaround_time_hours', nbins=50, title='Distribution of Laboratory Test Turnaround Times (Hours)')

    # Average turnaround time by test type
    if 'test_type' in df.columns:
        avg_tat_by_type = df.groupby('test_type')['turnaround_time_hours'].mean().reset_index()
        fig_avg_tat_by_type = px.bar(avg_tat_by_type, x='test_type', y='turnaround_time_hours', title='Average Turnaround Time by Test Type (Hours)')
    else:
        fig_avg_tat_by_type = go.Figure().add_annotation(text="Test type data not available for average TAT by type.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'turnaround_time_distribution': fig_tat_distribution,
        'average_turnaround_time_by_test_type': fig_avg_tat_by_type
    }

    metrics = {
        "overall_avg_turnaround_time_hours": df['turnaround_time_hours'].mean(),
        "total_tests_with_tat": len(df)
    }

    return {"metrics": metrics, "plots": plots}

def emergency_services_and_response_time_analysis(df):
    expected = ['incident_id', 'response_time_minutes', 'incident_type', 'incident_date', 'patient_outcome_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Emergency Services and Response Time Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['incident_date'] = pd.to_datetime(df['incident_date'], errors='coerce')
    df.dropna(subset=['incident_id', 'response_time_minutes'], inplace=True)

    # Distribution of response times
    fig_response_time_dist = px.histogram(df, x='response_time_minutes', nbins=50, title='Distribution of Emergency Response Times (Minutes)')

    # Average response time by incident type
    if 'incident_type' in df.columns:
        avg_response_by_type = df.groupby('incident_type')['response_time_minutes'].mean().reset_index()
        fig_avg_response_type = px.bar(avg_response_by_type, x='incident_type', y='response_time_minutes', title='Average Response Time by Incident Type (Minutes)')
    else:
        fig_avg_response_type = go.Figure().add_annotation(text="Incident type data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'response_time_distribution': fig_response_time_dist,
        'average_response_time_by_incident_type': fig_avg_response_type
    }

    metrics = {
        "total_incidents": len(df),
        "overall_avg_response_time_minutes": df['response_time_minutes'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def patient_insurance_coverage_and_claim_denials_analysis(df):
    expected = ['patient_id', 'insurance_provider', 'claim_id', 'claim_status', 'denial_reason']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Patient Insurance Coverage and Claim Denials Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'insurance_provider', 'claim_status'], inplace=True)

    # Claim status distribution
    claim_status_counts = df['claim_status'].value_counts(normalize=True).reset_index()
    claim_status_counts.columns = ['status', 'proportion']

    # Top 10 claim denial reasons (if available)
    if 'denial_reason' in df.columns:
        denied_claims = df[df['claim_status'].astype(str).str.lower() == 'denied']
        denial_reasons = denied_claims['denial_reason'].value_counts().nlargest(10).reset_index()
        denial_reasons.columns = ['reason', 'count']
        fig_denial_reasons = px.bar(denial_reasons, x='reason', y='count', title='Top 10 Claim Denial Reasons')
    else:
        fig_denial_reasons = go.Figure().add_annotation(text="Denial reason data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_claim_status = px.pie(claim_status_counts, names='status', values='proportion', title='Claim Status Distribution')

    plots = {
        'claim_status_distribution': fig_claim_status,
        'top_denial_reasons': fig_denial_reasons
    }

    metrics = {
        "total_claims": len(df),
        "denial_rate_percent": claim_status_counts[claim_status_counts['status'].astype(str).str.lower() == 'denied']['proportion'].sum() * 100 if 'denied' in claim_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def provider_staffing_and_patient_load_analysis(df):
    expected = ['provider_id', 'department', 'staff_hours_worked', 'num_patients_seen']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Provider Staffing and Patient Load Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['provider_id', 'department'], inplace=True)

    # Total staff hours by department (if available)
    if 'staff_hours_worked' in df.columns:
        staff_hours_by_dept = df.groupby('department')['staff_hours_worked'].sum().reset_index()
        fig_staff_hours = px.bar(staff_hours_by_dept, x='department', y='staff_hours_worked', title='Total Staff Hours Worked by Department')
    else:
        fig_staff_hours = go.Figure().add_annotation(text="Staff hours worked data not available.",
                                                     xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average patients seen per provider by department (if available)
    if 'num_patients_seen' in df.columns:
        patients_seen_per_provider = df.groupby(['provider_id', 'department'])['num_patients_seen'].sum().reset_index()
        avg_patients_per_provider_by_dept = patients_seen_per_provider.groupby('department')['num_patients_seen'].mean().reset_index()
        fig_avg_patients_seen = px.bar(avg_patients_per_provider_by_dept, x='department', y='num_patients_seen', title='Average Patients Seen per Provider by Department')
    else:
        fig_avg_patients_seen = go.Figure().add_annotation(text="Number of patients seen data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'total_staff_hours_by_department': fig_staff_hours,
        'average_patients_seen_by_department': fig_avg_patients_seen
    }

    metrics = {
        "total_providers": df['provider_id'].nunique(),
        "num_unique_departments": df['department'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def healthcare_facility_distribution_and_service_area_analysis(df):
    expected = ['facility_id', 'facility_type', 'latitude', 'longitude', 'patient_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Healthcare Facility Distribution and Service Area Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['facility_id', 'latitude', 'longitude'], inplace=True)

    # Map of healthcare facilities by type
    fig_facility_map = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="facility_type",
                                         hover_name="facility_id", zoom=3, height=400,
                                         title='Healthcare Facility Distribution by Type')
    fig_facility_map.update_layout(mapbox_style="open-street-map")

    # Number of patients served by facility type (if patient_count available for facility)
    if 'patient_count' in df.columns:
        patients_by_facility_type = df.groupby('facility_type')['patient_count'].sum().reset_index()
        fig_patients_by_facility_type = px.bar(patients_by_facility_type, x='facility_type', y='patient_count', title='Total Patients Served by Facility Type')
    else:
        fig_patients_by_facility_type = go.Figure().add_annotation(text="Patient count data for facilities not available.",
                                                                    xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'healthcare_facility_map': fig_facility_map,
        'patients_served_by_facility_type': fig_patients_by_facility_type
    }

    metrics = {
        "total_facilities": len(df),
        "num_unique_facility_types": df['facility_type'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def clinical_trial_recruitment_and_dropout_rate_analysis(df):
    expected = ['trial_id', 'patient_id', 'recruitment_status', 'dropout_reason', 'enrollment_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Clinical Trial Recruitment and Dropout Rate Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['trial_id', 'patient_id', 'recruitment_status'], inplace=True)

    # Recruitment status distribution (e.g., Screened, Enrolled, Failed Screen, Dropout)
    recruitment_status_dist = df['recruitment_status'].value_counts(normalize=True).reset_index()
    recruitment_status_dist.columns = ['status', 'proportion']

    # Top 10 dropout reasons (if available)
    if 'dropout_reason' in df.columns:
        dropout_reasons = df[df['recruitment_status'].astype(str).str.lower() == 'dropout']['dropout_reason'].value_counts().nlargest(10).reset_index()
        dropout_reasons.columns = ['reason', 'count']
        fig_dropout_reasons = px.bar(dropout_reasons, x='reason', y='count', title='Top 10 Clinical Trial Dropout Reasons')
    else:
        fig_dropout_reasons = go.Figure().add_annotation(text="Dropout reason data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_recruitment_status = px.pie(recruitment_status_dist, names='status', values='proportion', title='Clinical Trial Recruitment Status Distribution')

    plots = {
        'recruitment_status_distribution': fig_recruitment_status,
        'top_dropout_reasons': fig_dropout_reasons
    }

    metrics = {
        "total_records": len(df),
        "dropout_rate_percent": recruitment_status_dist[recruitment_status_dist['status'].astype(str).str.lower() == 'dropout']['proportion'].sum() * 100 if 'dropout' in recruitment_status_dist['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def socioeconomic_factors_in_healthcare_access_analysis(df):
    expected = ['patient_id', 'income_level', 'education_level', 'healthcare_access_score', 'zip_code']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Socioeconomic Factors in Healthcare Access Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id'], inplace=True)

    # Healthcare access score by income level
    if 'income_level' in df.columns and 'healthcare_access_score' in df.columns:
        avg_access_by_income = df.groupby('income_level')['healthcare_access_score'].mean().reset_index()
        fig_access_by_income = px.bar(avg_access_by_income, x='income_level', y='healthcare_access_score', title='Average Healthcare Access Score by Income Level')
    else:
        fig_access_by_income = go.Figure().add_annotation(text="Income level or healthcare access score data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of education levels
    if 'education_level' in df.columns:
        education_dist = df['education_level'].value_counts(normalize=True).reset_index()
        education_dist.columns = ['education_level', 'proportion']
        fig_education_dist = px.pie(education_dist, names='education_level', values='proportion', title='Patient Education Level Distribution')
    else:
        fig_education_dist = go.Figure().add_annotation(text="Education level data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'healthcare_access_by_income_level': fig_access_by_income,
        'patient_education_level_distribution': fig_education_dist
    }

    metrics = {
        "total_patients": len(df),
        "avg_healthcare_access_score": df['healthcare_access_score'].mean() if 'healthcare_access_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def hospital_supply_chain_and_vendor_cost_analysis(df):
    expected = ['item_id', 'vendor_name', 'purchase_price', 'quantity_purchased', 'supply_category', 'purchase_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Hospital Supply Chain and Vendor Cost Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['item_id', 'purchase_price', 'quantity_purchased'], inplace=True)

    df['total_cost'] = df['purchase_price'] * df['quantity_purchased']

    # Total cost by vendor (top 10)
    total_cost_by_vendor = df.groupby('vendor_name')['total_cost'].sum().nlargest(10).reset_index()

    # Total cost by supply category
    if 'supply_category' in df.columns:
        total_cost_by_category = df.groupby('supply_category')['total_cost'].sum().reset_index()
        fig_cost_by_category = px.pie(total_cost_by_category, names='supply_category', values='total_cost', title='Total Supply Cost by Category')
    else:
        fig_cost_by_category = go.Figure().add_annotation(text="Supply category data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_cost_by_vendor = px.bar(total_cost_by_vendor, x='vendor_name', y='total_cost', title='Top 10 Vendors by Total Supply Cost')

    plots = {
        'total_cost_by_vendor': fig_cost_by_vendor,
        'total_cost_by_supply_category': fig_cost_by_category
    }

    metrics = {
        "total_supply_chain_cost": df['total_cost'].sum(),
        "num_unique_vendors": df['vendor_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def disease_specific_cost_of_care_analysis(df):
    expected = ['patient_id', 'disease_name', 'total_cost_of_care', 'treatment_type']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Disease-Specific Cost of Care Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'disease_name', 'total_cost_of_care'], inplace=True)

    # Average cost of care by disease (top 10 diseases)
    avg_cost_by_disease = df.groupby('disease_name')['total_cost_of_care'].mean().nlargest(10).reset_index()

    # Total cost of care distribution by disease (top 10 diseases)
    total_cost_by_disease = df.groupby('disease_name')['total_cost_of_care'].sum().nlargest(10).reset_index()

    fig_avg_cost_by_disease = px.bar(avg_cost_by_disease, x='disease_name', y='total_cost_of_care', title='Average Cost of Care by Disease (Top 10)')
    fig_total_cost_by_disease = px.pie(total_cost_by_disease, names='disease_name', values='total_cost_of_care', title='Total Cost of Care Distribution by Disease (Top 10)')

    plots = {
        'average_cost_by_disease': fig_avg_cost_by_disease,
        'total_cost_distribution_by_disease': fig_total_cost_by_disease
    }

    metrics = {
        "overall_total_cost_of_care": df['total_cost_of_care'].sum(),
        "num_unique_diseases": df['disease_name'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def treatment_trends_by_patient_age_group_analysis(df):
    expected = ['patient_id', 'age', 'treatment_type', 'diagnosis_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Treatment Trends by Patient Age Group Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'age', 'treatment_type'], inplace=True)

    age_bins = [0, 18, 45, 65, 85, df['age'].max() + 1]
    age_labels = ['0-17', '18-44', '45-64', '65-84', '85+']
    df['age_group'] = pd.cut(df['age'], bins=age_bins, labels=age_labels, right=False)

    # Treatment type prevalence by age group
    treatment_by_age_group = df.groupby(['age_group', 'treatment_type']).size().unstack(fill_value=0)
    fig_treatment_by_age = px.bar(treatment_by_age_group, barmode='stack', title='Treatment Type Prevalence by Age Group')

    # Top 5 most common treatments overall
    top_5_treatments = df['treatment_type'].value_counts().nlargest(5).index.tolist()

    # Time trend for one of the top treatments across all age groups (if date is available)
    if 'diagnosis_date' in df.columns:
        df['diagnosis_date'] = pd.to_datetime(df['diagnosis_date'], errors='coerce')
        df['month_year'] = df['diagnosis_date'].dt.to_period('M').dt.start_time

        if top_5_treatments:
            trend_data = df[df['treatment_type'] == top_5_treatments[0]].groupby('month_year').size().reset_index(name='count')
            trend_data = trend_data.sort_values('month_year')
            fig_treatment_trend = px.line(trend_data, x='month_year', y='count', title=f'Monthly Trend for {top_5_treatments[0]} Treatment')
        else:
            fig_treatment_trend = go.Figure().add_annotation(text="No top treatments found for trend analysis.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))
    else:
        fig_treatment_trend = go.Figure().add_annotation(text="Diagnosis date data not available for treatment trend analysis.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'treatment_prevalence_by_age_group': fig_treatment_by_age,
        'monthly_treatment_trend': fig_treatment_trend
    }

    metrics = {
        "total_patients": df['patient_id'].nunique(),
        "num_unique_treatment_types": df['treatment_type'].nunique()
    }

    return {"metrics": metrics, "plots": plots}

def emergency_department_triage_and_patient_outcome_analysis(df):
    expected = ['er_visit_id', 'triage_level', 'discharge_disposition', 'er_wait_time_minutes', 'patient_outcome_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Emergency Department Triage and Patient Outcome Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['er_visit_id', 'triage_level'], inplace=True)

    # Distribution of triage levels
    triage_level_dist = df['triage_level'].value_counts(normalize=True).reset_index()
    triage_level_dist.columns = ['triage_level', 'proportion']

    # Patient outcome status by triage level (if available)
    if 'patient_outcome_status' in df.columns:
        outcome_by_triage = df.groupby(['triage_level', 'patient_outcome_status']).size().unstack(fill_value=0)
        fig_outcome_by_triage = px.bar(outcome_by_triage, barmode='stack', title='Patient Outcome Status by Triage Level')
    else:
        fig_outcome_by_triage = go.Figure().add_annotation(text="Patient outcome status data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    fig_triage_level = px.pie(triage_level_dist, names='triage_level', values='proportion', title='Emergency Department Triage Level Distribution')

    plots = {
        'triage_level_distribution': fig_triage_level,
        'patient_outcome_by_triage_level': fig_outcome_by_triage
    }

    metrics = {
        "total_er_visits": len(df),
        "avg_wait_time_minutes": df['er_wait_time_minutes'].mean() if 'er_wait_time_minutes' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def population_health_risk_assessment_analysis(df):
    expected = ['patient_id', 'risk_score', 'risk_category', 'age', 'chronic_disease_count']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "Population Health Risk Assessment Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['patient_id', 'risk_score'], inplace=True)

    # Distribution of risk scores
    fig_risk_score_dist = px.histogram(df, x='risk_score', nbins=50, title='Distribution of Population Health Risk Scores')

    # Number of patients by risk category
    if 'risk_category' in df.columns:
        risk_category_counts = df['risk_category'].value_counts().reset_index()
        risk_category_counts.columns = ['risk_category', 'count']
        fig_risk_category = px.bar(risk_category_counts, x='risk_category', y='count', title='Number of Patients by Risk Category')
    else:
        fig_risk_category = go.Figure().add_annotation(text="Risk category data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'risk_score_distribution': fig_risk_score_dist,
        'patients_by_risk_category': fig_risk_category
    }

    metrics = {
        "total_patients_assessed": len(df),
        "avg_risk_score": df['risk_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}


# --- General Analysis Functions (as provided earlier, using placeholder names like analyze_sales_performance) ---
def analyze_sales_performance(df):
    expected = ['transaction_id', 'sales_amount', 'salesperson', 'date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {
            "warning": f"Missing columns for Sales Performance: {missing}",
            "key_metrics": get_key_metrics(df)
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])

    total_sales = df['sales_amount'].sum()
    avg_sale = df['sales_amount'].mean()
    top_salesperson = df.groupby('salesperson')['sales_amount'].sum().idxmax()

    hist = px.histogram(df, x='sales_amount', nbins=50, title='Distribution of Sales Amount')
    bar = px.bar(df.groupby('salesperson')['sales_amount'].sum().sort_values(ascending=False).head(10),
                 title='Top 10 Salespeople by Revenue')

    return {
        "metrics": {
            "total_sales": total_sales,
            "avg_sale": avg_sale,
            "top_salesperson": top_salesperson
        },
        "plots": {
            "sales_amount_distribution": hist,
            "top_salespeople": bar
        }
    }

def analyze_customer(df):
    expected = ['customer_id', 'purchase_amount', 'purchase_date', 'customer_segment']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        return {
            "warning": f"Missing columns for Customer Analysis: {missing}",
            "key_metrics": get_key_metrics(df)
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    total_customers = df['customer_id'].nunique()
    avg_purchase = df['purchase_amount'].mean()
    top_segment = df['customer_segment'].mode()[0]

    pie = px.pie(df, names='customer_segment', title='Customer Segment Distribution')
    hist = px.histogram(df, x='purchase_amount', color='customer_segment', barmode='overlay')

    return {
        "metrics": {
            "total_customers": total_customers,
            "avg_purchase": avg_purchase,
            "top_segment": top_segment
        },
        "plots": {
            "segment_distribution": pie,
            "purchase_amount_histogram": hist
        }
    }
def readmission_analysis(df):
    # Assumes: columns = ['readmission_id', 'patient_id', 'readmission_status', 'risk_score']
    df = df.copy()
    df['readmitted'] = df['readmission_status'].str.lower().str.contains('readmitted', na=False)
    df['risk_score'] = pd.to_numeric(df['risk_score'], errors='coerce')
    total_patients = df['patient_id'].nunique()
    readmission_rate = df['readmitted'].mean() * 100
    avg_risk_score = df['risk_score'].mean()
    # Pie
    pie_fig = px.pie(df, names='readmitted', title="Readmission Status Distribution")
    # Histogram
    hist_fig = px.histogram(df, x='risk_score', color='readmitted', barmode='overlay', title="Risk Score by Readmission")
    return {
        "metrics": {"total_patients": total_patients, "readmission_rate": readmission_rate, "avg_risk_score": avg_risk_score},
        "plots": {"readmission_pie": pie_fig, "risk_hist": hist_fig}
    }

def treatment_outcomes(df):
    # Assumes: columns = ['treatment_id', 'treatment_type', 'outcome_score']
    df = df.copy()
    df['outcome_score'] = pd.to_numeric(df['outcome_score'], errors='coerce')
    total = df['treatment_id'].nunique()
    avg_score = df['outcome_score'].mean()
    best_type = df.groupby('treatment_type')['outcome_score'].mean().idxmax()
    # Boxplot
    box_fig = px.box(df, x='treatment_type', y='outcome_score', title="Outcomes by Treatment Type")
    # Barplot
    bar_fig = px.bar(df.groupby('treatment_type')['outcome_score'].mean().reset_index(),
                     x='treatment_type', y='outcome_score', title="Avg Outcome Score per Treatment Type")
    return {
        "metrics": {"total_treatments": total, "avg_outcome_score": avg_score, "best_treatment_type": best_type},
        "plots": {"outcome_box": box_fig, "avg_by_type": bar_fig}
    }

def cost_analysis(df):
    # Assumes: columns = ['patient_id', 'diagnosis_code', 'avg_cost_of_care']
    df = df.copy()
    df['avg_cost_of_care'] = pd.to_numeric(df['avg_cost_of_care'], errors='coerce')
    avg_cost = df['avg_cost_of_care'].mean()
    most_expensive = df.groupby('diagnosis_code')['avg_cost_of_care'].mean().idxmax()
    # Barplot Top 10
    bar_fig = px.bar(
        df.groupby('diagnosis_code')['avg_cost_of_care'].mean().nlargest(10).reset_index(),
        x='diagnosis_code', y='avg_cost_of_care', title="Top 10 Most Expensive Diagnoses")
    # Boxplot
    box_fig = px.box(df, x='diagnosis_code', y='avg_cost_of_care', title="Cost Distribution by Diagnosis")
    return {
        "metrics": {"avg_cost": avg_cost, "most_expensive_diagnosis": most_expensive},
        "plots": {"expensive_bar": bar_fig, "cost_box": box_fig}
    }

def chronic_disease_analysis(df):
    # Assumes: columns = ['patient_id', 'age_group', 'bmi_status', 'chronic_condition_count']
    df = df.copy()
    df['chronic_condition_count'] = pd.to_numeric(df['chronic_condition_count'], errors='coerce')
    total = df['patient_id'].nunique()
    avg_chronic = df['chronic_condition_count'].mean()
    common_bmi = df['bmi_status'].mode()[0]
    # Box
    box_fig = px.box(df, x='age_group', y='chronic_condition_count', title="Chronic Conditions by Age Group")
    # Hist
    hist_fig = px.histogram(df, x='chronic_condition_count', color='bmi_status', title="Chronic Conditions by BMI")
    return {
        "metrics": {"total_patients": total, "avg_chronic_conditions": avg_chronic, "most_common_bmi": common_bmi},
        "plots": {"chronic_box": box_fig, "chronic_hist": hist_fig}
    }

def emergency_cases(df):
    # Assumes: columns = ['hospital_id', 'emergency_type', 'discharge_status']
    df = df.copy()
    total = df['emergency_type'].count()
    home = df[df['discharge_status'].str.contains('Discharged Home', case=False, na=False)].shape[0]
    rate = (home / total) * 100 if total else 0
    # Pie
    pie_fig = px.pie(df, names='discharge_status', title="Discharge Status Distribution")
    # Bar by type
    bar_fig = px.bar(df.groupby('emergency_type')['discharge_status'].value_counts().reset_index(name='count'),
                     x='emergency_type', y='count', color='discharge_status',
                     title="Discharge Status by Emergency Type")
    return {
        "metrics": {"total_visits": total, "discharged_home": home, "discharge_rate": rate},
        "plots": {"status_pie": pie_fig, "by_type_bar": bar_fig}
    }

def medication_analysis(df):
    # Assumes: columns = ['medication_id', 'patient_id', 'adherence_score', 'side_effect_type']
    df = df.copy()
    df['adherence_score'] = pd.to_numeric(df['adherence_score'], errors='coerce')
    total = df['patient_id'].nunique()
    avg_adh = df['adherence_score'].mean()
    common_side = df['side_effect_type'].mode()[0]
    # Hist
    hist_fig = px.histogram(df, x='adherence_score', color='side_effect_type', title="Adherence by Side Effect Type")
    # Pie
    pie_fig = px.pie(df, names='side_effect_type', title="Side Effect Type Distribution")
    return {
        "metrics": {"total_patients": total, "avg_adherence": avg_adh, "most_common_side_effect": common_side},
        "plots": {"adherence_hist": hist_fig, "sideeffect_pie": pie_fig}
    }

def hospital_resources(df):
    # Assumes: columns = ['hospital_id', 'city', 'state', 'bed_count']
    df = df.copy()
    df['bed_count'] = pd.to_numeric(df['bed_count'], errors='coerce')
    total_hosp = df['hospital_id'].nunique()
    total_beds = df['bed_count'].sum()
    avg_beds = df['bed_count'].mean()
    # Bar top 10 states
    bar_fig = px.bar(df.groupby('state')['bed_count'].sum().sort_values(ascending=False).head(10).reset_index(),
                     x='state', y='bed_count', title="Top 10 States by Bed Count")
    # Histogram
    hist_fig = px.histogram(df, x='bed_count', title="Distribution of Hospital Bed Counts")
    return {
        "metrics": {"total_hospitals": total_hosp, "total_beds": total_beds, "avg_beds_per_hospital": avg_beds},
        "plots": {"top_states_bar": bar_fig, "bed_hist": hist_fig}
    }

def patient_satisfaction(df):
    # Assumes: columns = ['survey_id', 'patient_id', 'satisfaction_score', 'feedback_category']
    df = df.copy()
    df['satisfaction_score'] = pd.to_numeric(df['satisfaction_score'], errors='coerce')
    total = df['survey_id'].nunique()
    avg_score = df['satisfaction_score'].mean()
    best_cat = df.groupby('feedback_category')['satisfaction_score'].mean().idxmax()
    # Hist
    hist_fig = px.histogram(df, x='satisfaction_score', title="Distribution of Satisfaction Scores")
    # Bar by category
    bar_fig = px.bar(df.groupby('feedback_category')['satisfaction_score'].mean().reset_index(),
                     x='feedback_category', y='satisfaction_score', title="Satisfaction by Feedback Category")
    return {
        "metrics": {"total_surveys": total, "avg_satisfaction": avg_score, "top_feedback_category": best_cat},
        "plots": {"satisfaction_hist": hist_fig, "by_category_bar": bar_fig}
    }

def infection_control(df):
    # Assumes: columns = ['hospital_id', 'infection_rate', 'hospital_type']
    df = df.copy()
    df['infection_rate'] = pd.to_numeric(df['infection_rate'], errors='coerce')
    avg_rate = df['infection_rate'].mean()
    low_hosp = df.loc[df['infection_rate'].idxmin(), 'hospital_id']
    high_hosp = df.loc[df['infection_rate'].idxmax(), 'hospital_id']
    # Hist
    hist_fig = px.histogram(df, x='infection_rate', color='hospital_type', barmode='overlay', title="Infection Rate by Hospital Type")
    # Bar
    bar_fig = px.bar(df.groupby('hospital_type')['infection_rate'].mean().reset_index(),
                     x='hospital_type', y='infection_rate', title="Average Infection Rate by Hospital Type")
    return {
        "metrics": {"avg_infection_rate": avg_rate, "lowest_rate_hospital": low_hosp, "highest_rate_hospital": high_hosp},
        "plots": {"infection_hist": hist_fig, "hospitaltype_bar": bar_fig}
    }

def patient_demographics_and_disease_prevalence_analysis(df):
    # Assumes: columns = ['patient_id', 'age', 'gender', 'diagnosis_code']
    df = df.copy()
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    total = df['patient_id'].nunique()
    avg_age = df['age'].mean()
    common_diag = df['diagnosis_code'].mode()[0]
    # Hist
    hist_fig = px.histogram(df, x='age', color='gender', barmode='overlay', title="Age Distribution by Gender")
    # Bar
    diag_counts = df['diagnosis_code'].value_counts().nlargest(10).reset_index()
    diag_counts.columns = ['diagnosis_code', 'count']
    bar_fig = px.bar(diag_counts, x='diagnosis_code', y='count', title="Top 10 Diagnoses")
    return {
        "metrics": {"total_patients": total, "avg_age": avg_age, "most_common_diagnosis": common_diag},
        "plots": {"age_hist": hist_fig, "diag_bar": bar_fig}
    }

def hospital_resource_and_capacity_analysis(df):
    # Alias for hospital_resources
    return hospital_resources(df)

def patient_length_of_stay_analysis(df):
    # Assumes: columns = ['patient_id', 'admission_date', 'discharge_date']
    df = df.copy()
    df['admission_date'] = pd.to_datetime(df['admission_date'], errors='coerce')
    df['discharge_date'] = pd.to_datetime(df['discharge_date'], errors='coerce')
    df.dropna(inplace=True)
    df['length_of_stay'] = (df['discharge_date'] - df['admission_date']).dt.days
    total = df['patient_id'].nunique()
    avg_stay = df['length_of_stay'].mean()
    max_stay = df['length_of_stay'].max()
    # Hist
    hist_fig = px.histogram(df, x='length_of_stay', nbins=30, title="Distribution of Length of Stay")
    # Line (if desired, average by month)
    los_by_month = df.groupby(df['admission_date'].dt.to_period('M'))['length_of_stay'].mean().reset_index()
    los_by_month['admission_date'] = los_by_month['admission_date'].astype(str)
    line_fig = px.line(los_by_month, x='admission_date', y='length_of_stay', title="Avg LOS by Month")
    return {
        "metrics": {"total_patients": total, "avg_length_of_stay": avg_stay, "max_length_of_stay": max_stay},
        "plots": {"stay_hist": hist_fig, "los_line": line_fig}
    }

def insurance_claim_and_reimbursement_analysis(df):
    # Assumes: columns = ['claim_id', 'patient_id', 'payer_type', 'billed_amount', 'reimbursement_amount']
    df = df.copy()
    df['billed_amount'] = pd.to_numeric(df['billed_amount'], errors='coerce')
    df['reimbursement_amount'] = pd.to_numeric(df['reimbursement_amount'], errors='coerce')
    total_claims = df['claim_id'].nunique()
    total_billed = df['billed_amount'].sum()
    total_reimbursed = df['reimbursement_amount'].sum()
    reimb_rate = (total_reimbursed / total_billed) * 100 if total_billed else 0
    # Bar by payer
    payer_bar = px.bar(df.groupby('payer_type')['reimbursement_amount'].sum().reset_index(),
                       x='payer_type', y='reimbursement_amount', title="Reimbursement by Payer Type")
    # Boxplot
    df['reimbursement_rate'] = df['reimbursement_amount'] / df['billed_amount']
    box_fig = px.box(df, x='payer_type', y='reimbursement_rate', title="Reimbursement Rate by Payer")
    return {
        "metrics": {"total_claims": total_claims, "total_billed": total_billed, "total_reimbursed": total_reimbursed, "reimbursement_rate": reimb_rate},
        "plots": {"payer_bar": payer_bar, "rate_box": box_fig}}
def patient_demographics(df):
    df = df.copy()
    if 'age' in df.columns:
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df_clean = df.dropna(subset=['age', 'gender'])
    total_patients = df['patient_id'].nunique() if 'patient_id' in df.columns else len(df)
    avg_age = df_clean['age'].mean() if not df_clean.empty else None
    gender_counts = df_clean['gender'].value_counts()
    most_common_gender = gender_counts.idxmax() if not gender_counts.empty else None
    age_hist = px.histogram(df_clean, x='age', nbins=20, title="Age Distribution of Patients")
    gender_pie = px.pie(df_clean, names='gender', title='Gender Distribution of Patients')
    return {
        "metrics": {
            "total_patients": total_patients,
            "average_age": avg_age,
            "most_common_gender": most_common_gender
        },
        "plots": {
            "age_hist": age_hist,
            "gender_pie": gender_pie
        }
    }

    

def main_backend(file, encoding='utf-8', category=None, analysis=None, specific_analysis_name=None):
    df = load_data(file, encoding)
    if df is None:
        return {"error": "Failed to load data"}

    # Mapping for category+analysis style general analyses
    general_analysis_mapping = {
        "Sales Performance": analyze_sales_performance, # Assuming this is mapped from 'sales_analysis'
        "Customer Analysis": analyze_customer,         # Assuming this is mapped from 'customer_analysis'
        # Add other general analysis functions here (e.g., "Inventory Analysis": analyze_inventory)
    }

    # Corrected mapping specific analyses by exact name to functions
    # THIS IS WHERE THE PREVIOUS ERROR WAS (inconsistent naming)
    specific_healthcare_function_mapping = {
        "patient_demographics": patient_demographics,
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
        "hospital_resource_and_capacity_analysis": hospital_resource_and_capacity_analysis,
        "patient_length_of_stay_analysis": patient_length_of_stay_analysis,
        "insurance_claim_and_reimbursement_analysis": insurance_claim_and_reimbursement_analysis,
        "treatment_effectiveness_and_patient_outcome_analysis": treatment_effectiveness_and_patient_outcome_analysis,
        "hospital_staffing_and_turnover_rate_analysis": hospital_staffing_and_turnover_rate_analysis,
        "prescription_drug_utilization_analysis": prescription_drug_utilization_analysis,
        "patient_appointment_scheduling_and_cancellation_analysis": patient_appointment_scheduling_and_cancellation_analysis,
        "geospatial_mortality_rate_and_public_health_analysis": geospatial_mortality_rate_and_public_health_analysis,
        "surgical_and_clinical_procedure_cost_analysis": surgical_and_clinical_procedure_cost_analysis,
        "electronic_health_record_(ehr)_system_performance_analysis": electronic_health_record_ehr_system_performance_analysis,
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
        "hospital-acquired_infection_rate_analysis": hospital_acquired_infection_rate_analysis,
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
    }


    result = None

    if category == "General Sales Analysis" and analysis:
        func = general_analysis_mapping.get(analysis)
        if func:
            try:
                result = func(df)
            except Exception as e:
                result = {"error": str(e), "message": f"Error running analysis '{analysis}'"}
        else:
            # fallback - return key metrics only if no matching function found
            result = {"key_metrics": get_key_metrics(df)}

    # Updated category for healthcare-specific analyses
    elif category == "Specific Healthcare Analysis" and specific_analysis_name: # Changed category name
        func = specific_healthcare_function_mapping.get(specific_analysis_name)
        if func:
            try:
                result = func(df)
            except Exception as e:
                result = {"error": str(e), "message": f"Error running analysis '{specific_analysis_name}' - {e}"}
        else:
            result = {"error": f"Function not found for analysis '{specific_analysis_name}'"}

    else:
        # if no category/analysis provided, return general key metrics
        result = {"key_metrics": get_key_metrics(df)}

    return result