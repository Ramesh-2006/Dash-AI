import pandas as pd
import numpy as np
# Removed: import matplotlib.pyplot as plt
# Removed: import seaborn as sns
# Removed all sklearn imports
from fuzzywuzzy import process
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

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
    "employee_attrition_prediction_and_factor_analysis", # Adjusted to attrition factor analysis
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
def show_key_metrics(df):
    """Display key metrics about the dataset"""
    print("\n--- Key Metrics ---")

    total_employees = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    print(f"Total Employees: {total_employees}")
    print(f"Total Columns: {len(df.columns)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

def show_missing_columns_warning(missing_cols, matched_cols=None):
    print("\n--- ⚠️ Required Columns Not Found ---")
    print("The following columns are needed for this analysis but weren't found in your data:")
    for col in missing_cols:
        match_info = f" (matched to: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""
        print(f" - {col}{match_info}")

def show_general_insights(df, title="General Insights"):
    """Show general data visualizations when no specific analysis is selected"""
    print(f"\n--- {title} ---")

    # Always show key metrics first
    show_key_metrics(df)

    # Numeric columns analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\n--- Numeric Features Analysis ---")
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
        print("\n--- Feature Correlations ---")
        corr = df[numeric_cols].corr()
        fig3 = px.imshow(corr, text_auto=True, aspect="auto",
                                 title="Correlation Between Numeric Features")
        fig3.show()

    # Categorical columns analysis
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\n--- Categorical Features Analysis ---")
        print("Available categorical features:")
        for i, col in enumerate(categorical_cols):
            print(f"{i}: {col}")
        try:
            selected_cat_col_idx = int(input("Select categorical feature to analyze (enter index): "))
            selected_cat_col = categorical_cols[selected_cat_col_idx]

            value_counts = df[selected_cat_col].value_counts().reset_index()
            value_counts.columns = ['Value', 'Count']

            fig4 = px.bar(value_counts.head(10), x='Value', y='Count',
                                  title=f"Top 10 Values in {selected_cat_col}")
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

def attrition_analysis(df):
    print("\n--- Attrition Analysis ---")
    expected = ['employee_id', 'attrition', 'age', 'department', 'tenure']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Ensure 'attrition' is numeric (0/1) for calculations
    if df['attrition'].dtype == 'object':
        df['attrition'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)

    # Metrics
    attrition_rate = df['attrition'].mean() * 100
    avg_tenure = df[df['attrition'] == 1]['tenure'].mean()

    print(f"Attrition Rate: {attrition_rate:.1f}%")
    print(f"Avg Tenure (Leavers): {avg_tenure:.1f} yrs")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='age', color='attrition', barmode='overlay',
                        title="Age Distribution by Attrition Status")
    fig1.show()

    fig2 = px.box(df, x='attrition', y='tenure',
                    title="Tenure Comparison")
    fig2.show()

def performance_analysis(df):
    print("\n--- Performance Analysis ---")
    expected = ['employee_id', 'performance_rating', 'department', 'manager_id']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "Performance Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_rating = df['performance_rating'].mean()
    top_performers = (df['performance_rating'] >= 4).sum() # Assuming rating scale out of 5

    print(f"Avg Rating: {avg_rating:.1f}/5")
    print(f"Number of Top Performers (Rating >= 4): {top_performers}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='performance_rating',
                        title="Performance Distribution")
    fig1.show()

    if 'department' in df:
        fig2 = px.box(df, x='department', y='performance_rating',
                        title="Performance by Department")
        fig2.show()

def compensation_analysis(df):
    print("\n--- Compensation Analysis ---")
    expected = ['employee_id', 'salary', 'bonus', 'department', 'job_level', 'gender']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_salary = df['salary'].mean()
    pay_gap = 0
    if 'gender' in df and len(df['gender'].unique()) > 1:
        gender_salaries = df.groupby('gender')['salary'].mean()
        if 'Female' in gender_salaries.index and 'Male' in gender_salaries.index:
            pay_gap = abs(gender_salaries.get('Female', 0) - gender_salaries.get('Male', 0))

    print(f"Avg Salary: ${avg_salary:,.0f}")
    print(f"Gender Pay Gap: ${pay_gap:,.0f}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.box(df, x='department', y='salary',
                    title="Salary Distribution by Department")
    fig1.show()

    if 'job_level' in df:
        fig2 = px.scatter(df, x='job_level', y='salary', color='gender',
                            title="Salary by Job Level")
        fig2.show()

def diversity_analysis(df):
    print("\n--- Diversity Analysis ---")
    expected = ['employee_id', 'gender', 'department', 'age', 'ethnicity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "Diversity Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    gender_dist = df['gender'].value_counts(normalize=True)
    diversity_score = len(df['ethnicity'].unique()) if 'ethnicity' in df else 0

    print(f"Female Ratio: {gender_dist.get('Female',0)*100:.1f}%")
    print(f"Number of Ethnicities: {diversity_score}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    if 'gender' in df:
        fig1 = px.pie(df, names='gender', title="Gender Distribution")
        fig1.show()

    if 'department' in df and 'gender' in df:
        fig2 = px.bar(df.groupby(['department','gender']).size().reset_index(name='count'),
                      x='department', y='count', color='gender',
                      title="Departmental Gender Distribution")
        fig2.show()

def training_analysis(df):
    print("\n--- Training Analysis ---")
    expected = ['employee_id', 'training_hours', 'training_completed', 'skill_gain']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_hours = df['training_hours'].mean()
    completion_rate = df['training_completed'].mean() * 100

    print(f"Avg Training Hours: {avg_hours:.1f}")
    print(f"Completion Rate: {completion_rate:.1f}%")
    print(f"Employees Trained: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='training_hours',
                        title="Training Hours Distribution")
    fig1.show()

    if 'skill_gain' in df:
        fig2 = px.scatter(df, x='training_hours', y='skill_gain',
                            title="Training Impact on Skills")
        fig2.show()

def engagement_analysis(df):
    print("\n--- Engagement Analysis ---")
    expected = ['employee_id', 'engagement_score', 'department', 'survey_date']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_engagement = df['engagement_score'].mean()
    low_engagement = (df['engagement_score'] < 3).sum() # Assuming score on a scale of 1-5 where <3 is low

    print(f"Avg Engagement: {avg_engagement:.1f}/5")
    print(f"Number of Low Engagement Employees: {low_engagement}")
    print(f"Survey Responses: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='engagement_score',
                        title="Engagement Score Distribution")
    fig1.show()

    if 'department' in df:
        fig2 = px.box(df, x='department', y='engagement_score',
                        title="Engagement by Department")
        fig2.show()

def recruitment_analysis(df):
    print("\n--- Recruitment Analysis ---")
    expected = ['hire_date', 'time_to_hire', 'source', 'department']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Convert dates
    if 'hire_date' in df and not pd.api.types.is_datetime64_any_dtype(df['hire_date']):
        df['hire_date'] = pd.to_datetime(df['hire_date'])

    # Metrics
    avg_time_to_hire = df['time_to_hire'].mean()
    top_source = df['source'].mode()[0] if not df['source'].empty else "N/A"

    print(f"Avg Time to Hire: {avg_time_to_hire:.1f} days")
    print(f"Top Source of Hires: {top_source}")
    print(f"Total Hires: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='time_to_hire',
                        title="Time to Hire Distribution")
    fig1.show()

    if 'source' in df:
        source_counts = df['source'].value_counts().reset_index()
        source_counts.columns = ['Source', 'Count']
        fig2 = px.bar(source_counts, x='Source', y='Count',
                        title="Hires by Source")
        fig2.show()

def productivity_analysis(df):
    print("\n--- Productivity Analysis ---")
    expected = ['employee_id', 'projects_completed', 'productivity_score', 'department']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_productivity = df['productivity_score'].mean()
    top_performers = (df['productivity_score'] >= 4).sum() # Assuming 1-5 scale where 4-5 is high

    print(f"Avg Productivity: {avg_productivity:.1f}/5")
    print(f"Number of Top Performers (Score >= 4): {top_performers}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='productivity_score',
                        title="Productivity Distribution")
    fig1.show()

    if 'projects_completed' in df:
        fig2 = px.scatter(df, x='projects_completed', y='productivity_score',
                            title="Projects Completed vs Productivity Score")
        fig2.show()

def retention_analysis(df):
    print("\n--- Retention Analysis ---")
    expected = ['employee_id', 'tenure', 'retention_risk', 'department']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_tenure = df['tenure'].mean()
    high_risk_employees = (df['retention_risk'] >= 0.7).sum() # Assuming risk score 0-1, >0.7 is high risk

    print(f"Avg Tenure: {avg_tenure:.1f} yrs")
    print(f"Number of High Risk Employees: {high_risk_employees}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='tenure',
                        title="Employee Tenure Distribution")
    fig1.show()

    if 'retention_risk' in df and 'department' in df:
        fig2 = px.box(df, x='department', y='retention_risk',
                        title="Retention Risk by Department")
        fig2.show()

def attendance_analysis(df):
    print("\n--- Attendance Analysis ---")
    expected = ['employee_id', 'absent_days', 'late_arrivals', 'department']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v:k for k,v in matched.items() if v})

    # Metrics
    avg_absent = df['absent_days'].mean()
    problem_employees = (df['absent_days'] > 5).sum() # Employees with > 5 absent days

    print(f"Avg Absent Days: {avg_absent:.1f}")
    print(f"Number of Problem Employees (>5 absent days): {problem_employees}")
    print(f"Total Employees: {len(df)}")

    # Visualizations
    fig1 = px.histogram(df, x='absent_days',
                        title="Absent Days Distribution")
    fig1.show()

    if 'late_arrivals' in df:
        fig2 = px.scatter(df, x='absent_days', y='late_arrivals',
                            title="Absenteeism vs Late Arrivals")
        fig2.show()

def employee_demographic_and_tenure_analysis(df):
    print("\n--- Employee Demographic and Tenure Analysis ---")
    expected = ['birthdate', 'hiredate', 'jobrole', 'gender', 'ethnicity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
    df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
    df.dropna(subset=['birthdate', 'hiredate'], inplace=True)

    current_date = datetime.datetime.now()
    df['age'] = (current_date - df['birthdate']).dt.days / 365.25
    df['tenure_years'] = (current_date - df['hiredate']).dt.days / 365.25

    # Metrics
    avg_age = df['age'].mean()
    avg_tenure = df['tenure_years'].mean()
    most_common_role = df['jobrole'].mode()[0] if not df['jobrole'].empty else "N/A"

    print(f"Average Employee Age: {avg_age:.1f}")
    print(f"Average Tenure (Years): {avg_tenure:.1f}")
    print(f"Most Common Job Role: {most_common_role}")

    # Visualizations
    fig1 = px.histogram(df, x='age', nbins=30, title="Distribution of Employee Ages")
    fig1.show()

    fig2 = px.histogram(df, x='tenure_years', nbins=30, title="Distribution of Employee Tenure")
    fig2.show()

def employee_profile_and_departmental_analysis(df):
    print("\n--- Employee Profile and Departmental Analysis ---")
    expected = ['workdept', 'job', 'edlevel', 'gender', 'ethnicity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})

    # Metrics
    num_departments = df['workdept'].nunique()
    num_jobs = df['job'].nunique()
    avg_ed_level = pd.to_numeric(df['edlevel'], errors='coerce').mean()

    print(f"Number of Departments: {num_departments}")
    print(f"Number of Unique Jobs: {num_jobs}")
    print(f"Average Education Level: {avg_ed_level:.2f}")

    # Visualizations
    dept_counts = df['workdept'].value_counts().reset_index()
    dept_counts.columns = ['Department', 'Count']
    fig1 = px.pie(dept_counts, names='Department', values='Count', title="Employee Distribution by Department", hole=0.4)
    fig1.show()

    if 'edlevel' in df.columns:
        fig2 = px.box(df, x='workdept', y='edlevel', title="Education Level Distribution by Department")
        fig2.show()
    else:
        print("Skipping 'Education Level Distribution by Department' chart: 'edlevel' column not found.")

def employee_compensation_and_tenure_analysis(df):
    print("\n--- Employee Compensation and Tenure Analysis ---")
    expected = ['position', 'dateofhire', 'yearsatcompany', 'monthlyincome', 'overtime']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['yearsatcompany', 'monthlyincome']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_income = df['monthlyincome'].mean()
    avg_tenure = df['yearsatcompany'].mean()
    income_tenure_corr = df['yearsatcompany'].corr(df['monthlyincome'])

    print(f"Average Monthly Income: ${avg_income:,.0f}")
    print(f"Average Tenure (Years): {avg_tenure:.1f}")
    print(f"Income/Tenure Correlation: {income_tenure_corr:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='yearsatcompany', y='monthlyincome', color='overtime',
                        title="Monthly Income vs. Years at Company",
                        labels={'yearsatcompany': 'Years at Company', 'monthlyincome': 'Monthly Income'},
                        trendline='ols', trendline_scope='overall')
    fig1.show()

    if 'position' in df.columns:
        fig2 = px.box(df, x='position', y='monthlyincome', title="Monthly Income by Position")
        fig2.show()
    else:
        print("Skipping 'Monthly Income by Position' chart: 'position' column not found.")

def employee_attrition_prediction_and_factor_analysis(df): # Refactored to Attrition Factor Analysis
    print("\n--- Employee Attrition Factor Analysis ---")
    expected = ['maritalstatus', 'jobrole', 'monthlyincome', 'jobsatisfaction', 'environmentsatisfaction', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['monthlyincome', 'jobsatisfaction', 'environmentsatisfaction']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    attrition_rate = df['attrition_flag'].mean() * 100
    avg_satisfaction_leavers = df[df['attrition_flag'] == 1]['jobsatisfaction'].mean() if 'jobsatisfaction' in df.columns else np.nan
    avg_satisfaction_stayers = df[df['attrition_flag'] == 0]['jobsatisfaction'].mean() if 'jobsatisfaction' in df.columns else np.nan

    print(f"Overall Attrition Rate: {attrition_rate:.2f}%")
    print(f"Avg. Job Satisfaction (Leavers): {avg_satisfaction_leavers:.2f}" if not pd.isna(avg_satisfaction_leavers) else "Avg. Job Satisfaction (Leavers): N/A")
    print(f"Avg. Job Satisfaction (Stayers): {avg_satisfaction_stayers:.2f}" if not pd.isna(avg_satisfaction_stayers) else "Avg. Job Satisfaction (Stayers): N/A")

    # Visualizations
    if 'jobrole' in df.columns and 'attrition_flag' in df.columns:
        attrition_by_role = df.groupby('jobrole')['attrition_flag'].mean().mul(100).sort_values().reset_index()
        fig1 = px.bar(attrition_by_role, x='attrition_flag', y='jobrole', orientation='h', title="Attrition Rate by Job Role")
        fig1.show()
    else:
        print("Skipping 'Attrition Rate by Job Role' chart: 'jobrole' or 'attrition_flag' column not found.")

    if 'monthlyincome' in df.columns and 'maritalstatus' in df.columns and 'attrition' in df.columns:
        fig2 = px.box(df, x='attrition', y='monthlyincome', color='maritalstatus', title="Monthly Income by Attrition and Marital Status")
        fig2.show()
    else:
        print("Skipping 'Monthly Income by Attrition and Marital Status' chart: Missing 'monthlyincome', 'maritalstatus', or 'attrition' column.")

def employee_distribution_and_service_length_analysis(df):
    print("\n--- Employee Distribution and Service Length Analysis ---")
    expected = ['jobtitle', 'storelocation', 'businessunit', 'division', 'lengthofservice']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['lengthofservice'] = pd.to_numeric(df['lengthofservice'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_service_length = df['lengthofservice'].mean()
    top_division = df['division'].mode()[0] if 'division' in df.columns and not df['division'].empty else "N/A"
    top_location = df['storelocation'].mode()[0] if 'storelocation' in df.columns and not df['storelocation'].empty else "N/A"

    print(f"Avg. Length of Service: {avg_service_length:.1f} years")
    print(f"Largest Division: {top_division}")
    print(f"Top Store Location: {top_location}")

    # Visualizations
    if 'division' in df.columns and 'businessunit' in df.columns and 'storelocation' in df.columns and 'lengthofservice' in df.columns:
        fig1 = px.treemap(df, path=[px.Constant("All Employees"), 'division', 'businessunit', 'storelocation'],
                          values='lengthofservice', color='lengthofservice',
                          title="Hierarchical View of Employee Distribution")
        fig1.show()
    else:
        print("Skipping 'Hierarchical View of Employee Distribution' chart: Missing hierarchical columns or 'lengthofservice'.")

    if 'division' in df.columns and 'lengthofservice' in df.columns:
        fig2 = px.violin(df, x='division', y='lengthofservice', box=True, title="Length of Service Distribution by Division")
        fig2.show()
    else:
        print("Skipping 'Length of Service Distribution by Division' chart: 'division' or 'lengthofservice' column not found.")

def employee_performance_and_compensation_analysis(df):
    print("\n--- Employee Performance and Compensation Analysis ---")
    expected = ['jobrole', 'monthlyincome', 'stockoptionlevel', 'performancerating', 'yearsatcompany', 'yearsincurrentrole']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col != 'jobrole' and col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_rating = df['performancerating'].mean() if 'performancerating' in df.columns else np.nan
    avg_income = df['monthlyincome'].mean() if 'monthlyincome' in df.columns else np.nan

    print(f"Average Performance Rating: {avg_rating:.2f}" if not pd.isna(avg_rating) else "Average Performance Rating: N/A")
    print(f"Average Monthly Income: ${avg_income:,.0f}" if not pd.isna(avg_income) else "Average Monthly Income: N/A")

    # Visualizations
    if 'performancerating' in df.columns and 'monthlyincome' in df.columns:
        fig1 = px.box(df, x='performancerating', y='monthlyincome', title="Monthly Income by Performance Rating")
        fig1.show()
    else:
        print("Skipping 'Monthly Income by Performance Rating' chart: Missing 'performancerating' or 'monthlyincome' column.")

    if 'yearsatcompany' in df.columns and 'monthlyincome' in df.columns and 'performancerating' in df.columns and 'stockoptionlevel' in df.columns:
        fig2 = px.scatter(df, x='yearsatcompany', y='monthlyincome', color='performancerating',
                            size='stockoptionlevel', title="Income vs. Tenure (Colored by Performance, Sized by Stock Options)")
        fig2.show()
    else:
        print("Skipping 'Income vs. Tenure' chart: Missing key columns for visualization.")

def employee_salary_and_attrition_analysis(df):
    print("\n--- Employee Salary and Attrition Analysis ---")
    expected = ['jobtitle', 'annualsalary', 'stockoptionlevel', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    df['annualsalary'] = pd.to_numeric(df['annualsalary'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    attrition_rate = df['attrition_flag'].mean() * 100
    avg_salary_leavers = df[df['attrition_flag']==1]['annualsalary'].mean()
    avg_salary_stayers = df[df['attrition_flag']==0]['annualsalary'].mean()

    print(f"Attrition Rate: {attrition_rate:.2f}%")
    print(f"Avg. Salary (Leavers): ${avg_salary_leavers:,.0f}")
    print(f"Avg. Salary (Stayers): ${avg_salary_stayers:,.0f}")

    # Visualizations
    fig1 = px.box(df, x='attrition', y='annualsalary', title="Annual Salary by Attrition Status")
    fig1.show()

    fig2 = px.histogram(df, x='annualsalary', color='attrition', barmode='overlay', title="Salary Distribution by Attrition Status")
    fig2.show()

def employee_salary_hike_and_promotion_factor_analysis(df):
    print("\n--- Employee Salary Hike and Promotion Factor Analysis ---")
    expected = ['education', 'jobrole', 'monthlyincome', 'percentsalaryhike', 'joblevel']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in expected:
        if col != 'jobrole' and col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_hike = df['percentsalaryhike'].mean()
    avg_job_level = df['joblevel'].mean()

    print(f"Average Salary Hike: {avg_hike:.2f}%")
    print(f"Average Job Level: {avg_job_level:.2f}")

    # Visualizations
    fig1 = px.box(df, x='joblevel', y='percentsalaryhike', title="Salary Hike Percentage by Job Level")
    fig1.show()

    fig2 = px.scatter(df, x='monthlyincome', y='percentsalaryhike', color='joblevel',
                        title="Salary Hike vs. Monthly Income")
    fig2.show()

def work_life_balance_and_job_satisfaction_impact_on_attrition(df):
    print("\n--- Work-Life Balance and Job Satisfaction Impact on Attrition ---")
    expected = ['jobsatisfaction', 'worklifebalance', 'totalworkingyears', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    df.dropna(inplace=True)

    print(f"Attrition Rate: {df['attrition_flag'].mean()*100:.2f}%")

    # Visualizations
    fig1 = px.box(df, x='attrition', y='worklifebalance', title="Work-Life Balance Score by Attrition Status")
    fig1.show()

    fig2 = px.box(df, x='attrition', y='jobsatisfaction', title="Job Satisfaction Score by Attrition Status")
    fig2.show()

def commute_distance_and_work_history_impact_on_attrition(df):
    print("\n--- Commute Distance and Work History Impact on Attrition ---")
    expected = ['distancefromhome', 'numcompaniesworked', 'totalworkingyears', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    fig1 = px.box(df, x='attrition', y='distancefromhome', title="Distance From Home by Attrition Status")
    fig1.show()

    fig2 = px.violin(df, x='numcompaniesworked', y='totalworkingyears', color='attrition',
                        title="Work History by Attrition Status")
    fig2.show()

def employee_performance_and_promotion_cycle_analysis(df):
    print("\n--- Employee Performance and Promotion Cycle Analysis ---")
    expected = ['jobinvolvement', 'performancerating', 'yearssincelastpromotion', 'yearsatcompany']
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
    avg_years_since_promo = df['yearssincelastpromotion'].mean()
    avg_perf_rating = df['performancerating'].mean()

    print(f"Avg. Years Since Last Promotion: {avg_years_since_promo:.2f}")
    print(f"Avg. Performance Rating: {avg_perf_rating:.2f}")

    # Visualizations
    fig1 = px.scatter(df, x='yearssincelastpromotion', y='performancerating',
                        title="Performance Rating vs. Years Since Last Promotion",
                        trendline='ols')
    fig1.show()

    fig2 = px.density_heatmap(df, x="yearsatcompany", y="yearssincelastpromotion",
                                 title="Heatmap of Years at Company vs. Years Since Promotion")
    fig2.show()

def employee_demographic_and_compensation_profile_analysis(df):
    print("\n--- Employee Demographic and Compensation Profile Analysis ---")
    expected = ['dob', 'maritalstatus', 'education', 'joblevel', 'jobrole', 'monthlyincome', 'gender'] # Added gender for more complete demographic profile
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (datetime.datetime.now() - df['dob']).dt.days / 365.25
    for col in ['education', 'joblevel', 'monthlyincome']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    fig1 = px.box(df, x='maritalstatus', y='monthlyincome', color='joblevel',
                    title="Monthly Income by Marital Status and Job Level")
    fig1.show()

    fig2 = px.scatter(df, x='age', y='monthlyincome', color='jobrole',
                        title="Age vs. Monthly Income by Job Role")
    fig2.show()

def factors_influencing_employee_attrition_analysis(df):
    print("\n--- Factors Influencing Employee Attrition Analysis ---")
    expected = ['jobrole', 'attrition', 'monthlyincome', 'distancefromhome', 'numcompaniesworked', 'education', 'gender'] # Added education, gender for more factors
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    df.dropna(inplace=True)

    print(f"Overall Attrition Rate: {df['attrition_flag'].mean()*100:.2f}%")

    # Visualizations
    fig1 = px.density_heatmap(df, x='distancefromhome', y='monthlyincome', z='attrition_flag', histfunc='avg',
                                 title="Attrition Rate Heatmap by Distance from Home and Monthly Income")
    fig1.show()

    if 'gender' in df.columns and 'education' in df.columns:
        attrition_by_demographics = df.groupby(['gender', 'education'])['attrition_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(attrition_by_demographics, x='gender', y='attrition_flag', color='education',
                        barmode='group', title="Attrition Rate by Gender and Education Level")
        fig2.show()
    else:
        print("Skipping 'Attrition Rate by Gender and Education Level' chart: Missing 'gender' or 'education' column.")

def employee_demographics_and_attrition_correlation_analysis(df):
    print("\n--- Employee Demographics and Attrition Correlation Analysis ---")
    expected = ['dob', 'hiredate', 'jobrole', 'monthlyincome', 'yearsatcompany', 'attrition', 'gender', 'ethnicity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})

    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')

    df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
    df['age'] = (datetime.datetime.now() - df['dob']).dt.days / 365.25

    for col in ['monthlyincome', 'yearsatcompany']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    if 'jobrole' in df.columns and 'age' in df.columns and 'attrition' in df.columns:
        fig1 = px.violin(df, x='jobrole', y='age', color='attrition', box=True,
                            title="Age Distribution by Job Role and Attrition Status")
        fig1.show()
    else:
        print("Skipping 'Age Distribution by Job Role and Attrition Status' chart: Missing 'jobrole', 'age', or 'attrition' column.")

    if 'yearsatcompany' in df.columns and 'monthlyincome' in df.columns and 'attrition' in df.columns:
        fig2 = px.scatter(df, x='yearsatcompany', y='monthlyincome', color='attrition',
                            title="Income vs. Tenure by Attrition Status")
        fig2.show()
    else:
        print("Skipping 'Income vs. Tenure by Attrition Status' chart: Missing 'yearsatcompany', 'monthlyincome', or 'attrition' column.")

def employee_performance_and_tenure_analysis(df):
    print("\n--- Employee Performance and Tenure Analysis ---")
    expected = ['position', 'startdate', 'manager', 'yearswithcompany', 'performancerating']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['yearswithcompany', 'performancerating']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_rating = df['performancerating'].mean() if 'performancerating' in df.columns else np.nan
    avg_tenure = df['yearswithcompany'].mean() if 'yearswithcompany' in df.columns else np.nan
    corr = df['yearswithcompany'].corr(df['performancerating']) if 'yearswithcompany' in df.columns and 'performancerating' in df.columns else np.nan

    print(f"Average Performance Rating: {avg_rating:.2f}" if not pd.isna(avg_rating) else "Average Performance Rating: N/A")
    print(f"Average Tenure (Years): {avg_tenure:.2f}" if not pd.isna(avg_tenure) else "Average Tenure (Years): N/A")
    print(f"Tenure/Performance Correlation: {corr:.2f}" if not pd.isna(corr) else "Tenure/Performance Correlation: N/A")

    # Visualizations
    if 'yearswithcompany' in df.columns and 'performancerating' in df.columns:
        fig1 = px.scatter(df, x='yearswithcompany', y='performancerating', trendline='ols',
                            title="Performance Rating vs. Years with Company")
        fig1.show()
    else:
        print("Skipping 'Performance Rating vs. Years with Company' chart: Missing 'yearswithcompany' or 'performancerating' column.")

    if 'position' in df.columns and 'performancerating' in df.columns:
        perf_by_position = df.groupby('position')['performancerating'].mean().reset_index()
        fig2 = px.bar(perf_by_position, x='position', y='performancerating', title="Average Performance Rating by Position")
        fig2.show()
    else:
        print("Skipping 'Average Performance Rating by Position' chart: 'position' or 'performancerating' column not found.")

def compensation_promotion_and_career_progression_analysis(df):
    print("\n--- Compensation, Promotion, and Career Progression Analysis ---")
    expected = ['educationfield', 'joblevel', 'jobrole', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['joblevel', 'monthlyincome', 'percentsalaryhike', 'totalworkingyears']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    income_joblevel_corr = df['joblevel'].corr(df['monthlyincome']) if 'joblevel' in df.columns and 'monthlyincome' in df.columns else np.nan
    print(f"Correlation between Job Level and Monthly Income: {income_joblevel_corr:.2f}" if not pd.isna(income_joblevel_corr) else "Correlation between Job Level and Monthly Income: N/A")

    # Visualizations
    if 'totalworkingyears' in df.columns and 'monthlyincome' in df.columns and 'joblevel' in df.columns:
        fig1 = px.scatter(df, x='totalworkingyears', y='monthlyincome', color='joblevel',
                            title="Monthly Income vs. Total Working Years by Job Level")
        fig1.show()
    else:
        print("Skipping 'Monthly Income vs. Total Working Years by Job Level' chart: Missing key columns.")

    if 'joblevel' in df.columns and 'percentsalaryhike' in df.columns:
        hike_by_level = df.groupby('joblevel')['percentsalaryhike'].mean().reset_index()
        fig2 = px.bar(hike_by_level, x='joblevel', y='percentsalaryhike', title="Average Salary Hike % by Job Level")
        fig2.show()
    else:
        print("Skipping 'Average Salary Hike % by Job Level' chart: Missing 'joblevel' or 'percentsalaryhike' column.")

def employee_profile_and_training_engagement_analysis(df):
    print("\n--- Employee Profile and Training Engagement Analysis ---")
    expected = ['maritalstatus', 'jobrole', 'monthlyincome', 'distancefromhome', 'trainingtimeslastyear']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    for col in ['monthlyincome', 'distancefromhome', 'trainingtimeslastyear']:
        if col in df.columns: # Ensure column exists
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_trainings = df['trainingtimeslastyear'].mean() if 'trainingtimeslastyear' in df.columns else np.nan
    print(f"Average Trainings Last Year: {avg_trainings:.2f}" if not pd.isna(avg_trainings) else "Average Trainings Last Year: N/A")

    # Visualizations
    if 'jobrole' in df.columns and 'trainingtimeslastyear' in df.columns:
        training_by_role = df.groupby('jobrole')['trainingtimeslastyear'].mean().reset_index()
        fig1 = px.bar(training_by_role, x='jobrole', y='trainingtimeslastyear', title="Average Trainings by Job Role")
        fig1.show()
    else:
        print("Skipping 'Average Trainings by Job Role' chart: 'jobrole' or 'trainingtimeslastyear' column not found.")

    if 'monthlyincome' in df.columns and 'trainingtimeslastyear' in df.columns and 'maritalstatus' in df.columns:
        fig2 = px.scatter(df, x='monthlyincome', y='trainingtimeslastyear', color='maritalstatus',
                            title="Training Frequency vs. Monthly Income")
        fig2.show()
    else:
        print("Skipping 'Training Frequency vs. Monthly Income' chart: Missing key columns.")

def attrition_factors_related_to_promotions_and_stock_options(df):
    print("\n--- Attrition Factors: Promotions and Stock Options ---")
    expected = ['jobrole', 'yearssincelastpromotion', 'yearsatcompany', 'stockoptionlevel', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['yearssincelastpromotion', 'yearsatcompany', 'stockoptionlevel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    if 'attrition' in df.columns and 'yearssincelastpromotion' in df.columns:
        fig1 = px.box(df, x='attrition', y='yearssincelastpromotion', title="Years Since Last Promotion by Attrition Status")
        fig1.show()
    else:
        print("Skipping 'Years Since Last Promotion by Attrition Status' chart: Missing 'attrition' or 'yearssincelastpromotion' column.")

    if 'stockoptionlevel' in df.columns and 'attrition_flag' in df.columns:
        attrition_by_stock = df.groupby('stockoptionlevel')['attrition_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(attrition_by_stock, x='stockoptionlevel', y='attrition_flag', title="Attrition Rate by Stock Option Level")
        fig2.show()
    else:
        print("Skipping 'Attrition Rate by Stock Option Level' chart: Missing 'stockoptionlevel' or 'attrition_flag' column.")

def comprehensive_employee_satisfaction_and_attrition_analysis(df):
    print("\n--- Comprehensive Employee Satisfaction and Attrition Analysis ---")
    expected = ['jobrole', 'monthlyincome', 'totalworkingyears', 'yearsatcompany', 'worklifebalance', 'jobsatisfaction', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['monthlyincome', 'totalworkingyears', 'yearsatcompany', 'worklifebalance', 'jobsatisfaction']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Create a composite satisfaction score if columns exist
    if 'worklifebalance' in df.columns and 'jobsatisfaction' in df.columns:
        df['satisfaction_score'] = df[['worklifebalance', 'jobsatisfaction']].mean(axis=1)
    else:
        df['satisfaction_score'] = np.nan
        print("Warning: Could not create composite satisfaction score. Missing 'worklifebalance' or 'jobsatisfaction' columns.")

    # Visualizations
    if 'jobsatisfaction' in df.columns and 'worklifebalance' in df.columns and 'attrition_flag' in df.columns:
        fig1 = px.density_heatmap(df, x="jobsatisfaction", y="worklifebalance", z="attrition_flag", histfunc="avg",
                                 title="Heatmap of Attrition Rate by Job Satisfaction and Work-Life Balance")
        fig1.show()
    else:
        print("Skipping 'Heatmap of Attrition Rate by Job Satisfaction and Work-Life Balance' chart: Missing key columns.")

    if 'totalworkingyears' in df.columns and 'satisfaction_score' in df.columns and 'attrition' in df.columns:
        fig2 = px.scatter(df, x='totalworkingyears', y='satisfaction_score', color='attrition',
                            title="Composite Satisfaction vs. Total Working Years by Attrition")
        fig2.show()
    else:
        print("Skipping 'Composite Satisfaction vs. Total Working Years by Attrition' chart: Missing key columns.")

def employee_compensation_structure_and_attrition_analysis(df):
    print("\n--- Employee Compensation Structure and Attrition Analysis ---")
    expected = ['jobrole', 'hourlyrate', 'monthlyincome', 'overtime', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['hourlyrate', 'monthlyincome']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    if 'overtime' in df.columns and 'attrition_flag' in df.columns:
        attrition_by_overtime = df.groupby('overtime')['attrition_flag'].mean().mul(100).reset_index()
        fig1 = px.pie(attrition_by_overtime, names='overtime', values='attrition_flag', hole=0.4,
                        title="Attrition Rate by Overtime Status")
        fig1.show()
    else:
        print("Skipping 'Attrition Rate by Overtime Status' chart: 'overtime' or 'attrition_flag' column not found.")

    if 'jobrole' in df.columns and 'monthlyincome' in df.columns and 'attrition' in df.columns:
        fig2 = px.box(df, x='jobrole', y='monthlyincome', color='attrition',
                        title="Monthly Income by Job Role and Attrition Status")
        fig2.show()
    else:
        print("Skipping 'Monthly Income by Job Role and Attrition Status' chart: Missing 'jobrole', 'monthlyincome', or 'attrition' column.")

def employee_performance_and_career_level_attrition_analysis(df):
    print("\n--- Employee Performance and Career Level Attrition Analysis ---")
    expected = ['joblevel', 'performancerating', 'stockoptionlevel', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['joblevel', 'performancerating', 'stockoptionlevel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    if 'joblevel' in df.columns and 'performancerating' in df.columns and 'attrition_flag' in df.columns:
        attrition_by_level_perf = df.groupby(['joblevel', 'performancerating'])['attrition_flag'].mean().mul(100).reset_index()
        fig1 = px.density_heatmap(attrition_by_level_perf, x='joblevel', y='performancerating', z='attrition_flag',
                                 title="Attrition Rate by Job Level and Performance Rating")
        fig1.show()
    else:
        print("Skipping 'Attrition Rate by Job Level and Performance Rating' chart: Missing key columns.")

    if 'joblevel' in df.columns and 'stockoptionlevel' in df.columns and 'attrition_flag' in df.columns:
        attrition_by_stock_level = df.groupby(['stockoptionlevel', 'joblevel'])['attrition_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(attrition_by_stock_level, x='joblevel', y='attrition_flag', color='stockoptionlevel',
                      barmode='group', title="Attrition Rate by Job Level and Stock Option Level")
        fig2.show()
    else:
        print("Skipping 'Attrition Rate by Job Level and Stock Option Level' chart: Missing key columns.")

def employee_salary_structure_analysis_by_department(df):
    print("\n--- Employee Salary Structure Analysis by Department ---")
    expected = ['jobrole', 'departmentid', 'annualsalary', 'gender']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['annualsalary'] = pd.to_numeric(df['annualsalary'], errors='coerce')
    df.dropna(inplace=True)

    # Metrics
    avg_salary = df['annualsalary'].mean()
    highest_paid_dept = df.groupby('departmentid')['annualsalary'].mean().idxmax() if 'departmentid' in df.columns else "N/A"

    print(f"Average Annual Salary: ${avg_salary:,.0f}")
    print(f"Highest Paying Department: {highest_paid_dept}")

    # Visualizations
    if 'departmentid' in df.columns and 'annualsalary' in df.columns and 'jobrole' in df.columns:
        fig1 = px.box(df, x='departmentid', y='annualsalary', color='jobrole',
                        title="Annual Salary Distribution by Department and Job Role")
        fig1.show()
    else:
        print("Skipping 'Annual Salary Distribution by Department and Job Role' chart: Missing key columns.")

    if 'gender' in df.columns and 'annualsalary' in df.columns and 'departmentid' in df.columns:
        gender_salary_by_dept = df.groupby(['departmentid', 'gender'])['annualsalary'].mean().unstack(fill_value=0)
        fig2 = px.bar(gender_salary_by_dept, x=gender_salary_by_dept.index, y=gender_salary_by_dept.columns,
                        barmode='group', title="Average Annual Salary by Department and Gender")
        fig2.show()
    else:
        print("Skipping 'Average Annual Salary by Department and Gender' chart: Missing 'gender', 'annualsalary', or 'departmentid' column.")

def management_and_its_impact_on_employee_performance_and_attrition(df):
    print("\n--- Management Impact on Performance and Attrition ---")
    expected = ['manager', 'jobrole', 'performancerating', 'attrition', 'employee_id'] # Added employee_id for team_size
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['performancerating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    if 'manager' in df.columns and 'employee_id' in df.columns and 'performancerating' in df.columns and 'attrition_flag' in df.columns:
        manager_kpis = df.groupby('manager').agg(
            team_size=('employee_id', 'count'),
            avg_performance=('performancerating', 'mean'),
            attrition_rate=('attrition_flag', 'mean')
        ).reset_index()
        manager_kpis['attrition_rate'] *= 100

        print("\n--- Manager KPIs ---")
        print(manager_kpis.round(2).to_string())

        # Visualizations
        fig1 = px.scatter(manager_kpis, x='avg_performance', y='attrition_rate', size='team_size',
                            hover_name='manager', title="Team Attrition Rate vs. Average Performance by Manager")
        fig1.show()
    else:
        print("Skipping 'Manager KPIs' analysis: Missing key columns for manager analysis.")

    if 'jobrole' in df.columns and 'performancerating' in df.columns and 'attrition' in df.columns:
        fig2 = px.box(df, x='jobrole', y='performancerating', color='attrition',
                        title="Performance Rating by Job Role and Attrition Status")
        fig2.show()
    else:
        print("Skipping 'Performance Rating by Job Role and Attrition Status' chart: Missing 'jobrole', 'performancerating', or 'attrition' column.")

def job_involvement_and_training_impact_on_employee_retention(df):
    print("\n--- Job Involvement and Training Impact on Retention ---")
    expected = ['jobinvolvement', 'trainingtimeslastyear', 'yearsatcompany', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['retention_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'no' else 0)
    else:
        # Assuming Attrition=1, No Attrition=0, so Retention = 1 - Attrition
        df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['jobinvolvement', 'trainingtimeslastyear', 'yearsatcompany']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    if 'jobinvolvement' in df.columns and 'retention_flag' in df.columns:
        retention_by_involvement = df.groupby('jobinvolvement')['retention_flag'].mean().mul(100).reset_index()
        fig1 = px.bar(retention_by_involvement, x='jobinvolvement', y='retention_flag', title="Retention Rate by Job Involvement Level")
        fig1.show()
    else:
        print("Skipping 'Retention Rate by Job Involvement Level' chart: Missing 'jobinvolvement' or 'retention_flag' column.")

    if 'trainingtimeslastyear' in df.columns and 'retention_flag' in df.columns:
        retention_by_training = df.groupby('trainingtimeslastyear')['retention_flag'].mean().mul(100).reset_index()
        fig2 = px.bar(retention_by_training, x='trainingtimeslastyear', y='retention_flag', title="Retention Rate by Number of Trainings Last Year")
        fig2.show()
    else:
        print("Skipping 'Retention Rate by Number of Trainings Last Year' chart: Missing 'trainingtimeslastyear' or 'retention_flag' column.")



def employee_lifecycle_and_attrition_trend_analysis(df):
    print("\n--- Employee Lifecycle and Attrition Trend Analysis ---")
    expected = ['hiredate', 'education', 'monthlyincome', 'yearsatcompany', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['monthlyincome', 'yearsatcompany']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    df['hire_year'] = df['hiredate'].dt.year
    attrition_by_hire_year = df.groupby('hire_year')['attrition_flag'].mean().mul(100).reset_index()

    # Visualization
    fig1 = px.line(attrition_by_hire_year, x='hire_year', y='attrition_flag', title="Attrition Rate by Hire Year Cohort")
    fig1.show()

    fig2 = px.box(df, x='attrition', y='yearsatcompany', title="Tenure at Company by Attrition Status")
    fig2.show()

def performance_and_workload_impact_on_employee_attrition(df):
    print("\n--- Performance and Workload Impact on Attrition ---")
    expected = ['overtime', 'totalworkingyears', 'performancerating', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['totalworkingyears', 'performancerating']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    fig1 = px.box(df, x='attrition', y='overtime', title="Overtime Status by Attrition Status")
    fig1.show()

    fig2 = px.bar(df.groupby('performancerating')['attrition_flag'].mean().mul(100).reset_index(),
                    x='performancerating', y='attrition_flag', title="Attrition Rate by Performance Rating")
    fig2.show()

def training_and_stock_options_effect_on_employee_retention(df):
    print("\n--- Training and Stock Options' Effect on Retention ---")
    expected = ['stockoptionlevel', 'trainingtimeslastyear', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['retention_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'no' else 0)
    else:
        # Assuming Attrition=1, No Attrition=0, so Retention = 1 - Attrition
        df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['stockoptionlevel', 'trainingtimeslastyear']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Visualizations
    fig1 = px.bar(df.groupby('trainingtimeslastyear')['retention_flag'].mean().mul(100).reset_index(),
                    x='trainingtimeslastyear', y='retention_flag', title="Retention Rate by Number of Trainings Last Year")
    fig1.show()

    fig2 = px.bar(df.groupby('stockoptionlevel')['retention_flag'].mean().mul(100).reset_index(),
                    x='stockoptionlevel', y='retention_flag', title="Retention Rate by Stock Option Level")
    fig2.show()

def performance_rating_correlation_with_employee_attrition(df):
    print("\n--- Performance Rating Correlation with Employee Attrition ---")
    expected = ['performancerating', 'yearsatcompany', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['performancerating', 'yearsatcompany']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    attrition_by_rating = df.groupby('performancerating')['attrition_flag'].mean().mul(100).reset_index()

    print("\n--- Attrition Rate by Performance Rating ---")
    print(attrition_by_rating.to_string())

    # Visualization
    fig = px.bar(attrition_by_rating, x='performancerating', y='attrition_flag',
                    title="Attrition Rate vs. Performance Rating",
                    labels={'performancerating': 'Performance Rating', 'attrition_flag': 'Attrition Rate (%)'})
    fig.show()

def job_satisfaction_determinants_for_employee_retention(df):
    print("\n--- Job Satisfaction Determinants for Employee Retention ---")
    expected = ['jobsatisfaction', 'monthlyincome', 'totalworkingyears', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['retention_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'no' else 0)
    else:
        df['retention_flag'] = 1 - pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['jobsatisfaction', 'monthlyincome', 'totalworkingyears']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    print("\n--- Average Monthly Income by Job Satisfaction Level for Retained Employees ---")
    retained_df = df[df['retention_flag'] == 1]
    income_by_satisfaction = retained_df.groupby('jobsatisfaction')['monthlyincome'].mean().reset_index()
    print(income_by_satisfaction.to_string())

    # Visualization
    fig = px.bar(income_by_satisfaction, x='jobsatisfaction', y='monthlyincome',
                    title="Average Income by Job Satisfaction (Retained Employees)")
    fig.show()

def employee_performance_training_and_attrition_link_analysis(df):
    print("\n--- Employee Performance, Training, and Attrition Link Analysis ---")
    expected = ['performancerating', 'trainingtimeslastyear', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    for col in ['performancerating', 'trainingtimeslastyear']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # Analysis
    attrition_rates = df.groupby(['performancerating', 'trainingtimeslastyear'])['attrition_flag'].mean().mul(100).reset_index()

    # Visualization
    fig = px.density_heatmap(attrition_rates, x='performancerating', y='trainingtimeslastyear', z='attrition_flag',
                                 title="Attrition Rate by Performance and Training Frequency")
    fig.show()

def employee_tenure_and_attrition_risk_analysis(df):
    print("\n--- Employee Tenure and Attrition Risk Analysis ---")
    expected = ['dob', 'hiredate', 'attrition']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['hiredate'] = pd.to_datetime(df['hiredate'], errors='coerce')
    if df['attrition'].dtype == 'object':
        df['attrition_flag'] = df['attrition'].apply(lambda x: 1 if str(x).lower() == 'yes' else 0)
    else:
        df['attrition_flag'] = pd.to_numeric(df['attrition'], errors='coerce')
    df.dropna(inplace=True)

    df['tenure_years'] = (datetime.datetime.now() - df['hiredate']).dt.days / 365.25

    # Visualization
    fig = px.histogram(df, x='tenure_years', color='attrition', barmode='overlay',
                            title="Distribution of Employee Tenure by Attrition Status")
    fig.show()

    # Create tenure bins to analyze risk
    df['tenure_bin'] = pd.cut(df['tenure_years'], bins=[0, 1, 3, 5, 10, 20, 40],
                                 labels=['0-1 Yr', '1-3 Yrs', '3-5 Yrs', '5-10 Yrs', '10-20 Yrs', '20+ Yrs'])
    attrition_by_tenure = df.groupby('tenure_bin')['attrition_flag'].mean().mul(100).reset_index()
    fig2 = px.bar(attrition_by_tenure, x='tenure_bin', y='attrition_flag',
                    title="Attrition Risk by Tenure Group")
    fig2.show()

# ========== MAIN APP ==========
def main():
    """Main function to run the HR Analytics script."""
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

    specific_employee_function_mapping = {
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
        "Work-Life Balance and Job Satisfaction's Effect on Attrition": work_life_balance_and_job_satisfaction_impact_on_attrition, # Points to existing function
        "Employee Lifecycle and Attrition Trend Analysis": employee_lifecycle_and_attrition_trend_analysis,
        "Performance and Workload Impact on Employee Attrition": performance_and_workload_impact_on_employee_attrition,
        "Training and Stock Options' Effect on Employee Retention": training_and_stock_options_effect_on_employee_retention,
        "Performance Rating Correlation with Employee Attrition": performance_rating_correlation_with_employee_attrition,
        "Job Satisfaction Determinants for Employee Retention": job_satisfaction_determinants_for_employee_retention,
        "Employee Performance, Training, and Attrition Link Analysis": employee_performance_training_and_attrition_link_analysis,
        "Employee Tenure and Attrition Risk Analysis": employee_tenure_and_attrition_risk_analysis,
    }

    # --- Analysis Selection ---
    print("\nSelect an Employee Analysis to Perform:")
    all_analysis_names = list(specific_employee_function_mapping.keys())
    for i, name in enumerate(all_analysis_names):
        print(f"{i+1}: {name.replace('_', ' ').title()}") # Nicer display name
    print(f"{len(all_analysis_names)+1}: General Insights (Data Overview)")

    choice_str = input(f"Enter the number of your choice (1-{len(all_analysis_names)+1}): ")
    try:
        choice_idx = int(choice_str) - 1
        if 0 <= choice_idx < len(all_analysis_names):
            selected_analysis_key = all_analysis_names[choice_idx]
            selected_function = specific_employee_function_mapping.get(selected_analysis_key)
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