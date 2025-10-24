import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
from scipy.stats import pearsonr, linregress
# from fuzzywuzzy import process # Uncomment if you want to use fuzzy matching for column names

# Helper functions (adapted to remove Streamlit)
def safe_numeric_conversion(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df.dropna(subset=[column_name])
    print(f"Warning: Column '{column_name}' not found for numeric conversion.")
    return df

def fuzzy_match_column(df, target_columns):
    matched = {}
    available = df.columns.tolist()
    for target in target_columns:
        if target in available:
            matched[target] = target
            continue
        match, score = process.extractOne(target, available) if available else (None, 0)
        matched[target] = match if score >= 70 else None
    return matched

def check_and_rename_columns(df, expected_cols_map):
    missing_cols = []
    renamed_df = df.copy()
    for standard_name, potential_names in expected_cols_map.items():
        found = False
        for p_name in potential_names:
            if p_name in renamed_df.columns:
                if p_name != standard_name:
                    renamed_df = renamed_df.rename(columns={p_name: standard_name})
                found = True
                break
        if not found and standard_name not in renamed_df.columns:
            missing_cols.append(standard_name)
    return renamed_df, missing_cols
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
def show_missing_columns_warning(missing_columns, matched_columns=None):
    print(f"\n--- WARNING: Required Columns Not Found ---")
    print(f"The following columns are needed but missing: {', '.join(missing_columns)}")
    if matched_columns:
        print("Expected column mappings attempted:")
        for key, value in matched_columns.items():
            if value is None:
                print(f"- '{key}' (e.g., '{key}' or a similar variation)")
    print("Analysis might be incomplete or aborted due to missing required data.")

def academic_performance_analysis(df):
    print("\n--- Academic Performance Analysis ---")
    expected = {
        'student_id': ['student_id', 'StudentId', 'ID'],
        'gpa': ['gpa', 'GPA', 'GradePointAverage'],
        'test_score': ['test_score', 'TestScore', 'ExamScore'],
        'attendance_rate': ['attendance_rate', 'AttendanceRate', 'AttendancePct']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
    df['test_score'] = pd.to_numeric(df['test_score'], errors='coerce')
    df['attendance_rate'] = pd.to_numeric(df['attendance_rate'], errors='coerce')
    df = df.dropna(subset=['gpa', 'test_score', 'attendance_rate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_gpa = df['gpa'].mean()
    at_risk = (df['gpa'] < 2.0).sum()
    strong_performers = (df['gpa'] >= 3.5).sum()
    
    print(f"Average GPA: {avg_gpa:.2f}")
    print(f"At-Risk Students (GPA < 2.0): {at_risk}")
    print(f"Strong Performers (GPA >= 3.5): {strong_performers}")
    
    if at_risk > 0:
        print(f"WARNING: {at_risk} At-Risk Students (GPA < 2.0) detected. These students may need academic intervention.")
    
    fig1 = px.scatter(df, x='attendance_rate', y='gpa', title="Attendance Rate vs GPA")
    fig2 = px.scatter(df, x='test_score', y='gpa', title="Test Scores vs GPA")

    return {
        "metrics": {
            "Average GPA": avg_gpa,
            "At-Risk Students": at_risk,
            "Strong Performers": strong_performers
        },
        "figures": {
            "Attendance_vs_GPA_Scatter": fig1,
            "Test_Scores_vs_GPA_Scatter": fig2
        }
    }

def demographic_analysis(df):
    print("\n--- Demographic Analysis ---")
    expected = {
        'student_id': ['student_id', 'StudentId', 'ID'],
        'gender': ['gender', 'Gender', 'Sex'],
        'ethnicity': ['ethnicity', 'Ethnicity', 'Race'],
        'age': ['age', 'Age'],
        'socioeconomic_status': ['socioeconomic_status', 'SES', 'PovertyStatus']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['gender', 'ethnicity', 'age', 'socioeconomic_status'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    gender_dist = df['gender'].value_counts(normalize=True)
    diversity_index = df['ethnicity'].nunique()
    
    print(f"Gender Distribution:\n{gender_dist.to_string()}")
    print(f"Ethnicity Diversity Index (Number of unique ethnicities): {diversity_index}")
    print(f"Total Students: {len(df)}")
    
    fig1 = px.pie(df, names='gender', title="Gender Distribution")
    fig2 = px.bar(df['ethnicity'].value_counts().reset_index(name='count').rename(columns={'index': 'Ethnicity'}),
                  x='Ethnicity', y='count', title="Ethnicity Distribution")

    return {
        "metrics": {
            "Gender Distribution (Normalized)": gender_dist.to_dict(),
            "Ethnicity Diversity Index": diversity_index,
            "Total Students": len(df)
        },
        "figures": {
            "Gender_Distribution_Pie": fig1,
            "Ethnicity_Distribution_Bar": fig2
        }
    }

def course_analysis(df):
    print("\n--- Course Analysis ---")
    expected = {
        'course_id': ['course_id', 'CourseID', 'CourseCode'],
        'enrollment_count': ['enrollment_count', 'EnrollmentCount', 'NumEnrolled'],
        'pass_rate': ['pass_rate', 'PassRate', 'SuccessRate'],
        'instructor': ['instructor', 'InstructorName', 'Faculty']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['enrollment_count'] = pd.to_numeric(df['enrollment_count'], errors='coerce')
    df['pass_rate'] = pd.to_numeric(df['pass_rate'], errors='coerce')
    df = df.dropna(subset=['enrollment_count', 'pass_rate', 'course_id', 'instructor'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_pass_rate = df['pass_rate'].mean()
    challenging_courses = (df['pass_rate'] < 0.6).sum()
    popular_course = df.loc[df['enrollment_count'].idxmax(), 'course_id']
    
    print(f"Average Pass Rate: {avg_pass_rate:.1%}")
    print(f"Number of Challenging Courses (Pass Rate < 60%): {challenging_courses}")
    print(f"Most Popular Course: {popular_course}")
    
    fig1 = px.bar(df.sort_values('pass_rate'), x='course_id', y='pass_rate', title="Course Pass Rates")
    fig2 = px.scatter(df, x='enrollment_count', y='pass_rate', hover_name='course_id', title="Enrollment vs Pass Rate")

    return {
        "metrics": {
            "Average Pass Rate": avg_pass_rate,
            "Challenging Courses": challenging_courses,
            "Most Popular Course": popular_course
        },
        "figures": {
            "Course_Pass_Rates_Bar": fig1,
            "Enrollment_vs_Pass_Rate_Scatter": fig2
        }
    }

def attendance_analysis(df):
    print("\n--- Attendance Analysis ---")
    expected = {
        'student_id': ['student_id', 'StudentId', 'ID'],
        'attendance_rate': ['attendance_rate', 'AttendanceRate', 'AttendancePct'],
        'grade_level': ['grade_level', 'GradeLevel', 'Grade'],
        'absences': ['absences', 'TotalAbsences', 'NumAbsences']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['attendance_rate'] = pd.to_numeric(df['attendance_rate'], errors='coerce')
    df['absences'] = pd.to_numeric(df['absences'], errors='coerce')
    df = df.dropna(subset=['attendance_rate', 'absences'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_attendance = df['attendance_rate'].mean()
    chronic_absentees = (df['attendance_rate'] < 0.8).sum()
    
    print(f"Average Attendance Rate: {avg_attendance:.1%}")
    print(f"Chronic Absentees (Attendance < 80%): {chronic_absentees}")
    
    if chronic_absentees > 0:
        print(f"WARNING: {chronic_absentees} Chronic Absentees (Attendance < 80%) detected. These students may need intervention.")
    
    fig1 = px.histogram(df, x='attendance_rate', nbins=20, title="Attendance Rate Distribution")
    
    fig2 = None
    if 'grade_level' in df.columns:
        fig2 = px.box(df, x='grade_level', y='attendance_rate', title="Attendance by Grade Level")

    return {
        "metrics": {
            "Average Attendance": avg_attendance,
            "Chronic Absentees": chronic_absentees
        },
        "figures": {
            "Attendance_Rate_Distribution_Histogram": fig1,
            "Attendance_by_Grade_Level_Box": fig2
        }
    }

def behavioral_analysis(df):
    print("\n--- Behavioral Analysis ---")
    expected = {
        'student_id': ['student_id', 'StudentId', 'ID'],
        'behavior_incidents': ['behavior_incidents', 'BehaviorIncidents', 'DisciplineIncidents'],
        'interventions': ['interventions', 'NumInterventions', 'InterventionCount'],
        'grade_level': ['grade_level', 'GradeLevel', 'Grade']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['behavior_incidents'] = pd.to_numeric(df['behavior_incidents'], errors='coerce')
    df['interventions'] = pd.to_numeric(df['interventions'], errors='coerce')
    df = df.dropna(subset=['behavior_incidents', 'interventions'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_incidents = df['behavior_incidents'].mean()
    high_incidents = (df['behavior_incidents'] > 5).sum()
    
    print(f"Average Incidents: {avg_incidents:.1f}")
    print(f"High Incident Students (>5 incidents): {high_incidents}")
    
    if high_incidents > 0:
        print(f"WARNING: {high_incidents} Students with >5 Behavioral Incidents detected. These students may need behavioral support.")
    
    fig1 = px.histogram(df, x='behavior_incidents', nbins=20, title="Behavioral Incidents Distribution")
    fig2 = px.scatter(df, x='behavior_incidents', y='interventions', title="Incidents vs Interventions")

    return {
        "metrics": {
            "Average Incidents": avg_incidents,
            "High Incident Students": high_incidents
        },
        "figures": {
            "Behavioral_Incidents_Histogram": fig1,
            "Incidents_vs_Interventions_Scatter": fig2
        }
    }

def program_evaluation(df):
    print("\n--- Program Evaluation ---")
    expected = {
        'program_id': ['program_id', 'ProgramID', 'ProgramName'],
        'participant_count': ['participant_count', 'ParticipantCount', 'NumParticipants'],
        'improvement_score': ['improvement_score', 'ImprovementScore', 'EffectivenessScore'],
        'cost_per_student': ['cost_per_student', 'CostPerStudent', 'ProgramCostPerStudent']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['participant_count'] = pd.to_numeric(df['participant_count'], errors='coerce')
    df['improvement_score'] = pd.to_numeric(df['improvement_score'], errors='coerce')
    df['cost_per_student'] = pd.to_numeric(df['cost_per_student'], errors='coerce')
    df = df.dropna(subset=['participant_count', 'improvement_score', 'cost_per_student', 'program_id'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_improvement = df['improvement_score'].mean()
    
    cost_effective = df.loc[df['improvement_score'].idxmax(), 'program_id'] if not df.empty else None
    expensive_programs = (df['cost_per_student'] > df['cost_per_student'].quantile(0.75)).sum()
    
    print(f"Average Improvement Score: {avg_improvement:.1f}")
    print(f"Most Effective Program (highest improvement score): {cost_effective}")
    print(f"Number of High-Cost Programs (top 25% cost): {expensive_programs}")
    
    fig1 = px.bar(df.sort_values('improvement_score', ascending=False),
                  x='program_id', y='improvement_score', title="Program Improvement Scores")
    fig2 = px.scatter(df, x='cost_per_student', y='improvement_score',
                      hover_name='program_id', size='participant_count', title="Cost vs Effectiveness")

    return {
        "metrics": {
            "Average Improvement": avg_improvement,
            "Most Effective Program": cost_effective,
            "High-Cost Programs": expensive_programs
        },
        "figures": {
            "Program_Improvement_Scores_Bar": fig1,
            "Cost_vs_Effectiveness_Scatter": fig2
        }
    }

def school_district_performance_and_socioeconomic_analysis(df):
    print("\n--- School District Performance and Socioeconomic Analysis ---")
    expected = {
        'district': ['district', 'DistrictName', 'SchoolDistrict'],
        'school': ['school', 'SchoolName'],
        'county': ['county', 'County'],
        'read': ['read', 'ReadingScore', 'AvgReadScore'],
        'math': ['math', 'MathScore', 'AvgMathScore'],
        'socioeconomic_index': ['socioeconomic_index', 'SESIndex', 'PovertyRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['read'] = pd.to_numeric(df['read'], errors='coerce')
    df['math'] = pd.to_numeric(df['math'], errors='coerce')
    df['socioeconomic_index'] = pd.to_numeric(df['socioeconomic_index'], errors='coerce')
    df = df.dropna(subset=['read', 'math', 'socioeconomic_index'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_math = df['math'].mean()
    avg_read = df['read'].mean()
    
    print(f"Average Math Score: {avg_math:.2f}")
    print(f"Average Reading Score: {avg_read:.2f}")

    fig1 = px.box(df, x='county', y=['math', 'read'], title='Test Scores by County')
    fig2 = px.histogram(df, x='district', y='math', color='school', title='Math Scores by District and School')
    
    math_by_district = df.groupby('district')['math'].mean().reset_index()
    fig3 = px.bar(math_by_district.sort_values('math', ascending=False).head(20), x='district', y='math', title='Average Math Score by District (Top 20)')
    
    fig4 = px.scatter(df, x='socioeconomic_index', y=(df['math'] + df['read']) / 2,
                      title='Average Test Score vs. Socioeconomic Index',
                      hover_data=['district', 'school'])

    return {
        "metrics": {
            "Average Math Score": avg_math,
            "Average Reading Score": avg_read
        },
        "figures": {
            "Test_Scores_by_County_Box": fig1,
            "Math_Scores_by_District_and_School_Histogram": fig2,
            "Average_Math_Score_by_District_Bar": fig3,
            "Test_Score_vs_Socioeconomic_Index_Scatter": fig4
        }
    }

def higher_education_institution_cost_of_attendance_analysis(df):
    print("\n--- Higher Education Institution Cost of Attendance Analysis ---")
    expected = {
        'Instnm': ['Instnm', 'InstitutionName', 'Name'],
        'City': ['City', 'CampusCity'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'Year': ['Year', 'AcademicYear'],
        'AverageCostOfAttendance': ['AverageCostOfAttendance', 'CostOfAttendance', 'TotalCost']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AverageCostOfAttendance'] = pd.to_numeric(df['AverageCostOfAttendance'], errors='coerce')
    df = df.dropna(subset=['AverageCostOfAttendance'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_cost = df['AverageCostOfAttendance'].mean()
    most_expensive = df.loc[df['AverageCostOfAttendance'].idxmax(), 'Instnm']
    
    print(f"Average Cost of Attendance: ${avg_cost:,.0f}")
    print(f"Most Expensive Institution: {most_expensive}")

    fig1 = px.histogram(df, x='AverageCostOfAttendance', title='Distribution of Cost of Attendance')
    
    cost_by_state = df.groupby('State')['AverageCostOfAttendance'].mean().reset_index()
    fig2 = px.bar(cost_by_state.sort_values('AverageCostOfAttendance', ascending=False).head(20), x='State', y='AverageCostOfAttendance', title='Average Cost of Attendance by State (Top 20)')
    
    fig3 = px.box(df, x='State', y='AverageCostOfAttendance', title='Cost of Attendance by State')

    return {
        "metrics": {
            "Average Cost of Attendance": avg_cost,
            "Most Expensive Institution": most_expensive
        },
        "figures": {
            "Cost_of_Attendance_Distribution_Histogram": fig1,
            "Average_Cost_by_State_Bar": fig2,
            "Cost_by_State_Box": fig3
        }
    }

def state_level_average_cost_of_attendance_trend_analysis(df):
    print("\n--- State-Level Average Cost of Attendance Trend Analysis ---")
    expected = {
        'STABBR': ['STABBR', 'StateAbbreviation', 'State'],
        'AverageCostOfAttendance': ['AverageCostOfAttendance', 'CostOfAttendance', 'TotalCost'],
        'Year': ['Year', 'AcademicYear']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AverageCostOfAttendance'] = pd.to_numeric(df['AverageCostOfAttendance'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['AverageCostOfAttendance', 'Year'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df_avg = df.groupby(['STABBR', 'Year'])['AverageCostOfAttendance'].mean().reset_index()
    
    fig1 = px.line(df_avg, x='Year', y='AverageCostOfAttendance', color='STABBR', title='Average Cost of Attendance Trend by State')
    
    avg_cost_all_years = df.groupby('STABBR')['AverageCostOfAttendance'].mean().reset_index()
    fig2 = px.bar(avg_cost_all_years.sort_values('AverageCostOfAttendance', ascending=False).head(20), x='STABBR', y='AverageCostOfAttendance', title='Overall Average Cost of Attendance by State (Top 20)')
    
    fig3 = px.box(df, x='Year', y='AverageCostOfAttendance', title='Cost of Attendance Distribution by Year')

    return {
        "metrics": {},
        "figures": {
            "Average_Cost_of_Attendance_Trend_Line": fig1,
            "Overall_Average_Cost_by_State_Bar": fig2,
            "Cost_of_Attendance_Distribution_by_Year_Box": fig3
        }
    }

def university_financials_and_student_outcome_analysis(df):
    print("\n--- University Financials and Student Outcome Analysis ---")
    expected = {
        'INSTNM': ['INSTNM', 'InstitutionName', 'Name'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'TuitionIncome': ['TuitionIncome', 'NetTuitionRevenue', 'TotalTuitionAndFees'],
        'CompletionRate': ['CompletionRate', 'GraduationRate', 'SuccessRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TuitionIncome'] = pd.to_numeric(df['TuitionIncome'], errors='coerce')
    df['CompletionRate'] = pd.to_numeric(df['CompletionRate'], errors='coerce')
    df = df.dropna(subset=['TuitionIncome', 'CompletionRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_completion = df['CompletionRate'].mean()
    highest_tuition_income_inst = df.loc[df['TuitionIncome'].idxmax(), 'INSTNM']
    
    print(f"Average Completion Rate: {avg_completion:.2f}%")
    print(f"Institution with Highest Tuition Income: {highest_tuition_income_inst}")

    fig1 = px.scatter(df, x='TuitionIncome', y='CompletionRate', hover_name='INSTNM', title='Completion Rate vs. Tuition Income')
    
    completion_by_state = df.groupby('State')['CompletionRate'].mean().reset_index()
    fig2 = px.bar(completion_by_state.sort_values('CompletionRate', ascending=False).head(20), x='State', y='CompletionRate', title='Average Completion Rate by State (Top 20)')
    
    fig3 = px.box(df, x='State', y='TuitionIncome', title='Tuition Income Distribution by State')

    return {
        "metrics": {
            "Average Completion Rate": avg_completion,
            "Highest Tuition Income Institution": highest_tuition_income_inst
        },
        "figures": {
            "Completion_Rate_vs_Tuition_Income_Scatter": fig1,
            "Average_Completion_Rate_by_State_Bar": fig2,
            "Tuition_Income_Distribution_by_State_Box": fig3
        }
    }

def university_enrollment_expenditure_and_graduation_rate_analysis(df):
    print("\n--- University Enrollment, Expenditure, and Graduation Rate Analysis ---")
    expected = {
        'INSTNM': ['INSTNM', 'InstitutionName', 'Name'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment'],
        'GraduationRate': ['GraduationRate', 'CompletionRate', 'GradRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['UndergraduateEnrollment', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_enrollment = df['UndergraduateEnrollment'].mean()
    avg_grad_rate = df['GraduationRate'].mean()
    
    print(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

    fig1 = px.scatter(df, x='UndergraduateEnrollment', y='GraduationRate', hover_name='INSTNM', title='Graduation Rate vs. Undergraduate Enrollment')
    
    grad_rate_by_state = df.groupby('State')['GraduationRate'].mean().reset_index()
    fig2 = px.bar(grad_rate_by_state.sort_values('GraduationRate', ascending=False).head(20), x='State', y='GraduationRate', title='Average Graduation Rate by State (Top 20)')
    
    fig3 = px.histogram(df, x='UndergraduateEnrollment', title='Distribution of Undergraduate Enrollment')

    return {
        "metrics": {
            "Average Undergraduate Enrollment": avg_enrollment,
            "Average Graduation Rate": avg_grad_rate
        },
        "figures": {
            "Graduation_Rate_vs_Enrollment_Scatter": fig1,
            "Average_Graduation_Rate_by_State_Bar": fig2,
            "Undergraduate_Enrollment_Distribution_Histogram": fig3
        }
    }

def college_admissions_and_graduation_rate_analysis(df):
    print("\n--- College Admissions and Graduation Rate Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'INSTNM', 'Name'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'City': ['City', 'CampusCity'],
        'GraduationRate': ['GraduationRate', 'CompletionRate', 'GradRate'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    df = df.dropna(subset=['GraduationRate', 'AcceptanceRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_grad_rate = df['GraduationRate'].mean()
    high_grad_rate_college = df.loc[df['GraduationRate'].idxmax(), 'InstitutionName']
    
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
    print(f"College with Highest Graduation Rate: {high_grad_rate_college}")

    fig1 = px.bar(df.groupby('State')['GraduationRate'].mean().nlargest(20).reset_index(), x='State', y='GraduationRate', title='Average Graduation Rate by State (Top 20)')
    fig2 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')
    fig3 = px.box(df, x='State', y='GraduationRate', title='Graduation Rate Distribution by State')
    fig4 = px.scatter(df, x='AcceptanceRate', y='GraduationRate', hover_name='InstitutionName', title='Graduation Rate vs. Acceptance Rate')

    return {
        "metrics": {
            "Average Graduation Rate": avg_grad_rate,
            "Highest Graduation Rate College": high_grad_rate_college
        },
        "figures": {
            "Average_Graduation_Rate_by_State_Bar": fig1,
            "Graduation_Rate_Distribution_Histogram": fig2,
            "Graduation_Rate_Distribution_by_State_Box": fig3,
            "Graduation_Rate_vs_Acceptance_Rate_Scatter": fig4
        }
    }

def school_level_student_teacher_ratio_and_class_size_analysis(df):
    print("\n--- School-Level Student-Teacher Ratio and Class Size Analysis ---")
    expected = {
        'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
        'SchoolName': ['SchoolName', 'Name'],
        'StudentTeacherRatio': ['StudentTeacherRatio', 'STRatio', 'Ratio'],
        'AverageClassSize': ['AverageClassSize', 'AvgClassSize', 'ClassSize']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StudentTeacherRatio'] = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')
    df['AverageClassSize'] = pd.to_numeric(df['AverageClassSize'], errors='coerce')
    df = df.dropna(subset=['StudentTeacherRatio', 'AverageClassSize'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_ratio = df['StudentTeacherRatio'].mean()
    min_ratio_school = df.loc[df['StudentTeacherRatio'].idxmin(), 'SchoolName']
    
    print(f"Average Student-Teacher Ratio: {avg_ratio:.2f}")
    print(f"School with Lowest Student-Teacher Ratio: {min_ratio_school}")

    fig1 = px.histogram(df, x='StudentTeacherRatio', nbins=30, title='Distribution of Student-Teacher Ratios')
    fig2 = px.box(df, y='StudentTeacherRatio', title='Student-Teacher Ratio Distribution')
    fig3 = px.bar(df.sort_values('StudentTeacherRatio').head(20), x='SchoolName', y='StudentTeacherRatio', title='Schools with the Lowest Student-Teacher Ratios (Top 20)')
    fig4 = px.scatter(df, x='StudentTeacherRatio', y='AverageClassSize', hover_name='SchoolName', title='Average Class Size vs. Student-Teacher Ratio')

    return {
        "metrics": {
            "Average Student-Teacher Ratio": avg_ratio,
            "School with Lowest Ratio": min_ratio_school
        },
        "figures": {
            "Student_Teacher_Ratio_Distribution_Histogram": fig1,
            "Student_Teacher_Ratio_Distribution_Box": fig2,
            "Lowest_Student_Teacher_Ratios_Bar": fig3,
            "Class_Size_vs_Ratio_Scatter": fig4
        }
    }

def college_enrollment_and_income_trend_analysis(df):
    print("\n--- College Enrollment and Income Trend Analysis ---")
    expected = {
        'CollegeID': ['CollegeID', 'InstitutionID', 'ID'],
        'CollegeName': ['CollegeName', 'InstitutionName', 'Name'],
        'GraduateIncome': ['GraduateIncome', 'PostGradEarnings', 'MedianEarnings'],
        'EnrollmentYear': ['EnrollmentYear', 'Year', 'AcademicYear'],
        'TotalEnrollment': ['TotalEnrollment', 'Enrollment', 'StudentCount']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['GraduateIncome'] = pd.to_numeric(df['GraduateIncome'], errors='coerce')
    df['EnrollmentYear'] = pd.to_numeric(df['EnrollmentYear'], errors='coerce')
    df['TotalEnrollment'] = pd.to_numeric(df['TotalEnrollment'], errors='coerce')
    df = df.dropna(subset=['GraduateIncome', 'EnrollmentYear', 'TotalEnrollment'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_income = df['GraduateIncome'].mean()
    top_income_college = df.loc[df['GraduateIncome'].idxmax(), 'CollegeName']
    
    print(f"Average Graduate Income: ${avg_income:,.0f}")
    print(f"College with Highest Graduate Income: {top_income_college}")

    avg_income_by_year = df.groupby('EnrollmentYear')['GraduateIncome'].mean().reset_index()
    fig1 = px.line(avg_income_by_year, x='EnrollmentYear', y='GraduateIncome', title='Average Graduate Income Trend Over Time')
    
    fig2 = px.box(df, x='EnrollmentYear', y='GraduateIncome', title='Graduate Income Distribution by Enrollment Year')
    
    fig3 = px.bar(df.groupby('CollegeName')['GraduateIncome'].mean().nlargest(20).reset_index(), x='CollegeName', y='GraduateIncome', title='Top 20 Colleges by Average Graduate Income')
    
    enrollment_by_year = df.groupby('EnrollmentYear')['TotalEnrollment'].sum().reset_index()
    fig4 = px.line(enrollment_by_year, x='EnrollmentYear', y='TotalEnrollment', title='Total Enrollment Trend Over Time')

    return {
        "metrics": {
            "Average Graduate Income": avg_income,
            "Highest Graduate Income College": top_income_college
        },
        "figures": {
            "Average_Graduate_Income_Trend_Line": fig1,
            "Graduate_Income_Distribution_by_Enrollment_Year_Box": fig2,
            "Top_20_Colleges_by_Avg_Graduate_Income_Bar": fig3,
            "Total_Enrollment_Trend_Line": fig4
        }
    }

def school_district_resource_adequacy_analysis(df):
    print("\n--- School District Resource Adequacy Analysis ---")
    expected = {
        'DistrictCode': ['DistrictCode', 'DistrictID', 'ID'],
        'DistrictName': ['DistrictName', 'Name'],
        'AdequacyIndex': ['AdequacyIndex', 'ResourceAdequacyScore', 'AdequacyScore'],
        'PerPupilExpenditure': ['PerPupilExpenditure', 'SpendingPerStudent']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AdequacyIndex'] = pd.to_numeric(df['AdequacyIndex'], errors='coerce')
    df['PerPupilExpenditure'] = pd.to_numeric(df['PerPupilExpenditure'], errors='coerce')
    df = df.dropna(subset=['AdequacyIndex', 'PerPupilExpenditure'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_index = df['AdequacyIndex'].mean()
    most_adequate_district = df.loc[df['AdequacyIndex'].idxmax(), 'DistrictName']
    
    print(f"Average Adequacy Index: {avg_index:.2f}")
    print(f"Most Adequate District: {most_adequate_district}")

    fig1 = px.histogram(df, x='AdequacyIndex', title='Distribution of Resource Adequacy Index')
    
    fig2 = px.bar(df.groupby('DistrictName')['AdequacyIndex'].mean().nlargest(20).reset_index(), x='DistrictName', y='AdequacyIndex', title='Top 20 Districts by Adequacy Index')
    
    fig3 = px.box(df, y='AdequacyIndex', title='Adequacy Index Distribution')
    
    fig4 = px.scatter(df, x='PerPupilExpenditure', y='AdequacyIndex', hover_name='DistrictName', title='Resource Adequacy Index vs. Per-Pupil Expenditure')

    return {
        "metrics": {
            "Average Adequacy Index": avg_index,
            "Most Adequate District": most_adequate_district
        },
        "figures": {
            "Resource_Adequacy_Index_Distribution_Histogram": fig1,
            "Top_20_Districts_by_Adequacy_Index_Bar": fig2,
            "Adequacy_Index_Distribution_Box": fig3,
            "Adequacy_Index_vs_Per_Pupil_Expenditure_Scatter": fig4
        }
    }

def higher_education_institution_roi_and_default_rate_analysis(df):
    print("\n--- Higher Education Institution ROI and Default Rate Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'LoanDefaultRate': ['LoanDefaultRate', 'DefaultRate', 'ThreeYearDefaultRate'],
        'ROI': ['ROI', 'ReturnOnInvestment', 'MedianEarningsAfter10Years']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['LoanDefaultRate'] = pd.to_numeric(df['LoanDefaultRate'], errors='coerce')
    df['ROI'] = pd.to_numeric(df['ROI'], errors='coerce')
    df = df.dropna(subset=['LoanDefaultRate', 'ROI'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_default_rate = df['LoanDefaultRate'].mean()
    highest_default_rate_inst = df.loc[df['LoanDefaultRate'].idxmax(), 'Name']
    
    print(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
    print(f"Institution with Highest Default Rate: {highest_default_rate_inst}")

    fig1 = px.histogram(df, x='LoanDefaultRate', title='Distribution of Loan Default Rates')
    fig2 = px.box(df, x='State', y='LoanDefaultRate', title='Loan Default Rate by State')
    
    default_rate_by_state = df.groupby('State')['LoanDefaultRate'].mean().reset_index()
    fig3 = px.bar(default_rate_by_state.sort_values('LoanDefaultRate', ascending=False).head(20), x='State', y='LoanDefaultRate', title='Average Loan Default Rate by State (Top 20)')
    
    fig4 = px.scatter(df, x='ROI', y='LoanDefaultRate', hover_name='Name', title='Loan Default Rate vs. ROI')

    return {
        "metrics": {
            "Average Loan Default Rate": avg_default_rate,
            "Highest Default Rate Institution": highest_default_rate_inst
        },
        "figures": {
            "Loan_Default_Rate_Distribution_Histogram": fig1,
            "Loan_Default_Rate_by_State_Box": fig2,
            "Average_Loan_Default_Rate_by_State_Bar": fig3,
            "Loan_Default_Rate_vs_ROI_Scatter": fig4
        }
    }

def school_district_budget_and_student_outcome_analysis(df):
    print("\n--- School District Budget and Student Outcome Analysis ---")
    expected = {
        'School District': ['School District', 'DistrictName', 'District'],
        'County': ['County'],
        'DropoutRate': ['DropoutRate', 'HighSchoolDropoutRate'],
        'PerPupilExpenditure': ['PerPupilExpenditure', 'SpendingPerStudent', 'BudgetPerStudent']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['DropoutRate'] = pd.to_numeric(df['DropoutRate'], errors='coerce')
    df['PerPupilExpenditure'] = pd.to_numeric(df['PerPupilExpenditure'], errors='coerce')
    df = df.dropna(subset=['DropoutRate', 'PerPupilExpenditure'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_dropout_rate = df['DropoutRate'].mean()
    highest_dropout_district = df.loc[df['DropoutRate'].idxmax(), 'School District']
    
    print(f"Average Dropout Rate: {avg_dropout_rate:.2f}%")
    print(f"Highest Dropout Rate District: {highest_dropout_district}")

    fig1 = px.histogram(df, x='DropoutRate', title='Distribution of Dropout Rates')
    
    dropout_by_county = df.groupby('County')['DropoutRate'].mean().reset_index()
    fig2 = px.bar(dropout_by_county.sort_values('DropoutRate', ascending=False).head(20), x='County', y='DropoutRate', title='Average Dropout Rate by County (Top 20)')
    
    fig3 = px.box(df, x='County', y='DropoutRate', title='Dropout Rate Distribution by County')
    
    fig4 = px.scatter(df, x='PerPupilExpenditure', y='DropoutRate', hover_name='School District', title='Dropout Rate vs. Per-Pupil Expenditure')

    return {
        "metrics": {
            "Average Dropout Rate": avg_dropout_rate,
            "Highest Dropout Rate District": highest_dropout_district
        },
        "figures": {
            "Dropout_Rates_Distribution_Histogram": fig1,
            "Average_Dropout_Rate_by_County_Bar": fig2,
            "Dropout_Rate_Distribution_by_County_Box": fig3,
            "Dropout_Rate_vs_Per_Pupil_Expenditure_Scatter": fig4
        }
    }

def university_selectivity_and_student_debt_analysis(df):
    print("\n--- University Selectivity and Student Debt Analysis ---")
    expected = {
        'UniversityID': ['UniversityID', 'InstID', 'ID'],
        'Name': ['Name', 'UniversityName', 'INSTNM'],
        'PublicPrivate': ['PublicPrivate', 'Control', 'InstitutionType'],
        'AvgStudentDebt': ['AvgStudentDebt', 'MedianDebt', 'StudentLoanDebt'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AvgStudentDebt'] = pd.to_numeric(df['AvgStudentDebt'], errors='coerce')
    df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    df = df.dropna(subset=['AvgStudentDebt', 'AcceptanceRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_debt = df['AvgStudentDebt'].mean()
    most_debt_university = df.loc[df['AvgStudentDebt'].idxmax(), 'Name']
    
    print(f"Average Student Debt: ${avg_debt:,.0f}")
    print(f"University with Highest Student Debt: {most_debt_university}")

    fig1 = px.histogram(df, x='AvgStudentDebt', title='Distribution of Average Student Debt')
    fig2 = px.box(df, x='PublicPrivate', y='AvgStudentDebt', title='Average Student Debt by Institution Type')
    
    avg_debt_by_type = df.groupby('PublicPrivate')['AvgStudentDebt'].mean().reset_index()
    fig3 = px.bar(avg_debt_by_type, x='PublicPrivate', y='AvgStudentDebt', title='Average Student Debt by Institution Type')
    
    fig4 = px.scatter(df, x='AcceptanceRate', y='AvgStudentDebt', hover_name='Name', title='Average Student Debt vs. Acceptance Rate (Selectivity)')

    return {
        "metrics": {
            "Average Student Debt": avg_debt,
            "Highest Student Debt University": most_debt_university
        },
        "figures": {
            "Avg_Student_Debt_Distribution_Histogram": fig1,
            "Avg_Student_Debt_by_Institution_Type_Box": fig2,
            "Avg_Student_Debt_by_Institution_Type_Bar": fig3,
            "Student_Debt_vs_Acceptance_Rate_Scatter": fig4
        }
    }

def college_admissions_graduation_and_salary_outcome_analysis(df):
    print("\n--- College Admissions, Graduation, and Salary Outcome Analysis ---")
    expected = {
        'CollegeName': ['CollegeName', 'INSTNM', 'Name'],
        'ApplicationsReceived': ['ApplicationsReceived', 'Applicants', 'TotalApplicants'],
        'GraduationRate': ['GraduationRate', 'CompletionRate'],
        'MidCareerSalary': ['MidCareerSalary', 'PostGraduationEarnings', 'MedianSalary']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['MidCareerSalary'] = pd.to_numeric(df['MidCareerSalary'], errors='coerce')
    df = df.dropna(subset=['ApplicationsReceived', 'GraduationRate', 'MidCareerSalary'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_salary = df['MidCareerSalary'].mean()
    high_salary_college = df.loc[df['MidCareerSalary'].idxmax(), 'CollegeName']
    
    print(f"Average Mid-Career Salary: ${avg_salary:,.0f}")
    print(f"College with Highest Mid-Career Salary: {high_salary_college}")

    fig1 = px.scatter(df, x='ApplicationsReceived', y='MidCareerSalary', hover_name='CollegeName', title='Mid-Career Salary vs. Applications Received')
    fig2 = px.histogram(df, x='MidCareerSalary', title='Distribution of Mid-Career Salaries')
    fig3 = px.box(df, y='MidCareerSalary', title='Mid-Career Salary Distribution by College')
    fig4 = px.scatter(df, x='GraduationRate', y='MidCareerSalary', hover_name='CollegeName', title='Mid-Career Salary vs. Graduation Rate')

    return {
        "metrics": {
            "Average Mid-Career Salary": avg_salary,
            "Highest Mid-Career Salary College": high_salary_college
        },
        "figures": {
            "Mid_Career_Salary_vs_Applications_Received_Scatter": fig1,
            "Mid_Career_Salaries_Distribution_Histogram": fig2,
            "Mid_Career_Salary_Distribution_by_College_Box": fig3,
            "Mid_Career_Salary_vs_Graduation_Rate_Scatter": fig4
        }
    }

def school_funding_and_local_income_level_analysis(df):
    print("\n--- School Funding and Local Income Level Analysis ---")
    expected = {
        'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
        'Name': ['Name', 'SchoolName'],
        'District': ['District', 'DistrictName', 'SchoolDistrict'],
        'MedianHouseholdIncome': ['MedianHouseholdIncome', 'LocalIncomeLevel', 'AvgHouseholdIncome'],
        'PerPupilFunding': ['PerPupilFunding', 'SpendingPerStudent', 'DistrictFunding']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['MedianHouseholdIncome'] = pd.to_numeric(df['MedianHouseholdIncome'], errors='coerce')
    df['PerPupilFunding'] = pd.to_numeric(df['PerPupilFunding'], errors='coerce')
    df = df.dropna(subset=['MedianHouseholdIncome', 'PerPupilFunding'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_income = df['MedianHouseholdIncome'].mean()
    high_income_school = df.loc[df['MedianHouseholdIncome'].idxmax(), 'Name']
    
    print(f"Average Median Household Income: ${avg_income:,.0f}")
    print(f"School in Highest Income Area: {high_income_school}")

    fig1 = px.histogram(df, x='MedianHouseholdIncome', title='Distribution of Median Household Income')
    fig2 = px.box(df, y='MedianHouseholdIncome', title='Median Household Income Distribution')
    
    income_by_district = df.groupby('District')['MedianHouseholdIncome'].mean().reset_index()
    fig3 = px.bar(income_by_district.sort_values('MedianHouseholdIncome', ascending=False).head(20), x='District', y='MedianHouseholdIncome', title='Average Median Household Income by District (Top 20)')
    
    fig4 = px.scatter(df, x='MedianHouseholdIncome', y='PerPupilFunding', hover_name='Name', title='Per-Pupil Funding vs. Median Household Income')

    return {
        "metrics": {
            "Average Median Household Income": avg_income,
            "Highest Income School": high_income_school
        },
        "figures": {
            "Median_Household_Income_Distribution_Histogram": fig1,
            "Median_Household_Income_Distribution_Box": fig2,
            "Average_Median_Household_Income_by_District_Bar": fig3,
            "Per_Pupil_Funding_vs_Median_Household_Income_Scatter": fig4
        }
    }

def pell_grant_recipient_graduation_and_loan_default_rate_analysis(df):
    print("\n--- Pell Grant Recipient Graduation and Loan Default Rate Analysis ---")
    expected = {
        'Institution': ['Institution', 'INSTNM', 'Name'],
        'State': ['State', 'STABBR'],
        'Year': ['Year', 'AcademicYear'],
        'PellGradRate': ['PellGradRate', 'PellRecipientGraduationRate'],
        'OverallGradRate': ['OverallGradRate', 'GraduationRate'],
        'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['PellGradRate'] = pd.to_numeric(df['PellGradRate'], errors='coerce')
    df['OverallGradRate'] = pd.to_numeric(df['OverallGradRate'], errors='coerce')
    df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['PellGradRate', 'OverallGradRate', 'DefaultRate', 'Year'])
    
    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    avg_pell_grad_rate = df['PellGradRate'].mean()
    avg_overall_grad_rate = df['OverallGradRate'].mean()
    avg_default_rate = df['DefaultRate'].mean()
    
    print(f"Average Pell Grant Recipient Graduation Rate: {avg_pell_grad_rate:.2f}%")
    print(f"Average Overall Graduation Rate: {avg_overall_grad_rate:.2f}%")
    print(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
    
    fig1 = px.line(df.groupby('Year').agg({'PellGradRate': 'mean', 'OverallGradRate': 'mean', 'DefaultRate': 'mean'}).reset_index(),
                   x='Year', y=['PellGradRate', 'OverallGradRate', 'DefaultRate'],
                   title='Pell Grant Recipient Trends Over Time')

    fig2 = px.bar(df.groupby('State')['DefaultRate'].mean().nlargest(20).reset_index(), x='State', y='DefaultRate', title='Average Default Rate by State (Top 20)')
    
    fig3 = px.box(df, x='Year', y='DefaultRate', title='Default Rate Distribution by Year')

    return {
        "metrics": {
            "Average Pell Grant Graduation Rate": avg_pell_grad_rate,
            "Average Overall Graduation Rate": avg_overall_grad_rate,
            "Average Default Rate": avg_default_rate
        },
        "figures": {
            "Pell_Grant_Recipient_Trends_Line": fig1,
            "Average_Default_Rate_by_State_Bar": fig2,
            "Default_Rate_Distribution_by_Year_Box": fig3
        }
    }

def college_selectivity_and_graduation_rate_analysis(df):
    print("\n--- College Selectivity and Graduation Rate Analysis ---")
    expected = {
        'CollegeID': ['CollegeID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'GraduationRate': ['GraduationRate', 'CompletionRate'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
        'AvgSAT': ['AvgSAT', 'SATScore', 'AverageSATScore']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    if 'AcceptanceRate' in df.columns:
        df['SelectivityScore'] = 1 - pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    elif 'AvgSAT' in df.columns:
        df['SelectivityScore'] = pd.to_numeric(df['AvgSAT'], errors='coerce')
    else:
        print("Warning: No clear selectivity metric found (AcceptanceRate or AvgSAT). Selectivity analysis might be limited.")
        df['SelectivityScore'] = np.nan

    df = safe_numeric_conversion(df, 'SelectivityScore')
    df = safe_numeric_conversion(df, 'GraduationRate')
    df = df.dropna(subset=['SelectivityScore', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_grad_rate = df['GraduationRate'].mean()
    high_grad_rate_college = df.loc[df['GraduationRate'].idxmax(), 'Name']
    
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
    print(f"College with Highest Graduation Rate: {high_grad_rate_college}")

    fig1 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')
    fig2 = px.box(df, x='State', y='GraduationRate', title='Graduation Rate by State')
    fig3 = px.bar(df.groupby('State')['GraduationRate'].mean().nlargest(20).reset_index(), x='State', y='GraduationRate', title='Top 20 States by Average Graduation Rate')
    fig4 = px.scatter(df, x='SelectivityScore', y='GraduationRate', hover_name='Name', title='Graduation Rate vs. College Selectivity')

    return {
        "metrics": {
            "Average Graduation Rate": avg_grad_rate,
            "Highest Graduation Rate College": high_grad_rate_college
        },
        "figures": {
            "Graduation_Rates_Distribution_Histogram": fig1,
            "Graduation_Rate_by_State_Box": fig2,
            "Top_20_States_by_Avg_Graduation_Rate_Bar": fig3,
            "Graduation_Rate_vs_Selectivity_Scatter": fig4
        }
    }

def school_district_demographics_and_student_teacher_ratio_analysis(df):
    print("\n--- School District Demographics and Student-Teacher Ratio Analysis ---")
    expected = {
        'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
        'DistrictName': ['DistrictName', 'Name'],
        'State': ['State', 'STABBR'],
        'FreeLunchRate': ['FreeLunchRate', 'FreeReducedLunchRate', 'PovertyRate'],
        'StudentCount': ['StudentCount', 'TotalEnrollment', 'Enrollment'],
        'TeacherCount': ['TeacherCount', 'TotalTeachers', 'FTETeachers']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FreeLunchRate'] = pd.to_numeric(df['FreeLunchRate'], errors='coerce')
    df['StudentCount'] = pd.to_numeric(df['StudentCount'], errors='coerce')
    df['TeacherCount'] = pd.to_numeric(df['TeacherCount'], errors='coerce')
    df = df.dropna(subset=['FreeLunchRate', 'StudentCount', 'TeacherCount'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['StudentTeacherRatio'] = df.apply(
        lambda row: row['StudentCount'] / row['TeacherCount'] if row['TeacherCount'] > 0 else np.nan,
        axis=1
    )
    df = df.dropna(subset=['StudentTeacherRatio'])

    avg_free_lunch_rate = df['FreeLunchRate'].mean()
    high_need_district = df.loc[df['FreeLunchRate'].idxmax(), 'DistrictName']
    avg_student_teacher_ratio = df['StudentTeacherRatio'].mean()

    print(f"Average Free Lunch Rate: {avg_free_lunch_rate:.2f}%")
    print(f"District with Highest Free Lunch Rate: {high_need_district}")
    print(f"Average Student-Teacher Ratio: {avg_student_teacher_ratio:.2f}")

    fig1 = px.histogram(df, x='FreeLunchRate', title='Distribution of Free Lunch Rates')
    fig2 = px.box(df, x='State', y='FreeLunchRate', title='Free Lunch Rate by State')
    fig3 = px.bar(df.groupby('DistrictName')['FreeLunchRate'].mean().nlargest(20).reset_index(), x='DistrictName', y='FreeLunchRate', title='Top 20 Districts by Free Lunch Rate')
    fig4 = px.scatter(df, x='FreeLunchRate', y='StudentTeacherRatio', hover_name='DistrictName', title='Student-Teacher Ratio vs. Free Lunch Rate')

    return {
        "metrics": {
            "Average Free Lunch Rate": avg_free_lunch_rate,
            "Highest Need District": high_need_district,
            "Average Student-Teacher Ratio": avg_student_teacher_ratio
        },
        "figures": {
            "Free_Lunch_Rates_Distribution_Histogram": fig1,
            "Free_Lunch_Rate_by_State_Box": fig2,
            "Top_20_Districts_by_Free_Lunch_Rate_Bar": fig3,
            "Student_Teacher_Ratio_vs_Free_Lunch_Rate_Scatter": fig4
        }
    }

def college_tuition_and_enrollment_statistics_analysis(df):
    print("\n--- College Tuition and Enrollment Statistics Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
        'Enrollment': ['Enrollment', 'TotalEnrollment', 'UndergraduateEnrollment'],
        'GraduationRate': ['GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['Tuition', 'Enrollment', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_tuition = df['Tuition'].mean()
    highest_tuition_college = df.loc[df['Tuition'].idxmax(), 'Name']
    
    print(f"Average Tuition: ${avg_tuition:,.0f}")
    print(f"College with Highest Tuition: {highest_tuition_college}")

    fig1 = px.scatter(df, x='Tuition', y='GraduationRate', hover_name='Name', title='Graduation Rate vs. Tuition')
    fig2 = px.histogram(df, x='Tuition', title='Distribution of Tuition')
    fig3 = px.box(df, y='Tuition', title='Tuition Distribution')
    fig4 = px.scatter(df, x='Enrollment', y='Tuition', hover_name='Name', title='Tuition vs. Enrollment')

    return {
        "metrics": {
            "Average Tuition": avg_tuition,
            "Highest Tuition College": highest_tuition_college
        },
        "figures": {
            "Graduation_Rate_vs_Tuition_Scatter": fig1,
            "Tuition_Distribution_Histogram": fig2,
            "Tuition_Distribution_Box": fig3,
            "Tuition_vs_Enrollment_Scatter": fig4
        }
    }

def school_special_needs_and_counselor_ratio_analysis(df):
    print("\n--- School Special Needs and Counselor Ratio Analysis ---")
    expected = {
        'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
        'Name': ['Name', 'SchoolName'],
        'State': ['State', 'STABBR'],
        'SpecialNeedsStudentCount': ['SpecialNeedsStudentCount', 'StudentsWithDisabilities', 'SpEdStudents'],
        'CounselorCount': ['CounselorCount', 'FTECounselors', 'TotalCounselors']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['SpecialNeedsStudentCount'] = pd.to_numeric(df['SpecialNeedsStudentCount'], errors='coerce')
    df['CounselorCount'] = pd.to_numeric(df['CounselorCount'], errors='coerce')
    df = df.dropna(subset=['SpecialNeedsStudentCount', 'CounselorCount'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['SpecialNeedsCounselorRatio'] = df.apply(
        lambda row: row['SpecialNeedsStudentCount'] / row['CounselorCount'] if row['CounselorCount'] > 0 else np.nan,
        axis=1
    )
    df = df.dropna(subset=['SpecialNeedsCounselorRatio'])

    avg_ratio = df['SpecialNeedsCounselorRatio'].mean()
    lowest_ratio_school = df.loc[df['SpecialNeedsCounselorRatio'].idxmin(), 'Name']
    
    print(f"Average Special Needs Student to Counselor Ratio: {avg_ratio:.2f}:1")
    print(f"School with Lowest Ratio: {lowest_ratio_school}")

    fig1 = px.histogram(df, x='SpecialNeedsCounselorRatio', title='Distribution of Special Needs Student-Counselor Ratios')
    fig2 = px.box(df, x='State', y='SpecialNeedsCounselorRatio', title='Special Needs Student-Counselor Ratio by State')
    fig3 = px.bar(df.groupby('State')['SpecialNeedsCounselorRatio'].mean().nlargest(20).reset_index(), x='State', y='SpecialNeedsCounselorRatio', title='Top 20 States by Average Special Needs Student-Counselor Ratio')

    return {
        "metrics": {
            "Average Special Needs Student to Counselor Ratio": avg_ratio,
            "Lowest Ratio School": lowest_ratio_school
        },
        "figures": {
            "Special_Needs_Counselor_Ratio_Distribution_Histogram": fig1,
            "Special_Needs_Counselor_Ratio_by_State_Box": fig2,
            "Top_20_States_by_Avg_Special_Needs_Counselor_Ratio_Bar": fig3
        }
    }

def university_graduation_rate_and_diversity_index_analysis(df):
    print("\n--- University Graduation Rate and Diversity Index Analysis ---")
    expected = {
        'UniversityState': ['UniversityState', 'State', 'STABBR'],
        'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
        'StudentDiversityIndex': ['StudentDiversityIndex', 'DiversityIndex', 'RacialEthnicDiversity'],
        'GraduationRate': ['GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StudentDiversityIndex'] = pd.to_numeric(df['StudentDiversityIndex'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['StudentDiversityIndex', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_diversity = df['StudentDiversityIndex'].mean()
    most_diverse_university = df.loc[df['StudentDiversityIndex'].idxmax(), 'UniversityName']
    
    print(f"Average Student Diversity Index: {avg_diversity:.2f}")
    print(f"Most Diverse University: {most_diverse_university}")

    fig1 = px.scatter(df, x='StudentDiversityIndex', y='GraduationRate', hover_name='UniversityName', title='Graduation Rate vs. Student Diversity Index')
    fig2 = px.box(df, x='UniversityState', y='StudentDiversityIndex', title='Diversity Index by State')
    fig3 = px.histogram(df, x='GraduationRate', color='UniversityState', title='Graduation Rate Distribution by State')

    return {
        "metrics": {
            "Average Student Diversity Index": avg_diversity,
            "Most Diverse University": most_diverse_university
        },
        "figures": {
            "Graduation_Rate_vs_Diversity_Index_Scatter": fig1,
            "Diversity_Index_by_State_Box": fig2,
            "Graduation_Rate_Distribution_by_State_Histogram": fig3
        }
    }

def post_graduation_earnings_and_debt_analysis(df):
    print("\n--- Post-Graduation Earnings and Debt Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'GraduationRate': ['GraduationRate', 'CompletionRate'],
        'MedianEarnings': ['MedianEarnings', 'PostGradEarnings', 'AvgSalary'],
        'MedianDebt': ['MedianDebt', 'AvgDebtUponGraduation', 'StudentLoanDebt']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['MedianEarnings'] = pd.to_numeric(df['MedianEarnings'], errors='coerce')
    df['MedianDebt'] = pd.to_numeric(df['MedianDebt'], errors='coerce')
    df = df.dropna(subset=['GraduationRate', 'MedianEarnings', 'MedianDebt'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_earnings = df['MedianEarnings'].mean()
    highest_earnings_institution = df.loc[df['MedianEarnings'].idxmax(), 'InstitutionName']
    avg_debt = df['MedianDebt'].mean()
    
    print(f"Average Median Earnings: ${avg_earnings:,.0f}")
    print(f"Highest Earnings Institution: {highest_earnings_institution}")
    print(f"Average Median Debt: ${avg_debt:,.0f}")

    fig1 = px.scatter(df, x='GraduationRate', y='MedianEarnings', hover_name='InstitutionName', title='Median Earnings vs. Graduation Rate')
    fig2 = px.histogram(df, x='MedianEarnings', title='Distribution of Median Earnings')
    fig3 = px.box(df, y='MedianEarnings', title='Median Earnings Distribution')
    fig4 = px.scatter(df, x='MedianDebt', y='MedianEarnings', hover_name='InstitutionName', title='Median Earnings vs. Median Debt')
    fig5 = px.histogram(df, x='MedianDebt', title='Distribution of Median Debt')

    return {
        "metrics": {
            "Average Median Earnings": avg_earnings,
            "Highest Earnings Institution": highest_earnings_institution,
            "Average Median Debt": avg_debt
        },
        "figures": {
            "Median_Earnings_vs_Graduation_Rate_Scatter": fig1,
            "Median_Earnings_Distribution_Histogram": fig2,
            "Median_Earnings_Distribution_Box": fig3,
            "Median_Earnings_vs_Median_Debt_Scatter": fig4,
            "Median_Debt_Distribution_Histogram": fig5
        }
    }

def school_district_test_score_and_graduation_rate_analysis(df):
    print("\n--- School District Test Score and Graduation Rate Analysis ---")
    expected = {
        'DistrictName': ['DistrictName', 'Name'],
        'SchoolDistrictID': ['SchoolDistrictID', 'DistrictID', 'ID'],
        'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate'],
        'StateTestScores': ['StateTestScores', 'AverageTestScore', 'DistrictAvgScore']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['StateTestScores'] = pd.to_numeric(df['StateTestScores'], errors='coerce')
    df = df.dropna(subset=['GraduationRate', 'StateTestScores'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_grad_rate = df['GraduationRate'].mean()
    avg_test_score = df['StateTestScores'].mean()

    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
    print(f"Average State Test Score: {avg_test_score:.2f}")

    fig1 = px.scatter(df, x='StateTestScores', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. State Test Scores')
    fig2 = px.box(df, x='DistrictName', y='GraduationRate', title='Graduation Rate by School District')
    fig3 = px.histogram(df, x='StateTestScores', title='Distribution of State Test Scores')

    return {
        "metrics": {
            "Average Graduation Rate": avg_grad_rate,
            "Average State Test Score": avg_test_score
        },
        "figures": {
            "Graduation_Rate_vs_State_Test_Scores_Scatter": fig1,
            "Graduation_Rate_by_School_District_Box": fig2,
            "State_Test_Scores_Distribution_Histogram": fig3
        }
    }

def college_admissions_and_loan_default_rate_correlation(df):
    print("\n--- College Admissions and Loan Default Rate Correlation ---")
    expected = {
        'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
        'InstitutionState': ['InstitutionState', 'State', 'STABBR'],
        'LoanDefaultRate': ['LoanDefaultRate', 'DefaultRate', 'ThreeYearDefaultRate'],
        'AdmissionsRate': ['AdmissionsRate', 'AcceptanceRate', 'AdmitRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['LoanDefaultRate'] = pd.to_numeric(df['LoanDefaultRate'], errors='coerce')
    df['AdmissionsRate'] = pd.to_numeric(df['AdmissionsRate'], errors='coerce')
    df = df.dropna(subset=['LoanDefaultRate', 'AdmissionsRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_default_rate = df['LoanDefaultRate'].mean()
    avg_admissions_rate = df['AdmissionsRate'].mean()

    print(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
    print(f"Average Admissions Rate: {avg_admissions_rate:.2f}%")

    fig1 = px.scatter(df, x='AdmissionsRate', y='LoanDefaultRate', hover_name='CollegeName', title='Loan Default Rate vs. Admissions Rate')
    fig2 = px.box(df, x='InstitutionState', y='LoanDefaultRate', title='Loan Default Rate by State')
    fig3 = px.box(df, y='AdmissionsRate', title='Admissions Rate Distribution')

    return {
        "metrics": {
            "Average Loan Default Rate": avg_default_rate,
            "Average Admissions Rate": avg_admissions_rate
        },
        "figures": {
            "Loan_Default_Rate_vs_Admissions_Rate_Scatter": fig1,
            "Loan_Default_Rate_by_State_Box": fig2,
            "Admissions_Rate_Distribution_Box": fig3
        }
    }

def college_admissions_funnel_and_yield_rate_analysis(df):
    print("\n--- College Admissions Funnel and Yield Rate Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'State': ['State', 'STABBR', 'InstitutionState'],
        'Year': ['Year', 'AcademicYear'],
        'ApplicationsReceived': ['ApplicationsReceived', 'Applicants', 'TotalApplicants'],
        'Admitted': ['Admitted', 'AcceptedStudents'],
        'Enrolled': ['Enrolled', 'MatriculatedStudents'],
        'Yield': ['Yield', 'YieldRate'],
        'Graduation': ['Graduation', 'GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
    df['Admitted'] = pd.to_numeric(df['Admitted'], errors='coerce')
    df['Enrolled'] = pd.to_numeric(df['Enrolled'], errors='coerce')
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df['Graduation'] = pd.to_numeric(df['Graduation'], errors='coerce')
    df = df.dropna(subset=['ApplicationsReceived', 'Admitted', 'Enrolled', 'Yield', 'Graduation'])
    
    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    avg_yield = df['Yield'].mean()
    avg_graduation = df['Graduation'].mean()
    
    print(f"Average Yield Rate: {avg_yield:.2f}%")
    print(f"Average Graduation Rate: {avg_graduation:.2f}%")
    
    total_applications = df['ApplicationsReceived'].sum()
    total_admitted = df['Admitted'].sum()
    total_enrolled = df['Enrolled'].sum()

    funnel_data = pd.DataFrame({
        'Stage': ['Applications Received', 'Admitted', 'Enrolled'],
        'Count': [total_applications, total_admitted, total_enrolled]
    })
    
    fig1 = px.funnel(funnel_data, x='Count', y='Stage', title='Overall College Admissions Funnel')

    fig2 = px.scatter(df, x='Yield', y='Graduation', hover_name='InstitutionName', title='Graduation Rate vs. Yield Rate')
    fig3 = px.box(df, x='State', y='Yield', title='Yield Rate by State')
    fig4 = px.histogram(df, x='Yield', color='State', title='Yield Rate Distribution by State')

    return {
        "metrics": {
            "Overall Acceptance Rate": total_admitted / total_applications if total_applications > 0 else 0,
            "Overall Yield Rate": total_enrolled / total_admitted if total_admitted > 0 else 0
        },
        "figures": {
            "Overall_College_Admissions_Funnel": fig1,
            "Graduation_Rate_vs_Yield_Rate_Scatter": fig2,
            "Yield_Rate_by_State_Box": fig3,
            "Yield_Rate_Distribution_by_State_Histogram": fig4
        }
    }

def school_board_spending_and_student_achievement_analysis(df):
    print("\n--- School Board Spending and Student Achievement Analysis ---")
    expected = {
        'SchoolBoardID': ['SchoolBoardID', 'BoardID', 'ID'],
        'DistrictName': ['DistrictName', 'Name', 'SchoolDistrict'],
        'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate'],
        'PerStudentSpending': ['PerStudentSpending', 'SpendingPerStudent', 'BudgetPerStudent']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['PerStudentSpending'] = pd.to_numeric(df['PerStudentSpending'], errors='coerce')
    df = df.dropna(subset=['GraduationRate', 'PerStudentSpending'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    avg_spending = df['PerStudentSpending'].mean()
    avg_grad_rate = df['GraduationRate'].mean()
    
    print(f"Average Per-Student Spending: ${avg_spending:,.0f}")
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
    
    fig1 = px.scatter(df, x='PerStudentSpending', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. Per-Student Spending')
    fig2 = px.bar(df.groupby('DistrictName')['PerStudentSpending'].mean().nlargest(20).reset_index(), x='DistrictName', y='PerStudentSpending', title='Average Per-Student Spending by District (Top 20)')
    fig3 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')

    return {
        "metrics": {
            "Average Per-Student Spending": avg_spending,
            "Average Graduation Rate": avg_grad_rate
        },
        "figures": {
            "Graduation_Rate_vs_Per_Student_Spending_Scatter": fig1,
            "Average_Per_Student_Spending_by_District_Bar": fig2,
            "Graduation_Rates_Distribution_Histogram": fig3
        }
    }

def university_enrollment_and_earnings_outcome_analysis(df):
    print("\n--- University Enrollment and Earnings Outcome Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'Region': ['Region', 'State'],
        'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment'],
        'Earnings25thPercentile': ['Earnings25thPercentile', 'MedianEarnings', 'PostGradEarnings']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
    df['Earnings25thPercentile'] = pd.to_numeric(df['Earnings25thPercentile'], errors='coerce')
    df = df.dropna(subset=['UndergraduateEnrollment', 'Earnings25thPercentile'])
    
    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    avg_enrollment = df['UndergraduateEnrollment'].mean()
    avg_earnings = df['Earnings25thPercentile'].mean()
    
    print(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
    print(f"Average Earnings (25th Percentile): ${avg_earnings:,.0f}")
    
    fig1 = px.scatter(df, x='UndergraduateEnrollment', y='Earnings25thPercentile', hover_name='Name', title='Earnings vs. Undergraduate Enrollment')
    fig2 = px.box(df, x='Region', y='Earnings25thPercentile', title='Earnings Distribution by Region')
    fig3 = px.bar(df.groupby('Region')['UndergraduateEnrollment'].sum().reset_index(), x='Region', y='UndergraduateEnrollment', title='Total Enrollment by Region')

    return {
        "metrics": {
            "Average Undergraduate Enrollment": avg_enrollment,
            "Average Earnings (25th Percentile)": avg_earnings
        },
        "figures": {
            "Earnings_vs_Undergraduate_Enrollment_Scatter": fig1,
            "Earnings_Distribution_by_Region_Box": fig2,
            "Total_Enrollment_by_Region_Bar": fig3
        }
    }

def college_retention_debt_and_earnings_analysis(df):
    print("\n--- College Retention, Debt, and Earnings Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'State': ['State', 'STABBR'],
        'RetentionRate': ['RetentionRate', 'FirstYearRetention'],
        'StudentDebt': ['StudentDebt', 'AvgStudentDebt', 'MedianDebt'],
        '10YearEarnings': ['10YearEarnings', 'PostGradEarnings', 'MedianEarnings']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}

    df['RetentionRate'] = pd.to_numeric(df['RetentionRate'], errors='coerce')
    df['StudentDebt'] = pd.to_numeric(df['StudentDebt'], errors='coerce')
    df['10YearEarnings'] = pd.to_numeric(df['10YearEarnings'], errors='coerce')
    df = df.dropna(subset=['RetentionRate', 'StudentDebt', '10YearEarnings'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_earnings = df['10YearEarnings'].mean()
    avg_debt = df['StudentDebt'].mean()
    
    print(f"Average 10-Year Earnings: ${avg_earnings:,.0f}")
    print(f"Average Student Debt: ${avg_debt:,.0f}")
    
    fig1 = px.scatter(df, x='StudentDebt', y='10YearEarnings', hover_name='InstitutionName', title='10-Year Earnings vs. Student Debt')
    fig2 = px.box(df, x='State', y='10YearEarnings', title='10-Year Earnings Distribution by State')
    fig3 = px.box(df, x='State', y='StudentDebt', title='Student Debt Distribution by State')
    fig4 = px.scatter(df, x='RetentionRate', y='10YearEarnings', hover_name='InstitutionName', title='10-Year Earnings vs. Retention Rate')
    fig5 = px.scatter(df, x='RetentionRate', y='StudentDebt', hover_name='InstitutionName', title='Student Debt vs. Retention Rate')

    return {
        "metrics": {
            "Average 10-Year Earnings": avg_earnings,
            "Average Student Debt": avg_debt
        },
        "figures": {
            "10_Year_Earnings_vs_Student_Debt_Scatter": fig1,
            "10_Year_Earnings_Distribution_by_State_Box": fig2,
            "Student_Debt_Distribution_by_State_Box": fig3,
            "10_Year_Earnings_vs_Retention_Rate_Scatter": fig4,
            "Student_Debt_vs_Retention_Rate_Scatter": fig5
        }
    }

def school_enrollment_and_disadvantaged_student_population_analysis(df):
    print("\n--- School Enrollment and Disadvantaged Student Population Analysis ---")
    expected = {
        'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
        'District': ['District', 'DistrictName', 'SchoolDistrict'],
        'Enrollment': ['Enrollment', 'TotalEnrollment', 'StudentCount'],
        'EconomicallyDisadvantagedRate': ['EconomicallyDisadvantagedRate', 'FreeReducedLunchRate', 'PovertyRate']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df['EconomicallyDisadvantagedRate'] = pd.to_numeric(df['EconomicallyDisadvantagedRate'], errors='coerce')
    df = df.dropna(subset=['Enrollment', 'EconomicallyDisadvantagedRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_disadvantaged_rate = df['EconomicallyDisadvantagedRate'].mean()
    high_disadvantaged_school = df.loc[df['EconomicallyDisadvantagedRate'].idxmax(), 'SchoolID']

    print(f"Average Disadvantaged Rate: {avg_disadvantaged_rate:.2f}%")
    print(f"School with Highest Disadvantaged Rate: {high_disadvantaged_school}")

    fig1 = px.scatter(df, x='Enrollment', y='EconomicallyDisadvantagedRate', hover_name='SchoolID', title='Disadvantaged Rate vs. Enrollment')
    fig2 = px.histogram(df, x='EconomicallyDisadvantagedRate', title='Distribution of Economically Disadvantaged Rate')
    
    disadvantaged_by_district = df.groupby('District')['EconomicallyDisadvantagedRate'].mean().reset_index()
    fig3 = px.bar(disadvantaged_by_district.sort_values('EconomicallyDisadvantagedRate', ascending=False).head(20), x='District', y='EconomicallyDisadvantagedRate', title='Average Disadvantaged Rate by District (Top 20)')

    return {
        "metrics": {
            "Average Disadvantaged Rate": avg_disadvantaged_rate,
            "Highest Disadvantaged School": high_disadvantaged_school
        },
        "figures": {
            "Disadvantaged_Rate_vs_Enrollment_Scatter": fig1,
            "Economically_Disadvantaged_Rate_Distribution_Histogram": fig2,
            "Average_Disadvantaged_Rate_by_District_Bar": fig3
        }
    }

def university_enrollment_and_faculty_count_analysis(df):
    print("\n--- University Enrollment and Faculty Count Analysis ---")
    expected = {
        'UniversityID': ['UniversityID', 'InstID', 'ID'],
        'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
        'Enrollment': ['Enrollment', 'TotalEnrollment', 'UndergraduateEnrollment'],
        'FacultyCount': ['FacultyCount', 'TotalFaculty', 'FTFaculty']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df['FacultyCount'] = pd.to_numeric(df['FacultyCount'], errors='coerce')
    df = df.dropna(subset=['Enrollment', 'FacultyCount'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_faculty_count = df['FacultyCount'].mean()
    avg_enrollment = df['Enrollment'].mean()
    
    print(f"Average Faculty Count: {avg_faculty_count:,.0f}")
    print(f"Average Enrollment: {avg_enrollment:,.0f}")

    df['student_faculty_ratio'] = df.apply(
        lambda row: row['Enrollment'] / row['FacultyCount'] if row['FacultyCount'] > 0 else np.nan,
        axis=1
    )
    df = df.dropna(subset=['student_faculty_ratio'])

    print(f"Average Student-Faculty Ratio: {df['student_faculty_ratio'].mean():.2f}:1")

    fig1 = px.scatter(df, x='Enrollment', y='FacultyCount', hover_name='UniversityName', title='Faculty Count vs. Enrollment')
    fig2 = px.histogram(df, x='FacultyCount', title='Distribution of Faculty Count')
    fig3 = px.box(df, y='student_faculty_ratio', title='Student-Faculty Ratio Distribution')

    return {
        "metrics": {
            "Average Faculty Count": avg_faculty_count,
            "Average Enrollment": avg_enrollment,
            "Average Student-Faculty Ratio": df['student_faculty_ratio'].mean()
        },
        "figures": {
            "Faculty_Count_vs_Enrollment_Scatter": fig1,
            "Faculty_Count_Distribution_Histogram": fig2,
            "Student_Faculty_Ratio_Distribution_Box": fig3
        }
    }

def state_education_system_performance_analysis(df):
    print("\n--- State Education System Performance Analysis ---")
    expected = {
        'State': ['State', 'STABBR', 'StateName'],
        'SchoolCount': ['SchoolCount', 'TotalSchoolsInState'],
        'FinancialAidRate': ['FinancialAidRate', 'AvgFinancialAidRate', 'StateFinancialAidPct'],
        'GraduationRate': ['GraduationRate', 'StateGraduationRate'],
        'AvgTestScore': ['AvgTestScore', 'StateAvgTestScore']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['SchoolCount'] = pd.to_numeric(df['SchoolCount'], errors='coerce')
    df['FinancialAidRate'] = pd.to_numeric(df['FinancialAidRate'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df['AvgTestScore'] = pd.to_numeric(df['AvgTestScore'], errors='coerce')
    df = df.dropna(subset=['SchoolCount', 'FinancialAidRate', 'GraduationRate', 'AvgTestScore'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_schools = df['SchoolCount'].sum()
    avg_aid_rate = df['FinancialAidRate'].mean()
    avg_state_grad_rate = df['GraduationRate'].mean()
    avg_state_test_score = df['AvgTestScore'].mean()
    
    print(f"Total Schools Represented: {total_schools:,.0f}")
    print(f"Average Financial Aid Rate: {avg_aid_rate:.2f}%")
    print(f"Average State Graduation Rate: {avg_state_grad_rate:.2f}%")
    print(f"Average State Test Score: {avg_state_test_score:.2f}")

    fig1 = px.scatter(df, x='SchoolCount', y='FinancialAidRate', color='State', title='Financial Aid Rate vs. School Count by State')
    fig2 = px.box(df, x='State', y='FinancialAidRate', title='Financial Aid Rate Distribution by State')
    fig3 = px.bar(df.groupby('State')['SchoolCount'].sum().nlargest(20).reset_index(), x='State', y='SchoolCount', title='Total School Count by State (Top 20)')
    fig4 = px.scatter(df, x='AvgTestScore', y='GraduationRate', color='State', title='State Graduation Rate vs. Average Test Score')

    return {
        "metrics": {
            "Total Schools": total_schools,
            "Average Financial Aid Rate": avg_aid_rate,
            "Average State Graduation Rate": avg_state_grad_rate,
            "Average State Test Score": avg_state_test_score
        },
        "figures": {
            "Financial_Aid_Rate_vs_School_Count_Scatter": fig1,
            "Financial_Aid_Rate_Distribution_by_State_Box": fig2,
            "Total_School_Count_by_State_Bar": fig3,
            "State_Graduation_Rate_vs_Avg_Test_Score_Scatter": fig4
        }
    }

def university_tuition_earnings_and_default_rate_trend_analysis(df):
    print("\n--- University Tuition, Earnings, and Default Rate Trend Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'Year': ['Year', 'AcademicYear'],
        'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
        'Earnings': ['Earnings', 'MedianEarnings', 'PostGraduationEarnings'],
        'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
    df['Earnings'] = pd.to_numeric(df['Earnings'], errors='coerce')
    df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['Tuition', 'Earnings', 'DefaultRate', 'Year'])
    df = df.sort_values(by=['InstitutionName', 'Year'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    trend_data = df.groupby('Year')[['Tuition', 'Earnings', 'DefaultRate']].mean().reset_index()
    fig1 = px.line(trend_data, x='Year', y=['Tuition', 'Earnings', 'DefaultRate'],
                   title='Average Trends Over Time (Tuition, Earnings, Default Rate)')
    
    unique_institutions = df['InstitutionName'].unique()
    if len(unique_institutions) > 1:
        if len(unique_institutions) > 10:
            top_institutions = df.groupby('InstitutionName')['Earnings'].mean().nlargest(5).index
            df_plot = df[df['InstitutionName'].isin(top_institutions)]
        else:
            df_plot = df

        fig2 = px.line(df_plot, x='Year', y='Tuition', color='InstitutionName',
                       title='Tuition Trends for Sample Universities')
        fig3 = px.line(df_plot, x='Year', y='Earnings', color='InstitutionName',
                       title='Earnings Trends for Sample Universities')
        fig4 = px.line(df_plot, x='Year', y='DefaultRate', color='InstitutionName',
                       title='Default Rate Trends for Sample Universities')
    else:
        fig2 = px.line(df, x='Year', y='Tuition', title='Tuition Trend')
        fig3 = px.line(df, x='Year', y='Earnings', title='Earnings Trend')
        fig4 = px.line(df, x='Year', y='DefaultRate', title='Default Rate Trend')

    fig5 = px.scatter(df, x='Tuition', y='DefaultRate', hover_name='InstitutionName', animation_frame='Year',
                      title='Default Rate vs. Tuition Over Time (Animated)',
                      animation_group='InstitutionName')

    return {
        "metrics": {},
        "figures": {
            "Average_Trends_Over_Time_Line": fig1,
            "Tuition_Trends_Sample_Universities_Line": fig2,
            "Earnings_Trends_Sample_Universities_Line": fig3,
            "Default_Rate_Trends_Sample_Universities_Line": fig4,
            "Default_Rate_vs_Tuition_Animated_Scatter": fig5
        }
    }

def school_board_data_analysis(df):
    print("\n--- School Board Data Analysis ---")
    expected = {
        'District': ['District', 'DistrictName', 'SchoolDistrict'],
        'SchoolBoard': ['SchoolBoard', 'BoardName', 'GoverningBody'],
        'Budget': ['Budget', 'TotalBudget', 'DistrictBudget'],
        'Students': ['Students', 'TotalStudents', 'Enrollment'],
        'MeetingAttendanceRate': ['MeetingAttendanceRate', 'BoardMeetingAttendance']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    if 'Budget' in df.columns:
        df['Budget'] = pd.to_numeric(df['Budget'], errors='coerce')
    if 'Students' in df.columns:
        df['Students'] = pd.to_numeric(df['Students'], errors='coerce')
    if 'MeetingAttendanceRate' in df.columns:
        df['MeetingAttendanceRate'] = pd.to_numeric(df['MeetingAttendanceRate'], errors='coerce')

    df_clean = df.copy()
    if 'Budget' in df_clean.columns and 'Students' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['Budget', 'Students'])
        df_clean['per_student_spending'] = df_clean.apply(
            lambda row: row['Budget'] / row['Students'] if row['Students'] > 0 else np.nan,
            axis=1
        )
        df_clean = df_clean.dropna(subset=['per_student_spending'])
    else:
        df_clean['per_student_spending'] = np.nan

    if df_clean.empty and not df.empty:
        print("Note: After cleaning, specific budget/student ratio data is insufficient.")
    elif df.empty:
        print("No data available for this analysis.")
        return {"message": "No data available."}


    metrics = {}
    figures = {}

    if 'Budget' in df.columns and not df['Budget'].dropna().empty:
        total_budget = df['Budget'].sum()
        metrics['Total Budget'] = total_budget
        print(f"Total Budget: ${total_budget:,.0f}")
        figures['Total_Budget_Histogram'] = px.histogram(df, x='Budget', title='Distribution of Total Budgets')

    if 'Students' in df.columns and not df['Students'].dropna().empty:
        total_students = df['Students'].sum()
        metrics['Total Students'] = total_students
        print(f"Total Students: {total_students:,.0f}")
        figures['Total_Students_Histogram'] = px.histogram(df, x='Students', title='Distribution of Total Students')

    if 'per_student_spending' in df_clean.columns and not df_clean['per_student_spending'].dropna().empty:
        figures['Per_Student_Spending_Bar'] = px.bar(df_clean.groupby('SchoolBoard')['per_student_spending'].mean().nlargest(20).reset_index(),
                                                        x='SchoolBoard', y='per_student_spending', title='Average Per-Student Spending by School Board (Top 20)')
        figures['Per_Student_Spending_Box'] = px.box(df_clean, y='per_student_spending', title='Per-Student Spending Distribution')
        print(f"Average Per-Student Spending: ${df_clean['per_student_spending'].mean():,.0f}")
        metrics['Average Per-Student Spending'] = df_clean['per_student_spending'].mean()
        
    if 'Students' in df_clean.columns and 'Budget' in df_clean.columns and not df_clean[['Students', 'Budget']].dropna().empty:
        figures['Budget_vs_Student_Enrollment_Scatter'] = px.scatter(df_clean, x='Students', y='Budget', color='SchoolBoard', title='Budget vs. Student Enrollment')

    if 'MeetingAttendanceRate' in df.columns and not df['MeetingAttendanceRate'].dropna().empty:
        avg_attendance_rate = df['MeetingAttendanceRate'].mean()
        metrics['Average Meeting Attendance Rate'] = avg_attendance_rate
        print(f"Average Meeting Attendance Rate: {avg_attendance_rate:.2f}%")
        figures['Meeting_Attendance_Rate_Histogram'] = px.histogram(df, x='MeetingAttendanceRate', title='Distribution of Meeting Attendance Rates')

    if not metrics and not figures:
        print("No sufficient data available after cleaning for any specific analysis within this function.")
        return {"message": "No sufficient data available for analysis."}

    return {
        "metrics": metrics,
        "figures": figures
    }

def college_selectivity_and_loan_repayment_rate_analysis(df):
    print("\n--- College Selectivity and Loan Repayment Rate Analysis ---")
    expected = {
        'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
        'Region': ['Region', 'State', 'STABBR'],
        'LoanRepaymentRate': ['LoanRepaymentRate', 'RepaymentRate', 'SixYearRepaymentRate'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
        'AvgSAT': ['AvgSAT', 'SATScore', 'AverageSATScore']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    if 'AcceptanceRate' in df.columns:
        df['SelectivityScore'] = 1 - pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    elif 'AvgSAT' in df.columns:
        df['SelectivityScore'] = pd.to_numeric(df['AvgSAT'], errors='coerce')
    else:
        print("Warning: No clear selectivity metric found (AcceptanceRate or AvgSAT). Selectivity analysis might be limited.")
        df['SelectivityScore'] = np.nan

    df = safe_numeric_conversion(df, 'SelectivityScore')
    df = safe_numeric_conversion(df, 'LoanRepaymentRate')
    df = df.dropna(subset=['SelectivityScore', 'LoanRepaymentRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_repayment_rate = df['LoanRepaymentRate'].mean()
    lowest_repayment_rate_college = df.loc[df['LoanRepaymentRate'].idxmin(), 'CollegeName']

    print(f"Average Loan Repayment Rate: {avg_repayment_rate:.2f}%")
    print(f"College with Lowest Repayment Rate: {lowest_repayment_rate_college}")

    fig1 = px.scatter(df, x='SelectivityScore', y='LoanRepaymentRate', hover_name='CollegeName', title='Loan Repayment Rate vs. Selectivity')
    fig2 = px.box(df, x='Region', y='LoanRepaymentRate', title='Loan Repayment Rate by Region')
    fig3 = px.histogram(df, x='LoanRepaymentRate', color='Region', title='Loan Repayment Rate Distribution by Region')

    return {
        "metrics": {
            "Average Loan Repayment Rate": avg_repayment_rate,
            "Lowest Repayment Rate College": lowest_repayment_rate_college
        },
        "figures": {
            "Loan_Repayment_Rate_vs_Selectivity_Scatter": fig1,
            "Loan_Repayment_Rate_by_Region_Box": fig2,
            "Loan_Repayment_Rate_Distribution_by_Region_Histogram": fig3
        }
    }

def faculty_composition_and_student_faculty_ratio_analysis(df):
    print("\n--- Faculty Composition and Student-Faculty Ratio Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'State': ['State', 'STABBR'],
        'TotalFaculty': ['TotalFaculty', 'FacultyCount'],
        'TenureTrackFaculty': ['TenureTrackFaculty', 'FTFaculty'],
        'AdjunctFaculty': ['AdjunctFaculty', 'PTFaculty', 'NonTenureTrackFaculty'],
        'TotalStudents': ['TotalStudents', 'Enrollment', 'UndergraduateEnrollment'],
        'GraduationRate': ['GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TotalFaculty'] = pd.to_numeric(df['TotalFaculty'], errors='coerce')
    df['TenureTrackFaculty'] = pd.to_numeric(df['TenureTrackFaculty'], errors='coerce')
    df['AdjunctFaculty'] = pd.to_numeric(df['AdjunctFaculty'], errors='coerce')
    df['TotalStudents'] = pd.to_numeric(df['TotalStudents'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    
    df['StudentFacultyRatio'] = df.apply(
        lambda row: row['TotalStudents'] / row['TotalFaculty'] if row['TotalFaculty'] > 0 else np.nan,
        axis=1
    )
    df['PercentTenureTrack'] = df.apply(
        lambda row: row['TenureTrackFaculty'] / row['TotalFaculty'] if row['TotalFaculty'] > 0 else np.nan,
        axis=1
    )
    df['PercentAdjunct'] = df.apply(
        lambda row: row['AdjunctFaculty'] / row['TotalFaculty'] if row['TotalFaculty'] > 0 else np.nan,
        axis=1
    )
    df = df.dropna(subset=['StudentFacultyRatio', 'PercentTenureTrack', 'PercentAdjunct', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_ratio = df['StudentFacultyRatio'].mean()
    best_ratio_institution = df.loc[df['StudentFacultyRatio'].idxmin(), 'InstitutionName']
    avg_percent_tenure_track = df['PercentTenureTrack'].mean()
    avg_percent_adjunct = df['PercentAdjunct'].mean()

    print(f"Average Student-Faculty Ratio: {avg_ratio:.2f}:1")
    print(f"Institution with Best Ratio: {best_ratio_institution}")
    print(f"Average Percent Tenure-Track Faculty: {avg_percent_tenure_track:.2%}")
    print(f"Average Percent Adjunct Faculty: {avg_percent_adjunct:.2%}")

    fig1 = px.scatter(df, x='StudentFacultyRatio', y='GraduationRate', hover_name='InstitutionName', title='Graduation Rate vs. Student-Faculty Ratio')
    fig2 = px.histogram(df, x='StudentFacultyRatio', title='Distribution of Student-Faculty Ratios')
    fig3 = px.box(df, x='State', y='StudentFacultyRatio', title='Student-Faculty Ratio by State')
    fig4 = px.scatter(df, x='PercentTenureTrack', y='StudentFacultyRatio', hover_name='InstitutionName', title='Student-Faculty Ratio vs. Percent Tenure-Track Faculty')
    fig5 = px.scatter(df, x='PercentAdjunct', y='StudentFacultyRatio', hover_name='InstitutionName', title='Student-Faculty Ratio vs. Percent Adjunct Faculty')

    return {
        "metrics": {
            "Average Student-Faculty Ratio": avg_ratio,
            "Best Ratio Institution": best_ratio_institution,
            "Average Percent Tenure-Track Faculty": avg_percent_tenure_track,
            "Average Percent Adjunct Faculty": avg_percent_adjunct
        },
        "figures": {
            "Graduation_Rate_vs_Student_Faculty_Ratio_Scatter": fig1,
            "Student_Faculty_Ratio_Distribution_Histogram": fig2,
            "Student_Faculty_Ratio_by_State_Box": fig3,
            "Student_Faculty_Ratio_vs_Percent_Tenure_Track_Scatter": fig4,
            "Student_Faculty_Ratio_vs_Percent_Adjunct_Scatter": fig5
        }
    }

def school_district_expenditure_and_test_score_analysis(df):
    print("\n--- School District Expenditure and Test Score Analysis ---")
    expected = {
        'SchoolDistrictID': ['SchoolDistrictID', 'DistrictID', 'ID'],
        'DistrictName': ['DistrictName', 'Name'],
        'PerStudentExpenditure': ['PerStudentExpenditure', 'SpendingPerStudent', 'DistrictExpenditure'],
        'StateTestScores': ['StateTestScores', 'AverageTestScore', 'DistrictAvgScore']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['PerStudentExpenditure'] = pd.to_numeric(df['PerStudentExpenditure'], errors='coerce')
    df['StateTestScores'] = pd.to_numeric(df['StateTestScores'], errors='coerce')
    df = df.dropna(subset=['PerStudentExpenditure', 'StateTestScores'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_expenditure = df['PerStudentExpenditure'].mean()
    avg_test_score = df['StateTestScores'].mean()

    print(f"Average Per-Student Expenditure: ${avg_expenditure:,.0f}")
    print(f"Average State Test Score: {avg_test_score:.2f}")

    fig1 = px.scatter(df, x='PerStudentExpenditure', y='StateTestScores', hover_name='DistrictName', title='State Test Scores vs. Per-Student Expenditure')
    fig2 = px.histogram(df, x='PerStudentExpenditure', title='Distribution of Per-Student Expenditure')
    fig3 = px.box(df, y='StateTestScores', title='State Test Score Distribution')

    return {
        "metrics": {
            "Average Per-Student Expenditure": avg_expenditure,
            "Average State Test Score": avg_test_score
        },
        "figures": {
            "State_Test_Scores_vs_Per_Student_Expenditure_Scatter": fig1,
            "Per_Student_Expenditure_Distribution_Histogram": fig2,
            "State_Test_Score_Distribution_Box": fig3
        }
    }

def college_selectivity_and_income_diversity_analysis(df):
    print("\n--- College Selectivity and Income Diversity Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'State': ['State', 'STABBR'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
        'IncomeDiversityIndex': ['IncomeDiversityIndex', 'PellGrantPercentage', 'PercentLowIncomeStudents', 'SESDiversity']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    df['IncomeDiversityIndex'] = pd.to_numeric(df['IncomeDiversityIndex'], errors='coerce')
    df = df.dropna(subset=['AcceptanceRate', 'IncomeDiversityIndex'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_acceptance_rate = df['AcceptanceRate'].mean()
    avg_income_diversity = df['IncomeDiversityIndex'].mean()

    print(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
    print(f"Average Income Diversity Index: {avg_income_diversity:.2f}")

    fig1 = px.scatter(df, x='AcceptanceRate', y='IncomeDiversityIndex', hover_name='InstitutionName', title='Income Diversity vs. Acceptance Rate')
    fig2 = px.box(df, x='State', y='IncomeDiversityIndex', title='Income Diversity Index by State')
    fig3 = px.histogram(df, x='AcceptanceRate', title='Distribution of Acceptance Rates')

    return {
        "metrics": {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Income Diversity Index": avg_income_diversity
        },
        "figures": {
            "Income_Diversity_vs_Acceptance_Rate_Scatter": fig1,
            "Income_Diversity_Index_by_State_Box": fig2,
            "Acceptance_Rates_Distribution_Histogram": fig3
        }
    }

def school_district_poverty_and_graduation_rate_correlation(df):
    print("\n--- School District Poverty and Graduation Rate Correlation ---")
    expected = {
        'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
        'DistrictName': ['DistrictName', 'Name'],
        'PovertyRate': ['PovertyRate', 'FreeReducedLunchRate', 'LowIncomeStudentRate'],
        'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['PovertyRate'] = pd.to_numeric(df['PovertyRate'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['PovertyRate', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_poverty_rate = df['PovertyRate'].mean()
    avg_grad_rate = df['GraduationRate'].mean()
    
    print(f"Average Poverty Rate: {avg_poverty_rate:.2f}%")
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

    fig1 = px.scatter(df, x='PovertyRate', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. Poverty Rate')
    fig2 = px.box(df, x='DistrictName', y='GraduationRate', title='Graduation Rate by School District')
    fig3 = px.histogram(df, x='PovertyRate', title='Distribution of Poverty Rates')

    return {
        "metrics": {
            "Average Poverty Rate": avg_poverty_rate,
            "Average Graduation Rate": avg_grad_rate
        },
        "figures": {
            "Graduation_Rate_vs_Poverty_Rate_Scatter": fig1,
            "Graduation_Rate_by_School_District_Box": fig2,
            "Poverty_Rates_Distribution_Histogram": fig3
        }
    }

def university_enrollment_and_retention_analysis(df):
    print("\n--- University Enrollment and Retention Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'Region': ['Region', 'State', 'STABBR'],
        'RetentionRate': ['RetentionRate', 'FirstYearRetention'],
        'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['RetentionRate'] = pd.to_numeric(df['RetentionRate'], errors='coerce')
    df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
    df = df.dropna(subset=['RetentionRate', 'UndergraduateEnrollment'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    avg_retention_rate = df['RetentionRate'].mean()
    avg_enrollment = df['UndergraduateEnrollment'].mean()
    
    print(f"Average Retention Rate: {avg_retention_rate:.2f}%")
    print(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
    
    fig1 = px.scatter(df, x='UndergraduateEnrollment', y='RetentionRate', hover_name='Name', title='Retention Rate vs. Undergraduate Enrollment')
    fig2 = px.box(df, x='Region', y='RetentionRate', title='Retention Rate by Region')
    fig3 = px.bar(df.groupby('Region')['UndergraduateEnrollment'].sum().reset_index(), x='Region', y='UndergraduateEnrollment', title='Total Enrollment by Region')

    return {
        "metrics": {
            "Average Retention Rate": avg_retention_rate,
            "Average Undergraduate Enrollment": avg_enrollment
        },
        "figures": {
            "Retention_Rate_vs_Undergraduate_Enrollment_Scatter": fig1,
            "Retention_Rate_by_Region_Box": fig2,
            "Total_Enrollment_by_Region_Bar": fig3
        }
    }

def college_application_and_enrollment_funnel_analysis(df):
    print("\n--- College Application and Enrollment Funnel Analysis ---")
    expected = {
        'InstitutionState': ['InstitutionState', 'State', 'STABBR'],
        'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
        'ApplicationsReceived': ['ApplicationsReceived', 'Applicants', 'TotalApplicants'],
        'Admitted': ['Admitted', 'AcceptedStudents'],
        'Enrolled': ['Enrolled', 'MatriculatedStudents'],
        'GraduationRate': ['GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
    df['Admitted'] = pd.to_numeric(df['Admitted'], errors='coerce')
    df['Enrolled'] = pd.to_numeric(df['Enrolled'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['ApplicationsReceived', 'Admitted', 'Enrolled', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}
    
    df['AcceptanceRate'] = df.apply(
        lambda row: row['Admitted'] / row['ApplicationsReceived'] if row['ApplicationsReceived'] > 0 else 0,
        axis=1
    )
    df['YieldRate'] = df.apply(
        lambda row: row['Enrolled'] / row['Admitted'] if row['Admitted'] > 0 else 0,
        axis=1
    )

    total_applications = df['ApplicationsReceived'].sum()
    total_admitted = df['Admitted'].sum()
    total_enrolled = df['Enrolled'].sum()
    
    funnel_data = pd.DataFrame({
        'Stage': ['Applications Received', 'Admitted', 'Enrolled'],
        'Count': [total_applications, total_admitted, total_enrolled]
    })
    
    fig1 = px.funnel(funnel_data, x='Count', y='Stage', title='Overall College Admissions Funnel')

    fig2 = px.box(df, x='InstitutionState', y='AcceptanceRate', title='Acceptance Rate by State')
    fig3 = px.box(df, x='InstitutionState', y='YieldRate', title='Yield Rate by State')
    fig4 = px.scatter(df, x='AcceptanceRate', y='YieldRate', hover_name='CollegeName', title='Yield Rate vs. Acceptance Rate')

    return {
        "metrics": {
            "Overall Acceptance Rate": total_admitted / total_applications if total_applications > 0 else 0,
            "Overall Yield Rate": total_enrolled / total_admitted if total_admitted > 0 else 0
        },
        "figures": {
            "Overall_College_Admissions_Funnel": fig1,
            "Acceptance_Rate_by_State_Box": fig2,
            "Yield_Rate_by_State_Box": fig3,
            "Yield_Rate_vs_Acceptance_Rate_Scatter": fig4
        }
    }

def school_district_staffing_and_student_enrollment_analysis(df):
    print("\n--- School District Staffing and Student Enrollment Analysis ---")
    expected = {
        'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
        'DistrictName': ['DistrictName', 'Name', 'SchoolDistrict'],
        'StudentEnrollment': ['StudentEnrollment', 'Enrollment', 'TotalStudents'],
        'StaffCount': ['StaffCount', 'TotalStaff', 'FTStaff'],
        'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StudentEnrollment'] = pd.to_numeric(df['StudentEnrollment'], errors='coerce')
    df['StaffCount'] = pd.to_numeric(df['StaffCount'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['StudentEnrollment', 'StaffCount', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['student_staff_ratio'] = df.apply(
        lambda row: row['StudentEnrollment'] / row['StaffCount'] if row['StaffCount'] > 0 else np.nan,
        axis=1
    )
    df = df.dropna(subset=['student_staff_ratio'])

    avg_ratio = df['student_staff_ratio'].mean()
    avg_grad_rate = df['GraduationRate'].mean()
    
    print(f"Average Student-Staff Ratio: {avg_ratio:.2f}")
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

    fig1 = px.scatter(df, x='student_staff_ratio', y='GraduationRate', hover_name='SchoolID', title='Graduation Rate vs. Student-Staff Ratio')
    fig2 = px.box(df, y='student_staff_ratio', title='Student-Staff Ratio Distribution')
    fig3 = px.bar(df.groupby('DistrictName')['GraduationRate'].mean().nlargest(20).reset_index(), x='DistrictName', y='GraduationRate', title='Average Graduation Rate by District (Top 20)')

    return {
        "metrics": {
            "Average Student-Staff Ratio": avg_ratio,
            "Average Graduation Rate": avg_grad_rate
        },
        "figures": {
            "Graduation_Rate_vs_Student_Staff_Ratio_Scatter": fig1,
            "Student_Staff_Ratio_Distribution_Box": fig2,
            "Average_Graduation_Rate_by_District_Bar": fig3
        }
    }

def college_sat_scores_and_7_year_graduation_rate_analysis(df):
    print("\n--- College SAT Scores and 7-Year Graduation Rate Analysis ---")
    expected = {
        'StateAbbrev': ['StateAbbrev', 'State', 'STABBR'],
        'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
        'SATScore': ['SATScore', 'AvgSAT', 'MedianSAT'],
        'Graduation7YearRate': ['Graduation7YearRate', '7YearGradRate', 'LongTermGraduationRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['SATScore'] = pd.to_numeric(df['SATScore'], errors='coerce')
    df['Graduation7YearRate'] = pd.to_numeric(df['Graduation7YearRate'], errors='coerce')
    df = df.dropna(subset=['SATScore', 'Graduation7YearRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_sat_score = df['SATScore'].mean()
    avg_grad_rate = df['Graduation7YearRate'].mean()

    print(f"Average SAT Score: {avg_sat_score:.0f}")
    print(f"Average 7-Year Graduation Rate: {avg_grad_rate:.2f}%")
    
    fig1 = px.scatter(df, x='SATScore', y='Graduation7YearRate', hover_name='CollegeName', title='7-Year Graduation Rate vs. SAT Score')
    fig2 = px.box(df, x='StateAbbrev', y='Graduation7YearRate', title='7-Year Graduation Rate by State')
    fig3 = px.histogram(df, x='SATScore', title='Distribution of SAT Scores')

    return {
        "metrics": {
            "Average SAT Score": avg_sat_score,
            "Average 7-Year Graduation Rate": avg_grad_rate
        },
        "figures": {
            "7_Year_Graduation_Rate_vs_SAT_Score_Scatter": fig1,
            "7_Year_Graduation_Rate_by_State_Box": fig2,
            "SAT_Scores_Distribution_Histogram": fig3
        }
    }

def school_attendance_and_graduation_rate_analysis(df):
    print("\n--- School Attendance and Graduation Rate Analysis ---")
    expected = {
        'DistrictName': ['DistrictName', 'Name'],
        'SchoolName': ['SchoolName', 'Name'],
        'AttendanceRate': ['AttendanceRate', 'AvgAttendance'],
        'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AttendanceRate'] = pd.to_numeric(df['AttendanceRate'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['AttendanceRate', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_attendance = df['AttendanceRate'].mean()
    avg_grad_rate = df['GraduationRate'].mean()

    print(f"Average Attendance Rate: {avg_attendance:.2f}%")
    print(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
    
    fig1 = px.scatter(df, x='AttendanceRate', y='GraduationRate', hover_name='SchoolName', title='Graduation Rate vs. Attendance Rate')
    fig2 = px.box(df, x='DistrictName', y='AttendanceRate', title='Attendance Rate by District')
    fig3 = px.box(df, y='GraduationRate', title='Graduation Rate Distribution')

    return {
        "metrics": {
            "Average Attendance Rate": avg_attendance,
            "Average Graduation Rate": avg_grad_rate
        },
        "figures": {
            "Graduation_Rate_vs_Attendance_Rate_Scatter": fig1,
            "Attendance_Rate_by_District_Box": fig2,
            "Graduation_Rate_Distribution_Box": fig3
        }
    }


def university_undergraduate_and_graduate_enrollment_analysis(df):
    print("\n--- University Undergraduate and Graduate Enrollment Analysis ---")
    expected = {
        'CollegeID': ['CollegeID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'State': ['State', 'STABBR'],
        'UndergradEnrollment': ['UndergradEnrollment', 'UGEnrollment', 'UndergraduateStudents'],
        'GradEnrollment': ['GradEnrollment', 'GraduateEnrollment', 'GraduateStudents']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['UndergradEnrollment'] = pd.to_numeric(df['UndergradEnrollment'], errors='coerce')
    df['GradEnrollment'] = pd.to_numeric(df['GradEnrollment'], errors='coerce')
    df = df.dropna(subset=['UndergradEnrollment', 'GradEnrollment'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_undergrad = df['UndergradEnrollment'].sum()
    total_grad = df['GradEnrollment'].sum()

    print(f"Total Undergraduate Enrollment: {total_undergrad:,.0f}")
    print(f"Total Graduate Enrollment: {total_grad:,.0f}")
    
    enrollment_data = pd.DataFrame({
        'Level': ['Undergraduate', 'Graduate'],
        'Count': [total_undergrad, total_grad]
    })
    fig1 = px.pie(enrollment_data, names='Level', values='Count', title='Total Enrollment by Level')

    fig2 = px.scatter(df, x='UndergradEnrollment', y='GradEnrollment', hover_name='Name', title='Graduate vs. Undergraduate Enrollment')
    fig3 = px.bar(df.groupby('State')['UndergradEnrollment'].sum().nlargest(20).reset_index(), x='State', y='UndergradEnrollment', title='Undergrad Enrollment by State (Top 20)')

    return {
        "metrics": {
            "Total Undergrad Enrollment": total_undergrad,
            "Total Grad Enrollment": total_grad
        },
        "figures": {
            "Total_Enrollment_by_Level_Pie": fig1,
            "Graduate_vs_Undergraduate_Enrollment_Scatter": fig2,
            "Undergrad_Enrollment_by_State_Bar": fig3
        }
    }


def college_selectivity_yield_and_default_rate_analysis(df):
    print("\n--- College Selectivity, Yield, and Default Rate Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'City': ['City', 'CampusCity'],
        'State': ['State', 'STABBR'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
        'Yield': ['Yield', 'YieldRate'],
        'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
    df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
    df = df.dropna(subset=['AcceptanceRate', 'Yield', 'DefaultRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_acceptance_rate = df['AcceptanceRate'].mean()
    avg_yield = df['Yield'].mean()
    avg_default_rate = df['DefaultRate'].mean()
    
    print(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
    print(f"Average Yield Rate: {avg_yield:.2f}%")
    print(f"Average Default Rate: {avg_default_rate:.2f}%")
    
    fig1 = px.scatter(df, x='Yield', y='DefaultRate', hover_name='InstitutionName', title='Default Rate vs. Yield Rate')
    fig2 = px.box(df, x='State', y='Yield', title='Yield Rate by State')
    fig3 = px.box(df, x='State', y='DefaultRate', title='Default Rate by State')
    fig4 = px.scatter(df, x='AcceptanceRate', y='Yield', hover_name='InstitutionName', title='Yield Rate vs. Acceptance Rate')

    return {
        "metrics": {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Yield Rate": avg_yield,
            "Average Default Rate": avg_default_rate
        },
        "figures": {
            "Default_Rate_vs_Yield_Rate_Scatter": fig1,
            "Yield_Rate_by_State_Box": fig2,
            "Default_Rate_by_State_Box": fig3,
            "Yield_Rate_vs_Acceptance_Rate_Scatter": fig4
        }
    }


def school_district_classification_and_expenditure_analysis(df):
    print("\n--- School District Classification and Expenditure Analysis ---")
    expected = {
        'DistrictName': ['DistrictName', 'Name'],
        'DistrictType': ['DistrictType', 'Classification', 'UrbanRural'],
        'Enrollment': ['Enrollment', 'TotalEnrollment', 'StudentCount'],
        'perStudentExpenditure': ['perStudentExpenditure', 'SpendingPerStudent', 'PerPupilExpenditure']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
    df['perStudentExpenditure'] = pd.to_numeric(df['perStudentExpenditure'], errors='coerce')
    df = df.dropna(subset=['Enrollment', 'perStudentExpenditure'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_expenditure = df['perStudentExpenditure'].mean()
    total_enrollment = df['Enrollment'].sum()
    
    print(f"Average Per-Student Expenditure: ${avg_expenditure:,.0f}")
    print(f"Total Enrollment Across Districts: {total_enrollment:,.0f}")
    
    fig1 = px.histogram(df, x='perStudentExpenditure', title='Distribution of Per-Student Expenditure')
    fig2 = px.scatter(df, x='Enrollment', y='perStudentExpenditure', hover_name='DistrictName', title='Per-Student Expenditure vs. Enrollment')
    fig3 = px.box(df, y='perStudentExpenditure', title='Per-Student Expenditure Distribution')
    
    fig4 = None
    if 'DistrictType' in df.columns and not df['DistrictType'].dropna().empty:
        fig4 = px.box(df, x='DistrictType', y='perStudentExpenditure', title='Per-Student Expenditure by District Type')

    return {
        "metrics": {
            "Average Per-Student Expenditure": avg_expenditure,
            "Total Enrollment": total_enrollment
        },
        "figures": {
            "Per_Student_Expenditure_Distribution_Histogram": fig1,
            "Per_Student_Expenditure_vs_Enrollment_Scatter": fig2,
            "Per_Student_Expenditure_Distribution_Box": fig3,
            "Per_Student_Expenditure_by_District_Type_Box": fig4
        }
    }


def college_scholarship_and_post_graduation_earnings_analysis(df):
    print("\n--- College Scholarship and Post-Graduation Earnings Analysis ---")
    expected = {
        'CollegeID': ['CollegeID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'State': ['State', 'STABBR'],
        'MedianEarnings': ['MedianEarnings', 'PostGradEarnings', 'AvgSalary'],
        'AvgScholarship': ['AvgScholarship', 'AverageGrantAid', 'ScholarshipAmount']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['MedianEarnings'] = pd.to_numeric(df['MedianEarnings'], errors='coerce')
    df['AvgScholarship'] = pd.to_numeric(df['AvgScholarship'], errors='coerce')
    df = df.dropna(subset=['MedianEarnings', 'AvgScholarship'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_earnings = df['MedianEarnings'].mean()
    avg_scholarship = df['AvgScholarship'].mean()

    print(f"Average Median Earnings: ${avg_earnings:,.0f}")
    print(f"Average Scholarship Amount: ${avg_scholarship:,.0f}")

    fig1 = px.scatter(df, x='AvgScholarship', y='MedianEarnings', hover_name='Name', title='Median Earnings vs. Average Scholarship Amount')
    fig2 = px.box(df, x='State', y='MedianEarnings', title='Median Earnings by State')
    fig3 = px.box(df, x='State', y='AvgScholarship', title='Average Scholarship Amount by State')

    return {
        "metrics": {
            "Average Median Earnings": avg_earnings,
            "Average Scholarship Amount": avg_scholarship
        },
        "figures": {
            "Median_Earnings_vs_Average_Scholarship_Amount_Scatter": fig1,
            "Median_Earnings_by_State_Box": fig2,
            "Average_Scholarship_Amount_by_State_Box": fig3
        }
    }


def university_campus_enrollment_and_tuition_analysis(df):
    print("\n--- University Campus Enrollment and Tuition Analysis ---")
    expected = {
        'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
        'CampusName': ['CampusName', 'BranchName', 'Campus'],
        'UndergradEnrollment': ['UndergradEnrollment', 'UGEnrollment', 'Enrollment'],
        'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
        'GraduationRate': ['GraduationRate', 'CompletionRate']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['UndergradEnrollment'] = pd.to_numeric(df['UndergradEnrollment'], errors='coerce')
    df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
    df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
    df = df.dropna(subset=['UndergradEnrollment', 'Tuition', 'GraduationRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_enrollment = df['UndergradEnrollment'].sum()
    avg_tuition = df['Tuition'].mean()
    
    print(f"Total Undergraduate Enrollment (across campuses): {total_enrollment:,.0f}")
    print(f"Average Tuition (across campuses): ${avg_tuition:,.0f}")
    
    fig1 = px.scatter(df, x='UndergradEnrollment', y='Tuition', color='CampusName', title='Tuition vs. Enrollment by Campus')
    fig2 = px.box(df, x='CampusName', y='Tuition', title='Tuition Distribution by Campus')
    fig3 = px.bar(df.groupby('CampusName')['GraduationRate'].mean().reset_index(), x='CampusName', y='GraduationRate', title='Average Graduation Rate by Campus')

    return {
        "metrics": {
            "Total Undergraduate Enrollment": total_enrollment,
            "Average Tuition": avg_tuition
        },
        "figures": {
            "Tuition_vs_Enrollment_by_Campus_Scatter": fig1,
            "Tuition_Distribution_by_Campus_Box": fig2,
            "Average_Graduation_Rate_by_Campus_Bar": fig3
        }
    }


def university_control_debt_and_long_term_income_analysis(df):
    print("\n--- University Control, Debt, and Long-Term Income Analysis ---")
    expected = {
        'UniversityState': ['UniversityState', 'State', 'STABBR'],
        'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
        'PublicPrivate': ['PublicPrivate', 'Control', 'InstitutionType'],
        'AvgStudentDebt': ['AvgStudentDebt', 'MedianDebt', 'StudentLoanDebt'],
        'AvgIncome25Yr': ['AvgIncome25Yr', 'LongTermEarnings', 'MedianEarnings25Yr']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AvgStudentDebt'] = pd.to_numeric(df['AvgStudentDebt'], errors='coerce')
    df['AvgIncome25Yr'] = pd.to_numeric(df['AvgIncome25Yr'], errors='coerce')
    df = df.dropna(subset=['AvgStudentDebt', 'AvgIncome25Yr'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_debt = df['AvgStudentDebt'].mean()
    avg_income = df['AvgIncome25Yr'].mean()
    
    print(f"Average Student Debt: ${avg_debt:,.0f}")
    print(f"Average 25-Year Income: ${avg_income:,.0f}")
    
    fig1 = px.scatter(df, x='AvgStudentDebt', y='AvgIncome25Yr', color='PublicPrivate', hover_name='UniversityName', title='25-Year Income vs. Student Debt by Institution Type')
    fig2 = px.box(df, x='PublicPrivate', y='AvgStudentDebt', title='Student Debt Distribution by Institution Type')
    fig3 = px.bar(df.groupby('UniversityState')['AvgIncome25Yr'].mean().nlargest(20).reset_index(), x='UniversityState', y='AvgIncome25Yr', title='Average 25-Year Income by State (Top 20)')
    fig4 = px.box(df, x='PublicPrivate', y='AvgIncome25Yr', title='25-Year Income Distribution by Institution Type')

    return {
        "metrics": {
            "Average Student Debt": avg_debt,
            "Average 25-Year Income": avg_income
        },
        "figures": {
            "25_Year_Income_vs_Student_Debt_Scatter": fig1,
            "Student_Debt_Distribution_by_Institution_Type_Box": fig2,
            "Average_25_Year_Income_by_State_Bar": fig3,
            "25_Year_Income_Distribution_by_Institution_Type_Box": fig4
        }
    }


def college_admissions_and_student_loan_analysis(df):
    print("\n--- College Admissions and Student Loan Analysis ---")
    expected = {
        'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
        'Name': ['Name', 'InstitutionName', 'INSTNM'],
        'Region': ['Region', 'State', 'STABBR'],
        'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
        'SubsidizedLoanPercent': ['SubsidizedLoanPercent', 'PercentSubsidizedLoans', 'SubsidizedLoanShare'],
        'UnsubsidizedLoanPercent': ['UnsubsidizedLoanPercent', 'PercentUnsubsidizedLoans', 'UnsubsidizedLoanShare']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
    df['SubsidizedLoanPercent'] = pd.to_numeric(df['SubsidizedLoanPercent'], errors='coerce')
    df['UnsubsidizedLoanPercent'] = pd.to_numeric(df['UnsubsidizedLoanPercent'], errors='coerce')
    df = df.dropna(subset=['AcceptanceRate', 'SubsidizedLoanPercent', 'UnsubsidizedLoanPercent'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_acceptance_rate = df['AcceptanceRate'].mean()
    avg_subsidized_loan_percent = df['SubsidizedLoanPercent'].mean()
    avg_unsubsidized_loan_percent = df['UnsubsidizedLoanPercent'].mean()

    print(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
    print(f"Average Subsidized Loan Percent: {avg_subsidized_loan_percent:.2f}%")
    print(f"Average Unsubsidized Loan Percent: {avg_unsubsidized_loan_percent:.2f}%")

    fig1 = px.scatter(df, x='AcceptanceRate', y='SubsidizedLoanPercent', hover_name='Name', title='Subsidized Loan Percent vs. Acceptance Rate')
    fig2 = px.box(df, x='Region', y='SubsidizedLoanPercent', title='Subsidized Loan Percent by Region')
    fig3 = px.bar(df.groupby('Region')['SubsidizedLoanPercent'].mean().nlargest(20).reset_index(), x='Region', y='SubsidizedLoanPercent', title='Average Subsidized Loan Percent by Region (Top 20)')
    fig4 = px.scatter(df, x='AcceptanceRate', y='UnsubsidizedLoanPercent', hover_name='Name', title='Unsubsidized Loan Percent vs. Acceptance Rate')

    return {
        "metrics": {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Subsidized Loan Percent": avg_subsidized_loan_percent,
            "Average Unsubsidized Loan Percent": avg_unsubsidized_loan_percent
        },
        "figures": {
            "Subsidized_Loan_Percent_vs_Acceptance_Rate_Scatter": fig1,
            "Subsidized_Loan_Percent_by_Region_Box": fig2,
            "Average_Subsidized_Loan_Percent_by_Region_Bar": fig3,
            "Unsubsidized_Loan_Percent_vs_Acceptance_Rate_Scatter": fig4
        }
    }


def school_district_attendance_and_absenteeism_analysis(df):
    print("\n--- School District Attendance and Absenteeism Analysis ---")
    expected = {
        'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
        'Name': ['Name', 'DistrictName'],
        'AttendanceRate': ['AttendanceRate', 'AvgAttendance'],
        'ChronicAbsenteeismRate': ['ChronicAbsenteeismRate', 'ChronicAbsenceRate']
    }
    df, missing = check_and_rename_columns(df, expected)
    
    if missing:
        show_missing_columns_warning(missing, expected)
        show_general_insights(df, "General Analysis")
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['AttendanceRate'] = pd.to_numeric(df['AttendanceRate'], errors='coerce')
    df['ChronicAbsenteeismRate'] = pd.to_numeric(df['ChronicAbsenteeismRate'], errors='coerce')
    df = df.dropna(subset=['AttendanceRate', 'ChronicAbsenteeismRate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_attendance = df['AttendanceRate'].mean()
    avg_absenteeism = df['ChronicAbsenteeismRate'].mean()

    print(f"Average Attendance Rate: {avg_attendance:.2f}%")
    print(f"Average Chronic Absenteeism Rate: {avg_absenteeism:.2f}%")

    fig1 = px.scatter(df, x='AttendanceRate', y='ChronicAbsenteeismRate', hover_name='Name', title='Chronic Absenteeism Rate vs. Attendance Rate')
    fig2 = px.histogram(df, x='ChronicAbsenteeismRate', title='Distribution of Chronic Absenteeism Rate')
    fig3 = px.box(df, y='ChronicAbsenteeismRate', title='Chronic Absenteeism Rate Distribution')

    return {
        "metrics": {
            "Average Attendance Rate": avg_attendance,
            "Average Chronic Absenteeism Rate": avg_absenteeism
        },
        "figures": {
            "Chronic_Absenteeism_Rate_vs_Attendance_Rate_Scatter": fig1,
            "Chronic_Absenteeism_Rate_Distribution_Histogram": fig2,
            "Chronic_Absenteeism_Rate_Distribution_Box": fig3
        }
    }
def show_missing_columns_warning(missing_cols, matched_cols=None):
    """Display warning for missing columns"""
    print("\n⚠ Required Columns Not Found!")
    print("The following columns are needed but missing:")
    for col in missing_cols:
        if matched_cols and matched_cols.get(col):
            print(f"- {col} (matched to: {matched_cols[col]})")
        else:
            print(f"- {col}")

def show_general_insights(df, title="General Insights"):
    """Show general data overview and insights"""
    print(f"\n===== {title} =====")
    
    # Basic data information
    total_rows = len(df)
    total_columns = len(df.columns)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Total Rows: {total_rows:,}")
    print(f"Total Columns: {total_columns}")
    print(f"Numeric Columns ({len(numeric_cols)}): {numeric_cols}")
    print(f"Categorical Columns ({len(categorical_cols)}): {categorical_cols}")
    
    # Missing values check
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        for col, count in missing_values[missing_values > 0].items():
            percentage = (count / total_rows) * 100
            print(f"- {col}: {count} ({percentage:.1f}%)")
    else:
        print("\n✅ No missing values found")
    
    # Numeric data summary
    if numeric_cols:
        print(f"\n===== Numeric Data Summary =====")
        numeric_summary = df[numeric_cols].describe()
        print(numeric_summary)
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        for col in numeric_cols:
            skewness = df[col].skew()
            print(f"- {col} skewness: {skewness:.2f}")
    
    # Categorical data summary
    if categorical_cols:
        print(f"\n===== Categorical Data Summary =====")
        for col in categorical_cols[:3]:  # Show first 3 categorical columns
            unique_count = df[col].nunique()
            print(f"\n{col} (Unique values: {unique_count}):")
            value_counts = df[col].value_counts().head(10)
            for value, count in value_counts.items():
                percentage = (count / total_rows) * 100
                print(f"  {value}: {count} ({percentage:.1f}%)")
            if unique_count > 10:
                print(f"  ... and {unique_count - 10} more unique values")
    
    # Data types
    print(f"\n===== Data Types =====")
    for col, dtype in df.dtypes.items():
        print(f"- {col}: {dtype}")
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True).sum()
    print(f"\nTotal Memory Usage: {memory_usage / 1024 / 1024:.2f} MB")

def academic_performance(df):
    """Analyze academic performance metrics with visualizations"""
    print("\n===== ACADEMIC PERFORMANCE ANALYSIS =====")
    
    # Expected columns for academic performance analysis
    expected = ['student_id', 'gpa', 'test_score', 'attendance_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched[col] is None]
    
    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return
    
    # Rename columns to standardized names
    df = df.rename(columns={v: k for k, v in matched.items() if v})
    
    # Calculate key metrics
    avg_gpa = df['gpa'].mean()
    median_gpa = df['gpa'].median()
    std_gpa = df['gpa'].std()
    min_gpa = df['gpa'].min()
    max_gpa = df['gpa'].max()
    
    # Performance categories
    at_risk = (df['gpa'] < 2.0).sum()
    needs_improvement = ((df['gpa'] >= 2.0) & (df['gpa'] < 3.0)).sum()
    good_performance = ((df['gpa'] >= 3.0) & (df['gpa'] < 3.5)).sum()
    strong_performers = (df['gpa'] >= 3.5).sum()
    
    # Test score metrics
    avg_test_score = df['test_score'].mean()
    
    # Attendance metrics
    avg_attendance = df['attendance_rate'].mean()
    poor_attendance = (df['attendance_rate'] < 0.8).sum()
    
    # Print comprehensive metrics
    print(f"\n📊 GPA STATISTICS:")
    print(f"Average GPA: {avg_gpa:.2f}")
    print(f"Median GPA: {median_gpa:.2f}")
    print(f"Standard Deviation: {std_gpa:.2f}")
    print(f"Range: {min_gpa:.2f} - {max_gpa:.2f}")
    
    print(f"\n🎯 PERFORMANCE CATEGORIES:")
    print(f"At-Risk Students (GPA < 2.0): {at_risk} ({at_risk/len(df)*100:.1f}%)")
    print(f"Needs Improvement (2.0 ≤ GPA < 3.0): {needs_improvement} ({needs_improvement/len(df)*100:.1f}%)")
    print(f"Good Performance (3.0 ≤ GPA < 3.5): {good_performance} ({good_performance/len(df)*100:.1f}%)")
    print(f"Strong Performers (GPA ≥ 3.5): {strong_performers} ({strong_performers/len(df)*100:.1f}%)")
    
    print(f"\n📝 TEST SCORES:")
    print(f"Average Test Score: {avg_test_score:.2f}")
    
    print(f"\n📅 ATTENDANCE:")
    print(f"Average Attendance Rate: {avg_attendance:.1%}")
    print(f"Poor Attendance (< 80%): {poor_attendance} ({poor_attendance/len(df)*100:.1f}%)")
    
    # Correlation analysis
    correlation_gpa_attendance = df['gpa'].corr(df['attendance_rate'])
    correlation_gpa_test = df['gpa'].corr(df['test_score'])
    correlation_attendance_test = df['attendance_rate'].corr(df['test_score'])
    
    print(f"\n🔗 CORRELATIONS:")
    print(f"GPA vs Attendance Rate: {correlation_gpa_attendance:.3f}")
    print(f"GPA vs Test Score: {correlation_gpa_test:.3f}")
    print(f"Attendance vs Test Score: {correlation_attendance_test:.3f}")
    
    # Visualizations
    print(f"\n📈 GENERATING VISUALIZATIONS...")
    
    # 1. GPA Distribution
    fig1 = px.histogram(df, x='gpa', nbins=20, 
                        title="GPA Distribution",
                        labels={'gpa': 'GPA', 'count': 'Number of Students'},
                        color_discrete_sequence=['#1f77b4'])
    fig1.add_vline(x=avg_gpa, line_dash="dash", line_color="red", 
                   annotation_text=f"Average: {avg_gpa:.2f}")
    fig1.show()
    
    # 2. Attendance Rate vs GPA
    fig2 = px.scatter(df, x='attendance_rate', y='gpa', 
                      title="Attendance Rate vs GPA",
                      labels={'attendance_rate': 'Attendance Rate', 'gpa': 'GPA'},
                      trendline="ols",
                      color_discrete_sequence=['#ff7f0e'])
    fig2.show()
    
    # 3. Test Score vs GPA
    fig3 = px.scatter(df, x='test_score', y='gpa',
                      title="Test Scores vs GPA",
                      labels={'test_score': 'Test Score', 'gpa': 'GPA'},
                      trendline="ols",
                      color_discrete_sequence=['#2ca02c'])
    fig3.show()
    
    # 4. Performance Category Distribution
    performance_data = pd.DataFrame({
        'Category': ['At-Risk (<2.0)', 'Needs Improvement (2.0-3.0)', 
                     'Good (3.0-3.5)', 'Strong (≥3.5)'],
        'Count': [at_risk, needs_improvement, good_performance, strong_performers]
    })
    
    fig4 = px.pie(performance_data, values='Count', names='Category',
                  title="Student Performance Distribution",
                  color_discrete_sequence=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    fig4.show()
    
    # 5. Box plot of GPA by attendance categories
    df['attendance_category'] = pd.cut(df['attendance_rate'], 
                                       bins=[0, 0.7, 0.8, 0.9, 1.0],
                                       labels=['Poor (<70%)', 'Fair (70-80%)', 
                                              'Good (80-90%)', 'Excellent (≥90%)'])
    
    fig5 = px.box(df, x='attendance_category', y='gpa',
                  title="GPA Distribution by Attendance Category",
                  labels={'attendance_category': 'Attendance Category', 'gpa': 'GPA'})
    fig5.show()
    
    # Insights and recommendations
    print(f"\n💡 KEY INSIGHTS:")
    if correlation_gpa_attendance > 0.5:
        print("- Strong positive correlation between attendance and GPA")
    elif correlation_gpa_attendance > 0.3:
        print("- Moderate positive correlation between attendance and GPA")
    
    if at_risk > 0:
        print(f"- {at_risk} students are at risk and may need intervention")
    
    if poor_attendance > 0:
        print(f"- {poor_attendance} students have poor attendance (< 80%)")
    
    print(f"\n📋 RECOMMENDATIONS:")
    if at_risk > 0:
        print("- Implement academic support programs for at-risk students")
    if poor_attendance > 0:
        print("- Develop attendance improvement strategies")
    if correlation_gpa_attendance > 0.3:
        print("- Focus on attendance as a key factor for academic success")
    
    return {
        "avg_gpa": avg_gpa,
        "at_risk_students": at_risk,
        "strong_performers": strong_performers,
        "avg_attendance": avg_attendance,
        "correlations": {
            "gpa_attendance": correlation_gpa_attendance,
            "gpa_test": correlation_gpa_test
        }
    }


def main_backend(file, encoding='utf-8', category=None, analysis=None, specific_analysis_name=None):
    # Load data
    df = load_data(file, encoding)
    if df is None:
        return {"error": "Failed to load data"}

    # Mapping of specific educational analyses to functions
    specific_education_function_mapping = {
        "Academic Performance": academic_performance,
        "Demographic Analysis": demographic_analysis,
        "Course Analysis": course_analysis,
        "Attendance Analysis": attendance_analysis,
        "Behavioral Analysis": behavioral_analysis,
        "Program Evaluation": program_evaluation,
        "School District Performance and Socioeconomic Analysis": school_district_performance_and_socioeconomic_analysis,
        "Higher Education Institution Cost of Attendance Analysis": higher_education_institution_cost_of_attendance_analysis,
        "State-Level Average Cost of Attendance Trend Analysis": state_level_average_cost_of_attendance_trend_analysis,
        "University Financials and Student Outcome Analysis": university_financials_and_student_outcome_analysis,
        "University Enrollment, Expenditure, and Graduation Rate Analysis": university_enrollment_expenditure_and_graduation_rate_analysis,
        "College Admissions and Graduation Rate Analysis": college_admissions_and_graduation_rate_analysis,
        "School-Level Student-Teacher Ratio and Class Size Analysis": school_level_student_teacher_ratio_and_class_size_analysis,
        "College Enrollment and Income Trend Analysis": college_enrollment_and_income_trend_analysis,
        "School District Resource Adequacy Analysis": school_district_resource_adequacy_analysis,
        "Higher Education Institution ROI and Default Rate Analysis": higher_education_institution_roi_and_default_rate_analysis,
        "School District Budget and Student Outcome Analysis": school_district_budget_and_student_outcome_analysis,
        "University Selectivity and Student Debt Analysis": university_selectivity_and_student_debt_analysis,
        "College Admissions, Graduation, and Salary Outcome Analysis": college_admissions_graduation_and_salary_outcome_analysis,
        "School Funding and Local Income Level Analysis": school_funding_and_local_income_level_analysis,
        "Pell Grant Recipient Graduation and Loan Default Rate Analysis": pell_grant_recipient_graduation_and_loan_default_rate_analysis,
        "College Selectivity and Graduation Rate Analysis": college_selectivity_and_graduation_rate_analysis,
        "School District Demographics and Student-Teacher Ratio Analysis": school_district_demographics_and_student_teacher_ratio_analysis,
        "College Tuition and Enrollment Statistics Analysis": college_tuition_and_enrollment_statistics_analysis,
        "School Special Needs and Counselor Ratio Analysis": school_special_needs_and_counselor_ratio_analysis,
        "University Graduation Rate and Diversity Index Analysis": university_graduation_rate_and_diversity_index_analysis,
        "Post-Graduation Earnings and Debt Analysis": post_graduation_earnings_and_debt_analysis,
        "School District Test Score and Graduation Rate Analysis": school_district_test_score_and_graduation_rate_analysis,
        "College Admissions and Loan Default Rate Correlation": college_admissions_and_loan_default_rate_correlation,
        "College Admissions Funnel and Yield Rate Analysis": college_admissions_funnel_and_yield_rate_analysis,
        "School Board Spending and Student Achievement Analysis": school_board_spending_and_student_achievement_analysis,
        "University Enrollment and Earnings Outcome Analysis": university_enrollment_and_earnings_outcome_analysis,
        "College Retention, Debt, and Earnings Analysis": college_retention_debt_and_earnings_analysis,
        "School Enrollment and Disadvantaged Student Population Analysis": school_enrollment_and_disadvantaged_student_population_analysis,
        "University Enrollment and Faculty Count Analysis": university_enrollment_and_faculty_count_analysis,
        "State Education System Performance Analysis": state_education_system_performance_analysis,
        "University Tuition, Earnings, and Default Rate Trend Analysis": university_tuition_earnings_and_default_rate_trend_analysis,
        "School Board Data Analysis": school_board_data_analysis,
        "College Selectivity and Loan Repayment Rate Analysis": college_selectivity_and_loan_repayment_rate_analysis,
        "Faculty Composition and Student-Faculty Ratio Analysis": faculty_composition_and_student_faculty_ratio_analysis,
        "School District Expenditure and Test Score Analysis": school_district_expenditure_and_test_score_analysis,
        "College Selectivity and Income Diversity Analysis": college_selectivity_and_income_diversity_analysis,
        "School District Poverty and Graduation Rate Correlation": school_district_poverty_and_graduation_rate_correlation,
        "University Enrollment and Retention Analysis": university_enrollment_and_retention_analysis,
        "College Application and Enrollment Funnel Analysis": college_application_and_enrollment_funnel_analysis,
        "School District Staffing and Student Enrollment Analysis": school_district_staffing_and_student_enrollment_analysis,
        "College SAT Scores and 7-Year Graduation Rate Analysis": college_sat_scores_and_7_year_graduation_rate_analysis,
        "School Attendance and Graduation Rate Analysis": school_attendance_and_graduation_rate_analysis,
        "University Undergraduate and Graduate Enrollment Analysis": university_undergraduate_and_graduate_enrollment_analysis,
        "College Selectivity, Yield, and Default Rate Analysis": college_selectivity_yield_and_default_rate_analysis,
        "School District Classification and Expenditure Analysis": school_district_classification_and_expenditure_analysis,
        "College Scholarship and Post-Graduation Earnings Analysis": college_scholarship_and_post_graduation_earnings_analysis,
        "University Campus Enrollment and Tuition Analysis": university_campus_enrollment_and_tuition_analysis,
        "University Control, Debt, and Long-Term Income Analysis": university_control_debt_and_long_term_income_analysis,
        "College Admissions and Student Loan Analysis": college_admissions_and_student_loan_analysis,
        "School District Attendance and Absenteeism Analysis": school_district_attendance_and_absenteeism_analysis,
    }

    result = None

    # Dispatch based on the category and analysis type
    if category == "General":
        if analysis == "Academic Performance":
            result = academic_performance(df)
        elif analysis == "Demographic Analysis":
            result = demographic_analysis(df)
        elif analysis == "Course Analysis":
            result = course_analysis(df)
        elif analysis == "Attendance Analysis":
            result = attendance_analysis(df)
        elif analysis == "Behavioral Analysis":
            result = behavioral_analysis(df)
        elif analysis == "Program Evaluation":
            result = program_evaluation(df)
        else:
            result = show_general_insights(df)

    elif category == "Specific" and specific_analysis_name:
        if specific_analysis_name == "School District Performance and Socioeconomic Analysis":
            result = school_district_performance_and_socioeconomic_analysis(df)
        elif specific_analysis_name == "Higher Education Institution Cost of Attendance Analysis":
            result = higher_education_institution_cost_of_attendance_analysis(df)
        elif specific_analysis_name == "State-Level Average Cost of Attendance Trend Analysis":
            result = state_level_average_cost_of_attendance_trend_analysis(df)
        elif specific_analysis_name == "University Financials and Student Outcome Analysis":
            result = university_financials_and_student_outcome_analysis(df)
        elif specific_analysis_name == "University Enrollment, Expenditure, and Graduation Rate Analysis":
            result = university_enrollment_expenditure_and_graduation_rate_analysis(df)
        elif specific_analysis_name == "College Admissions and Graduation Rate Analysis":
            result = college_admissions_and_graduation_rate_analysis(df)
        elif specific_analysis_name == "School-Level Student-Teacher Ratio and Class Size Analysis":
            result = school_level_student_teacher_ratio_and_class_size_analysis(df)
        elif specific_analysis_name == "College Enrollment and Income Trend Analysis":
            result = college_enrollment_and_income_trend_analysis(df)
        elif specific_analysis_name == "School District Resource Adequacy Analysis":
            result = school_district_resource_adequacy_analysis(df)
        elif specific_analysis_name == "Higher Education Institution ROI and Default Rate Analysis":
            result = higher_education_institution_roi_and_default_rate_analysis(df)
        elif specific_analysis_name == "School District Budget and Student Outcome Analysis":
            result = school_district_budget_and_student_outcome_analysis(df)
        elif specific_analysis_name == "University Selectivity and Student Debt Analysis":
            result = university_selectivity_and_student_debt_analysis(df)
        elif specific_analysis_name == "College Admissions, Graduation, and Salary Outcome Analysis":
            result = college_admissions_graduation_and_salary_outcome_analysis(df)
        elif specific_analysis_name == "School Funding and Local Income Level Analysis":
            result = school_funding_and_local_income_level_analysis(df)
        elif specific_analysis_name == "Pell Grant Recipient Graduation and Loan Default Rate Analysis":
            result = pell_grant_recipient_graduation_and_loan_default_rate_analysis(df)
        elif specific_analysis_name == "College Selectivity and Graduation Rate Analysis":
            result = college_selectivity_and_graduation_rate_analysis(df)
        elif specific_analysis_name == "School District Demographics and Student-Teacher Ratio Analysis":
            result = school_district_demographics_and_student_teacher_ratio_analysis(df)
        elif specific_analysis_name == "College Tuition and Enrollment Statistics Analysis":
            result = college_tuition_and_enrollment_statistics_analysis(df)
        elif specific_analysis_name == "School Special Needs and Counselor Ratio Analysis":
            result = school_special_needs_and_counselor_ratio_analysis(df)
        elif specific_analysis_name == "University Graduation Rate and Diversity Index Analysis":
            result = university_graduation_rate_and_diversity_index_analysis(df)
        elif specific_analysis_name == "Post-Graduation Earnings and Debt Analysis":
            result = post_graduation_earnings_and_debt_analysis(df)
        elif specific_analysis_name == "School District Test Score and Graduation Rate Analysis":
            result = school_district_test_score_and_graduation_rate_analysis(df)
        elif specific_analysis_name == "College Admissions and Loan Default Rate Correlation":
            result = college_admissions_and_loan_default_rate_correlation(df)
        elif specific_analysis_name == "College Admissions Funnel and Yield Rate Analysis":
            result = college_admissions_funnel_and_yield_rate_analysis(df)
        elif specific_analysis_name == "School Board Spending and Student Achievement Analysis":
            result = school_board_spending_and_student_achievement_analysis(df)
        elif specific_analysis_name == "University Enrollment and Earnings Outcome Analysis":
            result = university_enrollment_and_earnings_outcome_analysis(df)
        elif specific_analysis_name == "College Retention, Debt, and Earnings Analysis":
            result = college_retention_debt_and_earnings_analysis(df)
        elif specific_analysis_name == "School Enrollment and Disadvantaged Student Population Analysis":
            result = school_enrollment_and_disadvantaged_student_population_analysis(df)
        elif specific_analysis_name == "University Enrollment and Faculty Count Analysis":
            result = university_enrollment_and_faculty_count_analysis(df)
        elif specific_analysis_name == "State Education System Performance Analysis":
            result = state_education_system_performance_analysis(df)
        elif specific_analysis_name == "University Tuition, Earnings, and Default Rate Trend Analysis":
            result = university_tuition_earnings_and_default_rate_trend_analysis(df)
        elif specific_analysis_name == "School Board Data Analysis":
            result = school_board_data_analysis(df)
        elif specific_analysis_name == "College Selectivity and Loan Repayment Rate Analysis":
            result = college_selectivity_and_loan_repayment_rate_analysis(df)
        elif specific_analysis_name == "Faculty Composition and Student-Faculty Ratio Analysis":
            result = faculty_composition_and_student_faculty_ratio_analysis(df)
        elif specific_analysis_name == "School District Expenditure and Test Score Analysis":
            result = school_district_expenditure_and_test_score_analysis(df)
        elif specific_analysis_name == "College Selectivity and Income Diversity Analysis":
            result = college_selectivity_and_income_diversity_analysis(df)
        elif specific_analysis_name == "School District Poverty and Graduation Rate Correlation":
            result = school_district_poverty_and_graduation_rate_correlation(df)
        elif specific_analysis_name == "University Enrollment and Retention Analysis":
            result = university_enrollment_and_retention_analysis(df)
        elif specific_analysis_name == "College Application and Enrollment Funnel Analysis":
            result = college_application_and_enrollment_funnel_analysis(df)
        elif specific_analysis_name == "School District Staffing and Student Enrollment Analysis":
            result = school_district_staffing_and_student_enrollment_analysis(df)
        elif specific_analysis_name == "College SAT Scores and 7-Year Graduation Rate Analysis":
            result = college_sat_scores_and_7_year_graduation_rate_analysis(df)
        elif specific_analysis_name == "School Attendance and Graduation Rate Analysis":
            result = school_attendance_and_graduation_rate_analysis(df)
        elif specific_analysis_name == "University Undergraduate and Graduate Enrollment Analysis":
            result = university_undergraduate_and_graduate_enrollment_analysis(df)
        elif specific_analysis_name == "College Selectivity, Yield, and Default Rate Analysis":
            result = college_selectivity_yield_and_default_rate_analysis(df)
        elif specific_analysis_name == "School District Classification and Expenditure Analysis":
            result = school_district_classification_and_expenditure_analysis(df)
        elif specific_analysis_name == "College Scholarship and Post-Graduation Earnings Analysis":
            result = college_scholarship_and_post_graduation_earnings_analysis(df)
        elif specific_analysis_name == "University Campus Enrollment and Tuition Analysis":
            result = university_campus_enrollment_and_tuition_analysis(df)
        elif specific_analysis_name == "University Control, Debt, and Long-Term Income Analysis":
            result = university_control_debt_and_long_term_income_analysis(df)
        elif specific_analysis_name == "College Admissions and Student Loan Analysis":
            result = college_admissions_and_student_loan_analysis(df)
        elif specific_analysis_name == "School District Attendance and Absenteeism Analysis":
            result = school_district_attendance_and_absenteeism_analysis(df)
        else:
            try:
                func = specific_education_function_mapping.get(specific_analysis_name)
                if func:
                    result = func(df)
                else:
                    result = {"error": f"Function not found for analysis '{specific_analysis_name}'"}
            except Exception as e:
                result = {"error": str(e), "message": f"Error running analysis '{specific_analysis_name}'"}

    else:
        result = show_general_insights(df)

    return result
