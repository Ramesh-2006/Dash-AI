import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go

# ========== UTILITY FUNCTIONS ==========
def show_key_metrics(df):
    print("\n=== Key Metrics ===")
    total_records = len(df)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Total Records: {total_records}")
    print(f"Total Features: {len(df.columns)}")
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

def show_missing_columns_warning(missing_cols, matched_cols=None):
    print("\n⚠ Required Columns Not Found")
    print("The following columns are needed for this analysis but weren't found in your data:")
    for col in missing_cols:
        match_info = f" (matched to: {matched_cols[col]})" if matched_cols and matched_cols[col] else ""
        print(f" - {col}{match_info}")

def show_general_insights(df, title="General Insights"):
    print(f"\n=== {title} ===")
    show_key_metrics(df)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        print("\nNumeric Features Analysis")
        print("Available numeric features:")
        for i, col in enumerate(numeric_cols):
            print(f"{i}: {col}")
        selected_num_col_idx = int(input("Select numeric feature to analyze (enter index): "))
        selected_num_col = numeric_cols[selected_num_col_idx]

        hist_fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
        hist_fig.show()
        
        box_fig = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
        box_fig.show()
    else:
        print("[WARNING] No numeric columns found for analysis.")
    
    if len(numeric_cols) >= 2:
        print("\nFeature Correlations:")
        corr = df[numeric_cols].corr()
        corr_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
        corr_fig.show()
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Features Analysis")
        print("Available categorical features:")
        for i, col in enumerate(categorical_cols):
            print(f"{i}: {col}")
        selected_cat_col_idx = int(input("Select categorical feature to analyze (enter index): "))
        selected_cat_col = categorical_cols[selected_cat_col_idx]

        value_counts = df[selected_cat_col].value_counts().reset_index()
        value_counts.columns = ['Value', 'Count']
        bar_fig = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Distribution of {selected_cat_col}")
        bar_fig.show()
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

# ========== STUDENT ANALYSIS FUNCTIONS ==========

def student_academic_performance_and_influencing_factors_analysis(df: pd.DataFrame):
    print("\n=== Student Academic Performance and Influencing Factors Analysis ===")
    expected = ['school', 'sex', 'age', 'G3']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col, None) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['G3'] = pd.to_numeric(df['G3'], errors='coerce')
    df.dropna(subset=['G3', 'sex', 'school', 'age'], inplace=True)

    avg_grade = df['G3'].mean()
    most_common_school = df['school'].mode()[0]
    
    print(f"Average Final Grade (G3): {avg_grade:.2f}")
    print(f"Most Common School: {most_common_school}")
    
    box_fig = px.box(df, x='school', y='G3', title="Final Grade Distribution by School")
    box_fig.show()
    
    hist_fig = px.histogram(df, x='age', color='sex', barmode='group', title="Age Distribution by Sex")
    hist_fig.show()

def student_test_score_analysis_by_demographics(df: pd.DataFrame):
    print("\n=== Student Test Score Analysis by Demographics and Preparation ===")
    expected = ['gender', 'race_ethnicity', 'writing_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col, None) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['writing_score'] = pd.to_numeric(df['writing_score'], errors='coerce')
    df.dropna(subset=['writing_score', 'gender', 'race_ethnicity'], inplace=True)

    avg_score = df['writing_score'].mean()
    top_performer_gender = df.groupby('gender')['writing_score'].mean().idxmax()
    
    print(f"Average Writing Score: {avg_score:.2f}")
    print(f"Top Performing Gender: {top_performer_gender}")
    
    bar_fig = px.bar(df.groupby('race_ethnicity')['writing_score'].mean().reset_index(), x='race_ethnicity', y='writing_score', title="Average Writing Score by Race/Ethnicity")
    bar_fig.show()
    
    box_fig = px.box(df, x='gender', y='writing_score', title="Writing Score Distribution by Gender")
    box_fig.show()

def factors_affecting_student_final_grades(df: pd.DataFrame):
    print("\n=== Factors Affecting Student Final Grades ===")
    expected = ['student_id', 'gender', 'age', 'final_grade']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col, None) is None]

    if missing:
        show_missing_columns_warning(missing, matched)
        show_general_insights(df, "General Analysis")
        return

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
    df.dropna(subset=['final_grade', 'gender', 'age'], inplace=True)

    avg_grade = df['final_grade'].mean()
    print(f"Average Final Grade: {avg_grade:.2f}")
    
    violin_fig = px.violin(df, x='gender', y='final_grade', color='age', title="Final Grade Distribution by Gender and Age")
    violin_fig.show()
def Student_Test_Score_Analysis_by_Demographics_and_Preparation(df):
    expected = ['student_id', 'test_score', 'gender', 'race_ethnicity', 'test_preparation_course']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'test_score'], inplace=True)

    # Average test score by race/ethnicity
    if 'race_ethnicity' in df.columns:
        avg_score_by_race = df.groupby('race_ethnicity')['test_score'].mean().sort_values(ascending=False).reset_index()
        fig_score_by_race = px.bar(avg_score_by_race, x='race_ethnicity', y='test_score',
                                   title='Average Test Score by Race/Ethnicity')
    else:
        fig_score_by_race = go.Figure().add_annotation(text="Race/Ethnicity data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average test score by test preparation course completion
    if 'test_preparation_course' in df.columns:
        avg_score_by_prep = df.groupby('test_preparation_course')['test_score'].mean().reset_index()
        fig_score_by_prep = px.bar(avg_score_by_prep, x='test_preparation_course', y='test_score',
                                   title='Average Test Score by Test Preparation Course Completion')
    else:
        fig_score_by_prep = go.Figure().add_annotation(text="Test preparation data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_test_score_by_race_ethnicity': fig_score_by_race,
        'average_test_score_by_preparation': fig_score_by_prep
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "overall_avg_test_score": df['test_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Factors_Affecting_Student_Final_Grades(df):
    expected = ['student_id', 'final_grade', 'study_time_weekly_hours', 'absences', 'parental_support_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between study time and final grade
    if 'study_time_weekly_hours' in df.columns:
        fig_study_time_grade = px.scatter(df, x='study_time_weekly_hours', y='final_grade',
                                         title='Final Grade vs. Weekly Study Time', trendline="ols")
    else:
        fig_study_time_grade = go.Figure().add_annotation(text="Study time data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Impact of absences on final grade
    if 'absences' in df.columns:
        avg_grade_by_absences = df.groupby('absences')['final_grade'].mean().reset_index()
        fig_absences_grade = px.line(avg_grade_by_absences, x='absences', y='final_grade',
                                     title='Average Final Grade by Number of Absences')
    else:
        fig_absences_grade = go.Figure().add_annotation(text="Absences data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_study_time': fig_study_time_grade,
        'average_final_grade_by_absences': fig_absences_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_study_time_grade": df[['study_time_weekly_hours', 'final_grade']].corr().iloc[0, 1] if 'study_time_weekly_hours' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Student_Academic_Performance_Summary_Analysis(df):
    expected = ['student_id', 'math_score', 'reading_score', 'writing_score', 'overall_gpa']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id'], inplace=True)

    # Average scores across different subjects
    scores_data = {}
    if 'math_score' in df.columns: scores_data['Math'] = df['math_score'].mean()
    if 'reading_score' in df.columns: scores_data['Reading'] = df['reading_score'].mean()
    if 'writing_score' in df.columns: scores_data['Writing'] = df['writing_score'].mean()
    avg_scores_df = pd.DataFrame(scores_data.items(), columns=['Subject', 'Average_Score'])
    fig_avg_subject_scores = px.bar(avg_scores_df, x='Subject', y='Average_Score',
                                    title='Average Scores Across Different Subjects')

    # Distribution of overall GPA
    if 'overall_gpa' in df.columns:
        fig_gpa_distribution = px.histogram(df, x='overall_gpa', nbins=20, title='Distribution of Overall GPA')
    else:
        fig_gpa_distribution = go.Figure().add_annotation(text="Overall GPA data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_scores_by_subject': fig_avg_subject_scores,
        'overall_gpa_distribution': fig_gpa_distribution
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "overall_avg_gpa": df['overall_gpa'].mean() if 'overall_gpa' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Study_Habits_and_Their_Impact_on_Final_Scores(df):
    expected = ['student_id', 'final_score', 'study_time_category', 'internet_access_hours_daily', 'tutoring_support']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Average final score by study time category
    if 'study_time_category' in df.columns:
        avg_score_by_study_time = df.groupby('study_time_category')['final_score'].mean().sort_values(ascending=False).reset_index()
        fig_score_by_study_time = px.bar(avg_score_by_study_time, x='study_time_category', y='final_score',
                                         title='Average Final Score by Study Time Category')
    else:
        fig_score_by_study_time = go.Figure().add_annotation(text="Study time category data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Impact of tutoring support on final scores
    if 'tutoring_support' in df.columns:
        avg_score_by_tutoring = df.groupby('tutoring_support')['final_score'].mean().reset_index()
        fig_score_by_tutoring = px.bar(avg_score_by_tutoring, x='tutoring_support', y='final_score',
                                       title='Average Final Score by Tutoring Support')
    else:
        fig_score_by_tutoring = go.Figure().add_annotation(text="Tutoring support data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_score_by_study_time_category': fig_score_by_study_time,
        'average_final_score_by_tutoring_support': fig_score_by_tutoring
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Student_Engagement_and_Performance_Analysis(df):
    expected = ['student_id', 'final_grade', 'attendance_rate', 'participation_score', 'extra_curricular_activities']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between attendance rate and final grade
    if 'attendance_rate' in df.columns:
        fig_attendance_grade = px.scatter(df, x='attendance_rate', y='final_grade',
                                         title='Final Grade vs. Attendance Rate', trendline="ols")
    else:
        fig_attendance_grade = go.Figure().add_annotation(text="Attendance rate data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Impact of participation score on final grade
    if 'participation_score' in df.columns:
        fig_participation_grade = px.scatter(df, x='participation_score', y='final_grade',
                                            title='Final Grade vs. Participation Score', trendline="ols")
    else:
        fig_participation_grade = go.Figure().add_annotation(text="Participation score data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_attendance_rate': fig_attendance_grade,
        'final_grade_vs_participation_score': fig_participation_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_attendance_grade": df[['attendance_rate', 'final_grade']].corr().iloc[0, 1] if 'attendance_rate' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Impact_of_Test_Preparation_on_Academic_Scores(df):
    expected = ['student_id', 'pre_test_score', 'post_test_score', 'test_preparation_course_completed']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id'], inplace=True)

    # Average score improvement for students who completed preparation vs. those who didn't
    if 'pre_test_score' in df.columns and 'post_test_score' in df.columns and 'test_preparation_course_completed' in df.columns:
        df['score_improvement'] = df['post_test_score'] - df['pre_test_score']
        avg_improvement_by_prep = df.groupby('test_preparation_course_completed')['score_improvement'].mean().reset_index()
        fig_improvement_by_prep = px.bar(avg_improvement_by_prep, x='test_preparation_course_completed', y='score_improvement',
                                         title='Average Score Improvement by Test Preparation Status')
    else:
        fig_improvement_by_prep = go.Figure().add_annotation(text="Pre/Post test score or preparation status data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of post-test scores by preparation status
    if 'post_test_score' in df.columns and 'test_preparation_course_completed' in df.columns:
        fig_post_score_dist = px.histogram(df, x='post_test_score', color='test_preparation_course_completed',
                                          barmode='overlay', title='Post-Test Score Distribution by Preparation Status')
    else:
        fig_post_score_dist = go.Figure().add_annotation(text="Post-test score or preparation status data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_score_improvement_by_preparation_status': fig_improvement_by_prep,
        'post_test_score_distribution_by_preparation_status': fig_post_score_dist
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "overall_avg_score_improvement": df['score_improvement'].mean() if 'score_improvement' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Learning_Management_System_LMS_Usage_and_Grade_Correlation(df):
    expected = ['student_id', 'final_grade', 'lms_login_frequency', 'lms_resource_views', 'assignment_submission_on_time_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between LMS login frequency and final grade
    if 'lms_login_frequency' in df.columns:
        fig_lms_login_grade = px.scatter(df, x='lms_login_frequency', y='final_grade',
                                        title='Final Grade vs. LMS Login Frequency', trendline="ols")
    else:
        fig_lms_login_grade = go.Figure().add_annotation(text="LMS login frequency data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between LMS resource views and final grade
    if 'lms_resource_views' in df.columns:
        fig_lms_resource_grade = px.scatter(df, x='lms_resource_views', y='final_grade',
                                           title='Final Grade vs. LMS Resource Views', trendline="ols")
    else:
        fig_lms_resource_grade = go.Figure().add_annotation(text="LMS resource views data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_lms_login_frequency': fig_lms_login_grade,
        'final_grade_vs_lms_resource_views': fig_lms_resource_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_lms_login_grade": df[['lms_login_frequency', 'final_grade']].corr().iloc[0, 1] if 'lms_login_frequency' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Demographic_and_Health_Factors_on_Student_Scores(df):
    expected = ['student_id', 'final_score', 'gender', 'age', 'health_status', 'absences']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Average final score by gender
    if 'gender' in df.columns:
        avg_score_by_gender = df.groupby('gender')['final_score'].mean().reset_index()
        fig_score_by_gender = px.bar(avg_score_by_gender, x='gender', y='final_score',
                                     title='Average Final Score by Gender')
    else:
        fig_score_by_gender = go.Figure().add_annotation(text="Gender data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final score by health status
    if 'health_status' in df.columns:
        avg_score_by_health = df.groupby('health_status')['final_score'].mean().reset_index()
        fig_score_by_health = px.bar(avg_score_by_health, x='health_status', y='final_score',
                                     title='Average Final Score by Health Status')
    else:
        fig_score_by_health = go.Figure().add_annotation(text="Health status data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_score_by_gender': fig_score_by_gender,
        'average_final_score_by_health_status': fig_score_by_health
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Social_Factors_and_Internet_Usage_Impact_on_Student_Performance(df):
    expected = ['student_id', 'final_grade', 'social_activities_level', 'internet_usage_hours_daily', 'family_relationship_quality']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by social activities level
    if 'social_activities_level' in df.columns:
        avg_grade_by_social_activities = df.groupby('social_activities_level')['final_grade'].mean().reset_index()
        fig_grade_by_social_activities = px.bar(avg_grade_by_social_activities, x='social_activities_level', y='final_grade',
                                               title='Average Final Grade by Social Activities Level')
    else:
        fig_grade_by_social_activities = go.Figure().add_annotation(text="Social activities level data not available.",
                                                                    xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between internet usage and final grade
    if 'internet_usage_hours_daily' in df.columns:
        fig_internet_usage_grade = px.scatter(df, x='internet_usage_hours_daily', y='final_grade',
                                             title='Final Grade vs. Daily Internet Usage Hours', trendline="ols")
    else:
        fig_internet_usage_grade = go.Figure().add_annotation(text="Internet usage data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_social_activities': fig_grade_by_social_activities,
        'final_grade_vs_internet_usage': fig_internet_usage_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_internet_usage_grade": df[['internet_usage_hours_daily', 'final_grade']].corr().iloc[0, 1] if 'internet_usage_hours_daily' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Student_Pass_Fail_Prediction_Analysis(df):
    expected = ['student_id', 'final_grade', 'pass_status', 'g1_grade', 'g2_grade', 'study_time_weekly_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade', 'pass_status'], inplace=True)

    # Pass/Fail status distribution
    pass_fail_counts = df['pass_status'].value_counts(normalize=True).reset_index()
    pass_fail_counts.columns = ['status', 'proportion']

    # Average final grade for Pass vs. Fail students
    avg_grade_by_status = df.groupby('pass_status')['final_grade'].mean().reset_index()
    fig_avg_grade_by_status = px.bar(avg_grade_by_status, x='pass_status', y='final_grade',
                                     title='Average Final Grade for Pass vs. Fail Students')

    fig_pass_fail_pie = px.pie(pass_fail_counts, names='status', values='proportion', title='Student Pass/Fail Status Distribution')

    plots = {
        'pass_fail_status_distribution': fig_pass_fail_pie,
        'average_final_grade_by_pass_fail_status': fig_avg_grade_by_status
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "pass_rate_percent": pass_fail_counts[pass_fail_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100 if 'pass' in pass_fail_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def Impact_of_Past_Grades_and_Study_Time_on_Current_Performance(df):
    expected = ['student_id', 'g1_grade', 'g2_grade', 'final_grade_g3', 'study_time_weekly_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade_g3'], inplace=True)

    # Correlation between G1/G2 grades and G3 (final) grade
    if 'g1_grade' in df.columns and 'g2_grade' in df.columns:
        fig_g1_g3_scatter = px.scatter(df, x='g1_grade', y='final_grade_g3', title='G3 Final Grade vs. G1 Grade', trendline="ols")
        fig_g2_g3_scatter = px.scatter(df, x='g2_grade', y='final_grade_g3', title='G3 Final Grade vs. G2 Grade', trendline="ols")
        plots = {
            'g3_vs_g1_grade_scatter': fig_g1_g3_scatter,
            'g3_vs_g2_grade_scatter': fig_g2_g3_scatter
        }
    else:
        plots = {
            'past_grades_warning': go.Figure().add_annotation(text="G1/G2 grades data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))
        }

    metrics = {
        "overall_avg_g3_grade": df['final_grade_g3'].mean(),
        "correlation_g1_g3": df[['g1_grade', 'final_grade_g3']].corr().iloc[0, 1] if 'g1_grade' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Extracurricular_Activities_and_Academic_Grade_Analysis(df):
    expected = ['student_id', 'final_grade', 'extra_curricular_activities', 'study_time_weekly_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade for students participating in extracurriculars vs. not
    if 'extra_curricular_activities' in df.columns:
        avg_grade_by_extracurriculars = df.groupby('extra_curricular_activities')['final_grade'].mean().reset_index()
        fig_grade_by_extracurriculars = px.bar(avg_grade_by_extracurriculars, x='extra_curricular_activities', y='final_grade',
                                              title='Average Final Grade by Extracurricular Activities Participation')
    else:
        fig_grade_by_extracurriculars = go.Figure().add_annotation(text="Extracurricular activities data not available.",
                                                                   xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Distribution of study time for students with and without extracurriculars
    if 'study_time_weekly_hours' in df.columns and 'extra_curricular_activities' in df.columns:
        fig_study_time_extracurriculars = px.box(df, x='extra_curricular_activities', y='study_time_weekly_hours',
                                                title='Weekly Study Time Distribution by Extracurricular Activities')
    else:
        fig_study_time_extracurriculars = go.Figure().add_annotation(text="Study time or extracurricular data not available.",
                                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_extracurriculars': fig_grade_by_extracurriculars,
        'study_time_distribution_by_extracurriculars': fig_study_time_extracurriculars
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Family_and_Internet_Support_on_Student_Final_Scores(df):
    expected = ['student_id', 'final_score', 'family_educational_support', 'internet_access_quality', 'parental_involvement_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Average final score by family educational support
    if 'family_educational_support' in df.columns:
        avg_score_by_family_support = df.groupby('family_educational_support')['final_score'].mean().reset_index()
        fig_score_by_family_support = px.bar(avg_score_by_family_support, x='family_educational_support', y='final_score',
                                            title='Average Final Score by Family Educational Support')
    else:
        fig_score_by_family_support = go.Figure().add_annotation(text="Family educational support data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final score by internet access quality
    if 'internet_access_quality' in df.columns:
        avg_score_by_internet_quality = df.groupby('internet_access_quality')['final_score'].mean().reset_index()
        fig_score_by_internet_quality = px.bar(avg_score_by_internet_quality, x='internet_access_quality', y='final_score',
                                              title='Average Final Score by Internet Access Quality')
    else:
        fig_score_by_internet_quality = go.Figure().add_annotation(text="Internet access quality data not available.",
                                                                   xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_score_by_family_educational_support': fig_score_by_family_support,
        'average_final_score_by_internet_access_quality': fig_score_by_internet_quality
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Parental_Background_and_Study_Time_on_Final_Grades(df):
    expected = ['student_id', 'final_grade', 'mother_education_level', 'father_education_level', 'study_time_weekly_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by mother's education level
    if 'mother_education_level' in df.columns:
        avg_grade_by_mother_edu = df.groupby('mother_education_level')['final_grade'].mean().sort_values(ascending=False).reset_index()
        fig_grade_by_mother_edu = px.bar(avg_grade_by_mother_edu, x='mother_education_level', y='final_grade',
                                         title="Average Final Grade by Mother's Education Level")
    else:
        fig_grade_by_mother_edu = go.Figure().add_annotation(text="Mother's education level data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between study time and final grade, possibly faceted by parental education
    if 'study_time_weekly_hours' in df.columns and 'mother_education_level' in df.columns:
        fig_study_time_grade_faceted = px.scatter(df, x='study_time_weekly_hours', y='final_grade',
                                                  color='mother_education_level', title='Final Grade vs. Study Time by Mother\'s Education Level',
                                                  facet_col='mother_education_level', facet_col_wrap=2, trendline="ols")
    else:
        fig_study_time_grade_faceted = go.Figure().add_annotation(text="Study time or mother's education data not available for faceted plot.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_mother_education_level': fig_grade_by_mother_edu,
        'final_grade_vs_study_time_faceted_by_mother_education': fig_study_time_grade_faceted
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Family_Relationships_and_Student_Grade_Analysis(df):
    expected = ['student_id', 'final_grade', 'family_relationship_quality_score', 'parents_status_cohabiting_or_apart']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by family relationship quality score
    if 'family_relationship_quality_score' in df.columns:
        avg_grade_by_family_relation = df.groupby('family_relationship_quality_score')['final_grade'].mean().reset_index()
        fig_grade_by_family_relation = px.bar(avg_grade_by_family_relation, x='family_relationship_quality_score', y='final_grade',
                                              title='Average Final Grade by Family Relationship Quality Score')
    else:
        fig_grade_by_family_relation = go.Figure().add_annotation(text="Family relationship quality data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by parents' status (cohabiting or apart)
    if 'parents_status_cohabiting_or_apart' in df.columns:
        avg_grade_by_parents_status = df.groupby('parents_status_cohabiting_or_apart')['final_grade'].mean().reset_index()
        fig_grade_by_parents_status = px.bar(avg_grade_by_parents_status, x='parents_status_cohabiting_or_apart', y='final_grade',
                                             title='Average Final Grade by Parents Status')
    else:
        fig_grade_by_parents_status = go.Figure().add_annotation(text="Parents status data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_family_relationship_quality': fig_grade_by_family_relation,
        'average_final_grade_by_parents_status': fig_grade_by_parents_status
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Assessment_Scores_and_Attendance_Impact_on_Final_Grade(df):
    expected = ['student_id', 'final_grade', 'midterm_score', 'quiz_average_score', 'attendance_rate_percent']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between midterm score and final grade
    if 'midterm_score' in df.columns:
        fig_midterm_final_grade = px.scatter(df, x='midterm_score', y='final_grade',
                                            title='Final Grade vs. Midterm Score', trendline="ols")
    else:
        fig_midterm_final_grade = go.Figure().add_annotation(text="Midterm score data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between attendance rate and final grade
    if 'attendance_rate_percent' in df.columns:
        fig_attendance_final_grade = px.scatter(df, x='attendance_rate_percent', y='final_grade',
                                                title='Final Grade vs. Attendance Rate (%)', trendline="ols")
    else:
        fig_attendance_final_grade = go.Figure().add_annotation(text="Attendance rate data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_midterm_score': fig_midterm_final_grade,
        'final_grade_vs_attendance_rate': fig_attendance_final_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_midterm_final": df[['midterm_score', 'final_grade']].corr().iloc[0, 1] if 'midterm_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Lifestyle_Factors_and_Their_Correlation_with_Student_GPA(df):
    expected = ['student_id', 'gpa', 'sleep_hours_daily', 'physical_activity_hours_weekly', 'diet_quality_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'gpa'], inplace=True)

    # Correlation between sleep hours and GPA
    if 'sleep_hours_daily' in df.columns:
        fig_sleep_gpa = px.scatter(df, x='sleep_hours_daily', y='gpa',
                                  title='GPA vs. Daily Sleep Hours', trendline="ols")
    else:
        fig_sleep_gpa = go.Figure().add_annotation(text="Sleep hours data not available.",
                                                   xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between physical activity and GPA
    if 'physical_activity_hours_weekly' in df.columns:
        fig_physical_activity_gpa = px.scatter(df, x='physical_activity_hours_weekly', y='gpa',
                                              title='GPA vs. Weekly Physical Activity Hours', trendline="ols")
    else:
        fig_physical_activity_gpa = go.Figure().add_annotation(text="Physical activity data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'gpa_vs_daily_sleep_hours': fig_sleep_gpa,
        'gpa_vs_weekly_physical_activity_hours': fig_physical_activity_gpa
    }

    metrics = {
        "overall_avg_gpa": df['gpa'].mean(),
        "correlation_sleep_gpa": df[['sleep_hours_daily', 'gpa']].corr().iloc[0, 1] if 'sleep_hours_daily' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Educational_Support_Systems_Impact_on_Student_Grades(df):
    expected = ['student_id', 'final_grade', 'school_support', 'extra_paid_classes', 'tutoring_provided_by_school']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by school support
    if 'school_support' in df.columns:
        avg_grade_by_school_support = df.groupby('school_support')['final_grade'].mean().reset_index()
        fig_grade_by_school_support = px.bar(avg_grade_by_school_support, x='school_support', y='final_grade',
                                             title='Average Final Grade by School Support')
    else:
        fig_grade_by_school_support = go.Figure().add_annotation(text="School support data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by extra paid classes
    if 'extra_paid_classes' in df.columns:
        avg_grade_by_paid_classes = df.groupby('extra_paid_classes')['final_grade'].mean().reset_index()
        fig_grade_by_paid_classes = px.bar(avg_grade_by_paid_classes, x='extra_paid_classes', y='final_grade',
                                           title='Average Final Grade by Extra Paid Classes')
    else:
        fig_grade_by_paid_classes = go.Figure().add_annotation(text="Extra paid classes data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_school_support': fig_grade_by_school_support,
        'average_final_grade_by_extra_paid_classes': fig_grade_by_paid_classes
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Impact_of_Paid_Classes_and_School_Support_on_Performance(df):
    expected = ['student_id', 'overall_score', 'paid_classes_taken', 'school_provided_support']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'overall_score'], inplace=True)

    # Average overall score by paid classes taken
    if 'paid_classes_taken' in df.columns:
        avg_score_by_paid_classes = df.groupby('paid_classes_taken')['overall_score'].mean().reset_index()
        fig_score_by_paid_classes = px.bar(avg_score_by_paid_classes, x='paid_classes_taken', y='overall_score',
                                           title='Average Overall Score by Paid Classes Taken')
    else:
        fig_score_by_paid_classes = go.Figure().add_annotation(text="Paid classes data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average overall score by school provided support
    if 'school_provided_support' in df.columns:
        avg_score_by_school_support = df.groupby('school_provided_support')['overall_score'].mean().reset_index()
        fig_score_by_school_support = px.bar(avg_score_by_school_support, x='school_provided_support', y='overall_score',
                                             title='Average Overall Score by School Provided Support')
    else:
        fig_score_by_school_support = go.Figure().add_annotation(text="School support data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_overall_score_by_paid_classes': fig_score_by_paid_classes,
        'average_overall_score_by_school_support': fig_score_by_school_support
    }

    metrics = {
        "overall_avg_score": df['overall_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Student_Performance_and_Pass_Status_Prediction(df):
    expected = ['student_id', 'final_grade', 'pass_status', 'g1_score', 'g2_score', 'study_time_weekly']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade', 'pass_status'], inplace=True)

    # Pass status distribution
    pass_status_counts = df['pass_status'].value_counts(normalize=True).reset_index()
    pass_status_counts.columns = ['status', 'proportion']

    # Average final grade by pass status
    avg_grade_by_pass_status = df.groupby('pass_status')['final_grade'].mean().reset_index()
    fig_avg_grade_by_pass_status = px.bar(avg_grade_by_pass_status, x='pass_status', y='final_grade',
                                         title='Average Final Grade by Pass Status')

    fig_pass_status_pie = px.pie(pass_status_counts, names='status', values='proportion', title='Student Pass Status Distribution')

    plots = {
        'pass_status_distribution': fig_pass_status_pie,
        'average_final_grade_by_pass_status': fig_avg_grade_by_pass_status
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "overall_pass_rate_percent": pass_status_counts[pass_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100 if 'pass' in pass_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def Health_Absences_and_Travel_Time_on_Final_Grades_G3(df):
    expected = ['student_id', 'g3_final_grade', 'health_status', 'absences', 'travel_time_to_school_minutes']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'g3_final_grade'], inplace=True)

    # Average G3 grade by health status
    if 'health_status' in df.columns:
        avg_grade_by_health = df.groupby('health_status')['g3_final_grade'].mean().reset_index()
        fig_grade_by_health = px.bar(avg_grade_by_health, x='health_status', y='g3_final_grade',
                                     title='Average G3 Final Grade by Health Status')
    else:
        fig_grade_by_health = go.Figure().add_annotation(text="Health status data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between absences and G3 grade
    if 'absences' in df.columns:
        fig_absences_grade = px.scatter(df, x='absences', y='g3_final_grade',
                                        title='G3 Final Grade vs. Absences', trendline="ols")
    else:
        fig_absences_grade = go.Figure().add_annotation(text="Absences data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_g3_grade_by_health_status': fig_grade_by_health,
        'g3_final_grade_vs_absences': fig_absences_grade
    }

    metrics = {
        "overall_avg_g3_grade": df['g3_final_grade'].mean(),
        "correlation_absences_g3": df[['absences', 'g3_final_grade']].corr().iloc[0, 1] if 'absences' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Impact_of_Extra_Paid_Classes_and_School_Support_on_Grades(df):
    # This function name is very similar to 'Impact_of_Paid_Classes_and_School_Support_on_Performance'
    # I will assume 'grades' refers to 'final_grade' for consistency if available.
    expected = ['student_id', 'final_grade', 'extra_paid_classes', 'school_support_services']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by extra paid classes
    if 'extra_paid_classes' in df.columns:
        avg_grade_by_paid_classes = df.groupby('extra_paid_classes')['final_grade'].mean().reset_index()
        fig_grade_by_paid_classes = px.bar(avg_grade_by_paid_classes, x='extra_paid_classes', y='final_grade',
                                           title='Average Final Grade by Extra Paid Classes')
    else:
        fig_grade_by_paid_classes = go.Figure().add_annotation(text="Extra paid classes data not available.",
                                                               xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by school support services
    if 'school_support_services' in df.columns:
        avg_grade_by_school_support = df.groupby('school_support_services')['final_grade'].mean().reset_index()
        fig_grade_by_school_support = px.bar(avg_grade_by_school_support, x='school_support_services', y='final_grade',
                                             title='Average Final Grade by School Support Services')
    else:
        fig_grade_by_school_support = go.Figure().add_annotation(text="School support services data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_extra_paid_classes': fig_grade_by_paid_classes,
        'average_final_grade_by_school_support_services': fig_grade_by_school_support
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Social_and_Health_Factors_Affecting_Student_Scores(df):
    expected = ['student_id', 'final_score', 'social_activities_participation', 'health_condition_status', 'absences']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Average final score by social activities participation
    if 'social_activities_participation' in df.columns:
        avg_score_by_social = df.groupby('social_activities_participation')['final_score'].mean().reset_index()
        fig_score_by_social = px.bar(avg_score_by_social, x='social_activities_participation', y='final_score',
                                     title='Average Final Score by Social Activities Participation')
    else:
        fig_score_by_social = go.Figure().add_annotation(text="Social activities participation data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final score by health condition status
    if 'health_condition_status' in df.columns:
        avg_score_by_health = df.groupby('health_condition_status')['final_score'].mean().reset_index()
        fig_score_by_health = px.bar(avg_score_by_health, x='health_condition_status', y='final_score',
                                     title='Average Final Score by Health Condition Status')
    else:
        fig_score_by_health = go.Figure().add_annotation(text="Health condition status data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_score_by_social_activities': fig_score_by_social,
        'average_final_score_by_health_condition_status': fig_score_by_health
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Physical_Attributes_and_Commutes_Effect_on_Grades(df):
    expected = ['student_id', 'final_grade', 'travel_time_to_school_minutes', 'physical_attributes_score', 'gym_attendance_weekly']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between travel time and final grade
    if 'travel_time_to_school_minutes' in df.columns:
        fig_travel_time_grade = px.scatter(df, x='travel_time_to_school_minutes', y='final_grade',
                                          title='Final Grade vs. Travel Time to School (Minutes)', trendline="ols")
    else:
        fig_travel_time_grade = go.Figure().add_annotation(text="Travel time data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between physical attributes score and final grade
    if 'physical_attributes_score' in df.columns:
        fig_physical_attr_grade = px.scatter(df, x='physical_attributes_score', y='final_grade',
                                            title='Final Grade vs. Physical Attributes Score', trendline="ols")
    else:
        fig_physical_attr_grade = go.Figure().add_annotation(text="Physical attributes score data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_travel_time': fig_travel_time_grade,
        'final_grade_vs_physical_attributes_score': fig_physical_attr_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_travel_time_grade": df[['travel_time_to_school_minutes', 'final_grade']].corr().iloc[0, 1] if 'travel_time_to_school_minutes' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Student_Performance_and_Pass_Fail_Classification(df):
    expected = ['student_id', 'final_score', 'pass_fail_status', 'exam_score_midterm', 'exam_score_final']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score', 'pass_fail_status'], inplace=True)

    # Pass/Fail status distribution
    pass_fail_status_counts = df['pass_fail_status'].value_counts(normalize=True).reset_index()
    pass_fail_status_counts.columns = ['status', 'proportion']

    # Average final score for Pass vs. Fail students
    avg_score_by_status = df.groupby('pass_fail_status')['final_score'].mean().reset_index()
    fig_avg_score_by_status = px.bar(avg_score_by_status, x='pass_fail_status', y='final_score',
                                     title='Average Final Score for Pass vs. Fail Students')

    fig_pass_fail_pie = px.pie(pass_fail_status_counts, names='status', values='proportion', title='Student Pass/Fail Status Distribution')

    plots = {
        'pass_fail_status_distribution': fig_pass_fail_pie,
        'average_final_score_by_pass_fail_status': fig_avg_score_by_status
    }

    metrics = {
        "total_students": len(df),
        "pass_rate_percent": pass_fail_status_counts[pass_fail_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100 if 'pass' in pass_fail_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def Digital_Engagement_and_Parental_Support_on_Student_GPA(df):
    expected = ['student_id', 'gpa', 'digital_engagement_score', 'parental_support_score', 'internet_access_quality']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'gpa'], inplace=True)

    # Correlation between digital engagement score and GPA
    if 'digital_engagement_score' in df.columns:
        fig_digital_engagement_gpa = px.scatter(df, x='digital_engagement_score', y='gpa',
                                               title='GPA vs. Digital Engagement Score', trendline="ols")
    else:
        fig_digital_engagement_gpa = go.Figure().add_annotation(text="Digital engagement score data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between parental support score and GPA
    if 'parental_support_score' in df.columns:
        fig_parental_support_gpa = px.scatter(df, x='parental_support_score', y='gpa',
                                             title='GPA vs. Parental Support Score', trendline="ols")
    else:
        fig_parental_support_gpa = go.Figure().add_annotation(text="Parental support score data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'gpa_vs_digital_engagement_score': fig_digital_engagement_gpa,
        'gpa_vs_parental_support_score': fig_parental_support_gpa
    }

    metrics = {
        "overall_avg_gpa": df['gpa'].mean(),
        "correlation_digital_engagement_gpa": df[['digital_engagement_score', 'gpa']].corr().iloc[0, 1] if 'digital_engagement_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Daily_Habits_and_Their_Influence_on_Student_Grades(df):
    expected = ['student_id', 'final_grade', 'daily_study_hours', 'daily_sleep_hours', 'daily_screen_time_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between daily study hours and final grade
    if 'daily_study_hours' in df.columns:
        fig_study_hours_grade = px.scatter(df, x='daily_study_hours', y='final_grade',
                                          title='Final Grade vs. Daily Study Hours', trendline="ols")
    else:
        fig_study_hours_grade = go.Figure().add_annotation(text="Daily study hours data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between daily sleep hours and final grade
    if 'daily_sleep_hours' in df.columns:
        fig_sleep_hours_grade = px.scatter(df, x='daily_sleep_hours', y='final_grade',
                                          title='Final Grade vs. Daily Sleep Hours', trendline="ols")
    else:
        fig_sleep_hours_grade = go.Figure().add_annotation(text="Daily sleep hours data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_daily_study_hours': fig_study_hours_grade,
        'final_grade_vs_daily_sleep_hours': fig_sleep_hours_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_daily_study_grade": df[['daily_study_hours', 'final_grade']].corr().iloc[0, 1] if 'daily_study_hours' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Demographic_Factors_and_Test_Preparation_on_Student_Scores(df):
    expected = ['student_id', 'overall_score', 'gender', 'race_ethnicity', 'parental_education_level', 'test_preparation_completed']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'overall_score'], inplace=True)

    # Average overall score by gender
    if 'gender' in df.columns:
        avg_score_by_gender = df.groupby('gender')['overall_score'].mean().reset_index()
        fig_score_by_gender = px.bar(avg_score_by_gender, x='gender', y='overall_score',
                                     title='Average Overall Score by Gender')
    else:
        fig_score_by_gender = go.Figure().add_annotation(text="Gender data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average overall score by test preparation completed
    if 'test_preparation_completed' in df.columns:
        avg_score_by_prep = df.groupby('test_preparation_completed')['overall_score'].mean().reset_index()
        fig_score_by_prep = px.bar(avg_score_by_prep, x='test_preparation_completed', y='overall_score',
                                   title='Average Overall Score by Test Preparation Completed')
    else:
        fig_score_by_prep = go.Figure().add_annotation(text="Test preparation data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_overall_score_by_gender': fig_score_by_gender,
        'average_overall_score_by_test_preparation_completed': fig_score_by_prep
    }

    metrics = {
        "overall_avg_score": df['overall_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Study_Time_and_Absences_on_Final_Academic_Performance(df):
    expected = ['student_id', 'final_academic_performance', 'study_time_hours_per_week', 'number_of_absences']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_academic_performance'], inplace=True)

    # Correlation between study time and final academic performance
    if 'study_time_hours_per_week' in df.columns:
        fig_study_time_performance = px.scatter(df, x='study_time_hours_per_week', y='final_academic_performance',
                                               title='Final Academic Performance vs. Weekly Study Time', trendline="ols")
    else:
        fig_study_time_performance = go.Figure().add_annotation(text="Study time data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between number of absences and final academic performance
    if 'number_of_absences' in df.columns:
        fig_absences_performance = px.scatter(df, x='number_of_absences', y='final_academic_performance',
                                              title='Final Academic Performance vs. Number of Absences', trendline="ols")
    else:
        fig_absences_performance = go.Figure().add_annotation(text="Number of absences data not available.",
                                                              xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_academic_performance_vs_study_time': fig_study_time_performance,
        'final_academic_performance_vs_absences': fig_absences_performance
    }

    metrics = {
        "overall_avg_performance": df['final_academic_performance'].mean(),
        "correlation_study_time_performance": df[['study_time_hours_per_week', 'final_academic_performance']].corr().iloc[0, 1] if 'study_time_hours_per_week' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Social_Activities_and_Health_on_Final_Student_Grades_G3(df):
    expected = ['student_id', 'g3_final_grade', 'social_activities', 'health_status_rating', 'absences']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'g3_final_grade'], inplace=True)

    # Average G3 grade by social activities
    if 'social_activities' in df.columns:
        avg_grade_by_social = df.groupby('social_activities')['g3_final_grade'].mean().reset_index()
        fig_grade_by_social = px.bar(avg_grade_by_social, x='social_activities', y='g3_final_grade',
                                     title='Average G3 Final Grade by Social Activities Participation')
    else:
        fig_grade_by_social = go.Figure().add_annotation(text="Social activities data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average G3 grade by health status rating
    if 'health_status_rating' in df.columns:
        avg_grade_by_health = df.groupby('health_status_rating')['g3_final_grade'].mean().reset_index()
        fig_grade_by_health = px.bar(avg_grade_by_health, x='health_status_rating', y='g3_final_grade',
                                     title='Average G3 Final Grade by Health Status Rating')
    else:
        fig_grade_by_health = go.Figure().add_annotation(text="Health status data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_g3_grade_by_social_activities': fig_grade_by_social,
        'average_g3_grade_by_health_status_rating': fig_grade_by_health
    }

    metrics = {
        "overall_avg_g3_grade": df['g3_final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Longitudinal_Academic_Performance_Analysis_G1_G2_G3(df):
    expected = ['student_id', 'g1_grade', 'g2_grade', 'g3_grade']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id'], inplace=True)

    # Average grades across G1, G2, G3
    grades_data = {}
    if 'g1_grade' in df.columns: grades_data['G1'] = df['g1_grade'].mean()
    if 'g2_grade' in df.columns: grades_data['G2'] = df['g2_grade'].mean()
    if 'g3_grade' in df.columns: grades_data['G3'] = df['g3_grade'].mean()
    avg_grades_df = pd.DataFrame(grades_data.items(), columns=['Grade_Period', 'Average_Grade'])
    fig_avg_grades_over_time = px.line(avg_grades_df, x='Grade_Period', y='Average_Grade',
                                       title='Average Grades Across G1, G2, and G3')

    # Distribution of final (G3) grades
    if 'g3_grade' in df.columns:
        fig_g3_distribution = px.histogram(df, x='g3_grade', nbins=20, title='Distribution of Final (G3) Grades')
    else:
        fig_g3_distribution = go.Figure().add_annotation(text="G3 grade data not available.",
                                                        xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_grades_over_time': fig_avg_grades_over_time,
        'g3_grade_distribution': fig_g3_distribution
    }

    metrics = {
        "total_students": df['student_id'].nunique(),
        "overall_avg_g3_grade": df['g3_grade'].mean() if 'g3_grade' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Student_Performance_Analysis_based_on_Demographics(df):
    expected = ['student_id', 'overall_grade', 'gender', 'race_ethnicity', 'parental_income_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'overall_grade'], inplace=True)

    # Average overall grade by gender
    if 'gender' in df.columns:
        avg_grade_by_gender = df.groupby('gender')['overall_grade'].mean().reset_index()
        fig_grade_by_gender = px.bar(avg_grade_by_gender, x='gender', y='overall_grade',
                                     title='Average Overall Grade by Gender')
    else:
        fig_grade_by_gender = go.Figure().add_annotation(text="Gender data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average overall grade by race/ethnicity
    if 'race_ethnicity' in df.columns:
        avg_grade_by_race = df.groupby('race_ethnicity')['overall_grade'].mean().sort_values(ascending=False).reset_index()
        fig_grade_by_race = px.bar(avg_grade_by_race, x='race_ethnicity', y='overall_grade',
                                   title='Average Overall Grade by Race/Ethnicity')
    else:
        fig_grade_by_race = go.Figure().add_annotation(text="Race/Ethnicity data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_overall_grade_by_gender': fig_grade_by_gender,
        'average_overall_grade_by_race_ethnicity': fig_grade_by_race
    }

    metrics = {
        "overall_avg_grade": df['overall_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Student_Performance_Category_Prediction_Analysis(df):
    expected = ['student_id', 'overall_score', 'performance_category', 'study_hours_weekly', 'past_academic_record']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'overall_score', 'performance_category'], inplace=True)

    # Performance category distribution
    performance_category_counts = df['performance_category'].value_counts(normalize=True).reset_index()
    performance_category_counts.columns = ['category', 'proportion']

    # Average overall score by performance category
    avg_score_by_category = df.groupby('performance_category')['overall_score'].mean().reset_index()
    fig_avg_score_by_category = px.bar(avg_score_by_category, x='performance_category', y='overall_score',
                                      title='Average Overall Score by Performance Category')

    fig_performance_category_pie = px.pie(performance_category_counts, names='category', values='proportion', title='Student Performance Category Distribution')

    plots = {
        'performance_category_distribution': fig_performance_category_pie,
        'average_overall_score_by_performance_category': fig_avg_score_by_category
    }

    metrics = {
        "total_students": len(df),
        "overall_avg_score": df['overall_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Ethnicity_and_Parental_Education_s_Role_in_Student_Grades(df):
    expected = ['student_id', 'final_grade', 'ethnicity', 'mother_education_level', 'father_education_level']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by ethnicity
    if 'ethnicity' in df.columns:
        avg_grade_by_ethnicity = df.groupby('ethnicity')['final_grade'].mean().sort_values(ascending=False).reset_index()
        fig_grade_by_ethnicity = px.bar(avg_grade_by_ethnicity, x='ethnicity', y='final_grade',
                                        title='Average Final Grade by Ethnicity')
    else:
        fig_grade_by_ethnicity = go.Figure().add_annotation(text="Ethnicity data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by mother's education level
    if 'mother_education_level' in df.columns:
        avg_grade_by_mother_edu = df.groupby('mother_education_level')['final_grade'].mean().sort_values(ascending=False).reset_index()
        fig_grade_by_mother_edu = px.bar(avg_grade_by_mother_edu, x='mother_education_level', y='final_grade',
                                         title="Average Final Grade by Mother's Education Level")
    else:
        fig_grade_by_mother_edu = go.Figure().add_annotation(text="Mother's education data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_ethnicity': fig_grade_by_ethnicity,
        'average_final_grade_by_mother_education_level': fig_grade_by_mother_edu
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Behavioral_and_Engagement_Scores_on_Academic_Outcomes(df):
    expected = ['student_id', 'academic_outcome_score', 'behavioral_score', 'engagement_score', 'attendance_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'academic_outcome_score'], inplace=True)

    # Correlation between behavioral score and academic outcome score
    if 'behavioral_score' in df.columns:
        fig_behavioral_academic = px.scatter(df, x='behavioral_score', y='academic_outcome_score',
                                            title='Academic Outcome Score vs. Behavioral Score', trendline="ols")
    else:
        fig_behavioral_academic = go.Figure().add_annotation(text="Behavioral score data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between engagement score and academic outcome score
    if 'engagement_score' in df.columns:
        fig_engagement_academic = px.scatter(df, x='engagement_score', y='academic_outcome_score',
                                            title='Academic Outcome Score vs. Engagement Score', trendline="ols")
    else:
        fig_engagement_academic = go.Figure().add_annotation(text="Engagement score data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'academic_outcome_vs_behavioral_score': fig_behavioral_academic,
        'academic_outcome_vs_engagement_score': fig_engagement_academic
    }

    metrics = {
        "overall_avg_academic_outcome": df['academic_outcome_score'].mean(),
        "correlation_behavioral_academic": df[['behavioral_score', 'academic_outcome_score']].corr().iloc[0, 1] if 'behavioral_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Continuous_Assessment_and_Study_Time_on_Final_Grade(df):
    expected = ['student_id', 'final_grade', 'continuous_assessment_average', 'weekly_study_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between continuous assessment average and final grade
    if 'continuous_assessment_average' in df.columns:
        fig_continuous_assessment_grade = px.scatter(df, x='continuous_assessment_average', y='final_grade',
                                                    title='Final Grade vs. Continuous Assessment Average', trendline="ols")
    else:
        fig_continuous_assessment_grade = go.Figure().add_annotation(text="Continuous assessment average data not available.",
                                                                     xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between weekly study hours and final grade
    if 'weekly_study_hours' in df.columns:
        fig_weekly_study_grade = px.scatter(df, x='weekly_study_hours', y='final_grade',
                                           title='Final Grade vs. Weekly Study Hours', trendline="ols")
    else:
        fig_weekly_study_grade = go.Figure().add_annotation(text="Weekly study hours data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_continuous_assessment_average': fig_continuous_assessment_grade,
        'final_grade_vs_weekly_study_hours': fig_weekly_study_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_continuous_assessment_grade": df[['continuous_assessment_average', 'final_grade']].corr().iloc[0, 1] if 'continuous_assessment_average' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Screen_Time_and_Sleep_s_Impact_on_Student_Anxiety_and_Grades(df):
    expected = ['student_id', 'final_grade', 'daily_screen_time_hours', 'daily_sleep_hours', 'anxiety_level_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between daily screen time and final grade
    if 'daily_screen_time_hours' in df.columns:
        fig_screen_time_grade = px.scatter(df, x='daily_screen_time_hours', y='final_grade',
                                           title='Final Grade vs. Daily Screen Time Hours', trendline="ols")
    else:
        fig_screen_time_grade = go.Figure().add_annotation(text="Daily screen time data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between daily sleep hours and anxiety level score
    if 'daily_sleep_hours' in df.columns and 'anxiety_level_score' in df.columns:
        fig_sleep_anxiety = px.scatter(df, x='daily_sleep_hours', y='anxiety_level_score',
                                       title='Anxiety Level Score vs. Daily Sleep Hours', trendline="ols")
    else:
        fig_sleep_anxiety = go.Figure().add_annotation(text="Daily sleep hours or anxiety level data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_daily_screen_time': fig_screen_time_grade,
        'anxiety_level_vs_daily_sleep_hours': fig_sleep_anxiety
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_screen_time_grade": df[['daily_screen_time_hours', 'final_grade']].corr().iloc[0, 1] if 'daily_screen_time_hours' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Midterm_Performance_and_Engagement_as_Predictors_of_Final_Grades(df):
    expected = ['student_id', 'final_grade', 'midterm_exam_score', 'engagement_score_lms', 'attendance_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between midterm exam score and final grade
    if 'midterm_exam_score' in df.columns:
        fig_midterm_final = px.scatter(df, x='midterm_exam_score', y='final_grade',
                                       title='Final Grade vs. Midterm Exam Score', trendline="ols")
    else:
        fig_midterm_final = go.Figure().add_annotation(text="Midterm exam score data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between engagement score (LMS) and final grade
    if 'engagement_score_lms' in df.columns:
        fig_engagement_final = px.scatter(df, x='engagement_score_lms', y='final_grade',
                                          title='Final Grade vs. LMS Engagement Score', trendline="ols")
    else:
        fig_engagement_final = go.Figure().add_annotation(text="LMS engagement score data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_midterm_exam_score': fig_midterm_final,
        'final_grade_vs_lms_engagement_score': fig_engagement_final
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_midterm_final": df[['midterm_exam_score', 'final_grade']].corr().iloc[0, 1] if 'midterm_exam_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Socioeconomic_Status_and_Its_Effect_on_Student_GPA(df):
    expected = ['student_id', 'gpa', 'parental_income_level', 'parental_education_level', 'free_reduced_lunch_status']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'gpa'], inplace=True)

    # Average GPA by parental income level
    if 'parental_income_level' in df.columns:
        avg_gpa_by_income = df.groupby('parental_income_level')['gpa'].mean().sort_values(ascending=False).reset_index()
        fig_gpa_by_income = px.bar(avg_gpa_by_income, x='parental_income_level', y='gpa',
                                   title='Average GPA by Parental Income Level')
    else:
        fig_gpa_by_income = go.Figure().add_annotation(text="Parental income level data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average GPA by free/reduced lunch status
    if 'free_reduced_lunch_status' in df.columns:
        avg_gpa_by_lunch_status = df.groupby('free_reduced_lunch_status')['gpa'].mean().reset_index()
        fig_gpa_by_lunch_status = px.bar(avg_gpa_by_lunch_status, x='free_reduced_lunch_status', y='gpa',
                                         title='Average GPA by Free/Reduced Lunch Status')
    else:
        fig_gpa_by_lunch_status = go.Figure().add_annotation(text="Free/reduced lunch status data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_gpa_by_parental_income_level': fig_gpa_by_income,
        'average_gpa_by_free_reduced_lunch_status': fig_gpa_by_lunch_status
    }

    metrics = {
        "overall_avg_gpa": df['gpa'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def LMS_Activity_and_Quiz_Scores_Correlation_with_Final_Score(df):
    expected = ['student_id', 'final_score', 'lms_activity_score', 'average_quiz_score', 'assignment_completion_rate']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Correlation between LMS activity score and final score
    if 'lms_activity_score' in df.columns:
        fig_lms_activity_final = px.scatter(df, x='lms_activity_score', y='final_score',
                                           title='Final Score vs. LMS Activity Score', trendline="ols")
    else:
        fig_lms_activity_final = go.Figure().add_annotation(text="LMS activity score data not available.",
                                                            xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between average quiz score and final score
    if 'average_quiz_score' in df.columns:
        fig_quiz_score_final = px.scatter(df, x='average_quiz_score', y='final_score',
                                          title='Final Score vs. Average Quiz Score', trendline="ols")
    else:
        fig_quiz_score_final = go.Figure().add_annotation(text="Average quiz score data not available.",
                                                          xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_score_vs_lms_activity_score': fig_lms_activity_final,
        'final_score_vs_average_quiz_score': fig_quiz_score_final
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean(),
        "correlation_lms_activity_final": df[['lms_activity_score', 'final_score']].corr().iloc[0, 1] if 'lms_activity_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Exam_Score_and_Pass_Status_Prediction(df):
    expected = ['student_id', 'exam_score', 'pass_status', 'study_time_hours', 'previous_exam_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'exam_score', 'pass_status'], inplace=True)

    # Pass status distribution
    pass_status_counts = df['pass_status'].value_counts(normalize=True).reset_index()
    pass_status_counts.columns = ['status', 'proportion']

    # Average exam score for Pass vs. Fail students
    avg_exam_score_by_status = df.groupby('pass_status')['exam_score'].mean().reset_index()
    fig_avg_exam_score_by_status = px.bar(avg_exam_score_by_status, x='pass_status', y='exam_score',
                                         title='Average Exam Score for Pass vs. Fail Students')

    fig_pass_status_pie = px.pie(pass_status_counts, names='status', values='proportion', title='Exam Pass Status Distribution')

    plots = {
        'pass_status_distribution': fig_pass_status_pie,
        'average_exam_score_by_pass_status': fig_avg_exam_score_by_status
    }

    metrics = {
        "total_students": len(df),
        "overall_avg_exam_score": df['exam_score'].mean(),
        "pass_rate_percent": pass_status_counts[pass_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100 if 'pass' in pass_status_counts['status'].astype(str).str.lower().values else 0
    }

    return {"metrics": metrics, "plots": plots}

def Extracurriculars_and_Study_Hours_on_Average_Score(df):
    expected = ['student_id', 'average_score', 'extra_curricular_activities_participation', 'weekly_study_hours']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "general Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'average_score'], inplace=True)

    # Average score by extracurricular activities participation
    if 'extra_curricular_activities_participation' in df.columns:
        avg_score_by_extracurriculars = df.groupby('extra_curricular_activities_participation')['average_score'].mean().reset_index()
        fig_score_by_extracurriculars = px.bar(avg_score_by_extracurriculars, x='extra_curricular_activities_participation', y='average_score',
                                              title='Average Score by Extracurricular Activities Participation')
    else:
        fig_score_by_extracurriculars = go.Figure().add_annotation(text="Extracurricular activities data not available.",
                                                                   xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between weekly study hours and average score
    if 'weekly_study_hours' in df.columns:
        fig_study_hours_score = px.scatter(df, x='weekly_study_hours', y='average_score',
                                          title='Average Score vs. Weekly Study Hours', trendline="ols")
    else:
        fig_study_hours_score = go.Figure().add_annotation(text="Weekly study hours data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_score_by_extracurriculars_participation': fig_score_by_extracurriculars,
        'average_score_vs_weekly_study_hours': fig_study_hours_score
    }

    metrics = {
        "overall_avg_score": df['average_score'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Health_and_Engagement_s_Influence_on_Final_Grades(df):
    expected = ['student_id', 'final_grade', 'health_status_category', 'engagement_level_in_class', 'absences']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by health status category
    if 'health_status_category' in df.columns:
        avg_grade_by_health_status = df.groupby('health_status_category')['final_grade'].mean().reset_index()
        fig_grade_by_health_status = px.bar(avg_grade_by_health_status, x='health_status_category', y='final_grade',
                                           title='Average Final Grade by Health Status Category')
    else:
        fig_grade_by_health_status = go.Figure().add_annotation(text="Health status category data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by engagement level in class
    if 'engagement_level_in_class' in df.columns:
        avg_grade_by_engagement = df.groupby('engagement_level_in_class')['final_grade'].mean().reset_index()
        fig_grade_by_engagement = px.bar(avg_grade_by_engagement, x='engagement_level_in_class', y='final_grade',
                                         title='Average Final Grade by Engagement Level in Class')
    else:
        fig_grade_by_engagement = go.Figure().add_annotation(text="Engagement level data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_health_status_category': fig_grade_by_health_status,
        'average_final_grade_by_engagement_level_in_class': fig_grade_by_engagement
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Student_Grade_Category_Classification_Analysis(df):
    expected = ['student_id', 'final_grade', 'grade_category', 'midterm_score', 'quiz_score_average']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade', 'grade_category'], inplace=True)

    # Grade category distribution
    grade_category_counts = df['grade_category'].value_counts(normalize=True).reset_index()
    grade_category_counts.columns = ['category', 'proportion']

    # Average final grade by grade category
    avg_final_grade_by_category = df.groupby('grade_category')['final_grade'].mean().reset_index()
    fig_avg_final_grade_category = px.bar(avg_final_grade_by_category, x='grade_category', y='final_grade',
                                         title='Average Final Grade by Grade Category')

    fig_grade_category_pie = px.pie(grade_category_counts, names='category', values='proportion', title='Student Grade Category Distribution')

    plots = {
        'grade_category_distribution': fig_grade_category_pie,
        'average_final_grade_by_grade_category': fig_avg_final_grade_category
    }

    metrics = {
        "total_students": len(df),
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Factors_Influencing_Overall_Student_Score(df):
    expected = ['student_id', 'overall_score', 'study_hours_per_week', 'absences_count', 'parental_support_level_score']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'overall_score'], inplace=True)

    # Correlation between study hours and overall score
    if 'study_hours_per_week' in df.columns:
        fig_study_hours_overall = px.scatter(df, x='study_hours_per_week', y='overall_score',
                                            title='Overall Score vs. Weekly Study Hours', trendline="ols")
    else:
        fig_study_hours_overall = go.Figure().add_annotation(text="Weekly study hours data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between parental support level score and overall score
    if 'parental_support_level_score' in df.columns:
        fig_parental_support_overall = px.scatter(df, x='parental_support_level_score', y='overall_score',
                                                 title='Overall Score vs. Parental Support Level Score', trendline="ols")
    else:
        fig_parental_support_overall = go.Figure().add_annotation(text="Parental support data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'overall_score_vs_weekly_study_hours': fig_study_hours_overall,
        'overall_score_vs_parental_support_level_score': fig_parental_support_overall
    }

    metrics = {
        "overall_avg_score": df['overall_score'].mean(),
        "correlation_study_hours_overall": df[['study_hours_per_week', 'overall_score']].corr().iloc[0, 1] if 'study_hours_per_week' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Study_Habits_and_Past_Performance_on_Final_Score(df):
    expected = ['student_id', 'final_score', 'weekly_study_hours', 'g1_grade_past', 'g2_grade_past']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Correlation between weekly study hours and final score
    if 'weekly_study_hours' in df.columns:
        fig_study_hours_final = px.scatter(df, x='weekly_study_hours', y='final_score',
                                          title='Final Score vs. Weekly Study Hours', trendline="ols")
    else:
        fig_study_hours_final = go.Figure().add_annotation(text="Weekly study hours data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between G2 grade (past performance) and final score
    if 'g2_grade_past' in df.columns:
        fig_g2_final_score = px.scatter(df, x='g2_grade_past', y='final_score',
                                        title='Final Score vs. G2 Grade (Past Performance)', trendline="ols")
    else:
        fig_g2_final_score = go.Figure().add_annotation(text="G2 grade data not available.",
                                                       xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_score_vs_weekly_study_hours': fig_study_hours_final,
        'final_score_vs_g2_grade_past': fig_g2_final_score
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean(),
        "correlation_study_hours_final": df[['weekly_study_hours', 'final_score']].corr().iloc[0, 1] if 'weekly_study_hours' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Test_Preparation_and_Demographics_Impact_on_Final_Grade(df):
    expected = ['student_id', 'final_grade', 'test_preparation_course_completed', 'gender', 'race_ethnicity']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Average final grade by test preparation course completion
    if 'test_preparation_course_completed' in df.columns:
        avg_grade_by_prep = df.groupby('test_preparation_course_completed')['final_grade'].mean().reset_index()
        fig_grade_by_prep = px.bar(avg_grade_by_prep, x='test_preparation_course_completed', y='final_grade',
                                   title='Average Final Grade by Test Preparation Course Completion')
    else:
        fig_grade_by_prep = go.Figure().add_annotation(text="Test preparation data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Average final grade by race/ethnicity
    if 'race_ethnicity' in df.columns:
        avg_grade_by_race = df.groupby('race_ethnicity')['final_grade'].mean().sort_values(ascending=False).reset_index()
        fig_grade_by_race = px.bar(avg_grade_by_race, x='race_ethnicity', y='final_grade',
                                   title='Average Final Grade by Race/Ethnicity')
    else:
        fig_grade_by_race = go.Figure().add_annotation(text="Race/Ethnicity data not available.",
                                                      xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_grade_by_test_preparation': fig_grade_by_prep,
        'average_final_grade_by_race_ethnicity': fig_grade_by_race
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean()
    }

    return {"metrics": metrics, "plots": plots}

def Course_Load_and_Absences_Effect_on_Student_GPA(df):
    expected = ['student_id', 'gpa', 'number_of_courses_taken', 'total_absences_in_semester']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'gpa'], inplace=True)

    # Correlation between number of courses taken and GPA
    if 'number_of_courses_taken' in df.columns:
        fig_course_load_gpa = px.scatter(df, x='number_of_courses_taken', y='gpa',
                                         title='GPA vs. Number of Courses Taken', trendline="ols")
    else:
        fig_course_load_gpa = go.Figure().add_annotation(text="Number of courses taken data not available.",
                                                         xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between total absences and GPA
    if 'total_absences_in_semester' in df.columns:
        fig_absences_gpa = px.scatter(df, x='total_absences_in_semester', y='gpa',
                                      title='GPA vs. Total Absences in Semester', trendline="ols")
    else:
        fig_absences_gpa = go.Figure().add_annotation(text="Total absences data not available.",
                                                     xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'gpa_vs_number_of_courses_taken': fig_course_load_gpa,
        'gpa_vs_total_absences_in_semester': fig_absences_gpa
    }

    metrics = {
        "overall_avg_gpa": df['gpa'].mean(),
        "correlation_course_load_gpa": df[['number_of_courses_taken', 'gpa']].corr().iloc[0, 1] if 'number_of_courses_taken' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Internet_Access_and_Activities_Impact_on_Final_Score(df):
    expected = ['student_id', 'final_score', 'internet_access_type', 'online_social_media_hours', 'online_learning_resource_usage']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_score'], inplace=True)

    # Average final score by internet access type
    if 'internet_access_type' in df.columns:
        avg_score_by_internet_access = df.groupby('internet_access_type')['final_score'].mean().reset_index()
        fig_score_by_internet_access = px.bar(avg_score_by_internet_access, x='internet_access_type', y='final_score',
                                              title='Average Final Score by Internet Access Type')
    else:
        fig_score_by_internet_access = go.Figure().add_annotation(text="Internet access type data not available.",
                                                                 xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between online learning resource usage and final score
    if 'online_learning_resource_usage' in df.columns:
        fig_online_resources_score = px.scatter(df, x='online_learning_resource_usage', y='final_score',
                                                title='Final Score vs. Online Learning Resource Usage', trendline="ols")
    else:
        fig_online_resources_score = go.Figure().add_annotation(text="Online learning resource usage data not available.",
                                                                xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'average_final_score_by_internet_access_type': fig_score_by_internet_access,
        'final_score_vs_online_learning_resource_usage': fig_online_resources_score
    }

    metrics = {
        "overall_avg_final_score": df['final_score'].mean(),
        "correlation_online_resources_final": df[['online_learning_resource_usage', 'final_score']].corr().iloc[0, 1] if 'online_learning_resource_usage' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}

def Test_Validity_and_Study_Time_s_Effect_on_Final_Grades(df):
    expected = ['student_id', 'final_grade', 'test_validity_score', 'weekly_study_hours', 'exam_difficulty_rating']
    matched = fuzzy_match_column(df, expected)
    missing = [col for col in expected if matched.get(col) is None]
    if missing:
        return {
            "missing_columns": show_missing_columns_warning(missing, matched),
            "insights": show_general_insights(df, "General Analysis")
        }

    df = df.rename(columns={v: k for k, v in matched.items() if v})
    df.dropna(subset=['student_id', 'final_grade'], inplace=True)

    # Correlation between test validity score and final grade
    if 'test_validity_score' in df.columns:
        fig_test_validity_grade = px.scatter(df, x='test_validity_score', y='final_grade',
                                             title='Final Grade vs. Test Validity Score', trendline="ols")
    else:
        fig_test_validity_grade = go.Figure().add_annotation(text="Test validity score data not available.",
                                                             xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    # Correlation between weekly study hours and final grade
    if 'weekly_study_hours' in df.columns:
        fig_study_hours_grade = px.scatter(df, x='weekly_study_hours', y='final_grade',
                                          title='Final Grade vs. Weekly Study Hours', trendline="ols")
    else:
        fig_study_hours_grade = go.Figure().add_annotation(text="Weekly study hours data not available.",
                                                           xref="paper", yref="paper", showarrow=False, font=dict(size=14))

    plots = {
        'final_grade_vs_test_validity_score': fig_test_validity_grade,
        'final_grade_vs_weekly_study_hours': fig_study_hours_grade
    }

    metrics = {
        "overall_avg_final_grade": df['final_grade'].mean(),
        "correlation_test_validity_final": df[['test_validity_score', 'final_grade']].corr().iloc[0, 1] if 'test_validity_score' in df.columns else 'N/A'
    }

    return {"metrics": metrics, "plots": plots}



def main():
    print("📚 Student Analytics Dashboard")
    file_path = input("Enter path to your student data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252): ")
    if not encoding:
        encoding = 'utf-8'
    df = load_data(file_path, encoding=encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    analysis_options = [

                    "Student Test Score Analysis by Demographics and Preparation",
                    "Factors Affecting Student Final Grades",
                    "Student Academic Performance Summary Analysis",
                    "Study Habits and Their Impact on Final Scores",
                    "Student Engagement and Performance Analysis",
                    "Impact of Test Preparation on Academic Scores",
                    "Learning Management System (LMS) Usage and Grade Correlation",
                    "Demographic and Health Factors on Student Scores",
                    "Social Factors and Internet Usage Impact on Student Performance",
                    "Student Pass/Fail Prediction Analysis",
                    "Impact of Past Grades and Study Time on Current Performance",
                    "Extracurricular Activities and Academic Grade Analysis",
                    "Family and Internet Support on Student Final Scores",
                    "Parental Background and Study Time on Final Grades",
                    "Family Relationships and Student Grade Analysis",
                    "Assessment Scores and Attendance Impact on Final Grade",
                    "Lifestyle Factors and Their Correlation with Student GPA",
                    "Educational Support Systems' Impact on Student Grades",
                    "Impact of Paid Classes and School Support on Performance",
                    "Student Performance and Pass Status Prediction",
                    "Health, Absences, and Travel Time on Final Grades (G3)",
                    "Impact of Extra Paid Classes and School Support on Grades",
                    "Social and Health Factors Affecting Student Scores",
                    "Physical Attributes and Commute's Effect on Grades",
                    "Student Performance and Pass/Fail Classification",
                    "Digital Engagement and Parental Support on Student GPA",
                    "Daily Habits and Their Influence on Student Grades",
                    "Demographic Factors and Test Preparation on Student Scores",
                    "Study Time and Absences on Final Academic Performance",
                    "Social Activities and Health on Final Student Grades (G3)",
                    "Longitudinal Academic Performance Analysis (G1, G2, G3)",
                    "Student Performance Analysis based on Demographics",
                    "Student Performance Category Prediction Analysis",
                    "Ethnicity and Parental Education's Role in Student Grades",
                    "Behavioral and Engagement Scores on Academic Outcomes",
                    "Continuous Assessment and Study Time on Final Grade",
                    "Screen Time and Sleep's Impact on Student Anxiety and Grades",
                    "Midterm Performance and Engagement as Predictors of Final Grades",
                    "Socioeconomic Status and Its Effect on Student GPA",
                    "LMS Activity and Quiz Scores' Correlation with Final Score",
                    "Exam Score and Pass Status Prediction",
                    "Extracurriculars and Study Hours on Average Score",
                    "Health and Engagement's Influence on Final Grades",
                    "Student Grade Category Classification Analysis",
                    "Factors Influencing Overall Student Score",
                    "Study Habits and Past Performance on Final Score",
                    "Test Preparation and Demographics' Impact on Final Grade",
                    "Course Load and Absences' Effect on Student GPA",
                    "Internet Access and Activities' Impact on Final Score",
                    "Test Validity and Study Time's Effect on Final Grades",

    
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

    if selected == "Student_Test_Score_Analysis_by_Demographics_and_Preparation":
        Student_Test_Score_Analysis_by_Demographics_and_Preparation(df)
    elif selected == "Factors_Affecting_Student_Final_Grades":
        Factors_Affecting_Student_Final_Grades(df)
    elif selected == "Student_Academic_Performance_Summary_Analysis":
        Student_Academic_Performance_Summary_Analysis(df)
    elif selected == "Study_Habits_and_Their_Impact_on_Final_Scores":
        Study_Habits_and_Their_Impact_on_Final_Scores(df)
    elif selected == "Student_Engagement_and_Performance_Analysis":
        Student_Engagement_and_Performance_Analysis(df)
    elif selected == "Impact_of_Test_Preparation_on_Academic_Scores":
        Impact_of_Test_Preparation_on_Academic_Scores(df)
    elif selected == "Learning_Management_System_LMS_Usage_and_Grade_Correlation":
        Learning_Management_System_LMS_Usage_and_Grade_Correlation(df)
    elif selected == "Demographic_and_Health_Factors_on_Student_Scores":
        Demographic_and_Health_Factors_on_Student_Scores(df)
    elif selected == "Social_Factors_and_Internet_Usage_Impact_on_Student_Performance":
        Social_Factors_and_Internet_Usage_Impact_on_Student_Performance(df)
    elif selected == "Student_Pass_Fail_Prediction_Analysis":
        Student_Pass_Fail_Prediction_Analysis(df)
    elif selected == "Impact_of_Past_Grades_and_Study_Time_on_Current_Performance":
        Impact_of_Past_Grades_and_Study_Time_on_Current_Performance(df)
    elif selected == "Extracurricular_Activities_and_Academic_Grade_Analysis":
        Extracurricular_Activities_and_Academic_Grade_Analysis(df)
    elif selected == "Family_and_Internet_Support_on_Student_Final_Scores":
        Family_and_Internet_Support_on_Student_Final_Scores(df)
    elif selected == "Parental_Background_and_Study_Time_on_Final_Grades":
        Parental_Background_and_Study_Time_on_Final_Grades(df)
    elif selected == "Family_Relationships_and_Student_Grade_Analysis":
        Family_Relationships_and_Student_Grade_Analysis(df)
    elif selected == "Assessment_Scores_and_Attendance_Impact_on_Final_Grade":
        Assessment_Scores_and_Attendance_Impact_on_Final_Grade(df)
    elif selected == "Lifestyle_Factors_and_Their_Correlation_with_Student_GPA":
        Lifestyle_Factors_and_Their_Correlation_with_Student_GPA(df)
    elif selected == "Educational_Support_Systems_Impact_on_Student_Grades":
        Educational_Support_Systems_Impact_on_Student_Grades(df)
    elif selected == "Impact_of_Paid_Classes_and_School_Support_on_Performance":
        Impact_of_Paid_Classes_and_School_Support_on_Performance(df)
    elif selected == "Student_Performance_and_Pass_Status_Prediction":
        Student_Performance_and_Pass_Status_Prediction(df)
    elif selected == "Health_Absences_and_Travel_Time_on_Final_Grades_G3":
        Health_Absences_and_Travel_Time_on_Final_Grades_G3(df)
    elif selected == "Impact_of_Extra_Paid_Classes_and_School_Support_on_Grades":
        Impact_of_Extra_Paid_Classes_and_School_Support_on_Grades(df)
    elif selected == "Social_and_Health_Factors_Affecting_Student_Scores":
        Social_and_Health_Factors_Affecting_Student_Scores(df)
    elif selected == "Physical_Attributes_and_Commutes_Effect_on_Grades":
        Physical_Attributes_and_Commutes_Effect_on_Grades(df)
    elif selected == "Student_Performance_and_Pass_Fail_Classification":
        Student_Performance_and_Pass_Fail_Classification(df)
    elif selected == "Digital_Engagement_and_Parental_Support_on_Student_GPA":
        Digital_Engagement_and_Parental_Support_on_Student_GPA(df)
    elif selected == "Daily_Habits_and_Their_Influence_on_Student_Grades":
        Daily_Habits_and_Their_Influence_on_Student_Grades(df)
    elif selected == "Demographic_Factors_and_Test_Preparation_on_Student_Scores":
        Demographic_Factors_and_Test_Preparation_on_Student_Scores(df)
    elif selected == "Study_Time_and_Absences_on_Final_Academic_Performance":
        Study_Time_and_Absences_on_Final_Academic_Performance(df)
    elif selected == "Social_Activities_and_Health_on_Final_Student_Grades_G3":
        Social_Activities_and_Health_on_Final_Student_Grades_G3(df)
    elif selected == "Longitudinal_Academic_Performance_Analysis_G1_G2_G3":
        Longitudinal_Academic_Performance_Analysis_G1_G2_G3(df)
    elif selected == "Student_Performance_Analysis_based_on_Demographics":
        Student_Performance_Analysis_based_on_Demographics(df)
    elif selected == "Student_Performance_Category_Prediction_Analysis":
        Student_Performance_Category_Prediction_Analysis(df)
    elif selected == "Ethnicity_and_Parental_Education_s_Role_in_Student_Grades":
        Ethnicity_and_Parental_Education_s_Role_in_Student_Grades(df)
    elif selected == "Behavioral_and_Engagement_Scores_on_Academic_Outcomes":
        Behavioral_and_Engagement_Scores_on_Academic_Outcomes(df)
    elif selected == "Continuous_Assessment_and_Study_Time_on_Final_Grade":
        Continuous_Assessment_and_Study_Time_on_Final_Grade(df)
    elif selected == "Screen_Time_and_Sleep_s_Impact_on_Student_Anxiety_and_Grades":
        Screen_Time_and_Sleep_s_Impact_on_Student_Anxiety_and_Grades(df)
    elif selected == "Midterm_Performance_and_Engagement_as_Predictors_of_Final_Grades":
        Midterm_Performance_and_Engagement_as_Predictors_of_Final_Grades(df)
    elif selected == "Socioeconomic_Status_and_Its_Effect_on_Student_GPA":
        Socioeconomic_Status_and_Its_Effect_on_Student_GPA(df)
    elif selected == "LMS_Activity_and_Quiz_Scores_Correlation_with_Final_Score":
        LMS_Activity_and_Quiz_Scores_Correlation_with_Final_Score(df)
    elif selected == "Exam_Score_and_Pass_Status_Prediction":
        Exam_Score_and_Pass_Status_Prediction(df)
    elif selected == "Extracurriculars_and_Study_Hours_on_Average_Score":
        Extracurriculars_and_Study_Hours_on_Average_Score(df)
    elif selected == "Health_and_Engagement_s_Influence_on_Final_Grades":
        Health_and_Engagement_s_Influence_on_Final_Grades(df)
    elif selected == "Student_Grade_Category_Classification_Analysis":
        Student_Grade_Category_Classification_Analysis(df)
    elif selected == "Factors_Influencing_Overall_Student_Score":
        Factors_Influencing_Overall_Student_Score(df)
    elif selected == "Study_Habits_and_Past_Performance_on_Final_Score":
        Study_Habits_and_Past_Performance_on_Final_Score(df)
    elif selected == "Test_Preparation_and_Demographics_Impact_on_Final_Grade":
        Test_Preparation_and_Demographics_Impact_on_Final_Grade(df)
    elif selected == "Course_Load_and_Absences_Effect_on_Student_GPA":
        Course_Load_and_Absences_Effect_on_Student_GPA(df)
    elif selected == "Internet_Access_and_Activities_Impact_on_Final_Score":
        Internet_Access_and_Activities_Impact_on_Final_Score(df)
    elif selected == "Test_Validity_and_Study_Time_s_Effect_on_Final_Grades":
        Test_Validity_and_Study_Time_s_Effect_on_Final_Grades(df)
    else:
        print(f"Analysis option '{selected}' not recognized or not implemented.")