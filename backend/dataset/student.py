import pandas as pd
import numpy as np
import plotly.express as px
from fuzzywuzzy import process
import warnings
import plotly.graph_objects as go
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ========== JSON TYPE CONVERSION UTILITIES ==========

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
        # First, dump to a string using the custom encoder
        json_str = json.dumps(data, cls=NumpyJSONEncoder)
        # Then, load the string back to a Python object
        return json.loads(json_str)
    except Exception as e:
        # Fallback for complex unhandled types
        print(f"Warning: Could not convert data to native types: {e}")
        return {} # Return empty dict on failure

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
            # Use process.extractOne which is good for this
            match, score = process.extractOne(target, available)
            matched[target] = match if score >= 70 else None # Use a 70% threshold
        except Exception:
            matched[target] = None
    
    return matched

def safe_rename(df, matched):
    """Renames dataframe columns based on fuzzy matches."""
    return df.rename(columns={v: k for k, v in matched.items() if v is not None})

# ========== FALLBACK & GENERAL ANALYSIS FUNCTIONS ==========

def show_general_insights(df, analysis_name="General Insights", missing_cols=None, matched_cols=None):
    """
    Provides comprehensive general insights with visualizations and metrics,
    including warnings for missing columns. Returns a structured dictionary.
    This function is non-interactive and API-ready.
    """
    analysis_type = "General Insights"
    visualizations = {}
    metrics = {}
    insights = []
    
    try:
        # Basic dataset information
        total_rows = len(df)
        total_columns = len(df.columns)
        
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        metrics = {
            "total_records": total_rows,
            "total_features": total_columns,
            "numeric_features_count": len(numeric_cols),
            "categorical_features_count": len(categorical_cols)
        }
        
        insights.append(f"Dataset contains {total_rows} records and {total_columns} features.")
        insights.append(f"Found {len(numeric_cols)} numeric and {len(categorical_cols)} categorical features.")

        # Add missing columns warning if provided
        if missing_cols and len(missing_cols) > 0:
            insights.append("--- ⚠️ Required Columns Not Found ---")
            insights.append(f"The analysis '{analysis_name}' could not run because some columns were missing.")
            for col in missing_cols:
                match_info = f" (best match: {matched_cols[col]})" if matched_cols and matched_cols.get(col) else " (no close match found)"
                insights.append(f" - Missing: '{col}'{match_info}")
            insights.append("Showing General Data Overview instead.")
            
        # --- Numeric Analysis (Non-Interactive) ---
        if len(numeric_cols) > 0:
            selected_num_col = numeric_cols[0] # Analyze the first numeric column
            insights.append(f"Analyzing first numeric feature: '{selected_num_col}'")
            try:
                hist_fig = px.histogram(df, x=selected_num_col, title=f"Distribution of {selected_num_col}")
                visualizations[f"distribution_{selected_num_col}"] = hist_fig.to_json()
                
                box_fig = px.box(df, y=selected_num_col, title=f"Box Plot of {selected_num_col}")
                visualizations[f"boxplot_{selected_num_col}"] = box_fig.to_json()
                
                metrics[f"{selected_num_col}_mean"] = df[selected_num_col].mean()
                metrics[f"{selected_num_col}_median"] = df[selected_num_col].median()
            except Exception as e:
                insights.append(f"Could not plot numeric column {selected_num_col}: {e}")
        else:
            insights.append("[WARNING] No numeric columns found for analysis.")
        
        # --- Correlation (Non-Interactive) ---
        if len(numeric_cols) >= 2:
            try:
                corr = df[numeric_cols].corr()
                corr_fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Between Numeric Features")
                visualizations["numeric_correlation_heatmap"] = corr_fig.to_json()
                metrics["correlation_matrix"] = corr.to_dict()
                insights.append("Generated correlation heatmap for numeric features.")
            except Exception as e:
                insights.append(f"Could not generate correlation heatmap: {e}")
        
        # --- Categorical Analysis (Non-Interactive) ---
        if len(categorical_cols) > 0:
            selected_cat_col = categorical_cols[0] # Analyze the first categorical column
            insights.append(f"Analyzing first categorical feature: '{selected_cat_col}'")
            try:
                # Get top 10 most frequent values
                value_counts = df[selected_cat_col].value_counts().reset_index()
                value_counts.columns = ['Value', 'Count']
                
                bar_fig = px.bar(value_counts.head(10), x='Value', y='Count', title=f"Top 10 Distribution of {selected_cat_col}")
                visualizations[f"distribution_{selected_cat_col}"] = bar_fig.to_json()
                
                metrics[f"{selected_cat_col}_unique_values"] = df[selected_cat_col].nunique()
                metrics[f"{selected_cat_col}_top_value"] = value_counts.iloc[0]['Value'] if not value_counts.empty else "N/A"
            except Exception as e:
                insights.append(f"Could not plot categorical column {selected_cat_col}: {e}")
        else:
            insights.append("[WARNING] No categorical columns found for analysis.")

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
            "insights": [f"An error occurred during general analysis: {e}"],
            "missing_columns": missing_cols or []
        }

def create_fallback_response(analysis_name, missing_cols, matched_cols, df):
    """
    Creates a structured response indicating missing columns and provides
    general insights as a fallback WITH VISUALIZATIONS.
    """
    print(f"--- ⚠️ Required Columns Not Found for {analysis_name} ---")
    print(f"Missing: {missing_cols}")
    print("Falling back to General Insights.")
    
    try:
        # Generate the fallback general insights
        general_insights_data = show_general_insights(
            df, 
            analysis_name, # Pass the original analysis name for the warning message
            missing_cols=missing_cols,
            matched_cols=matched_cols
        )
        
        # Create the fallback response WITH visualizations
        fallback_response = {
            "analysis_type": analysis_name, # Report the intended analysis type
            "status": "fallback",
            "message": f"Required columns were missing for '{analysis_name}'. Falling back to general insights.",
            "missing_columns": missing_cols,
            "matched_columns": matched_cols,
            "visualizations": general_insights_data.get("visualizations", {}),
            "metrics": general_insights_data.get("metrics", {}),
            "insights": general_insights_data.get("insights", [])
        }
        
        return fallback_response
        
    except Exception as e:
        # Absolute fallback in case general_insights fails
        return {
            "analysis_type": analysis_name,
            "status": "fallback_error",
            "message": f"Fallback to general insights failed: {e}",
            "missing_columns": missing_cols,
            "matched_columns": matched_cols,
            "visualizations": {},
            "metrics": {},
            "insights": [f"Original analysis failed due to missing columns: {missing_cols}", 
                        f"Fallback also failed: {e}"]
        }


# ========== REFACTORED STUDENT ANALYSIS FUNCTIONS ==========

def Student_Test_Score_Analysis_by_Demographics_and_Preparation(df):
    analysis_name = "Student Test Score Analysis by Demographics and Preparation"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'test_score', 'gender', 'race_ethnicity', 'test_preparation_course']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['test_score'] = pd.to_numeric(df['test_score'], errors='coerce')
        df.dropna(subset=['student_id', 'test_score'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = df['student_id'].nunique()
        metrics["overall_avg_test_score"] = df['test_score'].mean()
        insights.append(f"Analyzed {metrics['total_students']} students.")
        insights.append(f"Overall average test score: {metrics['overall_avg_test_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---
        
        # Analysis by race/ethnicity
        if 'race_ethnicity' in df.columns:
            avg_score_by_race = df.groupby('race_ethnicity')['test_score'].mean().sort_values(ascending=False).reset_index()
            fig_score_by_race = px.bar(avg_score_by_race, x='race_ethnicity', y='test_score',
                                     title='Average Test Score by Race/Ethnicity')
            visualizations['average_test_score_by_race_ethnicity'] = fig_score_by_race.to_json()
            metrics['avg_score_by_race'] = avg_score_by_race.to_dict('records')
            insights.append("Generated analysis by race/ethnicity.")
        else:
            insights.append("Skipping analysis by race/ethnicity: column not found.")

        # Analysis by test preparation
        if 'test_preparation_course' in df.columns:
            avg_score_by_prep = df.groupby('test_preparation_course')['test_score'].mean().reset_index()
            fig_score_by_prep = px.bar(avg_score_by_prep, x='test_preparation_course', y='test_score',
                                     title='Average Test Score by Test Preparation Course Completion')
            visualizations['average_test_score_by_preparation'] = fig_score_by_prep.to_json()
            metrics['avg_score_by_prep'] = avg_score_by_prep.to_dict('records')
            insights.append("Generated analysis by test preparation course.")
        else:
            insights.append("Skipping analysis by test preparation: column not found.")
        
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

def Factors_Affecting_Student_Final_Grades(df):
    analysis_name = "Factors Affecting Student Final Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'study_time_weekly_hours', 'absences', 'parental_support_level']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between study time and final grade
        if 'study_time_weekly_hours' in df.columns:
            df['study_time_weekly_hours'] = pd.to_numeric(df['study_time_weekly_hours'], errors='coerce')
            if not df['study_time_weekly_hours'].isnull().all():
                fig_study_time_grade = px.scatter(df, x='study_time_weekly_hours', y='final_grade',
                                                  title='Final Grade vs. Weekly Study Time', trendline="ols")
                visualizations['final_grade_vs_study_time'] = fig_study_time_grade.to_json()
                metrics["correlation_study_time_grade"] = df[['study_time_weekly_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between study time and final grade: {metrics['correlation_study_time_grade']:.2f}")
            else:
                 insights.append("Skipping study time analysis: column has no valid data.")
        else:
            insights.append("Skipping study time analysis: column not found.")

        # Impact of absences on final grade
        if 'absences' in df.columns:
            df['absences'] = pd.to_numeric(df['absences'], errors='coerce')
            if not df['absences'].isnull().all():
                avg_grade_by_absences = df.groupby('absences')['final_grade'].mean().reset_index()
                fig_absences_grade = px.line(avg_grade_by_absences, x='absences', y='final_grade',
                                             title='Average Final Grade by Number of Absences')
                visualizations['average_final_grade_by_absences'] = fig_absences_grade.to_json()
                insights.append("Generated analysis by number of absences.")
            else:
                 insights.append("Skipping absences analysis: column has no valid data.")
        else:
            insights.append("Skipping absences analysis: column not found.")

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

def Student_Academic_Performance_Summary_Analysis(df):
    analysis_name = "Student Academic Performance Summary Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'math_score', 'reading_score', 'writing_score', 'overall_gpa']
        matched = fuzzy_match_column(df, expected)

        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df.dropna(subset=['student_id'], inplace=True)

        metrics["total_students"] = df['student_id'].nunique()
        insights.append(f"Analyzed {metrics['total_students']} students.")
        
        # --- Metrics & Visualizations ---
        scores_data = {}
        if 'math_score' in df.columns: 
            df['math_score'] = pd.to_numeric(df['math_score'], errors='coerce')
            if not df['math_score'].isnull().all():
                scores_data['Math'] = df['math_score'].mean()
        if 'reading_score' in df.columns: 
            df['reading_score'] = pd.to_numeric(df['reading_score'], errors='coerce')
            if not df['reading_score'].isnull().all():
                scores_data['Reading'] = df['reading_score'].mean()
        if 'writing_score' in df.columns: 
            df['writing_score'] = pd.to_numeric(df['writing_score'], errors='coerce')
            if not df['writing_score'].isnull().all():
                scores_data['Writing'] = df['writing_score'].mean()

        if scores_data:
            avg_scores_df = pd.DataFrame(scores_data.items(), columns=['Subject', 'Average_Score'])
            fig_avg_subject_scores = px.bar(avg_scores_df, x='Subject', y='Average_Score',
                                          title='Average Scores Across Different Subjects')
            visualizations['average_scores_by_subject'] = fig_avg_subject_scores.to_json()
            metrics['average_subject_scores'] = avg_scores_df.to_dict('records')
            insights.append("Generated average scores by subject.")
        else:
            insights.append("No valid subject score columns (math_score, reading_score, writing_score) found or all were empty.")

        # Distribution of overall GPA
        if 'overall_gpa' in df.columns:
            df['overall_gpa'] = pd.to_numeric(df['overall_gpa'], errors='coerce')
            if not df['overall_gpa'].isnull().all():
                fig_gpa_distribution = px.histogram(df, x='overall_gpa', nbins=20, title='Distribution of Overall GPA')
                visualizations['overall_gpa_distribution'] = fig_gpa_distribution.to_json()
                metrics["overall_avg_gpa"] = df['overall_gpa'].mean()
                insights.append(f"Overall average GPA: {metrics['overall_avg_gpa']:.2f}")
            else:
                 insights.append("Skipping GPA analysis: column has no valid data.")
        else:
            insights.append("Skipping GPA analysis: column not found.")
            metrics["overall_avg_gpa"] = "N/A"

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

def Study_Habits_and_Their_Impact_on_Final_Scores(df):
    analysis_name = "Study Habits and Their Impact on Final Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'study_time_category', 'internet_access_hours_daily', 'tutoring_support']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final score by study time category
        if 'study_time_category' in df.columns:
            avg_score_by_study_time = df.groupby('study_time_category')['final_score'].mean().sort_values(ascending=False).reset_index()
            fig_score_by_study_time = px.bar(avg_score_by_study_time, x='study_time_category', y='final_score',
                                             title='Average Final Score by Study Time Category')
            visualizations['average_final_score_by_study_time_category'] = fig_score_by_study_time.to_json()
            metrics['avg_score_by_study_time'] = avg_score_by_study_time.to_dict('records')
            insights.append("Generated analysis by study time category.")
        else:
            insights.append("Skipping study time category analysis: column not found.")

        # Impact of tutoring support on final scores
        if 'tutoring_support' in df.columns:
            avg_score_by_tutoring = df.groupby('tutoring_support')['final_score'].mean().reset_index()
            fig_score_by_tutoring = px.bar(avg_score_by_tutoring, x='tutoring_support', y='final_score',
                                           title='Average Final Score by Tutoring Support')
            visualizations['average_final_score_by_tutoring_support'] = fig_score_by_tutoring.to_json()
            metrics['avg_score_by_tutoring'] = avg_score_by_tutoring.to_dict('records')
            insights.append("Generated analysis by tutoring support.")
        else:
            insights.append("Skipping tutoring support analysis: column not found.")

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

def Student_Engagement_and_Performance_Analysis(df):
    analysis_name = "Student Engagement and Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'attendance_rate', 'participation_score', 'extra_curricular_activities']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)

        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between attendance rate and final grade
        if 'attendance_rate' in df.columns:
            df['attendance_rate'] = pd.to_numeric(df['attendance_rate'], errors='coerce')
            if not df['attendance_rate'].isnull().all():
                fig_attendance_grade = px.scatter(df, x='attendance_rate', y='final_grade',
                                                  title='Final Grade vs. Attendance Rate', trendline="ols")
                visualizations['final_grade_vs_attendance_rate'] = fig_attendance_grade.to_json()
                metrics["correlation_attendance_grade"] = df[['attendance_rate', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between attendance and final grade: {metrics['correlation_attendance_grade']:.2f}")
            else:
                insights.append("Skipping attendance rate analysis: column has no valid data.")
        else:
            insights.append("Skipping attendance rate analysis: column not found.")
            metrics["correlation_attendance_grade"] = "N/A"

        # Impact of participation score on final grade
        if 'participation_score' in df.columns:
            df['participation_score'] = pd.to_numeric(df['participation_score'], errors='coerce')
            if not df['participation_score'].isnull().all():
                fig_participation_grade = px.scatter(df, x='participation_score', y='final_grade',
                                                     title='Final Grade vs. Participation Score', trendline="ols")
                visualizations['final_grade_vs_participation_score'] = fig_participation_grade.to_json()
                metrics["correlation_participation_grade"] = df[['participation_score', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between participation and final grade: {metrics['correlation_participation_grade']:.2f}")
            else:
                insights.append("Skipping participation score analysis: column has no valid data.")
        else:
            insights.append("Skipping participation score analysis: column not found.")

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

def Impact_of_Test_Preparation_on_Academic_Scores(df):
    analysis_name = "Impact of Test Preparation on Academic Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'pre_test_score', 'post_test_score', 'test_preparation_course_completed']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df.dropna(subset=['student_id'], inplace=True)

        metrics["total_students"] = df['student_id'].nunique()
        insights.append(f"Analyzed {metrics['total_students']} students.")

        # --- Visualizations & Deeper Metrics ---
        
        # Average score improvement
        if 'pre_test_score' in df.columns and 'post_test_score' in df.columns and 'test_preparation_course_completed' in df.columns:
            df['pre_test_score'] = pd.to_numeric(df['pre_test_score'], errors='coerce')
            df['post_test_score'] = pd.to_numeric(df['post_test_score'], errors='coerce')
            df.dropna(subset=['pre_test_score', 'post_test_score'], inplace=True)

            df['score_improvement'] = df['post_test_score'] - df['pre_test_score']
            avg_improvement_by_prep = df.groupby('test_preparation_course_completed')['score_improvement'].mean().reset_index()
            
            fig_improvement_by_prep = px.bar(avg_improvement_by_prep, x='test_preparation_course_completed', y='score_improvement',
                                             title='Average Score Improvement by Test Preparation Status')
            visualizations['average_score_improvement_by_preparation_status'] = fig_improvement_by_prep.to_json()

            metrics["overall_avg_score_improvement"] = df['score_improvement'].mean()
            metrics["avg_improvement_by_prep"] = avg_improvement_by_prep.to_dict('records')
            insights.append(f"Overall average score improvement: {metrics['overall_avg_score_improvement']:.2f}")
            
            # Distribution of post-test scores
            fig_post_score_dist = px.histogram(df, x='post_test_score', color='test_preparation_course_completed',
                                               barmode='overlay', title='Post-Test Score Distribution by Preparation Status')
            visualizations['post_test_score_distribution_by_preparation_status'] = fig_post_score_dist.to_json()
        else:
            insights.append("Skipping score improvement analysis: missing 'pre_test_score', 'post_test_score', or 'test_preparation_course_completed'.")
            metrics["overall_avg_score_improvement"] = "N/A"

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

def Learning_Management_System_LMS_Usage_and_Grade_Correlation(df):
    analysis_name = "Learning Management System (LMS) Usage and Grade Correlation"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'lms_login_frequency', 'lms_resource_views', 'assignment_submission_on_time_rate']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between LMS login frequency and final grade
        if 'lms_login_frequency' in df.columns:
            df['lms_login_frequency'] = pd.to_numeric(df['lms_login_frequency'], errors='coerce')
            if not df['lms_login_frequency'].isnull().all():
                fig_lms_login_grade = px.scatter(df, x='lms_login_frequency', y='final_grade',
                                                 title='Final Grade vs. LMS Login Frequency', trendline="ols")
                visualizations['final_grade_vs_lms_login_frequency'] = fig_lms_login_grade.to_json()
                metrics["correlation_lms_login_grade"] = df[['lms_login_frequency', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between LMS login frequency and final grade: {metrics['correlation_lms_login_grade']:.2f}")
            else:
                insights.append("Skipping LMS login analysis: column has no valid data.")
        else:
            insights.append("Skipping LMS login analysis: column not found.")
            metrics["correlation_lms_login_grade"] = "N/A"

        # Correlation between LMS resource views and final grade
        if 'lms_resource_views' in df.columns:
            df['lms_resource_views'] = pd.to_numeric(df['lms_resource_views'], errors='coerce')
            if not df['lms_resource_views'].isnull().all():
                fig_lms_resource_grade = px.scatter(df, x='lms_resource_views', y='final_grade',
                                                    title='Final Grade vs. LMS Resource Views', trendline="ols")
                visualizations['final_grade_vs_lms_resource_views'] = fig_lms_resource_grade.to_json()
                metrics["correlation_lms_resource_grade"] = df[['lms_resource_views', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between LMS resource views and final grade: {metrics['correlation_lms_resource_grade']:.2f}")
            else:
                insights.append("Skipping LMS resource views analysis: column has no valid data.")
        else:
            insights.append("Skipping LMS resource views analysis: column not found.")

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

def Demographic_and_Health_Factors_on_Student_Scores(df):
    analysis_name = "Demographic and Health Factors on Student Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'gender', 'age', 'health_status', 'absences']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final score by gender
        if 'gender' in df.columns:
            avg_score_by_gender = df.groupby('gender')['final_score'].mean().reset_index()
            fig_score_by_gender = px.bar(avg_score_by_gender, x='gender', y='final_score',
                                         title='Average Final Score by Gender')
            visualizations['average_final_score_by_gender'] = fig_score_by_gender.to_json()
            metrics['avg_score_by_gender'] = avg_score_by_gender.to_dict('records')
            insights.append("Generated analysis by gender.")
        else:
            insights.append("Skipping gender analysis: column not found.")

        # Average final score by health status
        if 'health_status' in df.columns:
            avg_score_by_health = df.groupby('health_status')['final_score'].mean().reset_index()
            fig_score_by_health = px.bar(avg_score_by_health, x='health_status', y='final_score',
                                         title='Average Final Score by Health Status')
            visualizations['average_final_score_by_health_status'] = fig_score_by_health.to_json()
            metrics['avg_score_by_health_status'] = avg_score_by_health.to_dict('records')
            insights.append("Generated analysis by health status.")
        else:
            insights.append("Skipping health status analysis: column not found.")

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

def Social_Factors_and_Internet_Usage_Impact_on_Student_Performance(df):
    analysis_name = "Social Factors and Internet Usage Impact on Student Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'social_activities_level', 'internet_usage_hours_daily', 'family_relationship_quality']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by social activities level
        if 'social_activities_level' in df.columns:
            avg_grade_by_social_activities = df.groupby('social_activities_level')['final_grade'].mean().reset_index()
            fig_grade_by_social_activities = px.bar(avg_grade_by_social_activities, x='social_activities_level', y='final_grade',
                                                    title='Average Final Grade by Social Activities Level')
            visualizations['average_final_grade_by_social_activities'] = fig_grade_by_social_activities.to_json()
            metrics['avg_grade_by_social_activities'] = avg_grade_by_social_activities.to_dict('records')
            insights.append("Generated analysis by social activities level.")
        else:
            insights.append("Skipping social activities analysis: column not found.")

        # Correlation between internet usage and final grade
        if 'internet_usage_hours_daily' in df.columns:
            df['internet_usage_hours_daily'] = pd.to_numeric(df['internet_usage_hours_daily'], errors='coerce')
            if not df['internet_usage_hours_daily'].isnull().all():
                fig_internet_usage_grade = px.scatter(df, x='internet_usage_hours_daily', y='final_grade',
                                                      title='Final Grade vs. Daily Internet Usage Hours', trendline="ols")
                visualizations['final_grade_vs_internet_usage'] = fig_internet_usage_grade.to_json()
                metrics["correlation_internet_usage_grade"] = df[['internet_usage_hours_daily', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between internet usage and final grade: {metrics['correlation_internet_usage_grade']:.2f}")
            else:
                insights.append("Skipping internet usage analysis: column has no valid data.")
        else:
            insights.append("Skipping internet usage analysis: column not found.")
            metrics["correlation_internet_usage_grade"] = "N/A"

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

def Student_Pass_Fail_Prediction_Analysis(df):
    analysis_name = "Student Pass/Fail Prediction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'pass_status', 'g1_grade', 'g2_grade', 'study_time_weekly_hours']
        matched = fuzzy_match_column(df, expected)

        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade', 'pass_status'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = df['student_id'].nunique()
        
        # Pass/Fail status distribution
        pass_fail_counts = df['pass_status'].value_counts(normalize=True).reset_index()
        pass_fail_counts.columns = ['status', 'proportion']
        
        pass_rate = pass_fail_counts[pass_fail_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100
        metrics["pass_rate_percent"] = pass_rate if pass_rate else 0
        metrics["pass_fail_distribution"] = pass_fail_counts.to_dict('records')
        
        insights.append(f"Analyzed {metrics['total_students']} students.")
        insights.append(f"Overall pass rate: {metrics['pass_rate_percent']:.2f}%")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade for Pass vs. Fail students
        avg_grade_by_status = df.groupby('pass_status')['final_grade'].mean().reset_index()
        fig_avg_grade_by_status = px.bar(avg_grade_by_status, x='pass_status', y='final_grade',
                                         title='Average Final Grade for Pass vs. Fail Students')
        visualizations['average_final_grade_by_pass_fail_status'] = fig_avg_grade_by_status.to_json()
        metrics['avg_grade_by_pass_fail_status'] = avg_grade_by_status.to_dict('records')

        fig_pass_fail_pie = px.pie(pass_fail_counts, names='status', values='proportion', title='Student Pass/Fail Status Distribution')
        visualizations['pass_fail_status_distribution'] = fig_pass_fail_pie.to_json()

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

def Impact_of_Past_Grades_and_Study_Time_on_Current_Performance(df):
    analysis_name = "Impact of Past Grades and Study Time on Current Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'g1_grade', 'g2_grade', 'final_grade_g3', 'study_time_weekly_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_grade_g3'] = pd.to_numeric(df['final_grade_g3'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade_g3'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_g3_grade"] = df['final_grade_g3'].mean()
        insights.append(f"Overall average final (G3) grade: {metrics['overall_avg_g3_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between G1/G2 grades and G3
        if 'g1_grade' in df.columns:
            df['g1_grade'] = pd.to_numeric(df['g1_grade'], errors='coerce')
            if not df['g1_grade'].isnull().all():
                fig_g1_g3_scatter = px.scatter(df, x='g1_grade', y='final_grade_g3', title='G3 Final Grade vs. G1 Grade', trendline="ols")
                visualizations['g3_vs_g1_grade_scatter'] = fig_g1_g3_scatter.to_json()
                metrics["correlation_g1_g3"] = df[['g1_grade', 'final_grade_g3']].corr().iloc[0, 1]
                insights.append(f"Correlation between G1 and G3 grades: {metrics['correlation_g1_g3']:.2f}")
            else:
                insights.append("Skipping G1 grade analysis: column has no valid data.")
        else:
            insights.append("Skipping G1 grade analysis: column not found.")
            metrics["correlation_g1_g3"] = "N/A"

        if 'g2_grade' in df.columns:
            df['g2_grade'] = pd.to_numeric(df['g2_grade'], errors='coerce')
            if not df['g2_grade'].isnull().all():
                fig_g2_g3_scatter = px.scatter(df, x='g2_grade', y='final_grade_g3', title='G3 Final Grade vs. G2 Grade', trendline="ols")
                visualizations['g3_vs_g2_grade_scatter'] = fig_g2_g3_scatter.to_json()
                metrics["correlation_g2_g3"] = df[['g2_grade', 'final_grade_g3']].corr().iloc[0, 1]
                insights.append(f"Correlation between G2 and G3 grades: {metrics['correlation_g2_g3']:.2f}")
            else:
                insights.append("Skipping G2 grade analysis: column has no valid data.")
        else:
            insights.append("Skipping G2 grade analysis: column not found.")

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

def Extracurricular_Activities_and_Academic_Grade_Analysis(df):
    analysis_name = "Extracurricular Activities and Academic Grade Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'extra_curricular_activities', 'study_time_weekly_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by extracurriculars
        if 'extra_curricular_activities' in df.columns:
            avg_grade_by_extracurriculars = df.groupby('extra_curricular_activities')['final_grade'].mean().reset_index()
            fig_grade_by_extracurriculars = px.bar(avg_grade_by_extracurriculars, x='extra_curricular_activities', y='final_grade',
                                                   title='Average Final Grade by Extracurricular Activities Participation')
            visualizations['average_final_grade_by_extracurriculars'] = fig_grade_by_extracurriculars.to_json()
            metrics['avg_grade_by_extracurriculars'] = avg_grade_by_extracurriculars.to_dict('records')
            insights.append("Generated analysis by extracurricular activities.")
        else:
            insights.append("Skipping extracurricular activities analysis: column not found.")

        # Distribution of study time by extracurriculars
        if 'study_time_weekly_hours' in df.columns and 'extra_curricular_activities' in df.columns:
            df['study_time_weekly_hours'] = pd.to_numeric(df['study_time_weekly_hours'], errors='coerce')
            if not df['study_time_weekly_hours'].isnull().all():
                fig_study_time_extracurriculars = px.box(df, x='extra_curricular_activities', y='study_time_weekly_hours',
                                                         title='Weekly Study Time Distribution by Extracurricular Activities')
                visualizations['study_time_distribution_by_extracurriculars'] = fig_study_time_extracurriculars.to_json()
                insights.append("Generated study time distribution by extracurriculars.")
            else:
                insights.append("Skipping study time by extracurriculars analysis: 'study_time_weekly_hours' has no valid data.")
        else:
            insights.append("Skipping study time by extracurriculars analysis: missing 'study_time_weekly_hours' or 'extra_curricular_activities'.")

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

def Family_and_Internet_Support_on_Student_Final_Scores(df):
    analysis_name = "Family and Internet Support on Student Final Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'family_educational_support', 'internet_access_quality', 'parental_involvement_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final score by family educational support
        if 'family_educational_support' in df.columns:
            avg_score_by_family_support = df.groupby('family_educational_support')['final_score'].mean().reset_index()
            fig_score_by_family_support = px.bar(avg_score_by_family_support, x='family_educational_support', y='final_score',
                                                 title='Average Final Score by Family Educational Support')
            visualizations['average_final_score_by_family_educational_support'] = fig_score_by_family_support.to_json()
            metrics['avg_score_by_family_support'] = avg_score_by_family_support.to_dict('records')
            insights.append("Generated analysis by family educational support.")
        else:
            insights.append("Skipping family educational support analysis: column not found.")

        # Average final score by internet access quality
        if 'internet_access_quality' in df.columns:
            avg_score_by_internet_quality = df.groupby('internet_access_quality')['final_score'].mean().reset_index()
            fig_score_by_internet_quality = px.bar(avg_score_by_internet_quality, x='internet_access_quality', y='final_score',
                                                   title='Average Final Score by Internet Access Quality')
            visualizations['average_final_score_by_internet_access_quality'] = fig_score_by_internet_quality.to_json()
            metrics['avg_score_by_internet_quality'] = avg_score_by_internet_quality.to_dict('records')
            insights.append("Generated analysis by internet access quality.")
        else:
            insights.append("Skipping internet access quality analysis: column not found.")

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

def Parental_Background_and_Study_Time_on_Final_Grades(df):
    analysis_name = "Parental Background and Study Time on Final Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'mother_education_level', 'father_education_level', 'study_time_weekly_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by mother's education level
        if 'mother_education_level' in df.columns:
            avg_grade_by_mother_edu = df.groupby('mother_education_level')['final_grade'].mean().sort_values(ascending=False).reset_index()
            fig_grade_by_mother_edu = px.bar(avg_grade_by_mother_edu, x='mother_education_level', y='final_grade',
                                             title="Average Final Grade by Mother's Education Level")
            visualizations['average_final_grade_by_mother_education_level'] = fig_grade_by_mother_edu.to_json()
            metrics['avg_grade_by_mother_education'] = avg_grade_by_mother_edu.to_dict('records')
            insights.append("Generated analysis by mother's education level.")
        else:
            insights.append("Skipping mother's education level analysis: column not found.")

        # Correlation between study time and final grade, faceted by parental education
        if 'study_time_weekly_hours' in df.columns and 'mother_education_level' in df.columns:
            df['study_time_weekly_hours'] = pd.to_numeric(df['study_time_weekly_hours'], errors='coerce')
            if not df['study_time_weekly_hours'].isnull().all():
                fig_study_time_grade_faceted = px.scatter(df, x='study_time_weekly_hours', y='final_grade',
                                                          color='mother_education_level', title='Final Grade vs. Study Time by Mother\'s Education Level',
                                                          facet_col='mother_education_level', facet_col_wrap=2, trendline="ols")
                visualizations['final_grade_vs_study_time_faceted_by_mother_education'] = fig_study_time_grade_faceted.to_json()
                insights.append("Generated faceted scatter plot for study time vs. grade by mother's education.")
            else:
                insights.append("Skipping faceted scatter plot: 'study_time_weekly_hours' has no valid data.")
        else:
            insights.append("Skipping faceted scatter plot: missing 'study_time_weekly_hours' or 'mother_education_level'.")

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

def Family_Relationships_and_Student_Grade_Analysis(df):
    analysis_name = "Family Relationships and Student Grade Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'family_relationship_quality_score', 'parents_status_cohabiting_or_apart']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by family relationship quality score
        if 'family_relationship_quality_score' in df.columns:
            df['family_relationship_quality_score'] = pd.to_numeric(df['family_relationship_quality_score'], errors='coerce')
            if not df['family_relationship_quality_score'].isnull().all():
                avg_grade_by_family_relation = df.groupby('family_relationship_quality_score')['final_grade'].mean().reset_index()
                fig_grade_by_family_relation = px.bar(avg_grade_by_family_relation, x='family_relationship_quality_score', y='final_grade',
                                                      title='Average Final Grade by Family Relationship Quality Score')
                visualizations['average_final_grade_by_family_relationship_quality'] = fig_grade_by_family_relation.to_json()
                metrics['avg_grade_by_family_relationship'] = avg_grade_by_family_relation.to_dict('records')
                insights.append("Generated analysis by family relationship quality.")
            else:
                insights.append("Skipping family relationship analysis: column has no valid data.")
        else:
            insights.append("Skipping family relationship analysis: column not found.")

        # Average final grade by parents' status
        if 'parents_status_cohabiting_or_apart' in df.columns:
            avg_grade_by_parents_status = df.groupby('parents_status_cohabiting_or_apart')['final_grade'].mean().reset_index()
            fig_grade_by_parents_status = px.bar(avg_grade_by_parents_status, x='parents_status_cohabiting_or_apart', y='final_grade',
                                                 title='Average Final Grade by Parents Status')
            visualizations['average_final_grade_by_parents_status'] = fig_grade_by_parents_status.to_json()
            metrics['avg_grade_by_parents_status'] = avg_grade_by_parents_status.to_dict('records')
            insights.append("Generated analysis by parents' status.")
        else:
            insights.append("Skipping parents' status analysis: column not found.")

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

def Assessment_Scores_and_Attendance_Impact_on_Final_Grade(df):
    analysis_name = "Assessment Scores and Attendance Impact on Final Grade"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'midterm_score', 'quiz_average_score', 'attendance_rate_percent']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between midterm score and final grade
        if 'midterm_score' in df.columns:
            df['midterm_score'] = pd.to_numeric(df['midterm_score'], errors='coerce')
            if not df['midterm_score'].isnull().all():
                fig_midterm_final_grade = px.scatter(df, x='midterm_score', y='final_grade',
                                                     title='Final Grade vs. Midterm Score', trendline="ols")
                visualizations['final_grade_vs_midterm_score'] = fig_midterm_final_grade.to_json()
                metrics["correlation_midterm_final"] = df[['midterm_score', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between midterm score and final grade: {metrics['correlation_midterm_final']:.2f}")
            else:
                insights.append("Skipping midterm score analysis: column has no valid data.")
        else:
            insights.append("Skipping midterm score analysis: column not found.")
            metrics["correlation_midterm_final"] = "N/A"

        # Correlation between attendance rate and final grade
        if 'attendance_rate_percent' in df.columns:
            df['attendance_rate_percent'] = pd.to_numeric(df['attendance_rate_percent'], errors='coerce')
            if not df['attendance_rate_percent'].isnull().all():
                fig_attendance_final_grade = px.scatter(df, x='attendance_rate_percent', y='final_grade',
                                                        title='Final Grade vs. Attendance Rate (%)', trendline="ols")
                visualizations['final_grade_vs_attendance_rate'] = fig_attendance_final_grade.to_json()
                metrics["correlation_attendance_final"] = df[['attendance_rate_percent', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between attendance rate and final grade: {metrics['correlation_attendance_final']:.2f}")
            else:
                insights.append("Skipping attendance rate analysis: column has no valid data.")
        else:
            insights.append("Skipping attendance rate analysis: column not found.")

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

def Lifestyle_Factors_and_Their_Correlation_with_Student_GPA(df):
    analysis_name = "Lifestyle Factors and Their Correlation with Student GPA"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'gpa', 'sleep_hours_daily', 'physical_activity_hours_weekly', 'diet_quality_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        df.dropna(subset=['student_id', 'gpa'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_gpa"] = df['gpa'].mean()
        insights.append(f"Overall average GPA: {metrics['overall_avg_gpa']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between sleep hours and GPA
        if 'sleep_hours_daily' in df.columns:
            df['sleep_hours_daily'] = pd.to_numeric(df['sleep_hours_daily'], errors='coerce')
            if not df['sleep_hours_daily'].isnull().all():
                fig_sleep_gpa = px.scatter(df, x='sleep_hours_daily', y='gpa',
                                           title='GPA vs. Daily Sleep Hours', trendline="ols")
                visualizations['gpa_vs_daily_sleep_hours'] = fig_sleep_gpa.to_json()
                metrics["correlation_sleep_gpa"] = df[['sleep_hours_daily', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between sleep hours and GPA: {metrics['correlation_sleep_gpa']:.2f}")
            else:
                insights.append("Skipping sleep hours analysis: column has no valid data.")
        else:
            insights.append("Skipping sleep hours analysis: column not found.")
            metrics["correlation_sleep_gpa"] = "N/A"

        # Correlation between physical activity and GPA
        if 'physical_activity_hours_weekly' in df.columns:
            df['physical_activity_hours_weekly'] = pd.to_numeric(df['physical_activity_hours_weekly'], errors='coerce')
            if not df['physical_activity_hours_weekly'].isnull().all():
                fig_physical_activity_gpa = px.scatter(df, x='physical_activity_hours_weekly', y='gpa',
                                                       title='GPA vs. Weekly Physical Activity Hours', trendline="ols")
                visualizations['gpa_vs_weekly_physical_activity_hours'] = fig_physical_activity_gpa.to_json()
                metrics["correlation_activity_gpa"] = df[['physical_activity_hours_weekly', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between physical activity and GPA: {metrics['correlation_activity_gpa']:.2f}")
            else:
                insights.append("Skipping physical activity analysis: column has no valid data.")
        else:
            insights.append("Skipping physical activity analysis: column not found.")

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

def Educational_Support_Systems_Impact_on_Student_Grades(df):
    analysis_name = "Educational Support Systems' Impact on Student Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'school_support', 'extra_paid_classes', 'tutoring_provided_by_school']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by school support
        if 'school_support' in df.columns:
            avg_grade_by_school_support = df.groupby('school_support')['final_grade'].mean().reset_index()
            fig_grade_by_school_support = px.bar(avg_grade_by_school_support, x='school_support', y='final_grade',
                                                 title='Average Final Grade by School Support')
            visualizations['average_final_grade_by_school_support'] = fig_grade_by_school_support.to_json()
            metrics['avg_grade_by_school_support'] = avg_grade_by_school_support.to_dict('records')
            insights.append("Generated analysis by school support.")
        else:
            insights.append("Skipping school support analysis: column not found.")

        # Average final grade by extra paid classes
        if 'extra_paid_classes' in df.columns:
            avg_grade_by_paid_classes = df.groupby('extra_paid_classes')['final_grade'].mean().reset_index()
            fig_grade_by_paid_classes = px.bar(avg_grade_by_paid_classes, x='extra_paid_classes', y='final_grade',
                                               title='Average Final Grade by Extra Paid Classes')
            visualizations['average_final_grade_by_extra_paid_classes'] = fig_grade_by_paid_classes.to_json()
            metrics['avg_grade_by_extra_paid_classes'] = avg_grade_by_paid_classes.to_dict('records')
            insights.append("Generated analysis by extra paid classes.")
        else:
            insights.append("Skipping extra paid classes analysis: column not found.")

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

def Impact_of_Paid_Classes_and_School_Support_on_Performance(df):
    analysis_name = "Impact of Paid Classes and School Support on Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'overall_score', 'paid_classes_taken', 'school_provided_support']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
        df.dropna(subset=['student_id', 'overall_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_score"] = df['overall_score'].mean()
        insights.append(f"Overall average score: {metrics['overall_avg_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average overall score by paid classes taken
        if 'paid_classes_taken' in df.columns:
            avg_score_by_paid_classes = df.groupby('paid_classes_taken')['overall_score'].mean().reset_index()
            fig_score_by_paid_classes = px.bar(avg_score_by_paid_classes, x='paid_classes_taken', y='overall_score',
                                               title='Average Overall Score by Paid Classes Taken')
            visualizations['average_overall_score_by_paid_classes'] = fig_score_by_paid_classes.to_json()
            metrics['avg_score_by_paid_classes'] = avg_score_by_paid_classes.to_dict('records')
            insights.append("Generated analysis by paid classes taken.")
        else:
            insights.append("Skipping paid classes analysis: column not found.")

        # Average overall score by school provided support
        if 'school_provided_support' in df.columns:
            avg_score_by_school_support = df.groupby('school_provided_support')['overall_score'].mean().reset_index()
            fig_score_by_school_support = px.bar(avg_score_by_school_support, x='school_provided_support', y='overall_score',
                                                 title='Average Overall Score by School Provided Support')
            visualizations['average_overall_score_by_school_support'] = fig_score_by_school_support.to_json()
            metrics['avg_score_by_school_support'] = avg_score_by_school_support.to_dict('records')
            insights.append("Generated analysis by school provided support.")
        else:
            insights.append("Skipping school provided support analysis: column not found.")

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

def Student_Performance_and_Pass_Status_Prediction(df):
    analysis_name = "Student Performance and Pass Status Prediction"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'pass_status', 'g1_score', 'g2_score', 'study_time_weekly']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade', 'pass_status'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = df['student_id'].nunique()
        
        # Pass status distribution
        pass_status_counts = df['pass_status'].value_counts(normalize=True).reset_index()
        pass_status_counts.columns = ['status', 'proportion']
        
        pass_rate = pass_status_counts[pass_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100
        metrics["overall_pass_rate_percent"] = pass_rate if pass_rate else 0
        metrics["pass_fail_distribution"] = pass_status_counts.to_dict('records')

        insights.append(f"Analyzed {metrics['total_students']} students.")
        insights.append(f"Overall pass rate: {metrics['overall_pass_rate_percent']:.2f}%")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by pass status
        avg_grade_by_pass_status = df.groupby('pass_status')['final_grade'].mean().reset_index()
        fig_avg_grade_by_pass_status = px.bar(avg_grade_by_pass_status, x='pass_status', y='final_grade',
                                              title='Average Final Grade by Pass Status')
        visualizations['average_final_grade_by_pass_status'] = fig_avg_grade_by_pass_status.to_json()
        metrics['avg_grade_by_pass_status'] = avg_grade_by_pass_status.to_dict('records')

        fig_pass_status_pie = px.pie(pass_status_counts, names='status', values='proportion', title='Student Pass Status Distribution')
        visualizations['pass_status_distribution'] = fig_pass_status_pie.to_json()

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

def Health_Absences_and_Travel_Time_on_Final_Grades_G3(df):
    analysis_name = "Health, Absences, and Travel Time on Final Grades (G3)"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'g3_final_grade', 'health_status', 'absences', 'travel_time_to_school_minutes']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['g3_final_grade'] = pd.to_numeric(df['g3_final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'g3_final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_g3_grade"] = df['g3_final_grade'].mean()
        insights.append(f"Overall average G3 grade: {metrics['overall_avg_g3_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average G3 grade by health status
        if 'health_status' in df.columns:
            avg_grade_by_health = df.groupby('health_status')['g3_final_grade'].mean().reset_index()
            fig_grade_by_health = px.bar(avg_grade_by_health, x='health_status', y='g3_final_grade',
                                         title='Average G3 Final Grade by Health Status')
            visualizations['average_g3_grade_by_health_status'] = fig_grade_by_health.to_json()
            metrics['avg_g3_grade_by_health_status'] = avg_grade_by_health.to_dict('records')
            insights.append("Generated analysis by health status.")
        else:
            insights.append("Skipping health status analysis: column not found.")

        # Correlation between absences and G3 grade
        if 'absences' in df.columns:
            df['absences'] = pd.to_numeric(df['absences'], errors='coerce')
            if not df['absences'].isnull().all():
                fig_absences_grade = px.scatter(df, x='absences', y='g3_final_grade',
                                                title='G3 Final Grade vs. Absences', trendline="ols")
                visualizations['g3_final_grade_vs_absences'] = fig_absences_grade.to_json()
                metrics["correlation_absences_g3"] = df[['absences', 'g3_final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between absences and G3 grade: {metrics['correlation_absences_g3']:.2f}")
            else:
                insights.append("Skipping absences analysis: column has no valid data.")
        else:
            insights.append("Skipping absences analysis: column not found.")
            metrics["correlation_absences_g3"] = "N/A"

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

def Impact_of_Extra_Paid_Classes_and_School_Support_on_Grades(df):
    analysis_name = "Impact of Extra Paid Classes and School Support on Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'extra_paid_classes', 'school_support_services']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by extra paid classes
        if 'extra_paid_classes' in df.columns:
            avg_grade_by_paid_classes = df.groupby('extra_paid_classes')['final_grade'].mean().reset_index()
            fig_grade_by_paid_classes = px.bar(avg_grade_by_paid_classes, x='extra_paid_classes', y='final_grade',
                                               title='Average Final Grade by Extra Paid Classes')
            visualizations['average_final_grade_by_extra_paid_classes'] = fig_grade_by_paid_classes.to_json()
            metrics['avg_grade_by_extra_paid_classes'] = avg_grade_by_paid_classes.to_dict('records')
            insights.append("Generated analysis by extra paid classes.")
        else:
            insights.append("Skipping extra paid classes analysis: column not found.")

        # Average final grade by school support services
        if 'school_support_services' in df.columns:
            avg_grade_by_school_support = df.groupby('school_support_services')['final_grade'].mean().reset_index()
            fig_grade_by_school_support = px.bar(avg_grade_by_school_support, x='school_support_services', y='final_grade',
                                                 title='Average Final Grade by School Support Services')
            visualizations['average_final_grade_by_school_support_services'] = fig_grade_by_school_support.to_json()
            metrics['avg_grade_by_school_support'] = avg_grade_by_school_support.to_dict('records')
            insights.append("Generated analysis by school support services.")
        else:
            insights.append("Skipping school support services analysis: column not found.")

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

def Social_and_Health_Factors_Affecting_Student_Scores(df):
    analysis_name = "Social and Health Factors Affecting Student Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'social_activities_participation', 'health_condition_status', 'absences']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final score by social activities participation
        if 'social_activities_participation' in df.columns:
            avg_score_by_social = df.groupby('social_activities_participation')['final_score'].mean().reset_index()
            fig_score_by_social = px.bar(avg_score_by_social, x='social_activities_participation', y='final_score',
                                         title='Average Final Score by Social Activities Participation')
            visualizations['average_final_score_by_social_activities'] = fig_score_by_social.to_json()
            metrics['avg_score_by_social_activities'] = avg_score_by_social.to_dict('records')
            insights.append("Generated analysis by social activities participation.")
        else:
            insights.append("Skipping social activities analysis: column not found.")

        # Average final score by health condition status
        if 'health_condition_status' in df.columns:
            avg_score_by_health = df.groupby('health_condition_status')['final_score'].mean().reset_index()
            fig_score_by_health = px.bar(avg_score_by_health, x='health_condition_status', y='final_score',
                                         title='Average Final Score by Health Condition Status')
            visualizations['average_final_score_by_health_condition_status'] = fig_score_by_health.to_json()
            metrics['avg_score_by_health_status'] = avg_score_by_health.to_dict('records')
            insights.append("Generated analysis by health condition status.")
        else:
            insights.append("Skipping health condition status analysis: column not found.")

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

def Physical_Attributes_and_Commutes_Effect_on_Grades(df):
    analysis_name = "Physical Attributes and Commute's Effect on Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'travel_time_to_school_minutes', 'physical_attributes_score', 'gym_attendance_weekly']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between travel time and final grade
        if 'travel_time_to_school_minutes' in df.columns:
            df['travel_time_to_school_minutes'] = pd.to_numeric(df['travel_time_to_school_minutes'], errors='coerce')
            if not df['travel_time_to_school_minutes'].isnull().all():
                fig_travel_time_grade = px.scatter(df, x='travel_time_to_school_minutes', y='final_grade',
                                                   title='Final Grade vs. Travel Time to School (Minutes)', trendline="ols")
                visualizations['final_grade_vs_travel_time'] = fig_travel_time_grade.to_json()
                metrics["correlation_travel_time_grade"] = df[['travel_time_to_school_minutes', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between travel time and final grade: {metrics['correlation_travel_time_grade']:.2f}")
            else:
                insights.append("Skipping travel time analysis: column has no valid data.")
        else:
            insights.append("Skipping travel time analysis: column not found.")
            metrics["correlation_travel_time_grade"] = "N/A"

        # Correlation between physical attributes score and final grade
        if 'physical_attributes_score' in df.columns:
            df['physical_attributes_score'] = pd.to_numeric(df['physical_attributes_score'], errors='coerce')
            if not df['physical_attributes_score'].isnull().all():
                fig_physical_attr_grade = px.scatter(df, x='physical_attributes_score', y='final_grade',
                                                     title='Final Grade vs. Physical Attributes Score', trendline="ols")
                visualizations['final_grade_vs_physical_attributes_score'] = fig_physical_attr_grade.to_json()
                metrics["correlation_physical_score_grade"] = df[['physical_attributes_score', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between physical attribute score and final grade: {metrics['correlation_physical_score_grade']:.2f}")
            else:
                insights.append("Skipping physical attributes score analysis: column has no valid data.")
        else:
            insights.append("Skipping physical attributes score analysis: column not found.")

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

def Student_Performance_and_Pass_Fail_Classification(df):
    analysis_name = "Student Performance and Pass/Fail Classification"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'pass_fail_status', 'exam_score_midterm', 'exam_score_final']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score', 'pass_fail_status'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = len(df)
        
        # Pass/Fail status distribution
        pass_fail_status_counts = df['pass_fail_status'].value_counts(normalize=True).reset_index()
        pass_fail_status_counts.columns = ['status', 'proportion']
        
        pass_rate = pass_fail_status_counts[pass_fail_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100
        metrics["pass_rate_percent"] = pass_rate if pass_rate else 0
        metrics["pass_fail_distribution"] = pass_fail_status_counts.to_dict('records')

        insights.append(f"Analyzed {metrics['total_students']} students.")
        insights.append(f"Overall pass rate: {metrics['pass_rate_percent']:.2f}%")

        # --- Visualizations & Deeper Metrics ---

        # Average final score for Pass vs. Fail students
        avg_score_by_status = df.groupby('pass_fail_status')['final_score'].mean().reset_index()
        fig_avg_score_by_status = px.bar(avg_score_by_status, x='pass_fail_status', y='final_score',
                                         title='Average Final Score for Pass vs. Fail Students')
        visualizations['average_final_score_by_pass_fail_status'] = fig_avg_score_by_status.to_json()
        metrics['avg_final_score_by_pass_fail_status'] = avg_score_by_status.to_dict('records')

        fig_pass_fail_pie = px.pie(pass_fail_status_counts, names='status', values='proportion', title='Student Pass/Fail Status Distribution')
        visualizations['pass_fail_status_distribution'] = fig_pass_fail_pie.to_json()

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

def Digital_Engagement_and_Parental_Support_on_Student_GPA(df):
    analysis_name = "Digital Engagement and Parental Support on Student GPA"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'gpa', 'digital_engagement_score', 'parental_support_score', 'internet_access_quality']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        df.dropna(subset=['student_id', 'gpa'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_gpa"] = df['gpa'].mean()
        insights.append(f"Overall average GPA: {metrics['overall_avg_gpa']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between digital engagement score and GPA
        if 'digital_engagement_score' in df.columns:
            df['digital_engagement_score'] = pd.to_numeric(df['digital_engagement_score'], errors='coerce')
            if not df['digital_engagement_score'].isnull().all():
                fig_digital_engagement_gpa = px.scatter(df, x='digital_engagement_score', y='gpa',
                                                        title='GPA vs. Digital Engagement Score', trendline="ols")
                visualizations['gpa_vs_digital_engagement_score'] = fig_digital_engagement_gpa.to_json()
                metrics["correlation_digital_engagement_gpa"] = df[['digital_engagement_score', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between digital engagement and GPA: {metrics['correlation_digital_engagement_gpa']:.2f}")
            else:
                insights.append("Skipping digital engagement analysis: column has no valid data.")
        else:
            insights.append("Skipping digital engagement analysis: column not found.")
            metrics["correlation_digital_engagement_gpa"] = "N/A"

        # Correlation between parental support score and GPA
        if 'parental_support_score' in df.columns:
            df['parental_support_score'] = pd.to_numeric(df['parental_support_score'], errors='coerce')
            if not df['parental_support_score'].isnull().all():
                fig_parental_support_gpa = px.scatter(df, x='parental_support_score', y='gpa',
                                                      title='GPA vs. Parental Support Score', trendline="ols")
                visualizations['gpa_vs_parental_support_score'] = fig_parental_support_gpa.to_json()
                metrics["correlation_parental_support_gpa"] = df[['parental_support_score', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between parental support and GPA: {metrics['correlation_parental_support_gpa']:.2f}")
            else:
                insights.append("Skipping parental support analysis: column has no valid data.")
        else:
            insights.append("Skipping parental support analysis: column not found.")

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

def Daily_Habits_and_Their_Influence_on_Student_Grades(df):
    analysis_name = "Daily Habits and Their Influence on Student Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'daily_study_hours', 'daily_sleep_hours', 'daily_screen_time_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between daily study hours and final grade
        if 'daily_study_hours' in df.columns:
            df['daily_study_hours'] = pd.to_numeric(df['daily_study_hours'], errors='coerce')
            if not df['daily_study_hours'].isnull().all():
                fig_study_hours_grade = px.scatter(df, x='daily_study_hours', y='final_grade',
                                                   title='Final Grade vs. Daily Study Hours', trendline="ols")
                visualizations['final_grade_vs_daily_study_hours'] = fig_study_hours_grade.to_json()
                metrics["correlation_daily_study_grade"] = df[['daily_study_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between daily study hours and final grade: {metrics['correlation_daily_study_grade']:.2f}")
            else:
                insights.append("Skipping daily study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping daily study hours analysis: column not found.")
            metrics["correlation_daily_study_grade"] = "N/A"

        # Correlation between daily sleep hours and final grade
        if 'daily_sleep_hours' in df.columns:
            df['daily_sleep_hours'] = pd.to_numeric(df['daily_sleep_hours'], errors='coerce')
            if not df['daily_sleep_hours'].isnull().all():
                fig_sleep_hours_grade = px.scatter(df, x='daily_sleep_hours', y='final_grade',
                                                   title='Final Grade vs. Daily Sleep Hours', trendline="ols")
                visualizations['final_grade_vs_daily_sleep_hours'] = fig_sleep_hours_grade.to_json()
                metrics["correlation_daily_sleep_grade"] = df[['daily_sleep_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between daily sleep hours and final grade: {metrics['correlation_daily_sleep_grade']:.2f}")
            else:
                insights.append("Skipping daily sleep hours analysis: column has no valid data.")
        else:
            insights.append("Skipping daily sleep hours analysis: column not found.")

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

def Demographic_Factors_and_Test_Preparation_on_Student_Scores(df):
    analysis_name = "Demographic Factors and Test Preparation on Student Scores"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'overall_score', 'gender', 'race_ethnicity', 'parental_education_level', 'test_preparation_completed']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
        df.dropna(subset=['student_id', 'overall_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_score"] = df['overall_score'].mean()
        insights.append(f"Overall average score: {metrics['overall_avg_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average overall score by gender
        if 'gender' in df.columns:
            avg_score_by_gender = df.groupby('gender')['overall_score'].mean().reset_index()
            fig_score_by_gender = px.bar(avg_score_by_gender, x='gender', y='overall_score',
                                         title='Average Overall Score by Gender')
            visualizations['average_overall_score_by_gender'] = fig_score_by_gender.to_json()
            metrics['avg_score_by_gender'] = avg_score_by_gender.to_dict('records')
            insights.append("Generated analysis by gender.")
        else:
            insights.append("Skipping gender analysis: column not found.")

        # Average overall score by test preparation completed
        if 'test_preparation_completed' in df.columns:
            avg_score_by_prep = df.groupby('test_preparation_completed')['overall_score'].mean().reset_index()
            fig_score_by_prep = px.bar(avg_score_by_prep, x='test_preparation_completed', y='overall_score',
                                       title='Average Overall Score by Test Preparation Completed')
            visualizations['average_overall_score_by_test_preparation_completed'] = fig_score_by_prep.to_json()
            metrics['avg_score_by_test_preparation'] = avg_score_by_prep.to_dict('records')
            insights.append("Generated analysis by test preparation status.")
        else:
            insights.append("Skipping test preparation analysis: column not found.")

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

def Study_Time_and_Absences_on_Final_Academic_Performance(df):
    analysis_name = "Study Time and Absences on Final Academic Performance"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_academic_performance', 'study_time_hours_per_week', 'number_of_absences']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_academic_performance'] = pd.to_numeric(df['final_academic_performance'], errors='coerce')
        df.dropna(subset=['student_id', 'final_academic_performance'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_performance"] = df['final_academic_performance'].mean()
        insights.append(f"Overall average final performance: {metrics['overall_avg_performance']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between study time and final academic performance
        if 'study_time_hours_per_week' in df.columns:
            df['study_time_hours_per_week'] = pd.to_numeric(df['study_time_hours_per_week'], errors='coerce')
            if not df['study_time_hours_per_week'].isnull().all():
                fig_study_time_performance = px.scatter(df, x='study_time_hours_per_week', y='final_academic_performance',
                                                        title='Final Academic Performance vs. Weekly Study Time', trendline="ols")
                visualizations['final_academic_performance_vs_study_time'] = fig_study_time_performance.to_json()
                metrics["correlation_study_time_performance"] = df[['study_time_hours_per_week', 'final_academic_performance']].corr().iloc[0, 1]
                insights.append(f"Correlation between study time and performance: {metrics['correlation_study_time_performance']:.2f}")
            else:
                insights.append("Skipping study time analysis: column has no valid data.")
        else:
            insights.append("Skipping study time analysis: column not found.")
            metrics["correlation_study_time_performance"] = "N/A"

        # Correlation between number of absences and final academic performance
        if 'number_of_absences' in df.columns:
            df['number_of_absences'] = pd.to_numeric(df['number_of_absences'], errors='coerce')
            if not df['number_of_absences'].isnull().all():
                fig_absences_performance = px.scatter(df, x='number_of_absences', y='final_academic_performance',
                                                      title='Final Academic Performance vs. Number of Absences', trendline="ols")
                visualizations['final_academic_performance_vs_absences'] = fig_absences_performance.to_json()
                metrics["correlation_absences_performance"] = df[['number_of_absences', 'final_academic_performance']].corr().iloc[0, 1]
                insights.append(f"Correlation between absences and performance: {metrics['correlation_absences_performance']:.2f}")
            else:
                insights.append("Skipping absences analysis: column has no valid data.")
        else:
            insights.append("Skipping absences analysis: column not found.")

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

def Social_Activities_and_Health_on_Final_Student_Grades_G3(df):
    analysis_name = "Social Activities and Health on Final Student Grades (G3)"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'g3_final_grade', 'social_activities', 'health_status_rating', 'absences']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['g3_final_grade'] = pd.to_numeric(df['g3_final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'g3_final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_g3_grade"] = df['g3_final_grade'].mean()
        insights.append(f"Overall average G3 grade: {metrics['overall_avg_g3_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average G3 grade by social activities
        if 'social_activities' in df.columns:
            avg_grade_by_social = df.groupby('social_activities')['g3_final_grade'].mean().reset_index()
            fig_grade_by_social = px.bar(avg_grade_by_social, x='social_activities', y='g3_final_grade',
                                         title='Average G3 Final Grade by Social Activities Participation')
            visualizations['average_g3_grade_by_social_activities'] = fig_grade_by_social.to_json()
            metrics['avg_g3_grade_by_social_activities'] = avg_grade_by_social.to_dict('records')
            insights.append("Generated analysis by social activities.")
        else:
            insights.append("Skipping social activities analysis: column not found.")

        # Average G3 grade by health status rating
        if 'health_status_rating' in df.columns:
            avg_grade_by_health = df.groupby('health_status_rating')['g3_final_grade'].mean().reset_index()
            fig_grade_by_health = px.bar(avg_grade_by_health, x='health_status_rating', y='g3_final_grade',
                                         title='Average G3 Final Grade by Health Status Rating')
            visualizations['average_g3_grade_by_health_status_rating'] = fig_grade_by_health.to_json()
            metrics['avg_g3_grade_by_health_status_rating'] = avg_grade_by_health.to_dict('records')
            insights.append("Generated analysis by health status rating.")
        else:
            insights.append("Skipping health status rating analysis: column not found.")

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

def Longitudinal_Academic_Performance_Analysis_G1_G2_G3(df):
    analysis_name = "Longitudinal Academic Performance Analysis (G1, G2, G3)"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'g1_grade', 'g2_grade', 'g3_grade']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df.dropna(subset=['student_id'], inplace=True)

        metrics["total_students"] = df['student_id'].nunique()
        insights.append(f"Analyzed {metrics['total_students']} students.")

        # --- Metrics & Visualizations ---
        grades_data = {}
        if 'g1_grade' in df.columns: 
            df['g1_grade'] = pd.to_numeric(df['g1_grade'], errors='coerce')
            if not df['g1_grade'].isnull().all():
                grades_data['G1'] = df['g1_grade'].mean()
        if 'g2_grade' in df.columns: 
            df['g2_grade'] = pd.to_numeric(df['g2_grade'], errors='coerce')
            if not df['g2_grade'].isnull().all():
                grades_data['G2'] = df['g2_grade'].mean()
        if 'g3_grade' in df.columns: 
            df['g3_grade'] = pd.to_numeric(df['g3_grade'], errors='coerce')
            if not df['g3_grade'].isnull().all():
                grades_data['G3'] = df['g3_grade'].mean()

        if grades_data:
            avg_grades_df = pd.DataFrame(grades_data.items(), columns=['Grade_Period', 'Average_Grade'])
            fig_avg_grades_over_time = px.line(avg_grades_df, x='Grade_Period', y='Average_Grade',
                                               title='Average Grades Across G1, G2, and G3')
            visualizations['average_grades_over_time'] = fig_avg_grades_over_time.to_json()
            metrics['average_grades_over_time'] = avg_grades_df.to_dict('records')
            insights.append("Generated longitudinal analysis of average grades (G1, G2, G3).")
        else:
            insights.append("No valid grade columns (g1_grade, g2_grade, g3_grade) found or all were empty.")

        # Distribution of final (G3) grades
        if 'g3_grade' in df.columns and not df['g3_grade'].isnull().all():
            fig_g3_distribution = px.histogram(df, x='g3_grade', nbins=20, title='Distribution of Final (G3) Grades')
            visualizations['g3_grade_distribution'] = fig_g3_distribution.to_json()
            metrics["overall_avg_g3_grade"] = df['g3_grade'].mean()
            insights.append(f"Overall average G3 grade: {metrics['overall_avg_g3_grade']:.2f}")
        else:
            insights.append("Skipping G3 distribution: column not found or empty.")
            metrics["overall_avg_g3_grade"] = "N/A"

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

def Student_Performance_Analysis_based_on_Demographics(df):
    analysis_name = "Student Performance Analysis based on Demographics"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'overall_grade', 'gender', 'race_ethnicity', 'parental_income_level']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['overall_grade'] = pd.to_numeric(df['overall_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'overall_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_grade"] = df['overall_grade'].mean()
        insights.append(f"Overall average grade: {metrics['overall_avg_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average overall grade by gender
        if 'gender' in df.columns:
            avg_grade_by_gender = df.groupby('gender')['overall_grade'].mean().reset_index()
            fig_grade_by_gender = px.bar(avg_grade_by_gender, x='gender', y='overall_grade',
                                         title='Average Overall Grade by Gender')
            visualizations['average_overall_grade_by_gender'] = fig_grade_by_gender.to_json()
            metrics['avg_grade_by_gender'] = avg_grade_by_gender.to_dict('records')
            insights.append("Generated analysis by gender.")
        else:
            insights.append("Skipping gender analysis: column not found.")

        # Average overall grade by race/ethnicity
        if 'race_ethnicity' in df.columns:
            avg_grade_by_race = df.groupby('race_ethnicity')['overall_grade'].mean().sort_values(ascending=False).reset_index()
            fig_grade_by_race = px.bar(avg_grade_by_race, x='race_ethnicity', y='overall_grade',
                                       title='Average Overall Grade by Race/Ethnicity')
            visualizations['average_overall_grade_by_race_ethnicity'] = fig_grade_by_race.to_json()
            metrics['avg_grade_by_race_ethnicity'] = avg_grade_by_race.to_dict('records')
            insights.append("Generated analysis by race/ethnicity.")
        else:
            insights.append("Skipping race/ethnicity analysis: column not found.")

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

def Student_Performance_Category_Prediction_Analysis(df):
    analysis_name = "Student Performance Category Prediction Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'overall_score', 'performance_category', 'study_hours_weekly', 'past_academic_record']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
        df.dropna(subset=['student_id', 'overall_score', 'performance_category'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = len(df)
        metrics["overall_avg_score"] = df['overall_score'].mean()
        
        # Performance category distribution
        performance_category_counts = df['performance_category'].value_counts(normalize=True).reset_index()
        performance_category_counts.columns = ['category', 'proportion']
        metrics["performance_category_distribution"] = performance_category_counts.to_dict('records')

        insights.append(f"Analyzed {metrics['total_students']} students with an average score of {metrics['overall_avg_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average overall score by performance category
        avg_score_by_category = df.groupby('performance_category')['overall_score'].mean().reset_index()
        fig_avg_score_by_category = px.bar(avg_score_by_category, x='performance_category', y='overall_score',
                                           title='Average Overall Score by Performance Category')
        visualizations['average_overall_score_by_performance_category'] = fig_avg_score_by_category.to_json()
        metrics['avg_score_by_performance_category'] = avg_score_by_category.to_dict('records')

        fig_performance_category_pie = px.pie(performance_category_counts, names='category', values='proportion', title='Student Performance Category Distribution')
        visualizations['performance_category_distribution'] = fig_performance_category_pie.to_json()

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

def Ethnicity_and_Parental_Education_s_Role_in_Student_Grades(df):
    analysis_name = "Ethnicity and Parental Education's Role in Student Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'ethnicity', 'mother_education_level', 'father_education_level']
        matched = fuzzy_match_column(df, expected)
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by ethnicity
        if 'ethnicity' in df.columns:
            avg_grade_by_ethnicity = df.groupby('ethnicity')['final_grade'].mean().sort_values(ascending=False).reset_index()
            fig_grade_by_ethnicity = px.bar(avg_grade_by_ethnicity, x='ethnicity', y='final_grade',
                                            title='Average Final Grade by Ethnicity')
            visualizations['average_final_grade_by_ethnicity'] = fig_grade_by_ethnicity.to_json()
            metrics['avg_grade_by_ethnicity'] = avg_grade_by_ethnicity.to_dict('records')
            insights.append("Generated analysis by ethnicity.")
        else:
            insights.append("Skipping ethnicity analysis: column not found.")

        # Average final grade by mother's education level
        if 'mother_education_level' in df.columns:
            avg_grade_by_mother_edu = df.groupby('mother_education_level')['final_grade'].mean().sort_values(ascending=False).reset_index()
            fig_grade_by_mother_edu = px.bar(avg_grade_by_mother_edu, x='mother_education_level', y='final_grade',
                                             title="Average Final Grade by Mother's Education Level")
            visualizations['average_final_grade_by_mother_education_level'] = fig_grade_by_mother_edu.to_json()
            metrics['avg_grade_by_mother_education'] = avg_grade_by_mother_edu.to_dict('records')
            insights.append("Generated analysis by mother's education level.")
        else:
            insights.append("Skipping mother's education level analysis: column not found.")

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

def Behavioral_and_Engagement_Scores_on_Academic_Outcomes(df):
    analysis_name = "Behavioral and Engagement Scores on Academic Outcomes"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'academic_outcome_score', 'behavioral_score', 'engagement_score', 'attendance_rate']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['academic_outcome_score'] = pd.to_numeric(df['academic_outcome_score'], errors='coerce')
        df.dropna(subset=['student_id', 'academic_outcome_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_academic_outcome"] = df['academic_outcome_score'].mean()
        insights.append(f"Overall average academic outcome score: {metrics['overall_avg_academic_outcome']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between behavioral score and academic outcome score
        if 'behavioral_score' in df.columns:
            df['behavioral_score'] = pd.to_numeric(df['behavioral_score'], errors='coerce')
            if not df['behavioral_score'].isnull().all():
                fig_behavioral_academic = px.scatter(df, x='behavioral_score', y='academic_outcome_score',
                                                     title='Academic Outcome Score vs. Behavioral Score', trendline="ols")
                visualizations['academic_outcome_vs_behavioral_score'] = fig_behavioral_academic.to_json()
                metrics["correlation_behavioral_academic"] = df[['behavioral_score', 'academic_outcome_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between behavioral score and academic outcome: {metrics['correlation_behavioral_academic']:.2f}")
            else:
                insights.append("Skipping behavioral score analysis: column has no valid data.")
        else:
            insights.append("Skipping behavioral score analysis: column not found.")
            metrics["correlation_behavioral_academic"] = "N/A"

        # Correlation between engagement score and academic outcome score
        if 'engagement_score' in df.columns:
            df['engagement_score'] = pd.to_numeric(df['engagement_score'], errors='coerce')
            if not df['engagement_score'].isnull().all():
                fig_engagement_academic = px.scatter(df, x='engagement_score', y='academic_outcome_score',
                                                     title='Academic Outcome Score vs. Engagement Score', trendline="ols")
                visualizations['academic_outcome_vs_engagement_score'] = fig_engagement_academic.to_json()
                metrics["correlation_engagement_academic"] = df[['engagement_score', 'academic_outcome_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between engagement score and academic outcome: {metrics['correlation_engagement_academic']:.2f}")
            else:
                insights.append("Skipping engagement score analysis: column has no valid data.")
        else:
            insights.append("Skipping engagement score analysis: column not found.")

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

def Continuous_Assessment_and_Study_Time_on_Final_Grade(df):
    analysis_name = "Continuous Assessment and Study Time on Final Grade"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'continuous_assessment_average', 'weekly_study_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between continuous assessment average and final grade
        if 'continuous_assessment_average' in df.columns:
            df['continuous_assessment_average'] = pd.to_numeric(df['continuous_assessment_average'], errors='coerce')
            if not df['continuous_assessment_average'].isnull().all():
                fig_continuous_assessment_grade = px.scatter(df, x='continuous_assessment_average', y='final_grade',
                                                             title='Final Grade vs. Continuous Assessment Average', trendline="ols")
                visualizations['final_grade_vs_continuous_assessment_average'] = fig_continuous_assessment_grade.to_json()
                metrics["correlation_continuous_assessment_grade"] = df[['continuous_assessment_average', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between continuous assessment and final grade: {metrics['correlation_continuous_assessment_grade']:.2f}")
            else:
                insights.append("Skipping continuous assessment analysis: column has no valid data.")
        else:
            insights.append("Skipping continuous assessment analysis: column not found.")
            metrics["correlation_continuous_assessment_grade"] = "N/A"

        # Correlation between weekly study hours and final grade
        if 'weekly_study_hours' in df.columns:
            df['weekly_study_hours'] = pd.to_numeric(df['weekly_study_hours'], errors='coerce')
            if not df['weekly_study_hours'].isnull().all():
                fig_weekly_study_grade = px.scatter(df, x='weekly_study_hours', y='final_grade',
                                                    title='Final Grade vs. Weekly Study Hours', trendline="ols")
                visualizations['final_grade_vs_weekly_study_hours'] = fig_weekly_study_grade.to_json()
                metrics["correlation_weekly_study_grade"] = df[['weekly_study_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between weekly study hours and final grade: {metrics['correlation_weekly_study_grade']:.2f}")
            else:
                insights.append("Skipping weekly study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping weekly study hours analysis: column not found.")

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

def Screen_Time_and_Sleep_s_Impact_on_Student_Anxiety_and_Grades(df):
    analysis_name = "Screen Time and Sleep's Impact on Student Anxiety and Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'daily_screen_time_hours', 'daily_sleep_hours', 'anxiety_level_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between daily screen time and final grade
        if 'daily_screen_time_hours' in df.columns:
            df['daily_screen_time_hours'] = pd.to_numeric(df['daily_screen_time_hours'], errors='coerce')
            if not df['daily_screen_time_hours'].isnull().all():
                fig_screen_time_grade = px.scatter(df, x='daily_screen_time_hours', y='final_grade',
                                                   title='Final Grade vs. Daily Screen Time Hours', trendline="ols")
                visualizations['final_grade_vs_daily_screen_time'] = fig_screen_time_grade.to_json()
                metrics["correlation_screen_time_grade"] = df[['daily_screen_time_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between screen time and final grade: {metrics['correlation_screen_time_grade']:.2f}")
            else:
                insights.append("Skipping screen time analysis: column has no valid data.")
        else:
            insights.append("Skipping screen time analysis: column not found.")
            metrics["correlation_screen_time_grade"] = "N/A"

        # Correlation between daily sleep hours and anxiety level score
        if 'daily_sleep_hours' in df.columns and 'anxiety_level_score' in df.columns:
            df['daily_sleep_hours'] = pd.to_numeric(df['daily_sleep_hours'], errors='coerce')
            df['anxiety_level_score'] = pd.to_numeric(df['anxiety_level_score'], errors='coerce')
            if not df['daily_sleep_hours'].isnull().all() and not df['anxiety_level_score'].isnull().all():
                fig_sleep_anxiety = px.scatter(df, x='daily_sleep_hours', y='anxiety_level_score',
                                               title='Anxiety Level Score vs. Daily Sleep Hours', trendline="ols")
                visualizations['anxiety_level_vs_daily_sleep_hours'] = fig_sleep_anxiety.to_json()
                metrics["correlation_sleep_anxiety"] = df[['daily_sleep_hours', 'anxiety_level_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between sleep hours and anxiety: {metrics['correlation_sleep_anxiety']:.2f}")
            else:
                insights.append("Skipping sleep/anxiety analysis: columns have no valid data.")
        else:
            insights.append("Skipping sleep/anxiety analysis: missing 'daily_sleep_hours' or 'anxiety_level_score'.")

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

def Midterm_Performance_and_Engagement_as_Predictors_of_Final_Grades(df):
    analysis_name = "Midterm Performance and Engagement as Predictors of Final Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'midterm_exam_score', 'engagement_score_lms', 'attendance_rate']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between midterm exam score and final grade
        if 'midterm_exam_score' in df.columns:
            df['midterm_exam_score'] = pd.to_numeric(df['midterm_exam_score'], errors='coerce')
            if not df['midterm_exam_score'].isnull().all():
                fig_midterm_final = px.scatter(df, x='midterm_exam_score', y='final_grade',
                                               title='Final Grade vs. Midterm Exam Score', trendline="ols")
                visualizations['final_grade_vs_midterm_exam_score'] = fig_midterm_final.to_json()
                metrics["correlation_midterm_final"] = df[['midterm_exam_score', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between midterm score and final grade: {metrics['correlation_midterm_final']:.2f}")
            else:
                insights.append("Skipping midterm score analysis: column has no valid data.")
        else:
            insights.append("Skipping midterm score analysis: column not found.")
            metrics["correlation_midterm_final"] = "N/A"

        # Correlation between engagement score (LMS) and final grade
        if 'engagement_score_lms' in df.columns:
            df['engagement_score_lms'] = pd.to_numeric(df['engagement_score_lms'], errors='coerce')
            if not df['engagement_score_lms'].isnull().all():
                fig_engagement_final = px.scatter(df, x='engagement_score_lms', y='final_grade',
                                                  title='Final Grade vs. LMS Engagement Score', trendline="ols")
                visualizations['final_grade_vs_lms_engagement_score'] = fig_engagement_final.to_json()
                metrics["correlation_engagement_final"] = df[['engagement_score_lms', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between LMS engagement and final grade: {metrics['correlation_engagement_final']:.2f}")
            else:
                insights.append("Skipping LMS engagement analysis: column has no valid data.")
        else:
            insights.append("Skipping LMS engagement analysis: column not found.")

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

def Socioeconomic_Status_and_Its_Effect_on_Student_GPA(df):
    analysis_name = "Socioeconomic Status and Its Effect on Student GPA"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'gpa', 'parental_income_level', 'parental_education_level', 'free_reduced_lunch_status']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        df.dropna(subset=['student_id', 'gpa'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_gpa"] = df['gpa'].mean()
        insights.append(f"Overall average GPA: {metrics['overall_avg_gpa']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average GPA by parental income level
        if 'parental_income_level' in df.columns:
            avg_gpa_by_income = df.groupby('parental_income_level')['gpa'].mean().sort_values(ascending=False).reset_index()
            fig_gpa_by_income = px.bar(avg_gpa_by_income, x='parental_income_level', y='gpa',
                                       title='Average GPA by Parental Income Level')
            visualizations['average_gpa_by_parental_income_level'] = fig_gpa_by_income.to_json()
            metrics['avg_gpa_by_parental_income'] = avg_gpa_by_income.to_dict('records')
            insights.append("Generated analysis by parental income level.")
        else:
            insights.append("Skipping parental income level analysis: column not found.")

        # Average GPA by free/reduced lunch status
        if 'free_reduced_lunch_status' in df.columns:
            avg_gpa_by_lunch_status = df.groupby('free_reduced_lunch_status')['gpa'].mean().reset_index()
            fig_gpa_by_lunch_status = px.bar(avg_gpa_by_lunch_status, x='free_reduced_lunch_status', y='gpa',
                                             title='Average GPA by Free/Reduced Lunch Status')
            visualizations['average_gpa_by_free_reduced_lunch_status'] = fig_gpa_by_lunch_status.to_json()
            metrics['avg_gpa_by_lunch_status'] = avg_gpa_by_lunch_status.to_dict('records')
            insights.append("Generated analysis by free/reduced lunch status.")
        else:
            insights.append("Skipping free/reduced lunch status analysis: column not found.")

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

def LMS_Activity_and_Quiz_Scores_Correlation_with_Final_Score(df):
    analysis_name = "LMS Activity and Quiz Scores' Correlation with Final Score"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'lms_activity_score', 'average_quiz_score', 'assignment_completion_rate']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between LMS activity score and final score
        if 'lms_activity_score' in df.columns:
            df['lms_activity_score'] = pd.to_numeric(df['lms_activity_score'], errors='coerce')
            if not df['lms_activity_score'].isnull().all():
                fig_lms_activity_final = px.scatter(df, x='lms_activity_score', y='final_score',
                                                    title='Final Score vs. LMS Activity Score', trendline="ols")
                visualizations['final_score_vs_lms_activity_score'] = fig_lms_activity_final.to_json()
                metrics["correlation_lms_activity_final"] = df[['lms_activity_score', 'final_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between LMS activity and final score: {metrics['correlation_lms_activity_final']:.2f}")
            else:
                insights.append("Skipping LMS activity score analysis: column has no valid data.")
        else:
            insights.append("Skipping LMS activity score analysis: column not found.")
            metrics["correlation_lms_activity_final"] = "N/A"

        # Correlation between average quiz score and final score
        if 'average_quiz_score' in df.columns:
            df['average_quiz_score'] = pd.to_numeric(df['average_quiz_score'], errors='coerce')
            if not df['average_quiz_score'].isnull().all():
                fig_quiz_score_final = px.scatter(df, x='average_quiz_score', y='final_score',
                                                  title='Final Score vs. Average Quiz Score', trendline="ols")
                visualizations['final_score_vs_average_quiz_score'] = fig_quiz_score_final.to_json()
                metrics["correlation_quiz_score_final"] = df[['average_quiz_score', 'final_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between quiz score and final score: {metrics['correlation_quiz_score_final']:.2f}")
            else:
                insights.append("Skipping average quiz score analysis: column has no valid data.")
        else:
            insights.append("Skipping average quiz score analysis: column not found.")

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

def Exam_Score_and_Pass_Status_Prediction(df):
    analysis_name = "Exam Score and Pass Status Prediction"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'exam_score', 'pass_status', 'study_time_hours', 'previous_exam_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
        df.dropna(subset=['student_id', 'exam_score', 'pass_status'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = len(df)
        metrics["overall_avg_exam_score"] = df['exam_score'].mean()
        
        # Pass status distribution
        pass_status_counts = df['pass_status'].value_counts(normalize=True).reset_index()
        pass_status_counts.columns = ['status', 'proportion']
        
        pass_rate = pass_status_counts[pass_status_counts['status'].astype(str).str.lower() == 'pass']['proportion'].sum() * 100
        metrics["pass_rate_percent"] = pass_rate if pass_rate else 0
        metrics["pass_fail_distribution"] = pass_status_counts.to_dict('records')

        insights.append(f"Analyzed {metrics['total_students']} students with an average exam score of {metrics['overall_avg_exam_score']:.2f}.")
        insights.append(f"Overall pass rate: {metrics['pass_rate_percent']:.2f}%")

        # --- Visualizations & Deeper Metrics ---

        # Average exam score for Pass vs. Fail students
        avg_exam_score_by_status = df.groupby('pass_status')['exam_score'].mean().reset_index()
        fig_avg_exam_score_by_status = px.bar(avg_exam_score_by_status, x='pass_status', y='exam_score',
                                              title='Average Exam Score for Pass vs. Fail Students')
        visualizations['average_exam_score_by_pass_status'] = fig_avg_exam_score_by_status.to_json()
        metrics['avg_exam_score_by_pass_status'] = avg_exam_score_by_status.to_dict('records')

        fig_pass_status_pie = px.pie(pass_status_counts, names='status', values='proportion', title='Exam Pass Status Distribution')
        visualizations['pass_status_distribution'] = fig_pass_status_pie.to_json()

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

def Extracurriculars_and_Study_Hours_on_Average_Score(df):
    analysis_name = "Extracurriculars and Study Hours on Average Score"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'average_score', 'extra_curricular_activities_participation', 'weekly_study_hours']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['average_score'] = pd.to_numeric(df['average_score'], errors='coerce')
        df.dropna(subset=['student_id', 'average_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_score"] = df['average_score'].mean()
        insights.append(f"Overall average score: {metrics['overall_avg_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average score by extracurricular activities participation
        if 'extra_curricular_activities_participation' in df.columns:
            avg_score_by_extracurriculars = df.groupby('extra_curricular_activities_participation')['average_score'].mean().reset_index()
            fig_score_by_extracurriculars = px.bar(avg_score_by_extracurriculars, x='extra_curricular_activities_participation', y='average_score',
                                                   title='Average Score by Extracurricular Activities Participation')
            visualizations['average_score_by_extracurriculars_participation'] = fig_score_by_extracurriculars.to_json()
            metrics['avg_score_by_extracurriculars'] = avg_score_by_extracurriculars.to_dict('records')
            insights.append("Generated analysis by extracurricular activities.")
        else:
            insights.append("Skipping extracurricular activities analysis: column not found.")

        # Correlation between weekly study hours and average score
        if 'weekly_study_hours' in df.columns:
            df['weekly_study_hours'] = pd.to_numeric(df['weekly_study_hours'], errors='coerce')
            if not df['weekly_study_hours'].isnull().all():
                fig_study_hours_score = px.scatter(df, x='weekly_study_hours', y='average_score',
                                                   title='Average Score vs. Weekly Study Hours', trendline="ols")
                visualizations['average_score_vs_weekly_study_hours'] = fig_study_hours_score.to_json()
                metrics["correlation_study_hours_score"] = df[['weekly_study_hours', 'average_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between weekly study hours and average score: {metrics['correlation_study_hours_score']:.2f}")
            else:
                insights.append("Skipping weekly study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping weekly study hours analysis: column not found.")

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

def Health_and_Engagement_s_Influence_on_Final_Grades(df):
    analysis_name = "Health and Engagement's Influence on Final Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'health_status_category', 'engagement_level_in_class', 'absences']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by health status category
        if 'health_status_category' in df.columns:
            avg_grade_by_health_status = df.groupby('health_status_category')['final_grade'].mean().reset_index()
            fig_grade_by_health_status = px.bar(avg_grade_by_health_status, x='health_status_category', y='final_grade',
                                                title='Average Final Grade by Health Status Category')
            visualizations['average_final_grade_by_health_status_category'] = fig_grade_by_health_status.to_json()
            metrics['avg_grade_by_health_status'] = avg_grade_by_health_status.to_dict('records')
            insights.append("Generated analysis by health status category.")
        else:
            insights.append("Skipping health status analysis: column not found.")

        # Average final grade by engagement level in class
        if 'engagement_level_in_class' in df.columns:
            avg_grade_by_engagement = df.groupby('engagement_level_in_class')['final_grade'].mean().reset_index()
            fig_grade_by_engagement = px.bar(avg_grade_by_engagement, x='engagement_level_in_class', y='final_grade',
                                             title='Average Final Grade by Engagement Level in Class')
            visualizations['average_final_grade_by_engagement_level_in_class'] = fig_grade_by_engagement.to_json()
            metrics['avg_grade_by_engagement'] = avg_grade_by_engagement.to_dict('records')
            insights.append("Generated analysis by engagement level.")
        else:
            insights.append("Skipping engagement level analysis: column not found.")

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

def Student_Grade_Category_Classification_Analysis(df):
    analysis_name = "Student Grade Category Classification Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'grade_category', 'midterm_score', 'quiz_score_average']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade', 'grade_category'], inplace=True)

        # --- Metrics ---
        metrics["total_students"] = len(df)
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()

        # Grade category distribution
        grade_category_counts = df['grade_category'].value_counts(normalize=True).reset_index()
        grade_category_counts.columns = ['category', 'proportion']
        metrics["grade_category_distribution"] = grade_category_counts.to_dict('records')
        
        insights.append(f"Analyzed {metrics['total_students']} students with an average final grade of {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by grade category
        avg_final_grade_by_category = df.groupby('grade_category')['final_grade'].mean().reset_index()
        fig_avg_final_grade_category = px.bar(avg_final_grade_by_category, x='grade_category', y='final_grade',
                                              title='Average Final Grade by Grade Category')
        visualizations['average_final_grade_by_grade_category'] = fig_avg_final_grade_category.to_json()
        metrics['avg_final_grade_by_grade_category'] = avg_final_grade_by_category.to_dict('records')

        fig_grade_category_pie = px.pie(grade_category_counts, names='category', values='proportion', title='Student Grade Category Distribution')
        visualizations['grade_category_distribution'] = fig_grade_category_pie.to_json()

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

def Factors_Influencing_Overall_Student_Score(df):
    analysis_name = "Factors Influencing Overall Student Score"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'overall_score', 'study_hours_per_week', 'absences_count', 'parental_support_level_score']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['overall_score'] = pd.to_numeric(df['overall_score'], errors='coerce')
        df.dropna(subset=['student_id', 'overall_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_score"] = df['overall_score'].mean()
        insights.append(f"Overall average score: {metrics['overall_avg_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between study hours and overall score
        if 'study_hours_per_week' in df.columns:
            df['study_hours_per_week'] = pd.to_numeric(df['study_hours_per_week'], errors='coerce')
            if not df['study_hours_per_week'].isnull().all():
                fig_study_hours_overall = px.scatter(df, x='study_hours_per_week', y='overall_score',
                                                     title='Overall Score vs. Weekly Study Hours', trendline="ols")
                visualizations['overall_score_vs_weekly_study_hours'] = fig_study_hours_overall.to_json()
                metrics["correlation_study_hours_overall"] = df[['study_hours_per_week', 'overall_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between study hours and score: {metrics['correlation_study_hours_overall']:.2f}")
            else:
                insights.append("Skipping weekly study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping weekly study hours analysis: column not found.")
            metrics["correlation_study_hours_overall"] = "N/A"

        # Correlation between parental support level score and overall score
        if 'parental_support_level_score' in df.columns:
            df['parental_support_level_score'] = pd.to_numeric(df['parental_support_level_score'], errors='coerce')
            if not df['parental_support_level_score'].isnull().all():
                fig_parental_support_overall = px.scatter(df, x='parental_support_level_score', y='overall_score',
                                                          title='Overall Score vs. Parental Support Level Score', trendline="ols")
                visualizations['overall_score_vs_parental_support_level_score'] = fig_parental_support_overall.to_json()
                metrics["correlation_parental_support_overall"] = df[['parental_support_level_score', 'overall_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between parental support and score: {metrics['correlation_parental_support_overall']:.2f}")
            else:
                insights.append("Skipping parental support analysis: column has no valid data.")
        else:
            insights.append("Skipping parental support analysis: column not found.")

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

def Study_Habits_and_Past_Performance_on_Final_Score(df):
    analysis_name = "Study Habits and Past Performance on Final Score"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'weekly_study_hours', 'g1_grade_past', 'g2_grade_past']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between weekly study hours and final score
        if 'weekly_study_hours' in df.columns:
            df['weekly_study_hours'] = pd.to_numeric(df['weekly_study_hours'], errors='coerce')
            if not df['weekly_study_hours'].isnull().all():
                fig_study_hours_final = px.scatter(df, x='weekly_study_hours', y='final_score',
                                                   title='Final Score vs. Weekly Study Hours', trendline="ols")
                visualizations['final_score_vs_weekly_study_hours'] = fig_study_hours_final.to_json()
                metrics["correlation_study_hours_final"] = df[['weekly_study_hours', 'final_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between study hours and final score: {metrics['correlation_study_hours_final']:.2f}")
            else:
                insights.append("Skipping weekly study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping weekly study hours analysis: column not found.")
            metrics["correlation_study_hours_final"] = "N/A"

        # Correlation between G2 grade (past performance) and final score
        if 'g2_grade_past' in df.columns:
            df['g2_grade_past'] = pd.to_numeric(df['g2_grade_past'], errors='coerce')
            if not df['g2_grade_past'].isnull().all():
                fig_g2_final_score = px.scatter(df, x='g2_grade_past', y='final_score',
                                                title='Final Score vs. G2 Grade (Past Performance)', trendline="ols")
                visualizations['final_score_vs_g2_grade_past'] = fig_g2_final_score.to_json()
                metrics["correlation_g2_past_final"] = df[['g2_grade_past', 'final_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between G2 past grade and final score: {metrics['correlation_g2_past_final']:.2f}")
            else:
                insights.append("Skipping G2 past grade analysis: column has no valid data.")
        else:
            insights.append("Skipping G2 past grade analysis: column not found.")

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

def Test_Preparation_and_Demographics_Impact_on_Final_Grade(df):
    analysis_name = "Test Preparation and Demographics' Impact on Final Grade"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'test_preparation_course_completed', 'gender', 'race_ethnicity']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final grade by test preparation course completion
        if 'test_preparation_course_completed' in df.columns:
            avg_grade_by_prep = df.groupby('test_preparation_course_completed')['final_grade'].mean().reset_index()
            fig_grade_by_prep = px.bar(avg_grade_by_prep, x='test_preparation_course_completed', y='final_grade',
                                       title='Average Final Grade by Test Preparation Course Completion')
            visualizations['average_final_grade_by_test_preparation'] = fig_grade_by_prep.to_json()
            metrics['avg_grade_by_test_preparation'] = avg_grade_by_prep.to_dict('records')
            insights.append("Generated analysis by test preparation status.")
        else:
            insights.append("Skipping test preparation analysis: column not found.")

        # Average final grade by race/ethnicity
        if 'race_ethnicity' in df.columns:
            avg_grade_by_race = df.groupby('race_ethnicity')['final_grade'].mean().sort_values(ascending=False).reset_index()
            fig_grade_by_race = px.bar(avg_grade_by_race, x='race_ethnicity', y='final_grade',
                                       title='Average Final Grade by Race/Ethnicity')
            visualizations['average_final_grade_by_race_ethnicity'] = fig_grade_by_race.to_json()
            metrics['avg_grade_by_race_ethnicity'] = avg_grade_by_race.to_dict('records')
            insights.append("Generated analysis by race/ethnicity.")
        else:
            insights.append("Skipping race/ethnicity analysis: column not found.")

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

def Course_Load_and_Absences_Effect_on_Student_GPA(df):
    analysis_name = "Course Load and Absences' Effect on Student GPA"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'gpa', 'number_of_courses_taken', 'total_absences_in_semester']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        df.dropna(subset=['student_id', 'gpa'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_gpa"] = df['gpa'].mean()
        insights.append(f"Overall average GPA: {metrics['overall_avg_gpa']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between number of courses taken and GPA
        if 'number_of_courses_taken' in df.columns:
            df['number_of_courses_taken'] = pd.to_numeric(df['number_of_courses_taken'], errors='coerce')
            if not df['number_of_courses_taken'].isnull().all():
                fig_course_load_gpa = px.scatter(df, x='number_of_courses_taken', y='gpa',
                                                 title='GPA vs. Number of Courses Taken', trendline="ols")
                visualizations['gpa_vs_number_of_courses_taken'] = fig_course_load_gpa.to_json()
                metrics["correlation_course_load_gpa"] = df[['number_of_courses_taken', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between course load and GPA: {metrics['correlation_course_load_gpa']:.2f}")
            else:
                insights.append("Skipping course load analysis: column has no valid data.")
        else:
            insights.append("Skipping course load analysis: column not found.")
            metrics["correlation_course_load_gpa"] = "N/A"

        # Correlation between total absences and GPA
        if 'total_absences_in_semester' in df.columns:
            df['total_absences_in_semester'] = pd.to_numeric(df['total_absences_in_semester'], errors='coerce')
            if not df['total_absences_in_semester'].isnull().all():
                fig_absences_gpa = px.scatter(df, x='total_absences_in_semester', y='gpa',
                                              title='GPA vs. Total Absences in Semester', trendline="ols")
                visualizations['gpa_vs_total_absences_in_semester'] = fig_absences_gpa.to_json()
                metrics["correlation_absences_gpa"] = df[['total_absences_in_semester', 'gpa']].corr().iloc[0, 1]
                insights.append(f"Correlation between absences and GPA: {metrics['correlation_absences_gpa']:.2f}")
            else:
                insights.append("Skipping absences analysis: column has no valid data.")
        else:
            insights.append("Skipping absences analysis: column not found.")

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

def Internet_Access_and_Activities_Impact_on_Final_Score(df):
    analysis_name = "Internet Access and Activities' Impact on Final Score"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_score', 'internet_access_type', 'online_social_media_hours', 'online_learning_resource_usage']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        
        df = safe_rename(df, matched)
        df['final_score'] = pd.to_numeric(df['final_score'], errors='coerce')
        df.dropna(subset=['student_id', 'final_score'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_score"] = df['final_score'].mean()
        insights.append(f"Overall average final score: {metrics['overall_avg_final_score']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Average final score by internet access type
        if 'internet_access_type' in df.columns:
            avg_score_by_internet_access = df.groupby('internet_access_type')['final_score'].mean().reset_index()
            fig_score_by_internet_access = px.bar(avg_score_by_internet_access, x='internet_access_type', y='final_score',
                                                  title='Average Final Score by Internet Access Type')
            visualizations['average_final_score_by_internet_access_type'] = fig_score_by_internet_access.to_json()
            metrics['avg_score_by_internet_access'] = avg_score_by_internet_access.to_dict('records')
            insights.append("Generated analysis by internet access type.")
        else:
            insights.append("Skipping internet access type analysis: column not found.")

        # Correlation between online learning resource usage and final score
        if 'online_learning_resource_usage' in df.columns:
            df['online_learning_resource_usage'] = pd.to_numeric(df['online_learning_resource_usage'], errors='coerce')
            if not df['online_learning_resource_usage'].isnull().all():
                fig_online_resources_score = px.scatter(df, x='online_learning_resource_usage', y='final_score',
                                                        title='Final Score vs. Online Learning Resource Usage', trendline="ols")
                visualizations['final_score_vs_online_learning_resource_usage'] = fig_online_resources_score.to_json()
                metrics["correlation_online_resources_final"] = df[['online_learning_resource_usage', 'final_score']].corr().iloc[0, 1]
                insights.append(f"Correlation between online resource use and final score: {metrics['correlation_online_resources_final']:.2f}")
            else:
                insights.append("Skipping online learning resource usage analysis: column has no valid data.")
        else:
            insights.append("Skipping online learning resource usage analysis: column not found.")
            metrics["correlation_online_resources_final"] = "N/A"

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

def Test_Validity_and_Study_Time_s_Effect_on_Final_Grades(df):
    analysis_name = "Test Validity and Study Time's Effect on Final Grades"
    visualizations = {}
    metrics = {}
    insights = []
    matched = {}

    try:
        expected = ['student_id', 'final_grade', 'test_validity_score', 'weekly_study_hours', 'exam_difficulty_rating']
        matched = fuzzy_match_column(df, expected)
        
        missing = [col for col in expected if matched.get(col) is None]
        if missing:
             return create_fallback_response(analysis_name, missing, matched, df)
        df = safe_rename(df, matched)
        df['final_grade'] = pd.to_numeric(df['final_grade'], errors='coerce')
        df.dropna(subset=['student_id', 'final_grade'], inplace=True)

        # --- Metrics ---
        metrics["overall_avg_final_grade"] = df['final_grade'].mean()
        insights.append(f"Overall average final grade: {metrics['overall_avg_final_grade']:.2f}")

        # --- Visualizations & Deeper Metrics ---

        # Correlation between test validity score and final grade
        if 'test_validity_score' in df.columns:
            df['test_validity_score'] = pd.to_numeric(df['test_validity_score'], errors='coerce')
            if not df['test_validity_score'].isnull().all():
                fig_test_validity_grade = px.scatter(df, x='test_validity_score', y='final_grade',
                                                     title='Final Grade vs. Test Validity Score', trendline="ols")
                visualizations['final_grade_vs_test_validity_score'] = fig_test_validity_grade.to_json()
                metrics["correlation_test_validity_final"] = df[['test_validity_score', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between test validity and final grade: {metrics['correlation_test_validity_final']:.2f}")
            else:
                insights.append("Skipping test validity score analysis: column has no valid data.")
        else:
            insights.append("Skipping test validity score analysis: column not found.")
            metrics["correlation_test_validity_final"] = "N/A"

        # Correlation between weekly study hours and final grade
        if 'weekly_study_hours' in df.columns:
            df['weekly_study_hours'] = pd.to_numeric(df['weekly_study_hours'], errors='coerce')
            if not df['weekly_study_hours'].isnull().all():
                fig_study_hours_grade = px.scatter(df, x='weekly_study_hours', y='final_grade',
                                                   title='Final Grade vs. Weekly Study Hours', trendline="ols")
                visualizations['final_grade_vs_weekly_study_hours'] = fig_study_hours_grade.to_json()
                metrics["correlation_study_hours_final"] = df[['weekly_study_hours', 'final_grade']].corr().iloc[0, 1]
                insights.append(f"Correlation between weekly study hours and final grade: {metrics['correlation_study_hours_final']:.2f}")
            else:
                insights.append("Skipping weekly study hours analysis: column has no valid data.")
        else:
            insights.append("Skipping weekly study hours analysis: column not found.")

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


# ========== MAIN DISPATCHER & COMMAND-LINE RUNNER ==========

# ========== ANALYSIS FUNCTION MAPPINGS ==========

# 1. MAPPING FOR "GENERAL" CATEGORY
# These are a subset of high-level analyses for a "General" dropdown
general_analysis_functions = {
    "Student Academic Performance Summary": Student_Academic_Performance_Summary_Analysis,
    "Test Score by Demographics": Student_Test_Score_Analysis_by_Demographics_and_Preparation,
    "Factors in Final Grades": Factors_Affecting_Student_Final_Grades,
    "Study Habits Impact": Study_Habits_and_Their_Impact_on_Final_Scores,
    "Student Engagement & Performance": Student_Engagement_and_Performance_Analysis,
    "Test Preparation Impact": Impact_of_Test_Preparation_on_Academic_Scores,
    "LMS Usage Correlation": Learning_Management_System_LMS_Usage_and_Grade_Correlation,
    "Pass/Fail Prediction": Student_Pass_Fail_Prediction_Analysis
}

# 2. MAPPING FOR "SPECIFIC" CATEGORY
# This maps all specific analyses from your analysis_options list
specific_student_function_mapping = {
    "Student Test Score Analysis by Demographics and Preparation": Student_Test_Score_Analysis_by_Demographics_and_Preparation,
    "Factors Affecting Student Final Grades": Factors_Affecting_Student_Final_Grades,
    "Student Academic Performance Summary Analysis": Student_Academic_Performance_Summary_Analysis,
    "Study Habits and Their Impact on Final Scores": Study_Habits_and_Their_Impact_on_Final_Scores,
    "Student Engagement and Performance Analysis": Student_Engagement_and_Performance_Analysis,
    "Impact of Test Preparation on Academic Scores": Impact_of_Test_Preparation_on_Academic_Scores,
    "Learning Management System (LMS) Usage and Grade Correlation": Learning_Management_System_LMS_Usage_and_Grade_Correlation,
    "Demographic and Health Factors on Student Scores": Demographic_and_Health_Factors_on_Student_Scores,
    "Social Factors and Internet Usage Impact on Student Performance": Social_Factors_and_Internet_Usage_Impact_on_Student_Performance,
    "Student Pass/Fail Prediction Analysis": Student_Pass_Fail_Prediction_Analysis,
    "Impact of Past Grades and Study Time on Current Performance": Impact_of_Past_Grades_and_Study_Time_on_Current_Performance,
    "Extracurricular Activities and Academic Grade Analysis": Extracurricular_Activities_and_Academic_Grade_Analysis,
    "Family and Internet Support on Student Final Scores": Family_and_Internet_Support_on_Student_Final_Scores,
    "Parental Background and Study Time on Final Grades": Parental_Background_and_Study_Time_on_Final_Grades,
    "Family Relationships and Student Grade Analysis": Family_Relationships_and_Student_Grade_Analysis,
    "Assessment Scores and Attendance Impact on Final Grade": Assessment_Scores_and_Attendance_Impact_on_Final_Grade,
    "Lifestyle Factors and Their Correlation with Student GPA": Lifestyle_Factors_and_Their_Correlation_with_Student_GPA,
    "Educational Support Systems' Impact on Student Grades": Educational_Support_Systems_Impact_on_Student_Grades,
    "Impact of Paid Classes and School Support on Performance": Impact_of_Paid_Classes_and_School_Support_on_Performance,
    "Student Performance and Pass Status Prediction": Student_Performance_and_Pass_Status_Prediction,
    "Health, Absences, and Travel Time on Final Grades (G3)": Health_Absences_and_Travel_Time_on_Final_Grades_G3,
    "Impact of Extra Paid Classes and School Support on Grades": Impact_of_Extra_Paid_Classes_and_School_Support_on_Grades,
    "Social and Health Factors Affecting Student Scores": Social_and_Health_Factors_Affecting_Student_Scores,
    "Physical Attributes and Commute's Effect on Grades": Physical_Attributes_and_Commutes_Effect_on_Grades,
    "Student Performance and Pass/Fail Classification": Student_Performance_and_Pass_Fail_Classification,
    "Digital Engagement and Parental Support on Student GPA": Digital_Engagement_and_Parental_Support_on_Student_GPA,
    "Daily Habits and Their Influence on Student Grades": Daily_Habits_and_Their_Influence_on_Student_Grades,
    "Demographic Factors and Test Preparation on Student Scores": Demographic_Factors_and_Test_Preparation_on_Student_Scores,
    "Study Time and Absences on Final Academic Performance": Study_Time_and_Absences_on_Final_Academic_Performance,
    "Social Activities and Health on Final Student Grades (G3)": Social_Activities_and_Health_on_Final_Student_Grades_G3,
    "Longitudinal Academic Performance Analysis (G1, G2, G3)": Longitudinal_Academic_Performance_Analysis_G1_G2_G3,
    "Student Performance Analysis based on Demographics": Student_Performance_Analysis_based_on_Demographics,
    "Student Performance Category Prediction Analysis": Student_Performance_Category_Prediction_Analysis,
    "Ethnicity and Parental Education's Role in Student Grades": Ethnicity_and_Parental_Education_s_Role_in_Student_Grades,
    "Behavioral and Engagement Scores on Academic Outcomes": Behavioral_and_Engagement_Scores_on_Academic_Outcomes,
    "Continuous Assessment and Study Time on Final Grade": Continuous_Assessment_and_Study_Time_on_Final_Grade,
    "Screen Time and Sleep's Impact on Student Anxiety and Grades": Screen_Time_and_Sleep_s_Impact_on_Student_Anxiety_and_Grades,
    "Midterm Performance and Engagement as Predictors of Final Grades": Midterm_Performance_and_Engagement_as_Predictors_of_Final_Grades,
    "Socioeconomic Status and Its Effect on Student GPA": Socioeconomic_Status_and_Its_Effect_on_Student_GPA,
    "LMS Activity and Quiz Scores' Correlation with Final Score": LMS_Activity_and_Quiz_Scores_Correlation_with_Final_Score,
    "Exam Score and Pass Status Prediction": Exam_Score_and_Pass_Status_Prediction,
    "Student Grade Category Classification Analysis": Student_Grade_Category_Classification_Analysis,
    "Factors Influencing Overall Student Score": Factors_Influencing_Overall_Student_Score,
    "Study Habits and Past Performance on Final Score": Study_Habits_and_Past_Performance_on_Final_Score,
    "Test Preparation and Demographics Impact on Final Grade": Test_Preparation_and_Demographics_Impact_on_Final_Grade,
    "Course Load and Absences Effect on Student GPA": Course_Load_and_Absences_Effect_on_Student_GPA,
    "Internet Access and Activities Impact on Final Score": Internet_Access_and_Activities_Impact_on_Final_Score,
    "Test Validity and Study Time's Effect on Final Grades": Test_Validity_and_Study_Time_s_Effect_on_Final_Grades
}


# ========== MAIN DISPATCHER FUNCTION ==========

def main_backend(df, category=None, general_analysis=None, specific_analysis_name=None):
    """
    Main dispatcher to route analysis requests based on UI selections.
    
    Args:
        df (pd.DataFrame): The dataframe to analyze.
        category (str): The main category selected (e.g., "General Student Analysis", "Specific Student Analysis").
        general_analysis (str): The name of the analysis from the "General" dropdown.
        specific_analysis_name (str): The name of the analysis from the "Specific" dropdown.
        
    Returns:
        dict: A dictionary containing the analysis results.
    """
    
    result = None

    try:
        if category == "General Retail Analysis":
            if not general_analysis or general_analysis == "--Select--":
                # Use the utility function from your script
                result = show_general_insights(df, "Initial Overview")
            else:
                func = general_analysis_functions.get(general_analysis)
                if func:
                    result = func(df)
                else:
                    # Fallback if name not found
                    result = show_general_insights(df, "Initial Overview")

        elif category == "Specific Retail Analysis":
            if specific_analysis_name and specific_analysis_name != "--Select--":
                func = specific_student_function_mapping.get(specific_analysis_name)
                if func:
                    result = func(df)
                else:
                    # Handle case where the specific name isn't in the map
                    result = {
                        "analysis_type": specific_analysis_name,
                        "status": "error",
                        "error_message": f"Analysis function for '{specific_analysis_name}' not found."
                    }
            else:
                # No specific analysis was selected, show general
                result = show_general_insights(df, "Specific Analysis Not Selected")
        else:
            # Default action if no category matches (e.g., initial load)
            result = show_general_insights(df, "Initial Overview")

    except Exception as e:
        # Broad exception handler for the dispatcher, using your script's structure
        return {
            "analysis_type": "Main Dispatcher",
            "status": "error",
            "error_message": f"An unexpected error occurred in main_backend: {str(e)}",
            "visualizations": {},
            "metrics": {},
            "insights": [f"Error: {e}"]
        }

    return result
if __name__ == "__main__":
    print("Running Retail Analysis Script in test mode...")
