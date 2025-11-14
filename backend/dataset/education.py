import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from fuzzywuzzy import process
from scipy.stats import pearsonr, linregress
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- New Helper Function for Type Conversion ---
def convert_to_native_types(data):
    """Recursively converts numpy and pandas types to native Python types for JSON serialization."""
    if isinstance(data, dict):
        return {k: convert_to_native_types(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_native_types(i) for i in data]
    if isinstance(data, np.integer):
        return int(data)
    if isinstance(data, np.floating):
        return float(data)
    if isinstance(data, np.bool_):
        return bool(data)
    if isinstance(data, (pd.Timestamp, pd.Timedelta)):
        return str(data)
    if isinstance(data, pd.Series):
        return convert_to_native_types(data.to_list())
    if isinstance(data, pd.DataFrame):
        return convert_to_native_types(data.to_dict())
    return data

# --- Original Helper Functions (Unchanged) ---
def safe_numeric_conversion(df, column_name):
    """Converts a column to numeric, coercing errors and dropping NaNs."""
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df.dropna(subset=[column_name])
    print(f"Warning: Column '{column_name}' not found for numeric conversion.")
    return df

def fuzzy_match_column(df, target_columns):
    """Finds the best fuzzy match for target columns in the DataFrame."""
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
    """
    Checks for expected columns and renames them to a standard name.
    Returns the renamed DataFrame and a list of missing standard_names.
    """
    missing_cols = []
    renamed_df = df.copy()
    for standard_name, potential_names in expected_cols_map.items():
        found = False
        if standard_name in renamed_df.columns: # Already correct
            found = True
        else:
            for p_name in potential_names:
                if p_name in renamed_df.columns:
                    renamed_df = renamed_df.rename(columns={p_name: standard_name})
                    found = True
                    break
        if not found:
            missing_cols.append(standard_name)
    return renamed_df, missing_cols

def load_data(file_path, encoding='utf-8'):
    """Loads data from CSV or Excel, trying multiple encodings for CSV."""
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=enc)
                    print(f"File loaded successfully with {enc} encoding.")
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

def show_missing_columns_warning(missing_columns, expected_cols_map):
    """Prints a standardized warning for missing columns (for logging)."""
    print(f"\n--- WARNING: Required Columns Not Found ---")
    print(f"The following columns are needed but missing: {', '.join(missing_columns)}")
    print("Expected column mappings attempted:")
    for standard_name in missing_columns:
        if standard_name in expected_cols_map:
            print(f"- '{standard_name}' (e.g., {', '.join(expected_cols_map[standard_name])})")
    print("Analysis might be incomplete or aborted due to missing required data.")


# --- NEW Refactored General Insights (Structured Output) ---
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
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
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
                    corr_matrix = df[numeric_cols].corr().round(2)
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
            "status": "success",
            "matched_columns": matched_cols or {},
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights,
            "missing_columns": missing_cols or []
        }
        
    except Exception as e:
        # Ultra-safe fallback with missing columns info
        basic_insights = [
            f"Basic dataset info: {len(df)} rows, {len(df.columns)} columns",
            f"Columns: {', '.join(df.columns.tolist()[:5])}{'...' if len(df.columns) > 5 else ''}",
            "Limited analysis due to data compatibility"
        ]
        
        # Add missing columns warning even in error case
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



# --- Refactored Analysis Functions ---

def academic_performance_analysis(df):
    print("\n--- Academic Performance Analysis ---")
    analysis_type = "Academic Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'student_id': ['student_id', 'StudentId', 'ID'],
            'gpa': ['gpa', 'GPA', 'GradePointAverage'],
            'test_score': ['test_score', 'TestScore', 'ExamScore'],
            'attendance_rate': ['attendance_rate', 'AttendanceRate', 'AttendancePct']
        }
        
        df, missing = check_and_rename_columns(df, expected)
        
        # Store matched columns for reporting
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}


        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result

        df['gpa'] = pd.to_numeric(df['gpa'], errors='coerce')
        df['test_score'] = pd.to_numeric(df['test_score'], errors='coerce')
        df['attendance_rate'] = pd.to_numeric(df['attendance_rate'], errors='coerce')
        df = df.dropna(subset=['gpa', 'test_score', 'attendance_rate'])

        if df.empty:
            return {
                "analysis_type": analysis_type,
                "status": "error",
                "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map,
                "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning for this analysis."]
            }

        avg_gpa = df['gpa'].mean()
        at_risk = (df['gpa'] < 2.0).sum()
        strong_performers = (df['gpa'] >= 3.5).sum()
        
        metrics = {
            "Average GPA": avg_gpa,
            "At-Risk Students": at_risk,
            "Strong Performers": strong_performers
        }
        
        insights.append(f"Average GPA: {avg_gpa:.2f}")
        insights.append(f"At-Risk Students (GPA < 2.0): {at_risk}")
        if at_risk > 0:
            insights.append(f"WARNING: {at_risk} At-Risk Students detected. These students may need academic intervention.")
        insights.append(f"Strong Performers (GPA >= 3.5): {strong_performers}")

        fig1 = px.scatter(df, x='attendance_rate', y='gpa', title="Attendance Rate vs GPA")
        fig2 = px.scatter(df, x='test_score', y='gpa', title="Test Scores vs GPA")
        
        visualizations = {
            "Attendance_vs_GPA_Scatter": fig1.to_json(),
            "Test_Scores_vs_GPA_Scatter": fig2.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }
    
    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type,
            "status": "error",
            "error": str(e),
            "matched_columns": matched_columns_map,
            "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def demographic_analysis(df):
    print("\n--- Demographic Analysis ---")
    analysis_type = "Demographic Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'student_id': ['student_id', 'StudentId', 'ID'],
            'gender': ['gender', 'Gender', 'Sex'],
            'ethnicity': ['ethnicity', 'Ethnicity', 'Race'],
            'age': ['age', 'Age'],
            'socioeconomic_status': ['socioeconomic_status', 'SES', 'PovertyStatus']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['age'] = pd.to_numeric(df['age'], errors='coerce')
        df = df.dropna(subset=['gender', 'ethnicity', 'age', 'socioeconomic_status'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        gender_dist = df['gender'].value_counts(normalize=True)
        diversity_index = df['ethnicity'].nunique()
        total_students = len(df)
        
        metrics = {
            "Gender Distribution (Normalized)": gender_dist.to_dict(),
            "Ethnicity Diversity Index": diversity_index,
            "Total Students": total_students
        }
        
        insights.append(f"Total Students Analyzed: {total_students}")
        insights.append(f"Ethnicity Diversity Index (Number of unique ethnicities): {diversity_index}")
        insights.append(f"Gender Distribution:\n{gender_dist.to_string()}")

        fig1 = px.pie(df, names='gender', title="Gender Distribution")
        fig2 = px.bar(df['ethnicity'].value_counts().reset_index(name='count').rename(columns={'index': 'Ethnicity'}),
                      x='Ethnicity', y='count', title="Ethnicity Distribution")

        visualizations = {
            "Gender_Distribution_Pie": fig1.to_json(),
            "Ethnicity_Distribution_Bar": fig2.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def course_analysis(df):
    print("\n--- Course Analysis ---")
    analysis_type = "Course Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'course_id': ['course_id', 'CourseID', 'CourseCode'],
            'enrollment_count': ['enrollment_count', 'EnrollmentCount', 'NumEnrolled'],
            'pass_rate': ['pass_rate', 'PassRate', 'SuccessRate'],
            'instructor': ['instructor', 'InstructorName', 'Faculty']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['enrollment_count'] = pd.to_numeric(df['enrollment_count'], errors='coerce')
        df['pass_rate'] = pd.to_numeric(df['pass_rate'], errors='coerce')
        df = df.dropna(subset=['enrollment_count', 'pass_rate', 'course_id', 'instructor'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_pass_rate = df['pass_rate'].mean()
        challenging_courses = (df['pass_rate'] < 0.6).sum()
        popular_course = df.loc[df['enrollment_count'].idxmax(), 'course_id']
        
        metrics = {
            "Average Pass Rate": avg_pass_rate,
            "Challenging Courses (<60% Pass Rate)": challenging_courses,
            "Most Popular Course": popular_course
        }
        
        insights.append(f"Average Pass Rate: {avg_pass_rate:.1%}")
        insights.append(f"Number of Challenging Courses (Pass Rate < 60%): {challenging_courses}")
        insights.append(f"Most Popular Course (by enrollment): {popular_course}")

        fig1 = px.bar(df.sort_values('pass_rate'), x='course_id', y='pass_rate', title="Course Pass Rates")
        fig2 = px.scatter(df, x='enrollment_count', y='pass_rate', hover_name='course_id', title="Enrollment vs Pass Rate")

        visualizations = {
            "Course_Pass_Rates_Bar": fig1.to_json(),
            "Enrollment_vs_Pass_Rate_Scatter": fig2.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def attendance_analysis(df):
    print("\n--- Attendance Analysis ---")
    analysis_type = "Attendance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'student_id': ['student_id', 'StudentId', 'ID'],
            'attendance_rate': ['attendance_rate', 'AttendanceRate', 'AttendancePct'],
            'grade_level': ['grade_level', 'GradeLevel', 'Grade'],
            'absences': ['absences', 'TotalAbsences', 'NumAbsences']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['attendance_rate'] = pd.to_numeric(df['attendance_rate'], errors='coerce')
        df['absences'] = pd.to_numeric(df['absences'], errors='coerce')
        df = df.dropna(subset=['attendance_rate', 'absences'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_attendance = df['attendance_rate'].mean()
        chronic_absentees = (df['attendance_rate'] < 0.8).sum()
        
        metrics = {
            "Average Attendance": avg_attendance,
            "Chronic Absentees (<80% Attendance)": chronic_absentees
        }
        
        insights.append(f"Average Attendance Rate: {avg_attendance:.1%}")
        insights.append(f"Chronic Absentees (Attendance < 80%): {chronic_absentees}")
        if chronic_absentees > 0:
            insights.append(f"WARNING: {chronic_absentees} Chronic Absentees detected. These students may need intervention.")
        
        fig1 = px.histogram(df, x='attendance_rate', nbins=20, title="Attendance Rate Distribution")
        visualizations["Attendance_Rate_Distribution_Histogram"] = fig1.to_json()
        
        if 'grade_level' in df.columns and not df['grade_level'].dropna().empty:
            fig2 = px.box(df, x='grade_level', y='attendance_rate', title="Attendance by Grade Level")
            visualizations["Attendance_by_Grade_Level_Box"] = fig2.to_json()

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def behavioral_analysis(df):
    print("\n--- Behavioral Analysis ---")
    analysis_type = "Behavioral Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'student_id': ['student_id', 'StudentId', 'ID'],
            'behavior_incidents': ['behavior_incidents', 'BehaviorIncidents', 'DisciplineIncidents'],
            'interventions': ['interventions', 'NumInterventions', 'InterventionCount'],
            'grade_level': ['grade_level', 'GradeLevel', 'Grade']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['behavior_incidents'] = pd.to_numeric(df['behavior_incidents'], errors='coerce')
        df['interventions'] = pd.to_numeric(df['interventions'], errors='coerce')
        df = df.dropna(subset=['behavior_incidents', 'interventions'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_incidents = df['behavior_incidents'].mean()
        high_incidents = (df['behavior_incidents'] > 5).sum()
        
        metrics = {
            "Average Incidents": avg_incidents,
            "High Incident Students (>5 incidents)": high_incidents
        }
        
        insights.append(f"Average Incidents: {avg_incidents:.1f}")
        insights.append(f"High Incident Students (>5 incidents): {high_incidents}")
        if high_incidents > 0:
            insights.append(f"WARNING: {high_incidents} Students with >5 Behavioral Incidents detected. These students may need behavioral support.")
        
        fig1 = px.histogram(df, x='behavior_incidents', nbins=20, title="Behavioral Incidents Distribution")
        fig2 = px.scatter(df, x='behavior_incidents', y='interventions', title="Incidents vs Interventions")

        visualizations = {
            "Behavioral_Incidents_Histogram": fig1.to_json(),
            "Incidents_vs_Interventions_Scatter": fig2.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def program_evaluation(df):
    print("\n--- Program Evaluation ---")
    analysis_type = "Program Evaluation"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'program_id': ['program_id', 'ProgramID', 'ProgramName'],
            'participant_count': ['participant_count', 'ParticipantCount', 'NumParticipants'],
            'improvement_score': ['improvement_score', 'ImprovementScore', 'EffectivenessScore'],
            'cost_per_student': ['cost_per_student', 'CostPerStudent', 'ProgramCostPerStudent']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['participant_count'] = pd.to_numeric(df['participant_count'], errors='coerce')
        df['improvement_score'] = pd.to_numeric(df['improvement_score'], errors='coerce')
        df['cost_per_student'] = pd.to_numeric(df['cost_per_student'], errors='coerce')
        df = df.dropna(subset=['participant_count', 'improvement_score', 'cost_per_student', 'program_id'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_improvement = df['improvement_score'].mean()
        cost_effective = df.loc[df['improvement_score'].idxmax(), 'program_id'] if not df.empty else None
        expensive_programs = (df['cost_per_student'] > df['cost_per_student'].quantile(0.75)).sum()
        
        metrics = {
            "Average Improvement": avg_improvement,
            "Most Effective Program": cost_effective,
            "High-Cost Programs (Top 25%)": expensive_programs
        }
        
        insights.append(f"Average Improvement Score: {avg_improvement:.1f}")
        insights.append(f"Most Effective Program (highest improvement score): {cost_effective}")
        insights.append(f"Number of High-Cost Programs (top 25% cost): {expensive_programs}")
        
        fig1 = px.bar(df.sort_values('improvement_score', ascending=False),
                      x='program_id', y='improvement_score', title="Program Improvement Scores")
        fig2 = px.scatter(df, x='cost_per_student', y='improvement_score',
                          hover_name='program_id', size='participant_count', title="Cost vs Effectiveness")

        visualizations = {
            "Program_Improvement_Scores_Bar": fig1.to_json(),
            "Cost_vs_Effectiveness_Scatter": fig2.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_performance_and_socioeconomic_analysis(df):
    print("\n--- School District Performance and Socioeconomic Analysis ---")
    analysis_type = "School District Performance and Socioeconomic Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'district': ['district', 'DistrictName', 'SchoolDistrict'],
            'school': ['school', 'SchoolName'],
            'county': ['county', 'County'],
            'read': ['read', 'ReadingScore', 'AvgReadScore'],
            'math': ['math', 'MathScore', 'AvgMathScore'],
            'socioeconomic_index': ['socioeconomic_index', 'SESIndex', 'PovertyRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['read'] = pd.to_numeric(df['read'], errors='coerce')
        df['math'] = pd.to_numeric(df['math'], errors='coerce')
        df['socioeconomic_index'] = pd.to_numeric(df['socioeconomic_index'], errors='coerce')
        df = df.dropna(subset=['read', 'math', 'socioeconomic_index'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_math = df['math'].mean()
        avg_read = df['read'].mean()
        
        metrics = {
            "Average Math Score": avg_math,
            "Average Reading Score": avg_read
        }
        
        insights.append(f"Average Math Score: {avg_math:.2f}")
        insights.append(f"Average Reading Score: {avg_read:.2f}")
        
        fig1 = px.box(df, x='county', y=['math', 'read'], title='Test Scores by County')
        fig2 = px.histogram(df, x='district', y='math', color='school', title='Math Scores by District and School')
        
        math_by_district = df.groupby('district')['math'].mean().reset_index()
        fig3 = px.bar(math_by_district.sort_values('math', ascending=False).head(20), x='district', y='math', title='Average Math Score by District (Top 20)')
        
        fig4 = px.scatter(df, x='socioeconomic_index', y=(df['math'] + df['read']) / 2,
                          title='Average Test Score vs. Socioeconomic Index',
                          hover_data=['district', 'school'])

        visualizations = {
            "Test_Scores_by_County_Box": fig1.to_json(),
            "Math_Scores_by_District_and_School_Histogram": fig2.to_json(),
            "Average_Math_Score_by_District_Bar": fig3.to_json(),
            "Test_Score_vs_Socioeconomic_Index_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def higher_education_institution_cost_of_attendance_analysis(df):
    print("\n--- Higher Education Institution Cost of Attendance Analysis ---")
    analysis_type = "Higher Education Institution Cost of Attendance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'Instnm': ['Instnm', 'InstitutionName', 'Name'],
            'City': ['City', 'CampusCity'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'Year': ['Year', 'AcademicYear'],
            'AverageCostOfAttendance': ['AverageCostOfAttendance', 'CostOfAttendance', 'TotalCost']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AverageCostOfAttendance'] = pd.to_numeric(df['AverageCostOfAttendance'], errors='coerce')
        df = df.dropna(subset=['AverageCostOfAttendance'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_cost = df['AverageCostOfAttendance'].mean()
        most_expensive = df.loc[df['AverageCostOfAttendance'].idxmax(), 'Instnm']
        
        metrics = {
            "Average Cost of Attendance": avg_cost,
            "Most Expensive Institution": most_expensive
        }
        
        insights.append(f"Average Cost of Attendance: ${avg_cost:,.0f}")
        insights.append(f"Most Expensive Institution: {most_expensive}")

        fig1 = px.histogram(df, x='AverageCostOfAttendance', title='Distribution of Cost of Attendance')
        
        cost_by_state = df.groupby('State')['AverageCostOfAttendance'].mean().reset_index()
        fig2 = px.bar(cost_by_state.sort_values('AverageCostOfAttendance', ascending=False).head(20), x='State', y='AverageCostOfAttendance', title='Average Cost of Attendance by State (Top 20)')
        
        fig3 = px.box(df, x='State', y='AverageCostOfAttendance', title='Cost of Attendance by State')

        visualizations = {
            "Cost_of_Attendance_Distribution_Histogram": fig1.to_json(),
            "Average_Cost_by_State_Bar": fig2.to_json(),
            "Cost_by_State_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def state_level_average_cost_of_attendance_trend_analysis(df):
    print("\n--- State-Level Average Cost of Attendance Trend Analysis ---")
    analysis_type = "State-Level Average Cost of Attendance Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'STABBR': ['STABBR', 'StateAbbreviation', 'State'],
            'AverageCostOfAttendance': ['AverageCostOfAttendance', 'CostOfAttendance', 'TotalCost'],
            'Year': ['Year', 'AcademicYear']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AverageCostOfAttendance'] = pd.to_numeric(df['AverageCostOfAttendance'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['AverageCostOfAttendance', 'Year'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        df_avg = df.groupby(['STABBR', 'Year'])['AverageCostOfAttendance'].mean().reset_index()
        
        fig1 = px.line(df_avg, x='Year', y='AverageCostOfAttendance', color='STABBR', title='Average Cost of Attendance Trend by State')
        
        avg_cost_all_years = df.groupby('STABBR')['AverageCostOfAttendance'].mean().reset_index()
        fig2 = px.bar(avg_cost_all_years.sort_values('AverageCostOfAttendance', ascending=False).head(20), x='STABBR', y='AverageCostOfAttendance', title='Overall Average Cost of Attendance by State (Top 20)')
        
        fig3 = px.box(df, x='Year', y='AverageCostOfAttendance', title='Cost of Attendance Distribution by Year')

        visualizations = {
            "Average_Cost_of_Attendance_Trend_Line": fig1.to_json(),
            "Overall_Average_Cost_by_State_Bar": fig2.to_json(),
            "Cost_of_Attendance_Distribution_by_Year_Box": fig3.to_json()
        }
        
        insights.append("Trend analysis of cost of attendance by state and year generated.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_financials_and_student_outcome_analysis(df):
    print("\n--- University Financials and Student Outcome Analysis ---")
    analysis_type = "University Financials and Student Outcome Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'INSTNM': ['INSTNM', 'InstitutionName', 'Name'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'TuitionIncome': ['TuitionIncome', 'NetTuitionRevenue', 'TotalTuitionAndFees'],
            'CompletionRate': ['CompletionRate', 'GraduationRate', 'SuccessRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['TuitionIncome'] = pd.to_numeric(df['TuitionIncome'], errors='coerce')
        df['CompletionRate'] = pd.to_numeric(df['CompletionRate'], errors='coerce')
        df = df.dropna(subset=['TuitionIncome', 'CompletionRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_completion = df['CompletionRate'].mean()
        highest_tuition_income_inst = df.loc[df['TuitionIncome'].idxmax(), 'INSTNM']
        
        metrics = {
            "Average Completion Rate": avg_completion,
            "Highest Tuition Income Institution": highest_tuition_income_inst
        }
        
        insights.append(f"Average Completion Rate: {avg_completion:.2f}%")
        insights.append(f"Institution with Highest Tuition Income: {highest_tuition_income_inst}")

        fig1 = px.scatter(df, x='TuitionIncome', y='CompletionRate', hover_name='INSTNM', title='Completion Rate vs. Tuition Income')
        
        completion_by_state = df.groupby('State')['CompletionRate'].mean().reset_index()
        fig2 = px.bar(completion_by_state.sort_values('CompletionRate', ascending=False).head(20), x='State', y='CompletionRate', title='Average Completion Rate by State (Top 20)')
        
        fig3 = px.box(df, x='State', y='TuitionIncome', title='Tuition Income Distribution by State')

        visualizations = {
            "Completion_Rate_vs_Tuition_Income_Scatter": fig1.to_json(),
            "Average_Completion_Rate_by_State_Bar": fig2.to_json(),
            "Tuition_Income_Distribution_by_State_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_enrollment_expenditure_and_graduation_rate_analysis(df):
    print("\n--- University Enrollment, Expenditure, and Graduation Rate Analysis ---")
    analysis_type = "University Enrollment, Expenditure, and Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'INSTNM': ['INSTNM', 'InstitutionName', 'Name'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment'],
            'GraduationRate': ['GraduationRate', 'CompletionRate', 'GradRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['UndergraduateEnrollment', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_enrollment = df['UndergraduateEnrollment'].mean()
        avg_grad_rate = df['GraduationRate'].mean()
        
        metrics = {
            "Average Undergraduate Enrollment": avg_enrollment,
            "Average Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

        fig1 = px.scatter(df, x='UndergraduateEnrollment', y='GraduationRate', hover_name='INSTNM', title='Graduation Rate vs. Undergraduate Enrollment')
        
        grad_rate_by_state = df.groupby('State')['GraduationRate'].mean().reset_index()
        fig2 = px.bar(grad_rate_by_state.sort_values('GraduationRate', ascending=False).head(20), x='State', y='GraduationRate', title='Average Graduation Rate by State (Top 20)')
        
        fig3 = px.histogram(df, x='UndergraduateEnrollment', title='Distribution of Undergraduate Enrollment')

        visualizations = {
            "Graduation_Rate_vs_Enrollment_Scatter": fig1.to_json(),
            "Average_Graduation_Rate_by_State_Bar": fig2.to_json(),
            "Undergraduate_Enrollment_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_admissions_and_graduation_rate_analysis(df):
    print("\n--- College Admissions and Graduation Rate Analysis ---")
    analysis_type = "College Admissions and Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'INSTNM', 'Name'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'City': ['City', 'CampusCity'],
            'GraduationRate': ['GraduationRate', 'CompletionRate', 'GradRate'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
        df = df.dropna(subset=['GraduationRate', 'AcceptanceRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_grad_rate = df['GraduationRate'].mean()
        high_grad_rate_college = df.loc[df['GraduationRate'].idxmax(), 'InstitutionName']
        
        metrics = {
            "Average Graduation Rate": avg_grad_rate,
            "Highest Graduation Rate College": high_grad_rate_college
        }
        
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
        insights.append(f"College with Highest Graduation Rate: {high_grad_rate_college}")

        fig1 = px.bar(df.groupby('State')['GraduationRate'].mean().nlargest(20).reset_index(), x='State', y='GraduationRate', title='Average Graduation Rate by State (Top 20)')
        fig2 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')
        fig3 = px.box(df, x='State', y='GraduationRate', title='Graduation Rate Distribution by State')
        fig4 = px.scatter(df, x='AcceptanceRate', y='GraduationRate', hover_name='InstitutionName', title='Graduation Rate vs. Acceptance Rate')

        visualizations = {
            "Average_Graduation_Rate_by_State_Bar": fig1.to_json(),
            "Graduation_Rate_Distribution_Histogram": fig2.to_json(),
            "Graduation_Rate_Distribution_by_State_Box": fig3.to_json(),
            "Graduation_Rate_vs_Acceptance_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_level_student_teacher_ratio_and_class_size_analysis(df):
    print("\n--- School-Level Student-Teacher Ratio and Class Size Analysis ---")
    analysis_type = "School-Level Student-Teacher Ratio and Class Size Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
            'SchoolName': ['SchoolName', 'Name'],
            'StudentTeacherRatio': ['StudentTeacherRatio', 'STRatio', 'Ratio'],
            'AverageClassSize': ['AverageClassSize', 'AvgClassSize', 'ClassSize']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['StudentTeacherRatio'] = pd.to_numeric(df['StudentTeacherRatio'], errors='coerce')
        df['AverageClassSize'] = pd.to_numeric(df['AverageClassSize'], errors='coerce')
        df = df.dropna(subset=['StudentTeacherRatio', 'AverageClassSize'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_ratio = df['StudentTeacherRatio'].mean()
        min_ratio_school = df.loc[df['StudentTeacherRatio'].idxmin(), 'SchoolName']
        
        metrics = {
            "Average Student-Teacher Ratio": avg_ratio,
            "School with Lowest Ratio": min_ratio_school
        }
        
        insights.append(f"Average Student-Teacher Ratio: {avg_ratio:.2f}")
        insights.append(f"School with Lowest Student-Teacher Ratio: {min_ratio_school}")

        fig1 = px.histogram(df, x='StudentTeacherRatio', nbins=30, title='Distribution of Student-Teacher Ratios')
        fig2 = px.box(df, y='StudentTeacherRatio', title='Student-Teacher Ratio Distribution')
        fig3 = px.bar(df.sort_values('StudentTeacherRatio').head(20), x='SchoolName', y='StudentTeacherRatio', title='Schools with the Lowest Student-Teacher Ratios (Top 20)')
        fig4 = px.scatter(df, x='StudentTeacherRatio', y='AverageClassSize', hover_name='SchoolName', title='Average Class Size vs. Student-Teacher Ratio')

        visualizations = {
            "Student_Teacher_Ratio_Distribution_Histogram": fig1.to_json(),
            "Student_Teacher_Ratio_Distribution_Box": fig2.to_json(),
            "Lowest_Student_Teacher_Ratios_Bar": fig3.to_json(),
            "Class_Size_vs_Ratio_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_enrollment_and_income_trend_analysis(df):
    print("\n--- College Enrollment and Income Trend Analysis ---")
    analysis_type = "College Enrollment and Income Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeID': ['CollegeID', 'InstitutionID', 'ID'],
            'CollegeName': ['CollegeName', 'InstitutionName', 'Name'],
            'GraduateIncome': ['GraduateIncome', 'PostGradEarnings', 'MedianEarnings'],
            'EnrollmentYear': ['EnrollmentYear', 'Year', 'AcademicYear'],
            'TotalEnrollment': ['TotalEnrollment', 'Enrollment', 'StudentCount']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['GraduateIncome'] = pd.to_numeric(df['GraduateIncome'], errors='coerce')
        df['EnrollmentYear'] = pd.to_numeric(df['EnrollmentYear'], errors='coerce')
        df['TotalEnrollment'] = pd.to_numeric(df['TotalEnrollment'], errors='coerce')
        df = df.dropna(subset=['GraduateIncome', 'EnrollmentYear', 'TotalEnrollment'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_income = df['GraduateIncome'].mean()
        top_income_college = df.loc[df['GraduateIncome'].idxmax(), 'CollegeName']
        
        metrics = {
            "Average Graduate Income": avg_income,
            "Highest Graduate Income College": top_income_college
        }
        
        insights.append(f"Average Graduate Income: ${avg_income:,.0f}")
        insights.append(f"College with Highest Graduate Income: {top_income_college}")

        avg_income_by_year = df.groupby('EnrollmentYear')['GraduateIncome'].mean().reset_index()
        fig1 = px.line(avg_income_by_year, x='EnrollmentYear', y='GraduateIncome', title='Average Graduate Income Trend Over Time')
        
        fig2 = px.box(df, x='EnrollmentYear', y='GraduateIncome', title='Graduate Income Distribution by Enrollment Year')
        
        fig3 = px.bar(df.groupby('CollegeName')['GraduateIncome'].mean().nlargest(20).reset_index(), x='CollegeName', y='GraduateIncome', title='Top 20 Colleges by Average Graduate Income')
        
        enrollment_by_year = df.groupby('EnrollmentYear')['TotalEnrollment'].sum().reset_index()
        fig4 = px.line(enrollment_by_year, x='EnrollmentYear', y='TotalEnrollment', title='Total Enrollment Trend Over Time')

        visualizations = {
            "Average_Graduate_Income_Trend_Line": fig1.to_json(),
            "Graduate_Income_Distribution_by_Enrollment_Year_Box": fig2.to_json(),
            "Top_20_Colleges_by_Avg_Graduate_Income_Bar": fig3.to_json(),
            "Total_Enrollment_Trend_Line": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_resource_adequacy_analysis(df):
    print("\n--- School District Resource Adequacy Analysis ---")
    analysis_type = "School District Resource Adequacy Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictCode': ['DistrictCode', 'DistrictID', 'ID'],
            'DistrictName': ['DistrictName', 'Name'],
            'AdequacyIndex': ['AdequacyIndex', 'ResourceAdequacyScore', 'AdequacyScore'],
            'PerPupilExpenditure': ['PerPupilExpenditure', 'SpendingPerStudent']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AdequacyIndex'] = pd.to_numeric(df['AdequacyIndex'], errors='coerce')
        df['PerPupilExpenditure'] = pd.to_numeric(df['PerPupilExpenditure'], errors='coerce')
        df = df.dropna(subset=['AdequacyIndex', 'PerPupilExpenditure'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_index = df['AdequacyIndex'].mean()
        most_adequate_district = df.loc[df['AdequacyIndex'].idxmax(), 'DistrictName']
        
        metrics = {
            "Average Adequacy Index": avg_index,
            "Most Adequate District": most_adequate_district
        }
        
        insights.append(f"Average Adequacy Index: {avg_index:.2f}")
        insights.append(f"Most Adequate District: {most_adequate_district}")

        fig1 = px.histogram(df, x='AdequacyIndex', title='Distribution of Resource Adequacy Index')
        
        fig2 = px.bar(df.groupby('DistrictName')['AdequacyIndex'].mean().nlargest(20).reset_index(), x='DistrictName', y='AdequacyIndex', title='Top 20 Districts by Adequacy Index')
        
        fig3 = px.box(df, y='AdequacyIndex', title='Adequacy Index Distribution')
        
        fig4 = px.scatter(df, x='PerPupilExpenditure', y='AdequacyIndex', hover_name='DistrictName', title='Resource Adequacy Index vs. Per-Pupil Expenditure')

        visualizations = {
            "Resource_Adequacy_Index_Distribution_Histogram": fig1.to_json(),
            "Top_20_Districts_by_Adequacy_Index_Bar": fig2.to_json(),
            "Adequacy_Index_Distribution_Box": fig3.to_json(),
            "Adequacy_Index_vs_Per_Pupil_Expenditure_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def higher_education_institution_roi_and_default_rate_analysis(df):
    print("\n--- Higher Education Institution ROI and Default Rate Analysis ---")
    analysis_type = "Higher Education Institution ROI and Default Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'LoanDefaultRate': ['LoanDefaultRate', 'DefaultRate', 'ThreeYearDefaultRate'],
            'ROI': ['ROI', 'ReturnOnInvestment', 'MedianEarningsAfter10Years']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['LoanDefaultRate'] = pd.to_numeric(df['LoanDefaultRate'], errors='coerce')
        df['ROI'] = pd.to_numeric(df['ROI'], errors='coerce')
        df = df.dropna(subset=['LoanDefaultRate', 'ROI'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_default_rate = df['LoanDefaultRate'].mean()
        highest_default_rate_inst = df.loc[df['LoanDefaultRate'].idxmax(), 'Name']
        
        metrics = {
            "Average Loan Default Rate": avg_default_rate,
            "Highest Default Rate Institution": highest_default_rate_inst
        }
        
        insights.append(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
        insights.append(f"Institution with Highest Default Rate: {highest_default_rate_inst}")

        fig1 = px.histogram(df, x='LoanDefaultRate', title='Distribution of Loan Default Rates')
        fig2 = px.box(df, x='State', y='LoanDefaultRate', title='Loan Default Rate by State')
        
        default_rate_by_state = df.groupby('State')['LoanDefaultRate'].mean().reset_index()
        fig3 = px.bar(default_rate_by_state.sort_values('LoanDefaultRate', ascending=False).head(20), x='State', y='LoanDefaultRate', title='Average Loan Default Rate by State (Top 20)')
        
        fig4 = px.scatter(df, x='ROI', y='LoanDefaultRate', hover_name='Name', title='Loan Default Rate vs. ROI')

        visualizations = {
            "Loan_Default_Rate_Distribution_Histogram": fig1.to_json(),
            "Loan_Default_Rate_by_State_Box": fig2.to_json(),
            "Average_Loan_Default_Rate_by_State_Bar": fig3.to_json(),
            "Loan_Default_Rate_vs_ROI_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_budget_and_student_outcome_analysis(df):
    print("\n--- School District Budget and Student Outcome Analysis ---")
    analysis_type = "School District Budget and Student Outcome Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'School District': ['School District', 'DistrictName', 'District'],
            'County': ['County'],
            'DropoutRate': ['DropoutRate', 'HighSchoolDropoutRate'],
            'PerPupilExpenditure': ['PerPupilExpenditure', 'SpendingPerStudent', 'BudgetPerStudent']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['DropoutRate'] = pd.to_numeric(df['DropoutRate'], errors='coerce')
        df['PerPupilExpenditure'] = pd.to_numeric(df['PerPupilExpenditure'], errors='coerce')
        df = df.dropna(subset=['DropoutRate', 'PerPupilExpenditure'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_dropout_rate = df['DropoutRate'].mean()
        highest_dropout_district = df.loc[df['DropoutRate'].idxmax(), 'School District']
        
        metrics = {
            "Average Dropout Rate": avg_dropout_rate,
            "Highest Dropout Rate District": highest_dropout_district
        }
        
        insights.append(f"Average Dropout Rate: {avg_dropout_rate:.2f}%")
        insights.append(f"Highest Dropout Rate District: {highest_dropout_district}")

        fig1 = px.histogram(df, x='DropoutRate', title='Distribution of Dropout Rates')
        
        dropout_by_county = df.groupby('County')['DropoutRate'].mean().reset_index()
        fig2 = px.bar(dropout_by_county.sort_values('DropoutRate', ascending=False).head(20), x='County', y='DropoutRate', title='Average Dropout Rate by County (Top 20)')
        
        fig3 = px.box(df, x='County', y='DropoutRate', title='Dropout Rate Distribution by County')
        
        fig4 = px.scatter(df, x='PerPupilExpenditure', y='DropoutRate', hover_name='School District', title='Dropout Rate vs. Per-Pupil Expenditure')

        visualizations = {
            "Dropout_Rates_Distribution_Histogram": fig1.to_json(),
            "Average_Dropout_Rate_by_County_Bar": fig2.to_json(),
            "Dropout_Rate_Distribution_by_County_Box": fig3.to_json(),
            "Dropout_Rate_vs_Per_Pupil_Expenditure_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_selectivity_and_student_debt_analysis(df):
    print("\n--- University Selectivity and Student Debt Analysis ---")
    analysis_type = "University Selectivity and Student Debt Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'UniversityID': ['UniversityID', 'InstID', 'ID'],
            'Name': ['Name', 'UniversityName', 'INSTNM'],
            'PublicPrivate': ['PublicPrivate', 'Control', 'InstitutionType'],
            'AvgStudentDebt': ['AvgStudentDebt', 'MedianDebt', 'StudentLoanDebt'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AvgStudentDebt'] = pd.to_numeric(df['AvgStudentDebt'], errors='coerce')
        df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
        df = df.dropna(subset=['AvgStudentDebt', 'AcceptanceRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_debt = df['AvgStudentDebt'].mean()
        most_debt_university = df.loc[df['AvgStudentDebt'].idxmax(), 'Name']
        
        metrics = {
            "Average Student Debt": avg_debt,
            "Highest Student Debt University": most_debt_university
        }
        
        insights.append(f"Average Student Debt: ${avg_debt:,.0f}")
        insights.append(f"University with Highest Student Debt: {most_debt_university}")

        fig1 = px.histogram(df, x='AvgStudentDebt', title='Distribution of Average Student Debt')
        fig2 = px.box(df, x='PublicPrivate', y='AvgStudentDebt', title='Average Student Debt by Institution Type')
        
        avg_debt_by_type = df.groupby('PublicPrivate')['AvgStudentDebt'].mean().reset_index()
        fig3 = px.bar(avg_debt_by_type, x='PublicPrivate', y='AvgStudentDebt', title='Average Student Debt by Institution Type')
        
        fig4 = px.scatter(df, x='AcceptanceRate', y='AvgStudentDebt', hover_name='Name', title='Average Student Debt vs. Acceptance Rate (Selectivity)')

        visualizations = {
            "Avg_Student_Debt_Distribution_Histogram": fig1.to_json(),
            "Avg_Student_Debt_by_Institution_Type_Box": fig2.to_json(),
            "Avg_Student_Debt_by_Institution_Type_Bar": fig3.to_json(),
            "Student_Debt_vs_Acceptance_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_admissions_graduation_and_salary_outcome_analysis(df):
    print("\n--- College Admissions, Graduation, and Salary Outcome Analysis ---")
    analysis_type = "College Admissions, Graduation, and Salary Outcome Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeName': ['CollegeName', 'INSTNM', 'Name'],
            'ApplicationsReceived': ['ApplicationsReceived', 'Applicants', 'TotalApplicants'],
            'GraduationRate': ['GraduationRate', 'CompletionRate'],
            'MidCareerSalary': ['MidCareerSalary', 'PostGraduationEarnings', 'MedianSalary']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['MidCareerSalary'] = pd.to_numeric(df['MidCareerSalary'], errors='coerce')
        df = df.dropna(subset=['ApplicationsReceived', 'GraduationRate', 'MidCareerSalary'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_salary = df['MidCareerSalary'].mean()
        high_salary_college = df.loc[df['MidCareerSalary'].idxmax(), 'CollegeName']
        
        metrics = {
            "Average Mid-Career Salary": avg_salary,
            "Highest Mid-Career Salary College": high_salary_college
        }
        
        insights.append(f"Average Mid-Career Salary: ${avg_salary:,.0f}")
        insights.append(f"College with Highest Mid-Career Salary: {high_salary_college}")

        fig1 = px.scatter(df, x='ApplicationsReceived', y='MidCareerSalary', hover_name='CollegeName', title='Mid-Career Salary vs. Applications Received')
        fig2 = px.histogram(df, x='MidCareerSalary', title='Distribution of Mid-Career Salaries')
        fig3 = px.box(df, y='MidCareerSalary', title='Mid-Career Salary Distribution by College')
        fig4 = px.scatter(df, x='GraduationRate', y='MidCareerSalary', hover_name='CollegeName', title='Mid-Career Salary vs. Graduation Rate')

        visualizations = {
            "Mid_Career_Salary_vs_Applications_Received_Scatter": fig1.to_json(),
            "Mid_Career_Salaries_Distribution_Histogram": fig2.to_json(),
            "Mid_Career_Salary_Distribution_by_College_Box": fig3.to_json(),
            "Mid_Career_Salary_vs_Graduation_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_funding_and_local_income_level_analysis(df):
    print("\n--- School Funding and Local Income Level Analysis ---")
    analysis_type = "School Funding and Local Income Level Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
            'Name': ['Name', 'SchoolName'],
            'District': ['District', 'DistrictName', 'SchoolDistrict'],
            'MedianHouseholdIncome': ['MedianHouseholdIncome', 'LocalIncomeLevel', 'AvgHouseholdIncome'],
            'PerPupilFunding': ['PerPupilFunding', 'SpendingPerStudent', 'DistrictFunding']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['MedianHouseholdIncome'] = pd.to_numeric(df['MedianHouseholdIncome'], errors='coerce')
        df['PerPupilFunding'] = pd.to_numeric(df['PerPupilFunding'], errors='coerce')
        df = df.dropna(subset=['MedianHouseholdIncome', 'PerPupilFunding'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_income = df['MedianHouseholdIncome'].mean()
        high_income_school = df.loc[df['MedianHouseholdIncome'].idxmax(), 'Name']
        
        metrics = {
            "Average Median Household Income": avg_income,
            "Highest Income School": high_income_school
        }
        
        insights.append(f"Average Median Household Income: ${avg_income:,.0f}")
        insights.append(f"School in Highest Income Area: {high_income_school}")

        fig1 = px.histogram(df, x='MedianHouseholdIncome', title='Distribution of Median Household Income')
        fig2 = px.box(df, y='MedianHouseholdIncome', title='Median Household Income Distribution')
        
        income_by_district = df.groupby('District')['MedianHouseholdIncome'].mean().reset_index()
        fig3 = px.bar(income_by_district.sort_values('MedianHouseholdIncome', ascending=False).head(20), x='District', y='MedianHouseholdIncome', title='Average Median Household Income by District (Top 20)')
        
        fig4 = px.scatter(df, x='MedianHouseholdIncome', y='PerPupilFunding', hover_name='Name', title='Per-Pupil Funding vs. Median Household Income')

        visualizations = {
            "Median_Household_Income_Distribution_Histogram": fig1.to_json(),
            "Median_Household_Income_Distribution_Box": fig2.to_json(),
            "Average_Median_Household_Income_by_District_Bar": fig3.to_json(),
            "Per_Pupil_Funding_vs_Median_Household_Income_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def pell_grant_recipient_graduation_and_loan_default_rate_analysis(df):
    print("\n--- Pell Grant Recipient Graduation and Loan Default Rate Analysis ---")
    analysis_type = "Pell Grant Recipient Graduation and Loan Default Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'Institution': ['Institution', 'INSTNM', 'Name'],
            'State': ['State', 'STABBR'],
            'Year': ['Year', 'AcademicYear'],
            'PellGradRate': ['PellGradRate', 'PellRecipientGraduationRate'],
            'OverallGradRate': ['OverallGradRate', 'GraduationRate'],
            'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result

        df['PellGradRate'] = pd.to_numeric(df['PellGradRate'], errors='coerce')
        df['OverallGradRate'] = pd.to_numeric(df['OverallGradRate'], errors='coerce')
        df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['PellGradRate', 'OverallGradRate', 'DefaultRate', 'Year'])
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
        avg_pell_grad_rate = df['PellGradRate'].mean()
        avg_overall_grad_rate = df['OverallGradRate'].mean()
        avg_default_rate = df['DefaultRate'].mean()
        
        metrics = {
            "Average Pell Grant Graduation Rate": avg_pell_grad_rate,
            "Average Overall Graduation Rate": avg_overall_grad_rate,
            "Average Default Rate": avg_default_rate
        }
        
        insights.append(f"Average Pell Grant Recipient Graduation Rate: {avg_pell_grad_rate:.2f}%")
        insights.append(f"Average Overall Graduation Rate: {avg_overall_grad_rate:.2f}%")
        insights.append(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
        
        fig1 = px.line(df.groupby('Year').agg({'PellGradRate': 'mean', 'OverallGradRate': 'mean', 'DefaultRate': 'mean'}).reset_index(),
                       x='Year', y=['PellGradRate', 'OverallGradRate', 'DefaultRate'],
                       title='Pell Grant Recipient Trends Over Time')

        fig2 = px.bar(df.groupby('State')['DefaultRate'].mean().nlargest(20).reset_index(), x='State', y='DefaultRate', title='Average Default Rate by State (Top 20)')
        
        fig3 = px.box(df, x='Year', y='DefaultRate', title='Default Rate Distribution by Year')

        visualizations = {
            "Pell_Grant_Recipient_Trends_Line": fig1.to_json(),
            "Average_Default_Rate_by_State_Bar": fig2.to_json(),
            "Default_Rate_Distribution_by_Year_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_selectivity_and_graduation_rate_analysis(df):
    print("\n--- College Selectivity and Graduation Rate Analysis ---")
    analysis_type = "College Selectivity and Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeID': ['CollegeID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'State': ['State', 'STABBR', 'InstitutionState'],
            'GraduationRate': ['GraduationRate', 'CompletionRate'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
            'AvgSAT': ['AvgSAT', 'SATScore', 'AverageSATScore']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        # This analysis can run with *either* AcceptanceRate or AvgSAT.
        # Check if at least one selectivity metric is present.
        has_acceptance = 'AcceptanceRate' in df.columns
        has_sat = 'AvgSAT' in df.columns
        
        if 'GraduationRate' not in df.columns or not (has_acceptance or has_sat):
             # Rerun missing list to be accurate
            _, missing_strict = check_and_rename_columns(df, {'GraduationRate': expected['GraduationRate'], 'AcceptanceRate': expected['AcceptanceRate']})
            show_missing_columns_warning(missing_strict, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing 'GraduationRate' or a selectivity metric ('AcceptanceRate' or 'AvgSAT').")
            return fallback_result
        
        if has_acceptance:
            df['SelectivityScore'] = 1 - pd.to_numeric(df['AcceptanceRate'], errors='coerce')
            insights.append("Using (1 - AcceptanceRate) as Selectivity Score.")
        elif has_sat:
            df['SelectivityScore'] = pd.to_numeric(df['AvgSAT'], errors='coerce')
            insights.append("Using AvgSAT as Selectivity Score.")
        
        df = safe_numeric_conversion(df, 'SelectivityScore')
        df = safe_numeric_conversion(df, 'GraduationRate')
        df = df.dropna(subset=['SelectivityScore', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_grad_rate = df['GraduationRate'].mean()
        high_grad_rate_college = df.loc[df['GraduationRate'].idxmax(), 'Name']
        
        metrics = {
            "Average Graduation Rate": avg_grad_rate,
            "Highest Graduation Rate College": high_grad_rate_college
        }
        
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
        insights.append(f"College with Highest Graduation Rate: {high_grad_rate_college}")

        fig1 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')
        fig2 = px.box(df, x='State', y='GraduationRate', title='Graduation Rate by State')
        fig3 = px.bar(df.groupby('State')['GraduationRate'].mean().nlargest(20).reset_index(), x='State', y='GraduationRate', title='Top 20 States by Average Graduation Rate')
        fig4 = px.scatter(df, x='SelectivityScore', y='GraduationRate', hover_name='Name', title='Graduation Rate vs. College Selectivity')

        visualizations = {
            "Graduation_Rates_Distribution_Histogram": fig1.to_json(),
            "Graduation_Rate_by_State_Box": fig2.to_json(),
            "Top_20_States_by_Avg_Graduation_Rate_Bar": fig3.to_json(),
            "Graduation_Rate_vs_Selectivity_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_demographics_and_student_teacher_ratio_analysis(df):
    print("\n--- School District Demographics and Student-Teacher Ratio Analysis ---")
    analysis_type = "School District Demographics and Student-Teacher Ratio Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
            'DistrictName': ['DistrictName', 'Name'],
            'State': ['State', 'STABBR'],
            'FreeLunchRate': ['FreeLunchRate', 'FreeReducedLunchRate', 'PovertyRate'],
            'StudentCount': ['StudentCount', 'TotalEnrollment', 'Enrollment'],
            'TeacherCount': ['TeacherCount', 'TotalTeachers', 'FTETeachers']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['FreeLunchRate'] = pd.to_numeric(df['FreeLunchRate'], errors='coerce')
        df['StudentCount'] = pd.to_numeric(df['StudentCount'], errors='coerce')
        df['TeacherCount'] = pd.to_numeric(df['TeacherCount'], errors='coerce')
        df = df.dropna(subset=['FreeLunchRate', 'StudentCount', 'TeacherCount'])
        
        df['StudentTeacherRatio'] = df.apply(
            lambda row: row['StudentCount'] / row['TeacherCount'] if row['TeacherCount'] > 0 else np.nan,
            axis=1
        )
        df = df.dropna(subset=['StudentTeacherRatio'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_free_lunch_rate = df['FreeLunchRate'].mean()
        high_need_district = df.loc[df['FreeLunchRate'].idxmax(), 'DistrictName']
        avg_student_teacher_ratio = df['StudentTeacherRatio'].mean()

        metrics = {
            "Average Free Lunch Rate": avg_free_lunch_rate,
            "Highest Need District": high_need_district,
            "Average Student-Teacher Ratio": avg_student_teacher_ratio
        }
        
        insights.append(f"Average Free Lunch Rate: {avg_free_lunch_rate:.2f}%")
        insights.append(f"District with Highest Free Lunch Rate: {high_need_district}")
        insights.append(f"Average Student-Teacher Ratio: {avg_student_teacher_ratio:.2f}")

        fig1 = px.histogram(df, x='FreeLunchRate', title='Distribution of Free Lunch Rates')
        fig2 = px.box(df, x='State', y='FreeLunchRate', title='Free Lunch Rate by State')
        fig3 = px.bar(df.groupby('DistrictName')['FreeLunchRate'].mean().nlargest(20).reset_index(), x='DistrictName', y='FreeLunchRate', title='Top 20 Districts by Free Lunch Rate')
        fig4 = px.scatter(df, x='FreeLunchRate', y='StudentTeacherRatio', hover_name='DistrictName', title='Student-Teacher Ratio vs. Free Lunch Rate')

        visualizations = {
            "Free_Lunch_Rates_Distribution_Histogram": fig1.to_json(),
            "Free_Lunch_Rate_by_State_Box": fig2.to_json(),
            "Top_20_Districts_by_Free_Lunch_Rate_Bar": fig3.to_json(),
            "Student_Teacher_Ratio_vs_Free_Lunch_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_tuition_and_enrollment_statistics_analysis(df):
    print("\n--- College Tuition and Enrollment Statistics Analysis ---")
    analysis_type = "College Tuition and Enrollment Statistics Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
            'Enrollment': ['Enrollment', 'TotalEnrollment', 'UndergraduateEnrollment'],
            'GraduationRate': ['GraduationRate', 'CompletionRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['Tuition', 'Enrollment', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_tuition = df['Tuition'].mean()
        highest_tuition_college = df.loc[df['Tuition'].idxmax(), 'Name']
        
        metrics = {
            "Average Tuition": avg_tuition,
            "Highest Tuition College": highest_tuition_college
        }
        
        insights.append(f"Average Tuition: ${avg_tuition:,.0f}")
        insights.append(f"College with Highest Tuition: {highest_tuition_college}")

        fig1 = px.scatter(df, x='Tuition', y='GraduationRate', hover_name='Name', title='Graduation Rate vs. Tuition')
        fig2 = px.histogram(df, x='Tuition', title='Distribution of Tuition')
        fig3 = px.box(df, y='Tuition', title='Tuition Distribution')
        fig4 = px.scatter(df, x='Enrollment', y='Tuition', hover_name='Name', title='Tuition vs. Enrollment')

        visualizations = {
            "Graduation_Rate_vs_Tuition_Scatter": fig1.to_json(),
            "Tuition_Distribution_Histogram": fig2.to_json(),
            "Tuition_Distribution_Box": fig3.to_json(),
            "Tuition_vs_Enrollment_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_special_needs_and_counselor_ratio_analysis(df):
    print("\n--- School Special Needs and Counselor Ratio Analysis ---")
    analysis_type = "School Special Needs and Counselor Ratio Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
            'Name': ['Name', 'SchoolName'],
            'State': ['State', 'STABBR'],
            'SpecialNeedsStudentCount': ['SpecialNeedsStudentCount', 'StudentsWithDisabilities', 'SpEdStudents'],
            'CounselorCount': ['CounselorCount', 'FTECounselors', 'TotalCounselors']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['SpecialNeedsStudentCount'] = pd.to_numeric(df['SpecialNeedsStudentCount'], errors='coerce')
        df['CounselorCount'] = pd.to_numeric(df['CounselorCount'], errors='coerce')
        df = df.dropna(subset=['SpecialNeedsStudentCount', 'CounselorCount'])

        df['SpecialNeedsCounselorRatio'] = df.apply(
            lambda row: row['SpecialNeedsStudentCount'] / row['CounselorCount'] if row['CounselorCount'] > 0 else np.nan,
            axis=1
        )
        df = df.dropna(subset=['SpecialNeedsCounselorRatio'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_ratio = df['SpecialNeedsCounselorRatio'].mean()
        lowest_ratio_school = df.loc[df['SpecialNeedsCounselorRatio'].idxmin(), 'Name']
        
        metrics = {
            "Average Special Needs Student to Counselor Ratio": avg_ratio,
            "Lowest Ratio School": lowest_ratio_school
        }
        
        insights.append(f"Average Special Needs Student to Counselor Ratio: {avg_ratio:.2f}:1")
        insights.append(f"School with Lowest Ratio: {lowest_ratio_school}")

        fig1 = px.histogram(df, x='SpecialNeedsCounselorRatio', title='Distribution of Special Needs Student-Counselor Ratios')
        fig2 = px.box(df, x='State', y='SpecialNeedsCounselorRatio', title='Special Needs Student-Counselor Ratio by State')
        fig3 = px.bar(df.groupby('State')['SpecialNeedsCounselorRatio'].mean().nlargest(20).reset_index(), x='State', y='SpecialNeedsCounselorRatio', title='Top 20 States by Average Special Needs Student-Counselor Ratio')

        visualizations = {
            "Special_Needs_Counselor_Ratio_Distribution_Histogram": fig1.to_json(),
            "Special_Needs_Counselor_Ratio_by_State_Box": fig2.to_json(),
            "Top_20_States_by_Avg_Special_Needs_Counselor_Ratio_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_graduation_rate_and_diversity_index_analysis(df):
    print("\n--- University Graduation Rate and Diversity Index Analysis ---")
    analysis_type = "University Graduation Rate and Diversity Index Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'UniversityState': ['UniversityState', 'State', 'STABBR'],
            'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
            'StudentDiversityIndex': ['StudentDiversityIndex', 'DiversityIndex', 'RacialEthnicDiversity'],
            'GraduationRate': ['GraduationRate', 'CompletionRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['StudentDiversityIndex'] = pd.to_numeric(df['StudentDiversityIndex'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['StudentDiversityIndex', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_diversity = df['StudentDiversityIndex'].mean()
        most_diverse_university = df.loc[df['StudentDiversityIndex'].idxmax(), 'UniversityName']
        
        metrics = {
            "Average Student Diversity Index": avg_diversity,
            "Most Diverse University": most_diverse_university
        }
        
        insights.append(f"Average Student Diversity Index: {avg_diversity:.2f}")
        insights.append(f"Most Diverse University: {most_diverse_university}")

        fig1 = px.scatter(df, x='StudentDiversityIndex', y='GraduationRate', hover_name='UniversityName', title='Graduation Rate vs. Student Diversity Index')
        fig2 = px.box(df, x='UniversityState', y='StudentDiversityIndex', title='Diversity Index by State')
        fig3 = px.histogram(df, x='GraduationRate', color='UniversityState', title='Graduation Rate Distribution by State')

        visualizations = {
            "Graduation_Rate_vs_Diversity_Index_Scatter": fig1.to_json(),
            "Diversity_Index_by_State_Box": fig2.to_json(),
            "Graduation_Rate_Distribution_by_State_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def post_graduation_earnings_and_debt_analysis(df):
    print("\n--- Post-Graduation Earnings and Debt Analysis ---")
    analysis_type = "Post-Graduation Earnings and Debt Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'GraduationRate': ['GraduationRate', 'CompletionRate'],
            'MedianEarnings': ['MedianEarnings', 'PostGradEarnings', 'AvgSalary'],
            'MedianDebt': ['MedianDebt', 'AvgDebtUponGraduation', 'StudentLoanDebt']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['MedianEarnings'] = pd.to_numeric(df['MedianEarnings'], errors='coerce')
        df['MedianDebt'] = pd.to_numeric(df['MedianDebt'], errors='coerce')
        df = df.dropna(subset=['GraduationRate', 'MedianEarnings', 'MedianDebt'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_earnings = df['MedianEarnings'].mean()
        highest_earnings_institution = df.loc[df['MedianEarnings'].idxmax(), 'InstitutionName']
        avg_debt = df['MedianDebt'].mean()
        
        metrics = {
            "Average Median Earnings": avg_earnings,
            "Highest Earnings Institution": highest_earnings_institution,
            "Average Median Debt": avg_debt
        }
        
        insights.append(f"Average Median Earnings: ${avg_earnings:,.0f}")
        insights.append(f"Highest Earnings Institution: {highest_earnings_institution}")
        insights.append(f"Average Median Debt: ${avg_debt:,.0f}")

        fig1 = px.scatter(df, x='GraduationRate', y='MedianEarnings', hover_name='InstitutionName', title='Median Earnings vs. Graduation Rate')
        fig2 = px.histogram(df, x='MedianEarnings', title='Distribution of Median Earnings')
        fig3 = px.box(df, y='MedianEarnings', title='Median Earnings Distribution')
        fig4 = px.scatter(df, x='MedianDebt', y='MedianEarnings', hover_name='InstitutionName', title='Median Earnings vs. Median Debt')
        fig5 = px.histogram(df, x='MedianDebt', title='Distribution of Median Debt')

        visualizations = {
            "Median_Earnings_vs_Graduation_Rate_Scatter": fig1.to_json(),
            "Median_Earnings_Distribution_Histogram": fig2.to_json(),
            "Median_Earnings_Distribution_Box": fig3.to_json(),
            "Median_Earnings_vs_Median_Debt_Scatter": fig4.to_json(),
            "Median_Debt_Distribution_Histogram": fig5.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_test_score_and_graduation_rate_analysis(df):
    print("\n--- School District Test Score and Graduation Rate Analysis ---")
    analysis_type = "School District Test Score and Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictName': ['DistrictName', 'Name'],
            'SchoolDistrictID': ['SchoolDistrictID', 'DistrictID', 'ID'],
            'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate'],
            'StateTestScores': ['StateTestScores', 'AverageTestScore', 'DistrictAvgScore']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['StateTestScores'] = pd.to_numeric(df['StateTestScores'], errors='coerce')
        df = df.dropna(subset=['GraduationRate', 'StateTestScores'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_grad_rate = df['GraduationRate'].mean()
        avg_test_score = df['StateTestScores'].mean()

        metrics = {
            "Average Graduation Rate": avg_grad_rate,
            "Average State Test Score": avg_test_score
        }
        
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
        insights.append(f"Average State Test Score: {avg_test_score:.2f}")

        fig1 = px.scatter(df, x='StateTestScores', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. State Test Scores')
        fig2 = px.box(df, x='DistrictName', y='GraduationRate', title='Graduation Rate by School District')
        fig3 = px.histogram(df, x='StateTestScores', title='Distribution of State Test Scores')

        visualizations = {
            "Graduation_Rate_vs_State_Test_Scores_Scatter": fig1.to_json(),
            "Graduation_Rate_by_School_District_Box": fig2.to_json(),
            "State_Test_Scores_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_admissions_and_loan_default_rate_correlation(df):
    print("\n--- College Admissions and Loan Default Rate Correlation ---")
    analysis_type = "College Admissions and Loan Default Rate Correlation"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
            'InstitutionState': ['InstitutionState', 'State', 'STABBR'],
            'LoanDefaultRate': ['LoanDefaultRate', 'DefaultRate', 'ThreeYearDefaultRate'],
            'AdmissionsRate': ['AdmissionsRate', 'AcceptanceRate', 'AdmitRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['LoanDefaultRate'] = pd.to_numeric(df['LoanDefaultRate'], errors='coerce')
        df['AdmissionsRate'] = pd.to_numeric(df['AdmissionsRate'], errors='coerce')
        df = df.dropna(subset=['LoanDefaultRate', 'AdmissionsRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_default_rate = df['LoanDefaultRate'].mean()
        avg_admissions_rate = df['AdmissionsRate'].mean()

        metrics = {
            "Average Loan Default Rate": avg_default_rate,
            "Average Admissions Rate": avg_admissions_rate
        }
        
        insights.append(f"Average Loan Default Rate: {avg_default_rate:.2f}%")
        insights.append(f"Average Admissions Rate: {avg_admissions_rate:.2f}%")

        fig1 = px.scatter(df, x='AdmissionsRate', y='LoanDefaultRate', hover_name='CollegeName', title='Loan Default Rate vs. Admissions Rate')
        fig2 = px.box(df, x='InstitutionState', y='LoanDefaultRate', title='Loan Default Rate by State')
        fig3 = px.box(df, y='AdmissionsRate', title='Admissions Rate Distribution')

        visualizations = {
            "Loan_Default_Rate_vs_Admissions_Rate_Scatter": fig1.to_json(),
            "Loan_Default_Rate_by_State_Box": fig2.to_json(),
            "Admissions_Rate_Distribution_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_admissions_funnel_and_yield_rate_analysis(df):
    print("\n--- College Admissions Funnel and Yield Rate Analysis ---")
    analysis_type = "College Admissions Funnel and Yield Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
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
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
        df['Admitted'] = pd.to_numeric(df['Admitted'], errors='coerce')
        df['Enrolled'] = pd.to_numeric(df['Enrolled'], errors='coerce')
        df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
        df['Graduation'] = pd.to_numeric(df['Graduation'], errors='coerce')
        df = df.dropna(subset=['ApplicationsReceived', 'Admitted', 'Enrolled', 'Yield', 'Graduation'])
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
        avg_yield = df['Yield'].mean()
        avg_graduation = df['Graduation'].mean()
        
        insights.append(f"Average Yield Rate: {avg_yield:.2f}%")
        insights.append(f"Average Graduation Rate: {avg_graduation:.2f}%")
        
        total_applications = df['ApplicationsReceived'].sum()
        total_admitted = df['Admitted'].sum()
        total_enrolled = df['Enrolled'].sum()

        metrics = {
            "Average Yield Rate": avg_yield,
            "Average Graduation Rate": avg_graduation,
            "Total Applications": total_applications,
            "Total Admitted": total_admitted,
            "Total Enrolled": total_enrolled,
            "Overall Acceptance Rate": total_admitted / total_applications if total_applications > 0 else 0,
            "Overall Yield Rate": total_enrolled / total_admitted if total_admitted > 0 else 0
        }

        funnel_data = pd.DataFrame({
            'Stage': ['Applications Received', 'Admitted', 'Enrolled'],
            'Count': [total_applications, total_admitted, total_enrolled]
        })
        
        fig1 = px.funnel(funnel_data, x='Count', y='Stage', title='Overall College Admissions Funnel')

        fig2 = px.scatter(df, x='Yield', y='Graduation', hover_name='InstitutionName', title='Graduation Rate vs. Yield Rate')
        fig3 = px.box(df, x='State', y='Yield', title='Yield Rate by State')
        fig4 = px.histogram(df, x='Yield', color='State', title='Yield Rate Distribution by State')

        visualizations = {
            "Overall_College_Admissions_Funnel": fig1.to_json(),
            "Graduation_Rate_vs_Yield_Rate_Scatter": fig2.to_json(),
            "Yield_Rate_by_State_Box": fig3.to_json(),
            "Yield_Rate_Distribution_by_State_Histogram": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_board_spending_and_student_achievement_analysis(df):
    print("\n--- School Board Spending and Student Achievement Analysis ---")
    analysis_type = "School Board Spending and Student Achievement Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolBoardID': ['SchoolBoardID', 'BoardID', 'ID'],
            'DistrictName': ['DistrictName', 'Name', 'SchoolDistrict'],
            'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate'],
            'PerStudentSpending': ['PerStudentSpending', 'SpendingPerStudent', 'BudgetPerStudent']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['PerStudentSpending'] = pd.to_numeric(df['PerStudentSpending'], errors='coerce')
        df = df.dropna(subset=['GraduationRate', 'PerStudentSpending'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
        avg_spending = df['PerStudentSpending'].mean()
        avg_grad_rate = df['GraduationRate'].mean()
        
        metrics = {
            "Average Per-Student Spending": avg_spending,
            "Average Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average Per-Student Spending: ${avg_spending:,.0f}")
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
        
        fig1 = px.scatter(df, x='PerStudentSpending', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. Per-Student Spending')
        fig2 = px.bar(df.groupby('DistrictName')['PerStudentSpending'].mean().nlargest(20).reset_index(), x='DistrictName', y='PerStudentSpending', title='Average Per-Student Spending by District (Top 20)')
        fig3 = px.histogram(df, x='GraduationRate', title='Distribution of Graduation Rates')

        visualizations = {
            "Graduation_Rate_vs_Per_Student_Spending_Scatter": fig1.to_json(),
            "Average_Per_Student_Spending_by_District_Bar": fig2.to_json(),
            "Graduation_Rates_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_enrollment_and_earnings_outcome_analysis(df):
    print("\n--- University Enrollment and Earnings Outcome Analysis ---")
    analysis_type = "University Enrollment and Earnings Outcome Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'Region': ['Region', 'State'],
            'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment'],
            'Earnings25thPercentile': ['Earnings25thPercentile', 'MedianEarnings', 'PostGradEarnings']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
        df['Earnings25thPercentile'] = pd.to_numeric(df['Earnings25thPercentile'], errors='coerce')
        df = df.dropna(subset=['UndergraduateEnrollment', 'Earnings25thPercentile'])
        
        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
        avg_enrollment = df['UndergraduateEnrollment'].mean()
        avg_earnings = df['Earnings25thPercentile'].mean()
        
        metrics = {
            "Average Undergraduate Enrollment": avg_enrollment,
            "Average Earnings (25th Percentile)": avg_earnings
        }
        
        insights.append(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
        insights.append(f"Average Earnings (25th Percentile): ${avg_earnings:,.0f}")
        
        fig1 = px.scatter(df, x='UndergraduateEnrollment', y='Earnings25thPercentile', hover_name='Name', title='Earnings vs. Undergraduate Enrollment')
        fig2 = px.box(df, x='Region', y='Earnings25thPercentile', title='Earnings Distribution by Region')
        fig3 = px.bar(df.groupby('Region')['UndergraduateEnrollment'].sum().reset_index(), x='Region', y='UndergraduateEnrollment', title='Total Enrollment by Region')

        visualizations = {
            "Earnings_vs_Undergraduate_Enrollment_Scatter": fig1.to_json(),
            "Earnings_Distribution_by_Region_Box": fig2.to_json(),
            "Total_Enrollment_by_Region_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_retention_debt_and_earnings_analysis(df):
    print("\n--- College Retention, Debt, and Earnings Analysis ---")
    analysis_type = "College Retention, Debt, and Earnings Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'State': ['State', 'STABBR'],
            'RetentionRate': ['RetentionRate', 'FirstYearRetention'],
            'StudentDebt': ['StudentDebt', 'AvgStudentDebt', 'MedianDebt'],
            '10YearEarnings': ['10YearEarnings', 'PostGradEarnings', 'MedianEarnings']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result

        df['RetentionRate'] = pd.to_numeric(df['RetentionRate'], errors='coerce')
        df['StudentDebt'] = pd.to_numeric(df['StudentDebt'], errors='coerce')
        df['10YearEarnings'] = pd.to_numeric(df['10YearEarnings'], errors='coerce')
        df = df.dropna(subset=['RetentionRate', 'StudentDebt', '10YearEarnings'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_earnings = df['10YearEarnings'].mean()
        avg_debt = df['StudentDebt'].mean()
        
        metrics = {
            "Average 10-Year Earnings": avg_earnings,
            "Average Student Debt": avg_debt
        }
        
        insights.append(f"Average 10-Year Earnings: ${avg_earnings:,.0f}")
        insights.append(f"Average Student Debt: ${avg_debt:,.0f}")
        
        fig1 = px.scatter(df, x='StudentDebt', y='10YearEarnings', hover_name='InstitutionName', title='10-Year Earnings vs. Student Debt')
        fig2 = px.box(df, x='State', y='10YearEarnings', title='10-Year Earnings Distribution by State')
        fig3 = px.box(df, x='State', y='StudentDebt', title='Student Debt Distribution by State')
        fig4 = px.scatter(df, x='RetentionRate', y='10YearEarnings', hover_name='InstitutionName', title='10-Year Earnings vs. Retention Rate')
        fig5 = px.scatter(df, x='RetentionRate', y='StudentDebt', hover_name='InstitutionName', title='Student Debt vs. Retention Rate')

        visualizations = {
            "10_Year_Earnings_vs_Student_Debt_Scatter": fig1.to_json(),
            "10_Year_Earnings_Distribution_by_State_Box": fig2.to_json(),
            "Student_Debt_Distribution_by_State_Box": fig3.to_json(),
            "10_Year_Earnings_vs_Retention_Rate_Scatter": fig4.to_json(),
            "Student_Debt_vs_Retention_Rate_Scatter": fig5.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_enrollment_and_disadvantaged_student_population_analysis(df):
    print("\n--- School Enrollment and Disadvantaged Student Population Analysis ---")
    analysis_type = "School Enrollment and Disadvantaged Student Population Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
            'District': ['District', 'DistrictName', 'SchoolDistrict'],
            'Enrollment': ['Enrollment', 'TotalEnrollment', 'StudentCount'],
            'EconomicallyDisadvantagedRate': ['EconomicallyDisadvantagedRate', 'FreeReducedLunchRate', 'PovertyRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['EconomicallyDisadvantagedRate'] = pd.to_numeric(df['EconomicallyDisadvantagedRate'], errors='coerce')
        df = df.dropna(subset=['Enrollment', 'EconomicallyDisadvantagedRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_disadvantaged_rate = df['EconomicallyDisadvantagedRate'].mean()
        high_disadvantaged_school = df.loc[df['EconomicallyDisadvantagedRate'].idxmax(), 'SchoolID']

        metrics = {
            "Average Disadvantaged Rate": avg_disadvantaged_rate,
            "Highest Disadvantaged School": high_disadvantaged_school
        }
        
        insights.append(f"Average Disadvantaged Rate: {avg_disadvantaged_rate:.2f}%")
        insights.append(f"School with Highest Disadvantaged Rate: {high_disadvantaged_school}")

        fig1 = px.scatter(df, x='Enrollment', y='EconomicallyDisadvantagedRate', hover_name='SchoolID', title='Disadvantaged Rate vs. Enrollment')
        fig2 = px.histogram(df, x='EconomicallyDisadvantagedRate', title='Distribution of Economically Disadvantaged Rate')
        
        disadvantaged_by_district = df.groupby('District')['EconomicallyDisadvantagedRate'].mean().reset_index()
        fig3 = px.bar(disadvantaged_by_district.sort_values('EconomicallyDisadvantagedRate', ascending=False).head(20), x='District', y='EconomicallyDisadvantagedRate', title='Average Disadvantaged Rate by District (Top 20)')

        visualizations = {
            "Disadvantaged_Rate_vs_Enrollment_Scatter": fig1.to_json(),
            "Economically_Disadvantaged_Rate_Distribution_Histogram": fig2.to_json(),
            "Average_Disadvantaged_Rate_by_District_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_enrollment_and_faculty_count_analysis(df):
    print("\n--- University Enrollment and Faculty Count Analysis ---")
    analysis_type = "University Enrollment and Faculty Count Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'UniversityID': ['UniversityID', 'InstID', 'ID'],
            'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
            'Enrollment': ['Enrollment', 'TotalEnrollment', 'UndergraduateEnrollment'],
            'FacultyCount': ['FacultyCount', 'TotalFaculty', 'FTFaculty']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['FacultyCount'] = pd.to_numeric(df['FacultyCount'], errors='coerce')
        df = df.dropna(subset=['Enrollment', 'FacultyCount'])
        
        df['student_faculty_ratio'] = df.apply(
            lambda row: row['Enrollment'] / row['FacultyCount'] if row['FacultyCount'] > 0 else np.nan,
            axis=1
        )
        df = df.dropna(subset=['student_faculty_ratio'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_faculty_count = df['FacultyCount'].mean()
        avg_enrollment = df['Enrollment'].mean()
        avg_student_faculty_ratio = df['student_faculty_ratio'].mean()

        metrics = {
            "Average Faculty Count": avg_faculty_count,
            "Average Enrollment": avg_enrollment,
            "Average Student-Faculty Ratio": avg_student_faculty_ratio
        }
        
        insights.append(f"Average Faculty Count: {avg_faculty_count:,.0f}")
        insights.append(f"Average Enrollment: {avg_enrollment:,.0f}")
        insights.append(f"Average Student-Faculty Ratio: {avg_student_faculty_ratio:.2f}:1")

        fig1 = px.scatter(df, x='Enrollment', y='FacultyCount', hover_name='UniversityName', title='Faculty Count vs. Enrollment')
        fig2 = px.histogram(df, x='FacultyCount', title='Distribution of Faculty Count')
        fig3 = px.box(df, y='student_faculty_ratio', title='Student-Faculty Ratio Distribution')

        visualizations = {
            "Faculty_Count_vs_Enrollment_Scatter": fig1.to_json(),
            "Faculty_Count_Distribution_Histogram": fig2.to_json(),
            "Student_Faculty_Ratio_Distribution_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def state_education_system_performance_analysis(df):
    print("\n--- State Education System Performance Analysis ---")
    analysis_type = "State Education System Performance Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'State': ['State', 'STABBR', 'StateName'],
            'SchoolCount': ['SchoolCount', 'TotalSchoolsInState'],
            'FinancialAidRate': ['FinancialAidRate', 'AvgFinancialAidRate', 'StateFinancialAidPct'],
            'GraduationRate': ['GraduationRate', 'StateGraduationRate'],
            'AvgTestScore': ['AvgTestScore', 'StateAvgTestScore']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['SchoolCount'] = pd.to_numeric(df['SchoolCount'], errors='coerce')
        df['FinancialAidRate'] = pd.to_numeric(df['FinancialAidRate'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df['AvgTestScore'] = pd.to_numeric(df['AvgTestScore'], errors='coerce')
        df = df.dropna(subset=['SchoolCount', 'FinancialAidRate', 'GraduationRate', 'AvgTestScore'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        total_schools = df['SchoolCount'].sum()
        avg_aid_rate = df['FinancialAidRate'].mean()
        avg_state_grad_rate = df['GraduationRate'].mean()
        avg_state_test_score = df['AvgTestScore'].mean()
        
        metrics = {
            "Total Schools": total_schools,
            "Average Financial Aid Rate": avg_aid_rate,
            "Average State Graduation Rate": avg_state_grad_rate,
            "Average State Test Score": avg_state_test_score
        }
        
        insights.append(f"Total Schools Represented: {total_schools:,.0f}")
        insights.append(f"Average Financial Aid Rate: {avg_aid_rate:.2f}%")
        insights.append(f"Average State Graduation Rate: {avg_state_grad_rate:.2f}%")
        insights.append(f"Average State Test Score: {avg_state_test_score:.2f}")

        fig1 = px.scatter(df, x='SchoolCount', y='FinancialAidRate', color='State', title='Financial Aid Rate vs. School Count by State')
        fig2 = px.box(df, x='State', y='FinancialAidRate', title='Financial Aid Rate Distribution by State')
        fig3 = px.bar(df.groupby('State')['SchoolCount'].sum().nlargest(20).reset_index(), x='State', y='SchoolCount', title='Total School Count by State (Top 20)')
        fig4 = px.scatter(df, x='AvgTestScore', y='GraduationRate', color='State', title='State Graduation Rate vs. Average Test Score')

        visualizations = {
            "Financial_Aid_Rate_vs_School_Count_Scatter": fig1.to_json(),
            "Financial_Aid_Rate_Distribution_by_State_Box": fig2.to_json(),
            "Total_School_Count_by_State_Bar": fig3.to_json(),
            "State_Graduation_Rate_vs_Avg_Test_Score_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_tuition_earnings_and_default_rate_trend_analysis(df):
    print("\n--- University Tuition, Earnings, and Default Rate Trend Analysis ---")
    analysis_type = "University Tuition, Earnings, and Default Rate Trend Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'Year': ['Year', 'AcademicYear'],
            'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
            'Earnings': ['Earnings', 'MedianEarnings', 'PostGraduationEarnings'],
            'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
        df['Earnings'] = pd.to_numeric(df['Earnings'], errors='coerce')
        df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Tuition', 'Earnings', 'DefaultRate', 'Year'])
        df = df.sort_values(by=['InstitutionName', 'Year'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        trend_data = df.groupby('Year')[['Tuition', 'Earnings', 'DefaultRate']].mean().reset_index()
        metrics["Average_Trends_Over_Time"] = trend_data.to_dict('records')
        insights.append("Generated average trends for Tuition, Earnings, and Default Rate over time.")
        
        fig1 = px.line(trend_data, x='Year', y=['Tuition', 'Earnings', 'DefaultRate'],
                       title='Average Trends Over Time (Tuition, Earnings, Default Rate)')
        visualizations["Average_Trends_Over_Time_Line"] = fig1.to_json()
        
        unique_institutions = df['InstitutionName'].unique()
        if len(unique_institutions) > 1:
            if len(unique_institutions) > 10:
                top_institutions = df.groupby('InstitutionName')['Earnings'].mean().nlargest(5).index
                df_plot = df[df['InstitutionName'].isin(top_institutions)]
                insights.append("Displaying trends for top 5 institutions by average earnings.")
            else:
                df_plot = df
                insights.append("Displaying trends for all institutions.")

            fig2 = px.line(df_plot, x='Year', y='Tuition', color='InstitutionName', title='Tuition Trends for Sample Universities')
            fig3 = px.line(df_plot, x='Year', y='Earnings', color='InstitutionName', title='Earnings Trends for Sample Universities')
            fig4 = px.line(df_plot, x='Year', y='DefaultRate', color='InstitutionName', title='Default Rate Trends for Sample Universities')
        else:
            fig2 = px.line(df, x='Year', y='Tuition', title='Tuition Trend')
            fig3 = px.line(df, x='Year', y='Earnings', title='Earnings Trend')
            fig4 = px.line(df, x='Year', y='DefaultRate', title='Default Rate Trend')
            insights.append("Displaying trend for the single institution in the dataset.")

        visualizations["Tuition_Trends_Sample_Universities_Line"] = fig2.to_json()
        visualizations["Earnings_Trends_Sample_Universities_Line"] = fig3.to_json()
        visualizations["Default_Rate_Trends_Sample_Universities_Line"] = fig4.to_json()
        
        fig5 = px.scatter(df, x='Tuition', y='DefaultRate', hover_name='InstitutionName', animation_frame='Year',
                          title='Default Rate vs. Tuition Over Time (Animated)',
                          animation_group='InstitutionName')
        visualizations["Default_Rate_vs_Tuition_Animated_Scatter"] = fig5.to_json()


        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_board_data_analysis(df):
    print("\n--- School Board Data Analysis ---")
    analysis_type = "School Board Data Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'District': ['District', 'DistrictName', 'SchoolDistrict'],
            'SchoolBoard': ['SchoolBoard', 'BoardName', 'GoverningBody'],
            'Budget': ['Budget', 'TotalBudget', 'DistrictBudget'],
            'Students': ['Students', 'TotalStudents', 'Enrollment'],
            'MeetingAttendanceRate': ['MeetingAttendanceRate', 'BoardMeetingAttendance']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}
        
        # This analysis is flexible. Run with what's available.
        if missing and not df.columns.tolist(): # No columns at all
             return {
                "analysis_type": analysis_type, "status": "error", "error": "No data found.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No data available."]
            }

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

        if 'Budget' in df.columns and not df['Budget'].dropna().empty:
            total_budget = df['Budget'].sum()
            metrics['Total Budget'] = total_budget
            insights.append(f"Total Budget: ${total_budget:,.0f}")
            visualizations['Total_Budget_Histogram'] = px.histogram(df, x='Budget', title='Distribution of Total Budgets').to_json()

        if 'Students' in df.columns and not df['Students'].dropna().empty:
            total_students = df['Students'].sum()
            metrics['Total Students'] = total_students
            insights.append(f"Total Students: {total_students:,.0f}")
            visualizations['Total_Students_Histogram'] = px.histogram(df, x='Students', title='Distribution of Total Students').to_json()

        if 'per_student_spending' in df_clean.columns and not df_clean['per_student_spending'].dropna().empty:
            avg_spending = df_clean['per_student_spending'].mean()
            metrics['Average Per-Student Spending'] = avg_spending
            insights.append(f"Average Per-Student Spending: ${avg_spending:,.0f}")
            if 'SchoolBoard' in df_clean.columns:
                visualizations['Per_Student_Spending_Bar'] = px.bar(df_clean.groupby('SchoolBoard')['per_student_spending'].mean().nlargest(20).reset_index(),
                                                                x='SchoolBoard', y='per_student_spending', title='Average Per-Student Spending by School Board (Top 20)').to_json()
            visualizations['Per_Student_Spending_Box'] = px.box(df_clean, y='per_student_spending', title='Per-Student Spending Distribution').to_json()
        
        if 'Students' in df_clean.columns and 'Budget' in df_clean.columns and not df_clean[['Students', 'Budget']].dropna().empty:
            color_col = 'SchoolBoard' if 'SchoolBoard' in df_clean.columns else None
            visualizations['Budget_vs_Student_Enrollment_Scatter'] = px.scatter(df_clean, x='Students', y='Budget', color=color_col, title='Budget vs. Student Enrollment').to_json()

        if 'MeetingAttendanceRate' in df.columns and not df['MeetingAttendanceRate'].dropna().empty:
            avg_attendance_rate = df['MeetingAttendanceRate'].mean()
            metrics['Average Meeting Attendance Rate'] = avg_attendance_rate
            insights.append(f"Average Meeting Attendance Rate: {avg_attendance_rate:.2f}%")
            visualizations['Meeting_Attendance_Rate_Histogram'] = px.histogram(df, x='MeetingAttendanceRate', title='Distribution of Meeting Attendance Rates').to_json()

        if not metrics and not visualizations:
            insights.append("No sufficient data available after cleaning for any specific analysis within this function.")
            return {
                "analysis_type": analysis_type, "status": "fallback", "error": "No sufficient data for analysis.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": insights
            }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_selectivity_and_loan_repayment_rate_analysis(df):
    print("\n--- College Selectivity and Loan Repayment Rate Analysis ---")
    analysis_type = "College Selectivity and Loan Repayment Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
            'Region': ['Region', 'State', 'STABBR'],
            'LoanRepaymentRate': ['LoanRepaymentRate', 'RepaymentRate', 'SixYearRepaymentRate'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
            'AvgSAT': ['AvgSAT', 'SATScore', 'AverageSATScore']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        has_acceptance = 'AcceptanceRate' in df.columns
        has_sat = 'AvgSAT' in df.columns
        
        if 'LoanRepaymentRate' not in df.columns or not (has_acceptance or has_sat):
            _, missing_strict = check_and_rename_columns(df, {'LoanRepaymentRate': expected['LoanRepaymentRate'], 'AcceptanceRate': expected['AcceptanceRate']})
            show_missing_columns_warning(missing_strict, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing 'LoanRepaymentRate' or a selectivity metric ('AcceptanceRate' or 'AvgSAT').")
            return fallback_result

        if has_acceptance:
            df['SelectivityScore'] = 1 - pd.to_numeric(df['AcceptanceRate'], errors='coerce')
            insights.append("Using (1 - AcceptanceRate) as Selectivity Score.")
        elif has_sat:
            df['SelectivityScore'] = pd.to_numeric(df['AvgSAT'], errors='coerce')
            insights.append("Using AvgSAT as Selectivity Score.")

        df = safe_numeric_conversion(df, 'SelectivityScore')
        df = safe_numeric_conversion(df, 'LoanRepaymentRate')
        df = df.dropna(subset=['SelectivityScore', 'LoanRepaymentRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_repayment_rate = df['LoanRepaymentRate'].mean()
        lowest_repayment_rate_college = df.loc[df['LoanRepaymentRate'].idxmin(), 'CollegeName']

        metrics = {
            "Average Loan Repayment Rate": avg_repayment_rate,
            "Lowest Repayment Rate College": lowest_repayment_rate_college
        }
        
        insights.append(f"Average Loan Repayment Rate: {avg_repayment_rate:.2f}%")
        insights.append(f"College with Lowest Repayment Rate: {lowest_repayment_rate_college}")

        fig1 = px.scatter(df, x='SelectivityScore', y='LoanRepaymentRate', hover_name='CollegeName', title='Loan Repayment Rate vs. Selectivity')
        fig2 = px.box(df, x='Region', y='LoanRepaymentRate', title='Loan Repayment Rate by Region')
        fig3 = px.histogram(df, x='LoanRepaymentRate', color='Region', title='Loan Repayment Rate Distribution by Region')

        visualizations = {
            "Loan_Repayment_Rate_vs_Selectivity_Scatter": fig1.to_json(),
            "Loan_Repayment_Rate_by_Region_Box": fig2.to_json(),
            "Loan_Repayment_Rate_Distribution_by_Region_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def faculty_composition_and_student_faculty_ratio_analysis(df):
    print("\n--- Faculty Composition and Student-Faculty Ratio Analysis ---")
    analysis_type = "Faculty Composition and Student-Faculty Ratio Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
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
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
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
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_ratio = df['StudentFacultyRatio'].mean()
        best_ratio_institution = df.loc[df['StudentFacultyRatio'].idxmin(), 'InstitutionName']
        avg_percent_tenure_track = df['PercentTenureTrack'].mean()
        avg_percent_adjunct = df['PercentAdjunct'].mean()

        metrics = {
            "Average Student-Faculty Ratio": avg_ratio,
            "Best Ratio Institution": best_ratio_institution,
            "Average Percent Tenure-Track Faculty": avg_percent_tenure_track,
            "Average Percent Adjunct Faculty": avg_percent_adjunct
        }
        
        insights.append(f"Average Student-Faculty Ratio: {avg_ratio:.2f}:1")
        insights.append(f"Institution with Best Ratio: {best_ratio_institution}")
        insights.append(f"Average Percent Tenure-Track Faculty: {avg_percent_tenure_track:.2%}")
        insights.append(f"Average Percent Adjunct Faculty: {avg_percent_adjunct:.2%}")

        fig1 = px.scatter(df, x='StudentFacultyRatio', y='GraduationRate', hover_name='InstitutionName', title='Graduation Rate vs. Student-Faculty Ratio')
        fig2 = px.histogram(df, x='StudentFacultyRatio', title='Distribution of Student-Faculty Ratios')
        fig3 = px.box(df, x='State', y='StudentFacultyRatio', title='Student-Faculty Ratio by State')
        fig4 = px.scatter(df, x='PercentTenureTrack', y='StudentFacultyRatio', hover_name='InstitutionName', title='Student-Faculty Ratio vs. Percent Tenure-Track Faculty')
        fig5 = px.scatter(df, x='PercentAdjunct', y='StudentFacultyRatio', hover_name='InstitutionName', title='Student-Faculty Ratio vs. Percent Adjunct Faculty')

        visualizations = {
            "Graduation_Rate_vs_Student_Faculty_Ratio_Scatter": fig1.to_json(),
            "Student_Faculty_Ratio_Distribution_Histogram": fig2.to_json(),
            "Student_Faculty_Ratio_by_State_Box": fig3.to_json(),
            "Student_Faculty_Ratio_vs_Percent_Tenure_Track_Scatter": fig4.to_json(),
            "Student_Faculty_Ratio_vs_Percent_Adjunct_Scatter": fig5.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_expenditure_and_test_score_analysis(df):
    print("\n--- School District Expenditure and Test Score Analysis ---")
    analysis_type = "School District Expenditure and Test Score Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolDistrictID': ['SchoolDistrictID', 'DistrictID', 'ID'],
            'DistrictName': ['DistrictName', 'Name'],
            'PerStudentExpenditure': ['PerStudentExpenditure', 'SpendingPerStudent', 'DistrictExpenditure'],
            'StateTestScores': ['StateTestScores', 'AverageTestScore', 'DistrictAvgScore']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['PerStudentExpenditure'] = pd.to_numeric(df['PerStudentExpenditure'], errors='coerce')
        df['StateTestScores'] = pd.to_numeric(df['StateTestScores'], errors='coerce')
        df = df.dropna(subset=['PerStudentExpenditure', 'StateTestScores'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_expenditure = df['PerStudentExpenditure'].mean()
        avg_test_score = df['StateTestScores'].mean()

        metrics = {
            "Average Per-Student Expenditure": avg_expenditure,
            "Average State Test Score": avg_test_score
        }
        
        insights.append(f"Average Per-Student Expenditure: ${avg_expenditure:,.0f}")
        insights.append(f"Average State Test Score: {avg_test_score:.2f}")

        fig1 = px.scatter(df, x='PerStudentExpenditure', y='StateTestScores', hover_name='DistrictName', title='State Test Scores vs. Per-Student Expenditure')
        fig2 = px.histogram(df, x='PerStudentExpenditure', title='Distribution of Per-Student Expenditure')
        fig3 = px.box(df, y='StateTestScores', title='State Test Score Distribution')

        visualizations = {
            "State_Test_Scores_vs_Per_Student_Expenditure_Scatter": fig1.to_json(),
            "Per_Student_Expenditure_Distribution_Histogram": fig2.to_json(),
            "State_Test_Score_Distribution_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_selectivity_and_income_diversity_analysis(df):
    print("\n--- College Selectivity and Income Diversity Analysis ---")
    analysis_type = "College Selectivity and Income Diversity Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'State': ['State', 'STABBR'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
            'IncomeDiversityIndex': ['IncomeDiversityIndex', 'PellGrantPercentage', 'PercentLowIncomeStudents', 'SESDiversity']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
        df['IncomeDiversityIndex'] = pd.to_numeric(df['IncomeDiversityIndex'], errors='coerce')
        df = df.dropna(subset=['AcceptanceRate', 'IncomeDiversityIndex'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_acceptance_rate = df['AcceptanceRate'].mean()
        avg_income_diversity = df['IncomeDiversityIndex'].mean()

        metrics = {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Income Diversity Index": avg_income_diversity
        }
        
        insights.append(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
        insights.append(f"Average Income Diversity Index: {avg_income_diversity:.2f}")

        fig1 = px.scatter(df, x='AcceptanceRate', y='IncomeDiversityIndex', hover_name='InstitutionName', title='Income Diversity vs. Acceptance Rate')
        fig2 = px.box(df, x='State', y='IncomeDiversityIndex', title='Income Diversity Index by State')
        fig3 = px.histogram(df, x='AcceptanceRate', title='Distribution of Acceptance Rates')

        visualizations = {
            "Income_Diversity_vs_Acceptance_Rate_Scatter": fig1.to_json(),
            "Income_Diversity_Index_by_State_Box": fig2.to_json(),
            "Acceptance_Rates_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_poverty_and_graduation_rate_correlation(df):
    print("\n--- School District Poverty and Graduation Rate Correlation ---")
    analysis_type = "School District Poverty and Graduation Rate Correlation"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
            'DistrictName': ['DistrictName', 'Name'],
            'PovertyRate': ['PovertyRate', 'FreeReducedLunchRate', 'LowIncomeStudentRate'],
            'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['PovertyRate'] = pd.to_numeric(df['PovertyRate'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['PovertyRate', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_poverty_rate = df['PovertyRate'].mean()
        avg_grad_rate = df['GraduationRate'].mean()
        
        metrics = {
            "Average Poverty Rate": avg_poverty_rate,
            "Average Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average Poverty Rate: {avg_poverty_rate:.2f}%")
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

        fig1 = px.scatter(df, x='PovertyRate', y='GraduationRate', hover_name='DistrictName', title='Graduation Rate vs. Poverty Rate')
        fig2 = px.box(df, x='DistrictName', y='GraduationRate', title='Graduation Rate by School District')
        fig3 = px.histogram(df, x='PovertyRate', title='Distribution of Poverty Rates')

        visualizations = {
            "Graduation_Rate_vs_Poverty_Rate_Scatter": fig1.to_json(),
            "Graduation_Rate_by_School_District_Box": fig2.to_json(),
            "Poverty_Rates_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_enrollment_and_retention_analysis(df):
    print("\n--- University Enrollment and Retention Analysis ---")
    analysis_type = "University Enrollment and Retention Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'Region': ['Region', 'State', 'STABBR'],
            'RetentionRate': ['RetentionRate', 'FirstYearRetention'],
            'UndergraduateEnrollment': ['UndergraduateEnrollment', 'UGEnrollment', 'Enrollment']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['RetentionRate'] = pd.to_numeric(df['RetentionRate'], errors='coerce')
        df['UndergraduateEnrollment'] = pd.to_numeric(df['UndergraduateEnrollment'], errors='coerce')
        df = df.dropna(subset=['RetentionRate', 'UndergraduateEnrollment'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
        avg_retention_rate = df['RetentionRate'].mean()
        avg_enrollment = df['UndergraduateEnrollment'].mean()
        
        metrics = {
            "Average Retention Rate": avg_retention_rate,
            "Average Undergraduate Enrollment": avg_enrollment
        }
        
        insights.append(f"Average Retention Rate: {avg_retention_rate:.2f}%")
        insights.append(f"Average Undergraduate Enrollment: {avg_enrollment:,.0f}")
        
        fig1 = px.scatter(df, x='UndergraduateEnrollment', y='RetentionRate', hover_name='Name', title='Retention Rate vs. Undergraduate Enrollment')
        fig2 = px.box(df, x='Region', y='RetentionRate', title='Retention Rate by Region')
        fig3 = px.bar(df.groupby('Region')['UndergraduateEnrollment'].sum().reset_index(), x='Region', y='UndergraduateEnrollment', title='Total Enrollment by Region')

        visualizations = {
            "Retention_Rate_vs_Undergraduate_Enrollment_Scatter": fig1.to_json(),
            "Retention_Rate_by_Region_Box": fig2.to_json(),
            "Total_Enrollment_by_Region_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_application_and_enrollment_funnel_analysis(df):
    print("\n--- College Application and Enrollment Funnel Analysis ---")
    analysis_type = "College Application and Enrollment Funnel Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionState': ['InstitutionState', 'State', 'STABBR'],
            'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
            'ApplicationsReceived': ['ApplicationsReceived', 'Applicants', 'TotalApplicants'],
            'Admitted': ['Admitted', 'AcceptedStudents'],
            'Enrolled': ['Enrolled', 'MatriculatedStudents'],
            'GraduationRate': ['GraduationRate', 'CompletionRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['ApplicationsReceived'] = pd.to_numeric(df['ApplicationsReceived'], errors='coerce')
        df['Admitted'] = pd.to_numeric(df['Admitted'], errors='coerce')
        df['Enrolled'] = pd.to_numeric(df['Enrolled'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['ApplicationsReceived', 'Admitted', 'Enrolled', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }
        
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
        
        metrics = {
            "Total Applications": total_applications,
            "Total Admitted": total_admitted,
            "Total Enrolled": total_enrolled,
            "Overall Acceptance Rate": total_admitted / total_applications if total_applications > 0 else 0,
            "Overall Yield Rate": total_enrolled / total_admitted if total_admitted > 0 else 0,
            "Average Acceptance Rate": df['AcceptanceRate'].mean(),
            "Average Yield Rate": df['YieldRate'].mean(),
            "Average Graduation Rate": df['GraduationRate'].mean(),
            "Number of Institutions": len(df)
        }
        
        insights.append(f"Total Applications Across All Institutions: {total_applications:,}")
        insights.append(f"Total Students Admitted: {total_admitted:,}")
        insights.append(f"Total Students Enrolled: {total_enrolled:,}")
        insights.append(f"Overall Acceptance Rate: {metrics['Overall Acceptance Rate']:.1%}")
        insights.append(f"Overall Yield Rate: {metrics['Overall Yield Rate']:.1%}")
        insights.append(f"Average Institutional Acceptance Rate: {metrics['Average Acceptance Rate']:.1%}")
        insights.append(f"Average Institutional Yield Rate: {metrics['Average Yield Rate']:.1%}")
        insights.append(f"Average Graduation Rate: {metrics['Average Graduation Rate']:.1%}")
        
        # Funnel visualization
        funnel_data = pd.DataFrame({
            'Stage': ['Applications Received', 'Admitted', 'Enrolled'],
            'Count': [total_applications, total_admitted, total_enrolled],
            'Percentage': [100, 
                          (total_admitted / total_applications * 100) if total_applications > 0 else 0,
                          (total_enrolled / total_applications * 100) if total_applications > 0 else 0]
        })
        
        fig1 = px.funnel(funnel_data, x='Count', y='Stage', title='College Admissions Funnel - Overall')
        
        # Acceptance Rate vs Yield Rate scatter plot
        fig2 = px.scatter(df, x='AcceptanceRate', y='YieldRate', 
                         hover_name='CollegeName', 
                         color='InstitutionState',
                         title='Acceptance Rate vs Yield Rate by Institution',
                         labels={'AcceptanceRate': 'Acceptance Rate', 'YieldRate': 'Yield Rate'})
        
        # Yield Rate vs Graduation Rate scatter plot
        fig3 = px.scatter(df, x='YieldRate', y='GraduationRate',
                         hover_name='CollegeName',
                         color='InstitutionState',
                         title='Yield Rate vs Graduation Rate',
                         labels={'YieldRate': 'Yield Rate', 'GraduationRate': 'Graduation Rate'})
        
        # Acceptance Rate distribution by state
        fig4 = px.box(df, x='InstitutionState', y='AcceptanceRate',
                     title='Acceptance Rate Distribution by State')
        
        # Yield Rate distribution by state
        fig5 = px.box(df, x='InstitutionState', y='YieldRate',
                     title='Yield Rate Distribution by State')
        
        # Top institutions by yield rate
        top_yield_institutions = df.nlargest(10, 'YieldRate')[['CollegeName', 'YieldRate', 'AcceptanceRate', 'GraduationRate']]
        fig6 = px.bar(top_yield_institutions, 
                     x='CollegeName', y='YieldRate',
                     title='Top 10 Institutions by Yield Rate',
                     hover_data=['AcceptanceRate', 'GraduationRate'])
        
        visualizations = {
            "Admissions_Funnel_Overall": fig1.to_json(),
            "Acceptance_Rate_vs_Yield_Rate_Scatter": fig2.to_json(),
            "Yield_Rate_vs_Graduation_Rate_Scatter": fig3.to_json(),
            "Acceptance_Rate_Distribution_by_State_Box": fig4.to_json(),
            "Yield_Rate_Distribution_by_State_Box": fig5.to_json(),
            "Top_Institutions_by_Yield_Rate_Bar": fig6.to_json()
        }

        # Additional insights based on analysis
        high_yield_institutions = df[df['YieldRate'] > 0.5]
        if len(high_yield_institutions) > 0:
            insights.append(f"{len(high_yield_institutions)} institutions have yield rates above 50%, indicating strong student commitment.")
        
        selective_institutions = df[df['AcceptanceRate'] < 0.2]
        if len(selective_institutions) > 0:
            insights.append(f"{len(selective_institutions)} highly selective institutions (acceptance rate < 20%).")
        
        # Correlation analysis
        acceptance_yield_corr = df['AcceptanceRate'].corr(df['YieldRate'])
        yield_graduation_corr = df['YieldRate'].corr(df['GraduationRate'])
        
        metrics["Acceptance_Yield_Correlation"] = acceptance_yield_corr
        metrics["Yield_Graduation_Correlation"] = yield_graduation_corr
        
        insights.append(f"Correlation between Acceptance Rate and Yield Rate: {acceptance_yield_corr:.3f}")
        insights.append(f"Correlation between Yield Rate and Graduation Rate: {yield_graduation_corr:.3f}")
        
        if acceptance_yield_corr < -0.3:
            insights.append("Strong negative correlation between acceptance and yield rates - more selective institutions tend to have higher yield.")
        elif acceptance_yield_corr > 0.3:
            insights.append("Positive correlation between acceptance and yield rates - less selective institutions also see higher yield.")
        
        if yield_graduation_corr > 0.3:
            insights.append("Positive correlation between yield and graduation rates - institutions with committed students tend to have better graduation outcomes.")

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, 
            "status": "error", 
            "error": str(e),
            "matched_columns": matched_columns_map, 
            "visualizations": {}, 
            "metrics": {}, 
            "insights": [f"An error occurred: {e}"]
        }
def school_district_staffing_and_student_enrollment_analysis(df):
    print("\n--- School District Staffing and Student Enrollment Analysis ---")
    analysis_type = "School District Staffing and Student Enrollment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'SchoolID': ['SchoolID', 'SchoolId', 'ID'],
            'DistrictName': ['DistrictName', 'Name', 'SchoolDistrict'],
            'StudentEnrollment': ['StudentEnrollment', 'Enrollment', 'TotalStudents'],
            'StaffCount': ['StaffCount', 'TotalStaff', 'FTStaff'],
            'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['StudentEnrollment'] = pd.to_numeric(df['StudentEnrollment'], errors='coerce')
        df['StaffCount'] = pd.to_numeric(df['StaffCount'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['StudentEnrollment', 'StaffCount', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        df['student_staff_ratio'] = df.apply(
            lambda row: row['StudentEnrollment'] / row['StaffCount'] if row['StaffCount'] > 0 else np.nan,
            axis=1
        )
        df = df.dropna(subset=['student_staff_ratio'])

        avg_ratio = df['student_staff_ratio'].mean()
        avg_grad_rate = df['GraduationRate'].mean()
        
        metrics = {
            "Average Student-Staff Ratio": avg_ratio,
            "Average Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average Student-Staff Ratio: {avg_ratio:.2f}")
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")

        fig1 = px.scatter(df, x='student_staff_ratio', y='GraduationRate', hover_name='SchoolID', title='Graduation Rate vs. Student-Staff Ratio')
        fig2 = px.box(df, y='student_staff_ratio', title='Student-Staff Ratio Distribution')
        fig3 = px.bar(df.groupby('DistrictName')['GraduationRate'].mean().nlargest(20).reset_index(), x='DistrictName', y='GraduationRate', title='Average Graduation Rate by District (Top 20)')

        visualizations = {
            "Graduation_Rate_vs_Student_Staff_Ratio_Scatter": fig1.to_json(),
            "Student_Staff_Ratio_Distribution_Box": fig2.to_json(),
            "Average_Graduation_Rate_by_District_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_sat_scores_and_7_year_graduation_rate_analysis(df):
    print("\n--- College SAT Scores and 7-Year Graduation Rate Analysis ---")
    analysis_type = "College SAT Scores and 7-Year Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'StateAbbrev': ['StateAbbrev', 'State', 'STABBR'],
            'CollegeName': ['CollegeName', 'Name', 'INSTNM'],
            'SATScore': ['SATScore', 'AvgSAT', 'MedianSAT'],
            'Graduation7YearRate': ['Graduation7YearRate', '7YearGradRate', 'LongTermGraduationRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['SATScore'] = pd.to_numeric(df['SATScore'], errors='coerce')
        df['Graduation7YearRate'] = pd.to_numeric(df['Graduation7YearRate'], errors='coerce')
        df = df.dropna(subset=['SATScore', 'Graduation7YearRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_sat_score = df['SATScore'].mean()
        avg_grad_rate = df['Graduation7YearRate'].mean()

        metrics = {
            "Average SAT Score": avg_sat_score,
            "Average 7-Year Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average SAT Score: {avg_sat_score:.0f}")
        insights.append(f"Average 7-Year Graduation Rate: {avg_grad_rate:.2f}%")
        
        fig1 = px.scatter(df, x='SATScore', y='Graduation7YearRate', hover_name='CollegeName', title='7-Year Graduation Rate vs. SAT Score')
        fig2 = px.box(df, x='StateAbbrev', y='Graduation7YearRate', title='7-Year Graduation Rate by State')
        fig3 = px.histogram(df, x='SATScore', title='Distribution of SAT Scores')

        visualizations = {
            "7_Year_Graduation_Rate_vs_SAT_Score_Scatter": fig1.to_json(),
            "7_Year_Graduation_Rate_by_State_Box": fig2.to_json(),
            "SAT_Scores_Distribution_Histogram": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_attendance_and_graduation_rate_analysis(df):
    print("\n--- School Attendance and Graduation Rate Analysis ---")
    analysis_type = "School Attendance and Graduation Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictName': ['DistrictName', 'Name'],
            'SchoolName': ['SchoolName', 'Name'],
            'AttendanceRate': ['AttendanceRate', 'AvgAttendance'],
            'GraduationRate': ['GraduationRate', 'HighSchoolGraduationRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AttendanceRate'] = pd.to_numeric(df['AttendanceRate'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['AttendanceRate', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_attendance = df['AttendanceRate'].mean()
        avg_grad_rate = df['GraduationRate'].mean()

        metrics = {
            "Average Attendance Rate": avg_attendance,
            "Average Graduation Rate": avg_grad_rate
        }
        
        insights.append(f"Average Attendance Rate: {avg_attendance:.2f}%")
        insights.append(f"Average Graduation Rate: {avg_grad_rate:.2f}%")
        
        fig1 = px.scatter(df, x='AttendanceRate', y='GraduationRate', hover_name='SchoolName', title='Graduation Rate vs. Attendance Rate')
        fig2 = px.box(df, x='DistrictName', y='AttendanceRate', title='Attendance Rate by District')
        fig3 = px.box(df, y='GraduationRate', title='Graduation Rate Distribution')

        visualizations = {
            "Graduation_Rate_vs_Attendance_Rate_Scatter": fig1.to_json(),
            "Attendance_Rate_by_District_Box": fig2.to_json(),
            "Graduation_Rate_Distribution_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_undergraduate_and_graduate_enrollment_analysis(df):
    print("\n--- University Undergraduate and Graduate Enrollment Analysis ---")
    analysis_type = "University Undergraduate and Graduate Enrollment Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeID': ['CollegeID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'State': ['State', 'STABBR'],
            'UndergradEnrollment': ['UndergradEnrollment', 'UGEnrollment', 'UndergraduateStudents'],
            'GradEnrollment': ['GradEnrollment', 'GraduateEnrollment', 'GraduateStudents']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['UndergradEnrollment'] = pd.to_numeric(df['UndergradEnrollment'], errors='coerce')
        df['GradEnrollment'] = pd.to_numeric(df['GradEnrollment'], errors='coerce')
        df = df.dropna(subset=['UndergradEnrollment', 'GradEnrollment'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        total_undergrad = df['UndergradEnrollment'].sum()
        total_grad = df['GradEnrollment'].sum()

        metrics = {
            "Total Undergraduate Enrollment": total_undergrad,
            "Total Graduate Enrollment": total_grad
        }
        
        insights.append(f"Total Undergraduate Enrollment: {total_undergrad:,.0f}")
        insights.append(f"Total Graduate Enrollment: {total_grad:,.0f}")
        
        enrollment_data = pd.DataFrame({
            'Level': ['Undergraduate', 'Graduate'],
            'Count': [total_undergrad, total_grad]
        })
        fig1 = px.pie(enrollment_data, names='Level', values='Count', title='Total Enrollment by Level')

        fig2 = px.scatter(df, x='UndergradEnrollment', y='GradEnrollment', hover_name='Name', title='Graduate vs. Undergraduate Enrollment')
        fig3 = px.bar(df.groupby('State')['UndergradEnrollment'].sum().nlargest(20).reset_index(), x='State', y='UndergradEnrollment', title='Undergrad Enrollment by State (Top 20)')

        visualizations = {
            "Total_Enrollment_by_Level_Pie": fig1.to_json(),
            "Graduate_vs_Undergraduate_Enrollment_Scatter": fig2.to_json(),
            "Undergrad_Enrollment_by_State_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_selectivity_yield_and_default_rate_analysis(df):
    print("\n--- College Selectivity, Yield, and Default Rate Analysis ---")
    analysis_type = "College Selectivity, Yield, and Default Rate Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'City': ['City', 'CampusCity'],
            'State': ['State', 'STABBR'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
            'Yield': ['Yield', 'YieldRate'],
            'DefaultRate': ['DefaultRate', 'LoanDefaultRate', 'ThreeYearDefaultRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
        df['Yield'] = pd.to_numeric(df['Yield'], errors='coerce')
        df['DefaultRate'] = pd.to_numeric(df['DefaultRate'], errors='coerce')
        df = df.dropna(subset=['AcceptanceRate', 'Yield', 'DefaultRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_acceptance_rate = df['AcceptanceRate'].mean()
        avg_yield = df['Yield'].mean()
        avg_default_rate = df['DefaultRate'].mean()
        
        metrics = {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Yield Rate": avg_yield,
            "Average Default Rate": avg_default_rate
        }
        
        insights.append(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
        insights.append(f"Average Yield Rate: {avg_yield:.2f}%")
        insights.append(f"Average Default Rate: {avg_default_rate:.2f}%")
        
        fig1 = px.scatter(df, x='Yield', y='DefaultRate', hover_name='InstitutionName', title='Default Rate vs. Yield Rate')
        fig2 = px.box(df, x='State', y='Yield', title='Yield Rate by State')
        fig3 = px.box(df, x='State', y='DefaultRate', title='Default Rate by State')
        fig4 = px.scatter(df, x='AcceptanceRate', y='Yield', hover_name='InstitutionName', title='Yield Rate vs. Acceptance Rate')

        visualizations = {
            "Default_Rate_vs_Yield_Rate_Scatter": fig1.to_json(),
            "Yield_Rate_by_State_Box": fig2.to_json(),
            "Default_Rate_by_State_Box": fig3.to_json(),
            "Yield_Rate_vs_Acceptance_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_classification_and_expenditure_analysis(df):
    print("\n--- School District Classification and Expenditure Analysis ---")
    analysis_type = "School District Classification and Expenditure Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictName': ['DistrictName', 'Name'],
            'DistrictType': ['DistrictType', 'Classification', 'UrbanRural'],
            'Enrollment': ['Enrollment', 'TotalEnrollment', 'StudentCount'],
            'perStudentExpenditure': ['perStudentExpenditure', 'SpendingPerStudent', 'PerPupilExpenditure']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['Enrollment'] = pd.to_numeric(df['Enrollment'], errors='coerce')
        df['perStudentExpenditure'] = pd.to_numeric(df['perStudentExpenditure'], errors='coerce')
        df = df.dropna(subset=['Enrollment', 'perStudentExpenditure'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_expenditure = df['perStudentExpenditure'].mean()
        total_enrollment = df['Enrollment'].sum()
        
        metrics = {
            "Average Per-Student Expenditure": avg_expenditure,
            "Total Enrollment Across Districts": total_enrollment
        }
        
        insights.append(f"Average Per-Student Expenditure: ${avg_expenditure:,.0f}")
        insights.append(f"Total Enrollment Across Districts: {total_enrollment:,.0f}")
        
        fig1 = px.histogram(df, x='perStudentExpenditure', title='Distribution of Per-Student Expenditure')
        fig2 = px.scatter(df, x='Enrollment', y='perStudentExpenditure', hover_name='DistrictName', title='Per-Student Expenditure vs. Enrollment')
        fig3 = px.box(df, y='perStudentExpenditure', title='Per-Student Expenditure Distribution')
        
        fig4 = None
        if 'DistrictType' in df.columns and not df['DistrictType'].dropna().empty:
            fig4 = px.box(df, x='DistrictType', y='perStudentExpenditure', title='Per-Student Expenditure by District Type')
            visualizations["Per_Student_Expenditure_by_District_Type_Box"] = fig4.to_json()

        visualizations.update({
            "Per_Student_Expenditure_Distribution_Histogram": fig1.to_json(),
            "Per_Student_Expenditure_vs_Enrollment_Scatter": fig2.to_json(),
            "Per_Student_Expenditure_Distribution_Box": fig3.to_json()
        })

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_scholarship_and_post_graduation_earnings_analysis(df):
    print("\n--- College Scholarship and Post-Graduation Earnings Analysis ---")
    analysis_type = "College Scholarship and Post-Graduation Earnings Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'CollegeID': ['CollegeID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'State': ['State', 'STABBR'],
            'MedianEarnings': ['MedianEarnings', 'PostGradEarnings', 'AvgSalary'],
            'AvgScholarship': ['AvgScholarship', 'AverageGrantAid', 'ScholarshipAmount']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['MedianEarnings'] = pd.to_numeric(df['MedianEarnings'], errors='coerce')
        df['AvgScholarship'] = pd.to_numeric(df['AvgScholarship'], errors='coerce')
        df = df.dropna(subset=['MedianEarnings', 'AvgScholarship'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_earnings = df['MedianEarnings'].mean()
        avg_scholarship = df['AvgScholarship'].mean()

        metrics = {
            "Average Median Earnings": avg_earnings,
            "Average Scholarship Amount": avg_scholarship
        }
        
        insights.append(f"Average Median Earnings: ${avg_earnings:,.0f}")
        insights.append(f"Average Scholarship Amount: ${avg_scholarship:,.0f}")

        fig1 = px.scatter(df, x='AvgScholarship', y='MedianEarnings', hover_name='Name', title='Median Earnings vs. Average Scholarship Amount')
        fig2 = px.box(df, x='State', y='MedianEarnings', title='Median Earnings by State')
        fig3 = px.box(df, x='State', y='AvgScholarship', title='Average Scholarship Amount by State')

        visualizations = {
            "Median_Earnings_vs_Average_Scholarship_Amount_Scatter": fig1.to_json(),
            "Median_Earnings_by_State_Box": fig2.to_json(),
            "Average_Scholarship_Amount_by_State_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_campus_enrollment_and_tuition_analysis(df):
    print("\n--- University Campus Enrollment and Tuition Analysis ---")
    analysis_type = "University Campus Enrollment and Tuition Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionName': ['InstitutionName', 'Name', 'INSTNM'],
            'CampusName': ['CampusName', 'BranchName', 'Campus'],
            'UndergradEnrollment': ['UndergradEnrollment', 'UGEnrollment', 'Enrollment'],
            'Tuition': ['Tuition', 'AvgTuition', 'NetPrice'],
            'GraduationRate': ['GraduationRate', 'CompletionRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['UndergradEnrollment'] = pd.to_numeric(df['UndergradEnrollment'], errors='coerce')
        df['Tuition'] = pd.to_numeric(df['Tuition'], errors='coerce')
        df['GraduationRate'] = pd.to_numeric(df['GraduationRate'], errors='coerce')
        df = df.dropna(subset=['UndergradEnrollment', 'Tuition', 'GraduationRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        total_enrollment = df['UndergradEnrollment'].sum()
        avg_tuition = df['Tuition'].mean()
        
        metrics = {
            "Total Undergraduate Enrollment (across campuses)": total_enrollment,
            "Average Tuition (across campuses)": avg_tuition
        }
        
        insights.append(f"Total Undergraduate Enrollment (across campuses): {total_enrollment:,.0f}")
        insights.append(f"Average Tuition (across campuses): ${avg_tuition:,.0f}")
        
        fig1 = px.scatter(df, x='UndergradEnrollment', y='Tuition', color='CampusName', title='Tuition vs. Enrollment by Campus')
        fig2 = px.box(df, x='CampusName', y='Tuition', title='Tuition Distribution by Campus')
        fig3 = px.bar(df.groupby('CampusName')['GraduationRate'].mean().reset_index(), x='CampusName', y='GraduationRate', title='Average Graduation Rate by Campus')

        visualizations = {
            "Tuition_vs_Enrollment_by_Campus_Scatter": fig1.to_json(),
            "Tuition_Distribution_by_Campus_Box": fig2.to_json(),
            "Average_Graduation_Rate_by_Campus_Bar": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def university_control_debt_and_long_term_income_analysis(df):
    print("\n--- University Control, Debt, and Long-Term Income Analysis ---")
    analysis_type = "University Control, Debt, and Long-Term Income Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'UniversityState': ['UniversityState', 'State', 'STABBR'],
            'UniversityName': ['UniversityName', 'Name', 'INSTNM'],
            'PublicPrivate': ['PublicPrivate', 'Control', 'InstitutionType'],
            'AvgStudentDebt': ['AvgStudentDebt', 'MedianDebt', 'StudentLoanDebt'],
            'AvgIncome25Yr': ['AvgIncome25Yr', 'LongTermEarnings', 'MedianEarnings25Yr']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AvgStudentDebt'] = pd.to_numeric(df['AvgStudentDebt'], errors='coerce')
        df['AvgIncome25Yr'] = pd.to_numeric(df['AvgIncome25Yr'], errors='coerce')
        df = df.dropna(subset=['AvgStudentDebt', 'AvgIncome25Yr'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_debt = df['AvgStudentDebt'].mean()
        avg_income = df['AvgIncome25Yr'].mean()
        
        metrics = {
            "Average Student Debt": avg_debt,
            "Average 25-Year Income": avg_income
        }
        
        insights.append(f"Average Student Debt: ${avg_debt:,.0f}")
        insights.append(f"Average 25-Year Income: ${avg_income:,.0f}")
        
        fig1 = px.scatter(df, x='AvgStudentDebt', y='AvgIncome25Yr', color='PublicPrivate', hover_name='UniversityName', title='25-Year Income vs. Student Debt by Institution Type')
        fig2 = px.box(df, x='PublicPrivate', y='AvgStudentDebt', title='Student Debt Distribution by Institution Type')
        fig3 = px.bar(df.groupby('UniversityState')['AvgIncome25Yr'].mean().nlargest(20).reset_index(), x='UniversityState', y='AvgIncome25Yr', title='Average 25-Year Income by State (Top 20)')
        fig4 = px.box(df, x='PublicPrivate', y='AvgIncome25Yr', title='25-Year Income Distribution by Institution Type')

        visualizations = {
            "25_Year_Income_vs_Student_Debt_Scatter": fig1.to_json(),
            "Student_Debt_Distribution_by_Institution_Type_Box": fig2.to_json(),
            "Average_25_Year_Income_by_State_Bar": fig3.to_json(),
            "25_Year_Income_Distribution_by_Institution_Type_Box": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def college_admissions_and_student_loan_analysis(df):
    print("\n--- College Admissions and Student Loan Analysis ---")
    analysis_type = "College Admissions and Student Loan Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'InstitutionID': ['InstitutionID', 'InstID', 'ID'],
            'Name': ['Name', 'InstitutionName', 'INSTNM'],
            'Region': ['Region', 'State', 'STABBR'],
            'AcceptanceRate': ['AcceptanceRate', 'AdmissionsRate', 'AdmitRate'],
            'SubsidizedLoanPercent': ['SubsidizedLoanPercent', 'PercentSubsidizedLoans', 'SubsidizedLoanShare'],
            'UnsubsidizedLoanPercent': ['UnsubsidizedLoanPercent', 'PercentUnsubsidizedLoans', 'UnsubsidizedLoanShare']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AcceptanceRate'] = pd.to_numeric(df['AcceptanceRate'], errors='coerce')
        df['SubsidizedLoanPercent'] = pd.to_numeric(df['SubsidizedLoanPercent'], errors='coerce')
        df['UnsubsidizedLoanPercent'] = pd.to_numeric(df['UnsubsidizedLoanPercent'], errors='coerce')
        df = df.dropna(subset=['AcceptanceRate', 'SubsidizedLoanPercent', 'UnsubsidizedLoanPercent'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_acceptance_rate = df['AcceptanceRate'].mean()
        avg_subsidized_loan_percent = df['SubsidizedLoanPercent'].mean()
        avg_unsubsidized_loan_percent = df['UnsubsidizedLoanPercent'].mean()

        metrics = {
            "Average Acceptance Rate": avg_acceptance_rate,
            "Average Subsidized Loan Percent": avg_subsidized_loan_percent,
            "Average Unsubsidized Loan Percent": avg_unsubsidized_loan_percent
        }
        
        insights.append(f"Average Acceptance Rate: {avg_acceptance_rate:.2f}%")
        insights.append(f"Average Subsidized Loan Percent: {avg_subsidized_loan_percent:.2f}%")
        insights.append(f"Average Unsubsidized Loan Percent: {avg_unsubsidized_loan_percent:.2f}%")

        fig1 = px.scatter(df, x='AcceptanceRate', y='SubsidizedLoanPercent', hover_name='Name', title='Subsidized Loan Percent vs. Acceptance Rate')
        fig2 = px.box(df, x='Region', y='SubsidizedLoanPercent', title='Subsidized Loan Percent by Region')
        fig3 = px.bar(df.groupby('Region')['SubsidizedLoanPercent'].mean().nlargest(20).reset_index(), x='Region', y='SubsidizedLoanPercent', title='Average Subsidized Loan Percent by Region (Top 20)')
        fig4 = px.scatter(df, x='AcceptanceRate', y='UnsubsidizedLoanPercent', hover_name='Name', title='Unsubsidized Loan Percent vs. Acceptance Rate')

        visualizations = {
            "Subsidized_Loan_Percent_vs_Acceptance_Rate_Scatter": fig1.to_json(),
            "Subsidized_Loan_Percent_by_Region_Box": fig2.to_json(),
            "Average_Subsidized_Loan_Percent_by_Region_Bar": fig3.to_json(),
            "Unsubsidized_Loan_Percent_vs_Acceptance_Rate_Scatter": fig4.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }

def school_district_attendance_and_absenteeism_analysis(df):
    print("\n--- School District Attendance and Absenteeism Analysis ---")
    analysis_type = "School District Attendance and Absenteeism Analysis"
    visualizations = {}
    metrics = {}
    insights = []
    matched_columns_map = {}

    try:
        expected = {
            'DistrictID': ['DistrictID', 'DistrictId', 'ID'],
            'Name': ['Name', 'DistrictName'],
            'AttendanceRate': ['AttendanceRate', 'AvgAttendance'],
            'ChronicAbsenteeismRate': ['ChronicAbsenteeismRate', 'ChronicAbsenceRate']
        }
        df, missing = check_and_rename_columns(df, expected)
        matched_columns_map = {k: v[0] for k, v in expected.items() if v[0] in df.columns or k in df.columns}

        if missing:
            show_missing_columns_warning(missing, expected)
            fallback_result = show_general_insights(df, "General Analysis (Fallback)")
            fallback_result["status"] = "fallback"
            fallback_result["insights"].append(f"Fell back to general insights. Missing required columns for {analysis_type}: {', '.join(missing)}")
            return fallback_result
        
        df['AttendanceRate'] = pd.to_numeric(df['AttendanceRate'], errors='coerce')
        df['ChronicAbsenteeismRate'] = pd.to_numeric(df['ChronicAbsenteeismRate'], errors='coerce')
        df = df.dropna(subset=['AttendanceRate', 'ChronicAbsenteeismRate'])

        if df.empty:
            return {
                "analysis_type": analysis_type, "status": "error", "error": "No sufficient data after cleaning.",
                "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": ["No sufficient data after cleaning."]
            }

        avg_attendance = df['AttendanceRate'].mean()
        avg_absenteeism = df['ChronicAbsenteeismRate'].mean()

        metrics = {
            "Average Attendance Rate": avg_attendance,
            "Average Chronic Absenteeism Rate": avg_absenteeism
        }
        
        insights.append(f"Average Attendance Rate: {avg_attendance:.2f}%")
        insights.append(f"Average Chronic Absenteeism Rate: {avg_absenteeism:.2f}%")

        fig1 = px.scatter(df, x='AttendanceRate', y='ChronicAbsenteeismRate', hover_name='Name', title='Chronic Absenteeism Rate vs. Attendance Rate')
        fig2 = px.histogram(df, x='ChronicAbsenteeismRate', title='Distribution of Chronic Absenteeism Rate')
        fig3 = px.box(df, y='ChronicAbsenteeismRate', title='Chronic Absenteeism Rate Distribution')

        visualizations = {
            "Chronic_Absenteeism_Rate_vs_Attendance_Rate_Scatter": fig1.to_json(),
            "Chronic_Absenteeism_Rate_Distribution_Histogram": fig2.to_json(),
            "Chronic_Absenteeism_Rate_Distribution_Box": fig3.to_json()
        }

        return {
            "analysis_type": analysis_type,
            "status": "success",
            "matched_columns": matched_columns_map,
            "visualizations": visualizations,
            "metrics": convert_to_native_types(metrics),
            "insights": insights
        }

    except Exception as e:
        print(f"[ERROR] in {analysis_type}: {e}")
        return {
            "analysis_type": analysis_type, "status": "error", "error": str(e),
            "matched_columns": matched_columns_map, "visualizations": {}, "metrics": {}, "insights": [f"An error occurred: {e}"]
        }
# Main function to run the analysis
def main_backend(file_path, encoding='utf-8', category=None, analysis=None, specific_analysis_name=None):
    """
    Main function to run educational data analysis
    
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
        return {"error": "Failed to load data file"}
    
    # Mapping of all analysis functions
    analysis_functions = {
        # General analyses
        "Academic Performance": academic_performance_analysis,
        "Demographic Analysis": demographic_analysis,
        "Course Analysis": course_analysis,
        "Attendance Analysis": attendance_analysis,
        "Behavioral Analysis": behavioral_analysis,
        "Program Evaluation": program_evaluation,
        
        # Specific analyses
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
    
    # Determine which analysis to run
    if category == "General" and analysis in analysis_functions:
        result = analysis_functions[analysis](df)
    elif category == "Specific" and specific_analysis_name in analysis_functions:
        result = analysis_functions[specific_analysis_name](df)
    else:
        # Default to general insights
        result = show_general_insights(df)
    
    return result

# Example usage
if __name__ == "__main__":
    # Example usage of the analysis functions
    file_path = "sample_education_data.csv"  # Replace with your actual file path
    
    # Run general insights
    result = main_backend(file_path)
    print("General Insights:", result)
    
    # Run specific analysis
    result = main_backend(
        file_path, 
        category="Specific", 
        specific_analysis_name="Academic Performance Analysis"
    )
    print("Academic Performance Analysis:", result)