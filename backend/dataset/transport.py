import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import pearsonr, linregress
# from fuzzywuzzy import process # Uncomment if you want to use fuzzy matching for column names

# Helper functions (adapted to remove Streamlit)
def safe_numeric_conversion(df, column_name):
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        return df.dropna(subset=[column_name])
    print(f"Warning: Column '{column_name}' not found for numeric conversion.")
    return df

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

def show_missing_columns_warning(missing_columns, matched_columns=None):
    print(f"\n--- WARNING: Required Columns Not Found ---")
    print(f"The following columns are needed but missing: {', '.join(missing_columns)}")
    if matched_columns:
        print("Expected column mappings attempted:")
        for key, value in matched_columns.items():
            if value is None:
                print(f"- '{key}' (e.g., '{key}' or a similar variation)")
    print("Analysis might be incomplete or aborted due to missing required data.")

def load_data(file_path, encoding='utf-8'):
    try:
        if file_path.endswith('.csv'):
            encodings = [encoding, 'latin1', 'iso-8859-1', 'cp1252']
            for enc in encodings:
                try:
                    return pd.read_csv(file_path, encoding=enc)
                except UnicodeDecodeError:
                    continue
            print("Failed to decode file. Try another encoding.")
            return None
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path)
        else:
            print("Unsupported file format")
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def show_general_insights(df, title="General Insights"):
    print(f"--- {title} ---")
    print("📊 Key Metrics:")
    print(f"Total rows: {len(df)}")
    print(f"Total columns: {len(df.columns)}")
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numeric Features: {len(numeric_cols)}")
    print(f"Categorical Features: {len(categorical_cols)}")

    if numeric_cols.any():
        print("\nNumeric Features Analysis:")
        print(df[numeric_cols].describe())
        for col in numeric_cols:
            fig = px.histogram(df, x=col, title=f"Distribution of {col}")
            fig.show()
            fig = px.box(df, y=col, title=f"Box Plot of {col}")
            fig.show()
        
        if len(numeric_cols) >= 2:
            print("\nFeature Correlations:")
            corr = df[numeric_cols].corr()
            print(corr)
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            fig_corr.show()
    
    if categorical_cols.any():
        print("\nCategorical Features Analysis:")
        for col in categorical_cols:
            print(f"\nValue counts for {col}:\n{df[col].value_counts()}")
            value_counts_df = df[col].value_counts().reset_index()
            value_counts_df.columns = [col, 'count']
            fig = px.bar(value_counts_df, x=col, y='count', title=f"Distribution of {col}")
            fig.show()

# --- Transportation Analysis Functions ---

def fleet_analysis(df):
    print("\n--- Fleet Analysis ---")
    expected = {
        'VehicleID': ['VehicleID', 'Vehicle_ID', 'ID'],
        'VehicleType': ['VehicleType', 'Type', 'Model'],
        'PurchaseYear': ['PurchaseYear', 'YearOfPurchase', 'AcquisitionYear'],
        'Mileage': ['Mileage', 'Odometer', 'CurrentMileage'],
        'Status': ['Status', 'OperationalStatus', 'Condition']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['PurchaseYear'] = pd.to_numeric(df['PurchaseYear'], errors='coerce')
    df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')
    df = df.dropna(subset=['PurchaseYear', 'Mileage', 'Status'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_vehicles = len(df)
    avg_mileage = df['Mileage'].mean()
    most_common_type = df['VehicleType'].mode()[0]
    
    print(f"Total Vehicles in Fleet: {total_vehicles}")
    print(f"Average Fleet Mileage: {avg_mileage:,.0f} miles")
    print(f"Most Common Vehicle Type: {most_common_type}")
    
    fig1 = px.histogram(df, x='VehicleType', title='Vehicle Type Distribution')
    fig1.show()
    
    fig2 = px.scatter(df, x='PurchaseYear', y='Mileage', color='VehicleType', hover_name='VehicleID',
                     title='Mileage by Purchase Year and Vehicle Type')
    fig2.show()
    
    fig3 = px.pie(df, names='Status', title='Operational Status Distribution')
    fig3.show()

    return {
        "metrics": {
            "Total Vehicles": total_vehicles,
            "Average Mileage": avg_mileage,
            "Most Common Vehicle Type": most_common_type
        },
        "figures": {
            "Vehicle_Type_Distribution_Histogram": fig1,
            "Mileage_by_Purchase_Year_Scatter": fig2,
            "Operational_Status_Distribution_Pie": fig3
        }
    }

def route_analysis(df):
    print("\n--- Route Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'Route_ID', 'ID'],
        'StartLocation': ['StartLocation', 'Origin'],
        'EndLocation': ['EndLocation', 'Destination'],
        'Distance': ['Distance', 'RouteDistance', 'Miles'],
        'AverageTravelTime': ['AverageTravelTime', 'AvgTime', 'TravelTimeHours']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Distance'] = pd.to_numeric(df['Distance'], errors='coerce')
    df['AverageTravelTime'] = pd.to_numeric(df['AverageTravelTime'], errors='coerce')
    df = df.dropna(subset=['Distance', 'AverageTravelTime'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_routes = len(df)
    avg_distance = df['Distance'].mean()
    longest_route = df.loc[df['Distance'].idxmax(), 'RouteID']
    
    print(f"Total Routes: {total_routes}")
    print(f"Average Route Distance: {avg_distance:.2f} miles")
    print(f"Longest Route: {longest_route}")
    
    fig1 = px.histogram(df, x='Distance', nbins=20, title='Route Distance Distribution')
    fig1.show()
    
    fig2 = px.scatter(df, x='Distance', y='AverageTravelTime', color='StartLocation', hover_name='RouteID',
                     title='Average Travel Time vs. Distance by Start Location')
    fig2.show()
    
    route_counts = df['StartLocation'].value_counts().reset_index()
    route_counts.columns = ['StartLocation', 'count']
    fig3 = px.bar(route_counts, x='StartLocation', y='count', title='Number of Routes Originating from Each Location')
    fig3.show()

    return {
        "metrics": {
            "Total Routes": total_routes,
            "Average Route Distance": avg_distance,
            "Longest Route": longest_route
        },
        "figures": {
            "Route_Distance_Distribution_Histogram": fig1,
            "Travel_Time_vs_Distance_Scatter": fig2,
            "Routes_by_Start_Location_Bar": fig3
        }
    }

def driver_analysis(df):
    print("\n--- Driver Analysis ---")
    expected = {
        'DriverID': ['DriverID', 'Driver_ID', 'ID'],
        'YearsExperience': ['YearsExperience', 'Experience', 'YearsDriving'],
        'SafetyScore': ['SafetyScore', 'AccidentRate', 'IncidentRate'],
        'HoursDrivenLastWeek': ['HoursDrivenLastWeek', 'WeeklyHours', 'HoursDriven']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['YearsExperience'] = pd.to_numeric(df['YearsExperience'], errors='coerce')
    df['SafetyScore'] = pd.to_numeric(df['SafetyScore'], errors='coerce')
    df['HoursDrivenLastWeek'] = pd.to_numeric(df['HoursDrivenLastWeek'], errors='coerce')
    df = df.dropna(subset=['YearsExperience', 'SafetyScore', 'HoursDrivenLastWeek'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_drivers = len(df)
    avg_experience = df['YearsExperience'].mean()
    highest_safety_score_driver = df.loc[df['SafetyScore'].idxmax(), 'DriverID']
    
    print(f"Total Drivers: {total_drivers}")
    print(f"Average Driver Experience: {avg_experience:.1f} years")
    print(f"Driver with Highest Safety Score: {highest_safety_score_driver}")
    
    fig1 = px.histogram(df, x='SafetyScore', nbins=20, title='Safety Score Distribution')
    fig1.show()
    
    fig2 = px.scatter(df, x='YearsExperience', y='SafetyScore', hover_name='DriverID',
                     title='Safety Score vs. Years of Experience')
    fig2.show()
    
    fig3 = px.box(df, y='HoursDrivenLastWeek', title='Weekly Hours Driven Distribution')
    fig3.show()

    return {
        "metrics": {
            "Total Drivers": total_drivers,
            "Average Experience": avg_experience,
            "Highest Safety Score Driver": highest_safety_score_driver
        },
        "figures": {
            "Safety_Score_Distribution_Histogram": fig1,
            "Safety_Score_vs_Experience_Scatter": fig2,
            "Weekly_Hours_Driven_Box": fig3
        }
    }

def fuel_analysis(df):
    print("\n--- Fuel Analysis ---")
    expected = {
        'VehicleID': ['VehicleID', 'Vehicle_ID', 'ID'],
        'FuelType': ['FuelType', 'Type'],
        'FuelConsumedGallons': ['FuelConsumedGallons', 'Gallons', 'FuelUsed'],
        'DistanceDriven': ['DistanceDriven', 'MilesDriven', 'Mileage'],
        'FuelCost': ['FuelCost', 'Cost', 'TotalFuelCost']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FuelConsumedGallons'] = pd.to_numeric(df['FuelConsumedGallons'], errors='coerce')
    df['DistanceDriven'] = pd.to_numeric(df['DistanceDriven'], errors='coerce')
    df['FuelCost'] = pd.to_numeric(df['FuelCost'], errors='coerce')
    df = df.dropna(subset=['FuelConsumedGallons', 'DistanceDriven', 'FuelCost'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_fuel_cost = df['FuelCost'].sum()
    avg_mpg = (df['DistanceDriven'] / df['FuelConsumedGallons']).mean()
    
    print(f"Total Fuel Cost: ${total_fuel_cost:,.2f}")
    print(f"Average MPG: {avg_mpg:.2f}")
    
    fig1 = px.histogram(df, x='FuelType', title='Fuel Type Distribution')
    fig1.show()
    
    fig2 = px.scatter(df, x='DistanceDriven', y='FuelConsumedGallons', color='FuelType', hover_name='VehicleID',
                     title='Fuel Consumption vs. Distance Driven by Fuel Type')
    fig2.show()
    
    fig3 = px.box(df, y='FuelCost', title='Fuel Cost Distribution')
    fig3.show()

    return {
        "metrics": {
            "Total Fuel Cost": total_fuel_cost,
            "Average MPG": avg_mpg
        },
        "figures": {
            "Fuel_Type_Distribution_Histogram": fig1,
            "Fuel_Consumption_vs_Distance_Scatter": fig2,
            "Fuel_Cost_Distribution_Box": fig3
        }
    }

def maintenance_analysis(df):
    print("\n--- Maintenance Analysis ---")
    expected = {
        'VehicleID': ['VehicleID', 'Vehicle_ID', 'ID'],
        'MaintenanceType': ['MaintenanceType', 'Type', 'ServiceType'],
        'MaintenanceCost': ['MaintenanceCost', 'Cost', 'RepairCost'],
        'DateOfMaintenance': ['DateOfMaintenance', 'Date', 'ServiceDate'],
        'MileageAtMaintenance': ['MileageAtMaintenance', 'OdometerAtService', 'ServiceMileage']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['MaintenanceCost'] = pd.to_numeric(df['MaintenanceCost'], errors='coerce')
    df['MileageAtMaintenance'] = pd.to_numeric(df['MileageAtMaintenance'], errors='coerce')
    df['DateOfMaintenance'] = pd.to_datetime(df['DateOfMaintenance'], errors='coerce')
    df = df.dropna(subset=['MaintenanceCost', 'MileageAtMaintenance', 'DateOfMaintenance'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_maintenance_cost = df['MaintenanceCost'].sum()
    avg_maintenance_cost_per_vehicle = df.groupby('VehicleID')['MaintenanceCost'].sum().mean()
    most_common_maintenance_type = df['MaintenanceType'].mode()[0]
    
    print(f"Total Maintenance Cost: ${total_maintenance_cost:,.2f}")
    print(f"Average Maintenance Cost per Vehicle: ${avg_maintenance_cost_per_vehicle:,.2f}")
    print(f"Most Common Maintenance Type: {most_common_maintenance_type}")
    
    fig1 = px.histogram(df, x='MaintenanceType', title='Maintenance Type Frequency')
    fig1.show()
    
    fig2 = px.line(df.groupby(df['DateOfMaintenance'].dt.to_period('M'))['MaintenanceCost'].sum().reset_index().rename(columns={'DateOfMaintenance': 'Month'}),
                   x='Month', y='MaintenanceCost', title='Monthly Maintenance Cost Trend')
    fig2.show()
    
    fig3 = px.box(df, y='MaintenanceCost', color='MaintenanceType', title='Maintenance Cost Distribution by Type')
    fig3.show()

    return {
        "metrics": {
            "Total Maintenance Cost": total_maintenance_cost,
            "Average Maintenance Cost per Vehicle": avg_maintenance_cost_per_vehicle,
            "Most Common Maintenance Type": most_common_maintenance_type
        },
        "figures": {
            "Maintenance_Type_Frequency_Histogram": fig1,
            "Monthly_Maintenance_Cost_Trend_Line": fig2,
            "Maintenance_Cost_Distribution_by_Type_Box": fig3
        }
    }

def delivery_analysis(df):
    print("\n--- Delivery Analysis ---")
    expected = {
        'DeliveryID': ['DeliveryID', 'Delivery_ID', 'ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'DeliveryStatus': ['DeliveryStatus', 'Status', 'CompletionStatus'],
        'DeliveryTimeSeconds': ['DeliveryTimeSeconds', 'DeliveryDuration', 'TimeTakenSeconds'],
        'CustomerRating': ['CustomerRating', 'Rating', 'ServiceRating']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['DeliveryTimeSeconds'] = pd.to_numeric(df['DeliveryTimeSeconds'], errors='coerce')
    df['CustomerRating'] = pd.to_numeric(df['CustomerRating'], errors='coerce')
    df = df.dropna(subset=['DeliveryTimeSeconds', 'CustomerRating'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_deliveries = len(df)
    avg_delivery_time_minutes = df['DeliveryTimeSeconds'].mean() / 60
    avg_customer_rating = df['CustomerRating'].mean()
    
    print(f"Total Deliveries: {total_deliveries}")
    print(f"Average Delivery Time: {avg_delivery_time_minutes:.1f} minutes")
    print(f"Average Customer Rating: {avg_customer_rating:.1f}")
    
    fig1 = px.histogram(df, x='DeliveryStatus', title='Delivery Status Distribution')
    fig1.show()
    
    fig2 = px.box(df, y='DeliveryTimeSeconds', title='Delivery Time Distribution (Seconds)')
    fig2.show()
    
    fig3 = px.histogram(df, x='CustomerRating', nbins=5, title='Customer Rating Distribution')
    fig3.show()

    return {
        "metrics": {
            "Total Deliveries": total_deliveries,
            "Average Delivery Time (minutes)": avg_delivery_time_minutes,
            "Average Customer Rating": avg_customer_rating
        },
        "figures": {
            "Delivery_Status_Distribution_Histogram": fig1,
            "Delivery_Time_Distribution_Box": fig2,
            "Customer_Rating_Distribution_Histogram": fig3
        }
    }

def cost_analysis(df):
    print("\n--- Cost Analysis ---")
    expected = {
        'VehicleID': ['VehicleID', 'Vehicle_ID', 'ID'],
        'CostCategory': ['CostCategory', 'Category', 'ExpenseType'],
        'Amount': ['Amount', 'Cost', 'ExpenseAmount'],
        'Month': ['Month', 'ReportingMonth', 'DateMonth']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce')
    df = df.dropna(subset=['Amount', 'CostCategory'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_cost = df['Amount'].sum()
    avg_cost_per_vehicle = df.groupby('VehicleID')['Amount'].sum().mean()
    highest_cost_category = df.groupby('CostCategory')['Amount'].sum().idxmax()
    
    print(f"Total Overall Cost: ${total_cost:,.2f}")
    print(f"Average Cost per Vehicle: ${avg_cost_per_vehicle:,.2f}")
    print(f"Highest Cost Category: {highest_cost_category}")
    
    fig1 = px.pie(df, names='CostCategory', values='Amount', title='Cost Distribution by Category')
    fig1.show()
    
    if 'Month' in df.columns:
        monthly_costs = df.groupby('Month')['Amount'].sum().reset_index()
        fig2 = px.line(monthly_costs, x='Month', y='Amount', title='Monthly Cost Trend')
        fig2.show()
    else:
        print("Note: 'Month' column not found for monthly trend analysis.")
    
    fig3 = px.box(df, y='Amount', color='CostCategory', title='Cost Distribution by Category')
    fig3.show()

    return {
        "metrics": {
            "Total Overall Cost": total_cost,
            "Average Cost per Vehicle": avg_cost_per_vehicle,
            "Highest Cost Category": highest_cost_category
        },
        "figures": {
            "Cost_Distribution_by_Category_Pie": fig1,
            "Monthly_Cost_Trend_Line": fig2 if 'Month' in df.columns else None,
            "Cost_Distribution_by_Category_Box": fig3
        }
    }

def safety_analysis(df):
    print("\n--- Safety Analysis ---")
    expected = {
        'IncidentID': ['IncidentID', 'ID', 'ReportID'],
        'IncidentType': ['IncidentType', 'Type', 'AccidentType'],
        'Date': ['Date', 'IncidentDate', 'ReportDate'],
        'Severity': ['Severity', 'ImpactLevel', 'DamageSeverity'],
        'Location': ['Location', 'IncidentLocation', 'Address']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'IncidentType', 'Severity'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_incidents = len(df)
    most_common_incident_type = df['IncidentType'].mode()[0]
    avg_severity = df['Severity'].mean() if pd.api.types.is_numeric_dtype(df['Severity']) else "N/A"
    
    print(f"Total Incidents Recorded: {total_incidents}")
    print(f"Most Common Incident Type: {most_common_incident_type}")
    print(f"Average Incident Severity: {avg_severity:.1f}" if pd.api.types.is_numeric_dtype(df['Severity']) else f"Average Incident Severity: {avg_severity}")
    
    fig1 = px.histogram(df, x='IncidentType', title='Incident Type Frequency')
    fig1.show()
    
    df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
    monthly_incidents = df.groupby('YearMonth').size().reset_index(name='Count')
    fig2 = px.line(monthly_incidents, x='YearMonth', y='Count', title='Monthly Incident Trend')
    fig2.show()
    
    fig3 = px.histogram(df, x='Severity', title='Incident Severity Distribution')
    fig3.show()

    return {
        "metrics": {
            "Total Incidents": total_incidents,
            "Most Common Incident Type": most_common_incident_type,
            "Average Incident Severity": avg_severity
        },
        "figures": {
            "Incident_Type_Frequency_Histogram": fig1,
            "Monthly_Incident_Trend_Line": fig2,
            "Incident_Severity_Distribution_Histogram": fig3
        }
    }

def commuter_transportation_mode_choice_analysis(df):
    print("\n--- Commuter Transportation Mode Choice Analysis ---")
    expected = {
        'UserID': ['UserID', 'CommuterID', 'ID'],
        'ModeChoice': ['ModeChoice', 'TransportationMode', 'PrimaryMode'],
        'CommuteDistance': ['CommuteDistance', 'DistanceMiles', 'DistanceKm'],
        'CommuteTime': ['CommuteTime', 'TimeMinutes', 'DurationHours'],
        'AgeGroup': ['AgeGroup', 'AgeCategory'],
        'IncomeLevel': ['IncomeLevel', 'HouseholdIncome']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['CommuteDistance'] = pd.to_numeric(df['CommuteDistance'], errors='coerce')
    df['CommuteTime'] = pd.to_numeric(df['CommuteTime'], errors='coerce')
    df = df.dropna(subset=['ModeChoice', 'CommuteDistance', 'CommuteTime'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    most_popular_mode = df['ModeChoice'].mode()[0]
    avg_commute_distance = df['CommuteDistance'].mean()
    avg_commute_time = df['CommuteTime'].mean()
    
    print(f"Most Popular Commute Mode: {most_popular_mode}")
    print(f"Average Commute Distance: {avg_commute_distance:.2f} miles")
    print(f"Average Commute Time: {avg_commute_time:.1f} minutes")
    
    fig1 = px.pie(df, names='ModeChoice', title='Commuter Mode Choice Distribution')
    fig1.show()
    
    fig2 = px.box(df, x='ModeChoice', y='CommuteTime', title='Commute Time Distribution by Mode')
    fig2.show()
    
    fig3 = px.scatter(df, x='CommuteDistance', y='CommuteTime', color='ModeChoice', hover_name='UserID',
                     title='Commute Time vs. Distance by Mode')
    fig3.show()

    if 'AgeGroup' in df.columns:
        fig4 = px.histogram(df, x='AgeGroup', color='ModeChoice', barmode='group', title='Mode Choice by Age Group')
        fig4.show()
    
    if 'IncomeLevel' in df.columns:
        fig5 = px.histogram(df, x='IncomeLevel', color='ModeChoice', barmode='group', title='Mode Choice by Income Level')
        fig5.show()

    return {
        "metrics": {
            "Most Popular Commute Mode": most_popular_mode,
            "Average Commute Distance": avg_commute_distance,
            "Average Commute Time": avg_commute_time
        },
        "figures": {
            "Mode_Choice_Distribution_Pie": fig1,
            "Commute_Time_by_Mode_Box": fig2,
            "Commute_Time_vs_Distance_Scatter": fig3,
            "Mode_Choice_by_Age_Group_Histogram": fig4 if 'AgeGroup' in df.columns else None,
            "Mode_Choice_by_Income_Level_Histogram": fig5 if 'IncomeLevel' in df.columns else None
        }
    }

def public_bus_performance_and_delay_analysis(df):
    print("\n--- Public Bus Performance and Delay Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'Route_ID', 'ID'],
        'BusID': ['BusID', 'Bus_ID', 'VehicleID'],
        'ScheduledArrivalTime': ['ScheduledArrivalTime', 'ScheduledTime', 'PlannedArrival'],
        'ActualArrivalTime': ['ActualArrivalTime', 'ActualTime', 'RealArrival'],
        'TripDuration': ['TripDuration', 'TravelTimeMinutes'],
        'PassengerCount': ['PassengerCount', 'Ridership', 'NumPassengers']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ScheduledArrivalTime'] = pd.to_datetime(df['ScheduledArrivalTime'], errors='coerce')
    df['ActualArrivalTime'] = pd.to_datetime(df['ActualArrivalTime'], errors='coerce')
    df['TripDuration'] = pd.to_numeric(df['TripDuration'], errors='coerce')
    df['PassengerCount'] = pd.to_numeric(df['PassengerCount'], errors='coerce')
    df = df.dropna(subset=['ScheduledArrivalTime', 'ActualArrivalTime', 'TripDuration', 'PassengerCount'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    df['DelayMinutes'] = (df['ActualArrivalTime'] - df['ScheduledArrivalTime']).dt.total_seconds() / 60
    
    avg_delay = df['DelayMinutes'].mean()
    on_time_percentage = (df['DelayMinutes'] <= 5).mean() * 100 # Assuming <= 5 min delay is on-time
    avg_ridership = df['PassengerCount'].mean()
    
    print(f"Average Bus Delay: {avg_delay:.2f} minutes")
    print(f"On-Time Performance: {on_time_percentage:.2f}%")
    print(f"Average Ridership per Trip: {avg_ridership:.1f} passengers")
    
    fig1 = px.histogram(df, x='DelayMinutes', nbins=30, title='Distribution of Bus Delays (Minutes)')
    fig1.show()
    
    fig2 = px.box(df, x='RouteID', y='DelayMinutes', title='Delay Distribution by Route')
    fig2.show()
    
    fig3 = px.scatter(df, x='TripDuration', y='PassengerCount', color='RouteID', hover_name='BusID',
                     title='Passenger Count vs. Trip Duration by Route')
    fig3.show()

    return {
        "metrics": {
            "Average Bus Delay": avg_delay,
            "On-Time Performance": on_time_percentage,
            "Average Ridership per Trip": avg_ridership
        },
        "figures": {
            "Bus_Delays_Distribution_Histogram": fig1,
            "Delay_Distribution_by_Route_Box": fig2,
            "Passenger_Count_vs_Trip_Duration_Scatter": fig3
        }
    }

def global_air_transport_passenger_trends_analysis(df):
    print("\n--- Global Air Transport Passenger Trends Analysis ---")
    expected = {
        'Country': ['Country', 'Region'],
        'Year': ['Year', 'CalendarYear'],
        'TotalPassengers': ['TotalPassengers', 'Passengers', 'NumPassengers'],
        'CargoTons': ['CargoTons', 'CargoWeight', 'FreightTons'] # Added cargo for completeness
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TotalPassengers'] = pd.to_numeric(df['TotalPassengers'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['CargoTons'] = pd.to_numeric(df['CargoTons'], errors='coerce')
    df = df.dropna(subset=['TotalPassengers', 'Year'])
    df = df.sort_values(by='Year')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_passengers_overall = df['TotalPassengers'].sum()
    latest_year_passengers = df.loc[df['Year'].idxmax(), 'TotalPassengers']
    
    print(f"Total Passengers (Overall): {total_passengers_overall:,.0f}")
    print(f"Latest Year ({df['Year'].max()}) Passengers: {latest_year_passengers:,.0f}")
    
    fig1 = px.line(df, x='Year', y='TotalPassengers', color='Country', title='Global Passenger Trends by Country')
    fig1.show()
    
    yearly_global_passengers = df.groupby('Year')['TotalPassengers'].sum().reset_index()
    fig2 = px.bar(yearly_global_passengers, x='Year', y='TotalPassengers', title='Total Global Air Passengers by Year')
    fig2.show()
    
    if 'CargoTons' in df.columns and not df['CargoTons'].dropna().empty:
        fig3 = px.line(df, x='Year', y='CargoTons', color='Country', title='Global Air Cargo Trends by Country')
        fig3.show()
    else:
        print("Note: 'CargoTons' column not found for cargo trend analysis.")

    return {
        "metrics": {
            "Total Passengers Overall": total_passengers_overall,
            "Latest Year Passengers": latest_year_passengers
        },
        "figures": {
            "Global_Passenger_Trends_Line": fig1,
            "Total_Global_Air_Passengers_Bar": fig2,
            "Global_Air_Cargo_Trends_Line": fig3 if 'CargoTons' in df.columns and not df['CargoTons'].dropna().empty else None
        }
    }

def airline_directory_and_operational_status_analysis(df):
    print("\n--- Airline Directory and Operational Status Analysis ---")
    expected = {
        'AirlineID': ['AirlineID', 'ID', 'AirlineCode'],
        'AirlineName': ['AirlineName', 'Name'],
        'Country': ['Country', 'BaseCountry'],
        'OperationalStatus': ['OperationalStatus', 'Status', 'ActiveStatus'],
        'FleetSize': ['FleetSize', 'NumAircraft', 'AircraftCount']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FleetSize'] = pd.to_numeric(df['FleetSize'], errors='coerce')
    df = df.dropna(subset=['OperationalStatus', 'FleetSize'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_airlines = len(df)
    active_airlines_count = (df['OperationalStatus'] == 'Active').sum()
    avg_fleet_size_active = df[df['OperationalStatus'] == 'Active']['FleetSize'].mean()
    
    print(f"Total Airlines in Directory: {total_airlines}")
    print(f"Active Airlines: {active_airlines_count}")
    print(f"Average Fleet Size of Active Airlines: {avg_fleet_size_active:.0f} aircraft")
    
    fig1 = px.pie(df, names='OperationalStatus', title='Airline Operational Status Distribution')
    fig1.show()
    
    airlines_by_country = df['Country'].value_counts().reset_index()
    airlines_by_country.columns = ['Country', 'Count']
    fig2 = px.bar(airlines_by_country.head(20), x='Country', y='Count', title='Number of Airlines by Country (Top 20)')
    fig2.show()
    
    fig3 = px.box(df, x='OperationalStatus', y='FleetSize', title='Fleet Size Distribution by Operational Status')
    fig3.show()

    return {
        "metrics": {
            "Total Airlines": total_airlines,
            "Active Airlines": active_airlines_count,
            "Average Fleet Size of Active Airlines": avg_fleet_size_active
        },
        "figures": {
            "Airline_Operational_Status_Pie": fig1,
            "Airlines_by_Country_Bar": fig2,
            "Fleet_Size_by_Operational_Status_Box": fig3
        }
    }

def public_transit_fare_and_journey_type_analysis(df):
    print("\n--- Public Transit Fare and Journey Type Analysis ---")
    expected = {
        'FareID': ['FareID', 'ID'],
        'JourneyType': ['JourneyType', 'TripType', 'PassType'],
        'FareAmount': ['FareAmount', 'Cost', 'Price'],
        'RouteID': ['RouteID', 'Route_ID'],
        'RidershipCount': ['RidershipCount', 'Passengers', 'NumRiders']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FareAmount'] = pd.to_numeric(df['FareAmount'], errors='coerce')
    df['RidershipCount'] = pd.to_numeric(df['RidershipCount'], errors='coerce')
    df = df.dropna(subset=['FareAmount', 'RidershipCount', 'JourneyType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_fare_amount = df['FareAmount'].mean()
    most_common_journey_type = df['JourneyType'].mode()[0]
    total_ridership_by_fare = df.groupby('JourneyType')['RidershipCount'].sum().sum() # Sum of ridership across all fare types
    
    print(f"Average Fare Amount: ${avg_fare_amount:.2f}")
    print(f"Most Common Journey Type: {most_common_journey_type}")
    print(f"Total Ridership by Fare Type: {total_ridership_by_fare:,.0f}")
    
    fig1 = px.histogram(df, x='JourneyType', y='RidershipCount', title='Ridership by Journey Type')
    fig1.show()
    
    fig2 = px.box(df, x='JourneyType', y='FareAmount', title='Fare Amount Distribution by Journey Type')
    fig2.show()
    
    if 'RouteID' in df.columns:
        ridership_by_route_fare = df.groupby(['RouteID', 'JourneyType'])['RidershipCount'].sum().reset_index()
        fig3 = px.bar(ridership_by_route_fare.sort_values('RidershipCount', ascending=False).head(20),
                      x='RouteID', y='RidershipCount', color='JourneyType', title='Top Routes by Ridership and Journey Type')
        fig3.show()
    else:
        print("Note: 'RouteID' column not found for route-specific ridership analysis.")

    return {
        "metrics": {
            "Average Fare Amount": avg_fare_amount,
            "Most Common Journey Type": most_common_journey_type,
            "Total Ridership by Fare Type": total_ridership_by_fare
        },
        "figures": {
            "Ridership_by_Journey_Type_Histogram": fig1,
            "Fare_Amount_by_Journey_Type_Box": fig2,
            "Top_Routes_by_Ridership_and_Journey_Type_Bar": fig3 if 'RouteID' in df.columns else None
        }
    }

def regional_vehicle_registration_trend_analysis(df):
    print("\n--- Regional Vehicle Registration Trend Analysis ---")
    expected = {
        'Region': ['Region', 'County', 'State'],
        'Year': ['Year', 'CalendarYear', 'RegistrationYear'],
        'VehicleType': ['VehicleType', 'Type', 'Category'],
        'RegisteredVehiclesCount': ['RegisteredVehiclesCount', 'Count', 'NumVehicles']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['RegisteredVehiclesCount'] = pd.to_numeric(df['RegisteredVehiclesCount'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df.dropna(subset=['RegisteredVehiclesCount', 'Year', 'Region', 'VehicleType'])
    df = df.sort_values(by='Year')

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_registered_vehicles = df['RegisteredVehiclesCount'].sum()
    latest_year_registrations = df[df['Year'] == df['Year'].max()]['RegisteredVehiclesCount'].sum()
    
    print(f"Total Registered Vehicles (Overall): {total_registered_vehicles:,.0f}")
    print(f"Latest Year ({df['Year'].max()}) Registrations: {latest_year_registrations:,.0f}")
    
    fig1 = px.line(df, x='Year', y='RegisteredVehiclesCount', color='Region',
                   title='Vehicle Registration Trends by Region')
    fig1.show()
    
    fig2 = px.bar(df.groupby('VehicleType')['RegisteredVehiclesCount'].sum().reset_index().sort_values('RegisteredVehiclesCount', ascending=False),
                  x='VehicleType', y='RegisteredVehiclesCount', title='Total Registered Vehicles by Type')
    fig2.show()
    
    regional_yearly_registrations = df.groupby(['Region', 'Year'])['RegisteredVehiclesCount'].sum().reset_index()
    fig3 = px.line(regional_yearly_registrations, x='Year', y='RegisteredVehiclesCount', color='Region', line_group='Region',
                   title='Regional Vehicle Registration Trends Over Time')
    fig3.show()

    return {
        "metrics": {
            "Total Registered Vehicles Overall": total_registered_vehicles,
            "Latest Year Registrations": latest_year_registrations
        },
        "figures": {
            "Vehicle_Registration_Trends_by_Region_Line": fig1,
            "Total_Registered_Vehicles_by_Type_Bar": fig2,
            "Regional_Vehicle_Registration_Trends_Over_Time_Line": fig3
        }
    }

def transportation_user_survey_response_analysis(df):
    print("\n--- Transportation User Survey Response Analysis ---")
    expected = {
        'SurveyID': ['SurveyID', 'ID'],
        'Age': ['Age', 'RespondentAge'],
        'Gender': ['Gender', 'RespondentGender'],
        'SatisfactionRating': ['SatisfactionRating', 'Rating', 'OverallSatisfaction'],
        'FrequencyOfUse': ['FrequencyOfUse', 'UsageFrequency'],
        'PrimaryTransportationMode': ['PrimaryTransportationMode', 'MainMode']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['SatisfactionRating'] = pd.to_numeric(df['SatisfactionRating'], errors='coerce')
    df = df.dropna(subset=['SatisfactionRating', 'Age', 'Gender', 'PrimaryTransportationMode'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_responses = len(df)
    avg_satisfaction = df['SatisfactionRating'].mean()
    most_freq_mode_in_survey = df['PrimaryTransportationMode'].mode()[0]
    
    print(f"Total Survey Responses: {total_responses}")
    print(f"Average Satisfaction Rating: {avg_satisfaction:.2f}")
    print(f"Most Frequent Primary Transportation Mode in Survey: {most_freq_mode_in_survey}")
    
    fig1 = px.histogram(df, x='SatisfactionRating', nbins=5, title='Satisfaction Rating Distribution')
    fig1.show()
    
    fig2 = px.histogram(df, x='PrimaryTransportationMode', color='Gender', barmode='group',
                       title='Primary Transportation Mode by Gender')
    fig2.show()
    
    if 'Age' in df.columns:
        fig3 = px.box(df, x='PrimaryTransportationMode', y='Age', title='Age Distribution by Primary Transportation Mode')
        fig3.show()

    return {
        "metrics": {
            "Total Survey Responses": total_responses,
            "Average Satisfaction Rating": avg_satisfaction,
            "Most Frequent Primary Transportation Mode": most_freq_mode_in_survey
        },
        "figures": {
            "Satisfaction_Rating_Distribution_Histogram": fig1,
            "Primary_Transportation_Mode_by_Gender_Histogram": fig2,
            "Age_Distribution_by_Primary_Transportation_Mode_Box": fig3 if 'Age' in df.columns else None
        }
    }

def bus_route_schedule_analysis(df):
    print("\n--- Bus Route Schedule Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'Route_ID', 'ID'],
        'TripID': ['TripID', 'Trip_ID'],
        'DepartureTime': ['DepartureTime', 'ScheduledDeparture'],
        'ArrivalTime': ['ArrivalTime', 'ScheduledArrival'],
        'ServiceDay': ['ServiceDay', 'DayOfWeek', 'Day'],
        'HeadwayMinutes': ['HeadwayMinutes', 'FrequencyMinutes', 'Interval']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['HeadwayMinutes'] = pd.to_numeric(df['HeadwayMinutes'], errors='coerce')
    df = df.dropna(subset=['DepartureTime', 'ArrivalTime', 'HeadwayMinutes', 'RouteID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_trips = len(df)
    avg_headway = df['HeadwayMinutes'].mean()
    most_frequent_route = df.groupby('RouteID')['HeadwayMinutes'].mean().nsmallest(1).index[0]
    
    print(f"Total Scheduled Trips: {total_trips}")
    print(f"Average Headway (Frequency) Across Routes: {avg_headway:.1f} minutes")
    print(f"Most Frequent Route (lowest average headway): {most_frequent_route}")
    
    fig1 = px.histogram(df, x='HeadwayMinutes', nbins=20, title='Distribution of Headway Minutes')
    fig1.show()
    
    fig2 = px.box(df, x='RouteID', y='HeadwayMinutes', title='Headway Distribution by Route')
    fig2.show()
    
    if 'ServiceDay' in df.columns:
        fig3 = px.histogram(df, x='ServiceDay', title='Scheduled Trips by Service Day')
        fig3.show()

    return {
        "metrics": {
            "Total Scheduled Trips": total_trips,
            "Average Headway": avg_headway,
            "Most Frequent Route": most_frequent_route
        },
        "figures": {
            "Headway_Distribution_Histogram": fig1,
            "Headway_Distribution_by_Route_Box": fig2,
            "Scheduled_Trips_by_Service_Day_Histogram": fig3 if 'ServiceDay' in df.columns else None
        }
    }

def public_transit_station_ridership_analysis(df):
    print("\n--- Public Transit Station Ridership Analysis ---")
    expected = {
        'StationID': ['StationID', 'StopID', 'ID'],
        'StationName': ['StationName', 'StopName', 'Name'],
        'Date': ['Date', 'RidershipDate'],
        'DailyRidership': ['DailyRidership', 'Ridership', 'Boardings'],
        'Line': ['Line', 'RouteName', 'ServiceLine']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['DailyRidership'] = pd.to_numeric(df['DailyRidership'], errors='coerce')
    df = df.dropna(subset=['Date', 'DailyRidership', 'StationName'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_ridership = df['DailyRidership'].sum()
    busiest_station = df.groupby('StationName')['DailyRidership'].sum().idxmax()
    avg_daily_ridership = df['DailyRidership'].mean()
    
    print(f"Total Ridership Recorded: {total_ridership:,.0f}")
    print(f"Busiest Station: {busiest_station}")
    print(f"Average Daily Ridership per Station: {avg_daily_ridership:.0f}")
    
    fig1 = px.histogram(df, x='DailyRidership', nbins=30, title='Distribution of Daily Ridership')
    fig1.show()
    
    top_stations = df.groupby('StationName')['DailyRidership'].sum().nlargest(20).reset_index()
    fig2 = px.bar(top_stations, x='StationName', y='DailyRidership', title='Top 20 Busiest Stations by Total Ridership')
    fig2.show()
    
    ridership_trend = df.groupby(df['Date'].dt.to_period('M'))['DailyRidership'].sum().reset_index().rename(columns={'Date':'Month'})
    fig3 = px.line(ridership_trend, x='Month', y='DailyRidership', title='Monthly Ridership Trend')
    fig3.show()

    if 'Line' in df.columns:
        fig4 = px.box(df, x='Line', y='DailyRidership', title='Daily Ridership by Transit Line')
        fig4.show()

    return {
        "metrics": {
            "Total Ridership Recorded": total_ridership,
            "Busiest Station": busiest_station,
            "Average Daily Ridership per Station": avg_daily_ridership
        },
        "figures": {
            "Daily_Ridership_Distribution_Histogram": fig1,
            "Top_20_Busiest_Stations_Bar": fig2,
            "Monthly_Ridership_Trend_Line": fig3,
            "Daily_Ridership_by_Transit_Line_Box": fig4 if 'Line' in df.columns else None
        }
    }

def county_level_transportation_infrastructure_and_commute_analysis(df):
    print("\n--- County-Level Transportation Infrastructure and Commute Analysis ---")
    expected = {
        'CountyName': ['CountyName', 'County', 'Name'],
        'State': ['State', 'STABBR'],
        'RoadMiles': ['RoadMiles', 'TotalRoadMiles', 'InfrastructureMiles'],
        'PublicTransitAccessScore': ['PublicTransitAccessScore', 'TransitScore', 'AccessScore'],
        'AverageCommuteTime': ['AverageCommuteTime', 'AvgCommuteMins', 'CommuteTimeHours'],
        'Population': ['Population', 'TotalPopulation', 'CountyPopulation']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['RoadMiles'] = pd.to_numeric(df['RoadMiles'], errors='coerce')
    df['PublicTransitAccessScore'] = pd.to_numeric(df['PublicTransitAccessScore'], errors='coerce')
    df['AverageCommuteTime'] = pd.to_numeric(df['AverageCommuteTime'], errors='coerce')
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df = df.dropna(subset=['RoadMiles', 'PublicTransitAccessScore', 'AverageCommuteTime', 'Population'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_road_miles = df['RoadMiles'].sum()
    avg_commute_time_overall = df['AverageCommuteTime'].mean()
    county_highest_transit_access = df.loc[df['PublicTransitAccessScore'].idxmax(), 'CountyName']
    
    print(f"Total Road Miles (Overall): {total_road_miles:,.0f} miles")
    print(f"Average Commute Time (Overall): {avg_commute_time_overall:.1f} minutes")
    print(f"County with Highest Public Transit Access: {county_highest_transit_access}")
    
    fig1 = px.scatter(df, x='RoadMiles', y='AverageCommuteTime', color='PublicTransitAccessScore', hover_name='CountyName',
                     title='Average Commute Time vs. Road Miles (colored by Transit Access)')
    fig1.show()
    
    fig2 = px.box(df, x='State', y='AverageCommuteTime', title='Average Commute Time by State')
    fig2.show()
    
    fig3 = px.bar(df.sort_values('PublicTransitAccessScore', ascending=False).head(20), x='CountyName', y='PublicTransitAccessScore',
                  title='Top 20 Counties by Public Transit Access Score')
    fig3.show()
    
    fig4 = px.scatter(df, x='Population', y='RoadMiles', size='PublicTransitAccessScore', hover_name='CountyName',
                     title='Road Miles vs. Population (sized by Transit Access)')
    fig4.show()

    return {
        "metrics": {
            "Total Road Miles Overall": total_road_miles,
            "Average Commute Time Overall": avg_commute_time_overall,
            "County with Highest Public Transit Access": county_highest_transit_access
        },
        "figures": {
            "Commute_Time_vs_Road_Miles_Scatter": fig1,
            "Average_Commute_Time_by_State_Box": fig2,
            "Top_20_Counties_by_Public_Transit_Access_Bar": fig3,
            "Road_Miles_vs_Population_Scatter": fig4
        }
    }

def transit_agency_information_analysis(df):
    print("\n--- Transit Agency Information Analysis ---")
    expected = {
        'AgencyID': ['AgencyID', 'ID'],
        'AgencyName': ['AgencyName', 'Name'],
        'AgencyURL': ['AgencyURL', 'URL'],
        'AgencyTimezone': ['AgencyTimezone', 'Timezone'],
        'AgencyFareURL': ['AgencyFareURL', 'FareURL', 'FaresPage']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    if df.empty:
        print("No data available for this analysis.")
        return {"message": "No data available."}

    total_agencies = len(df)
    unique_timezones = df['AgencyTimezone'].nunique()
    
    print(f"Total Transit Agencies: {total_agencies}")
    print(f"Unique Timezones Represented: {unique_timezones}")
    
    agency_counts_by_timezone = df['AgencyTimezone'].value_counts().reset_index()
    agency_counts_by_timezone.columns = ['Timezone', 'Count']
    fig1 = px.bar(agency_counts_by_timezone, x='Timezone', y='Count', title='Number of Agencies by Timezone')
    fig1.show()
    
    if 'AgencyURL' in df.columns:
        print("\nSample Agency URLs:")
        for url in df['AgencyURL'].head().tolist():
            print(f"- {url}")

    print("\nAgencies and their URLs/Timezones:")
    if 'AgencyName' in df.columns and 'AgencyURL' in df.columns and 'AgencyTimezone' in df.columns:
        for index, row in df.head(10).iterrows():
            print(f"  {row['AgencyName']} | URL: {row['AgencyURL']} | Timezone: {row['AgencyTimezone']}")
    elif 'AgencyName' in df.columns and 'AgencyURL' in df.columns:
        for index, row in df.head(10).iterrows():
            print(f"  {row['AgencyName']} | URL: {row['AgencyURL']}")
    else:
        print("Not enough columns to display agency details.")

    return {
        "metrics": {
            "Total Transit Agencies": total_agencies,
            "Unique Timezones Represented": unique_timezones
        },
        "figures": {
            "Agencies_by_Timezone_Bar": fig1
        },
        "details": {
            "Sample Agency URLs": df['AgencyURL'].head().tolist() if 'AgencyURL' in df.columns else None
        }
    }

def public_transit_route_definition_analysis(df):
    print("\n--- Public Transit Route Definition Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'Route_ID', 'ID'],
        'AgencyID': ['AgencyID', 'Agency_ID'],
        'RouteShortName': ['RouteShortName', 'ShortName', 'RouteNum'],
        'RouteLongName': ['RouteLongName', 'LongName', 'RouteDescription'],
        'RouteType': ['RouteType', 'Type', 'TransitType'],
        'RouteColor': ['RouteColor', 'Color']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_routes = len(df)
    unique_route_types = df['RouteType'].nunique()
    most_common_route_type = df['RouteType'].mode()[0]
    
    print(f"Total Defined Routes: {total_routes}")
    print(f"Unique Route Types: {unique_route_types}")
    print(f"Most Common Route Type: {most_common_route_type}")
    
    fig1 = px.histogram(df, x='RouteType', title='Distribution of Route Types')
    fig1.show()
    
    routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
    fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                  title='Number of Routes per Agency (Top 20)')
    fig2.show()
    
    if 'RouteShortName' in df.columns and 'RouteLongName' in df.columns:
        print("\nSample Routes:")
        for index, row in df.head(10).iterrows():
            print(f"- {row['RouteShortName']}: {row['RouteLongName']} (Type: {row['RouteType']})")

    return {
        "metrics": {
            "Total Defined Routes": total_routes,
            "Unique Route Types": unique_route_types,
            "Most Common Route Type": most_common_route_type
        },
        "figures": {
            "Route_Types_Distribution_Histogram": fig1,
            "Routes_per_Agency_Bar": fig2
        }
    }

def transit_trip_schedule_and_accessibility_analysis(df):
    print("\n--- Transit Trip Schedule and Accessibility Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID', 'ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'ServiceCode': ['ServiceCode', 'ServiceID'],
        'WheelchairAccessible': ['WheelchairAccessible', 'Accessible', 'WheelchairBoarding'],
        'BikesAllowed': ['BikesAllowed', 'BikeAccessible', 'BikeRack'],
        'TripStartTime': ['TripStartTime', 'StartTime', 'ScheduledStart'],
        'TripEndTime': ['TripEndTime', 'EndTime', 'ScheduledEnd']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TripStartTime'] = pd.to_datetime(df['TripStartTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['TripEndTime'] = pd.to_datetime(df['TripEndTime'], format='%H:%M:%S', errors='coerce').dt.time
    
    df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    
    df = df.dropna(subset=['TripID', 'RouteID', 'TripStartTime', 'TripEndTime', 'WheelchairAccessible', 'BikesAllowed'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_trips = len(df)
    accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
    bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
    
    print(f"Total Trips Scheduled: {total_trips}")
    print(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
    print(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
    
    fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
    fig1.show()
    
    fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
    fig2.show()
    
    # Example: Daily trip count
    if 'ServiceCode' in df.columns:
        trips_by_service = df.groupby('ServiceCode').size().reset_index(name='Count')
        fig3 = px.bar(trips_by_service, x='ServiceCode', y='Count', title='Number of Trips by Service Code')
        fig3.show()

    return {
        "metrics": {
            "Total Trips Scheduled": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        },
        "figures": {
            "Wheelchair_Accessibility_Pie": fig1,
            "Bike_Accessibility_Pie": fig2,
            "Trips_by_Service_Code_Bar": fig3 if 'ServiceCode' in df.columns else None
        }
    }

def transit_stop_location_and_information_analysis(df):
    print("\n--- Transit Stop Location and Information Analysis ---")
    expected = {
        'StopID': ['StopID', 'ID', 'Stop_ID'],
        'StopName': ['StopName', 'Name', 'LocationName'],
        'StopLat': ['StopLat', 'Latitude', 'Lat'],
        'StopLon': ['StopLon', 'Longitude', 'Lon'],
        'LocationType': ['LocationType', 'Type', 'StopType'],
        'WheelchairBoarding': ['WheelchairBoarding', 'AccessibleBoarding']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StopLat'] = pd.to_numeric(df['StopLat'], errors='coerce')
    df['StopLon'] = pd.to_numeric(df['StopLon'], errors='coerce')
    df['WheelchairBoarding'] = df['WheelchairBoarding'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['StopLat', 'StopLon', 'StopName', 'LocationType', 'WheelchairBoarding'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_stops = len(df)
    accessible_stops_pct = (df['WheelchairBoarding'] == 'Yes').mean() * 100
    unique_stop_types = df['LocationType'].nunique()
    
    print(f"Total Transit Stops: {total_stops}")
    print(f"Percentage of Wheelchair Accessible Stops: {accessible_stops_pct:.2f}%")
    print(f"Unique Location Types for Stops: {unique_stop_types}")
    
    fig1 = px.scatter_mapbox(df, lat='StopLat', lon='StopLon', hover_name='StopName', color='LocationType',
                             zoom=10, title='Transit Stop Locations by Type',
                             mapbox_style="carto-positron")
    fig1.show()
    
    fig2 = px.pie(df, names='WheelchairBoarding', title='Wheelchair Boarding Availability at Stops')
    fig2.show()
    
    fig3 = px.histogram(df, x='LocationType', title='Distribution of Stop Location Types')
    fig3.show()

    return {
        "metrics": {
            "Total Transit Stops": total_stops,
            "Wheelchair Accessible Stops (%)": accessible_stops_pct,
            "Unique Location Types for Stops": unique_stop_types
        },
        "figures": {
            "Transit_Stop_Locations_Map": fig1,
            "Wheelchair_Boarding_Availability_Pie": fig2,
            "Stop_Location_Types_Distribution_Histogram": fig3
        }
    }

def transit_stop_time_and_sequence_analysis(df):
    print("\n--- Transit Stop Time and Sequence Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'StopID': ['StopID', 'Stop_ID'],
        'ArrivalTime': ['ArrivalTime', 'ScheduledArrival', 'Arrival'],
        'DepartureTime': ['DepartureTime', 'ScheduledDeparture', 'Departure'],
        'StopSequence': ['StopSequence', 'Sequence', 'Order'],
        'TravelTimeFromPreviousStop': ['TravelTimeFromPreviousStop', 'TimeBetweenStops', 'SegmentTime']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
    df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_stop_times = len(df)
    avg_stops_per_trip = df.groupby('TripID').size().mean()
    
    print(f"Total Stop Times Records: {total_stop_times}")
    print(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
    
    fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences')
    fig1.show()
    
    # Calculate dwell time if both arrival and departure are available and meaningful
    df['DwellTimeSeconds'] = (pd.to_datetime(df['DepartureTime'].astype(str)) - pd.to_datetime(df['ArrivalTime'].astype(str))).dt.total_seconds()
    df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan) # Handle overnight trips or data errors
    df = df.dropna(subset=['DwellTimeSeconds'])

    if not df['DwellTimeSeconds'].empty:
        fig2 = px.histogram(df, x='DwellTimeSeconds', nbins=30, title='Distribution of Dwell Times at Stops (Seconds)')
        fig2.show()
    else:
        print("Note: Dwell time calculation not possible or no valid dwell times found.")

    if 'TravelTimeFromPreviousStop' in df.columns:
        df['TravelTimeFromPreviousStop'] = pd.to_numeric(df['TravelTimeFromPreviousStop'], errors='coerce')
        fig3 = px.histogram(df.dropna(subset=['TravelTimeFromPreviousStop']), x='TravelTimeFromPreviousStop',
                            title='Distribution of Travel Times Between Stops')
        fig3.show()

    return {
        "metrics": {
            "Total Stop Times Records": total_stop_times,
            "Average Number of Stops per Trip": avg_stops_per_trip
        },
        "figures": {
            "Stop_Sequences_Distribution_Histogram": fig1,
            "Dwell_Times_Distribution_Histogram": fig2 if 'DwellTimeSeconds' in df.columns and not df['DwellTimeSeconds'].empty else None,
            "Travel_Times_Between_Stops_Histogram": fig3 if 'TravelTimeFromPreviousStop' in df.columns else None
        }
    }

def transit_service_calendar_analysis(df):
    print("\n--- Transit Service Calendar Analysis ---")
    expected = {
        'ServiceID': ['ServiceID', 'ID', 'ServiceCode'],
        'StartDate': ['StartDate', 'ServiceStartDate', 'ValidFrom'],
        'EndDate': ['EndDate', 'ServiceEndDate', 'ValidUntil'],
        'Monday': ['Monday', 'ServiceOnMonday'],
        'Tuesday': ['Tuesday', 'ServiceOnTuesday'],
        'Wednesday': ['Wednesday', 'ServiceOnWednesday'],
        'Thursday': ['Thursday', 'ServiceOnThursday'],
        'Friday': ['Friday', 'ServiceOnFriday'],
        'Saturday': ['Saturday', 'ServiceOnSaturday'],
        'Sunday': ['Sunday', 'ServiceOnSunday']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
    df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
    
    # Convert day columns to boolean and handle missing
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        df[day] = pd.to_numeric(df[day], errors='coerce').fillna(0).astype(bool) # Assuming 0/1 or similar
    
    df = df.dropna(subset=['StartDate', 'EndDate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_service_ids = len(df)
    avg_service_duration_days = (df['EndDate'] - df['StartDate']).dt.days.mean()
    
    print(f"Total Service IDs Defined: {total_service_ids}")
    print(f"Average Service Duration: {avg_service_duration_days:.1f} days")
    
    # Calculate total days of service for each day of the week
    service_days_counts = {
        day: df[day].sum() for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }
    service_days_df = pd.DataFrame(service_days_counts.items(), columns=['Day', 'ServiceCount'])
    fig1 = px.bar(service_days_df, x='Day', y='ServiceCount', title='Total Service Days per Day of Week',
                  category_orders={"Day": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    fig1.show()
    
    fig2 = px.histogram(df, x=(df['EndDate'] - df['StartDate']).dt.days, title='Distribution of Service Durations (Days)')
    fig2.show()

    return {
        "metrics": {
            "Total Service IDs Defined": total_service_ids,
            "Average Service Duration (days)": avg_service_duration_days
        },
        "figures": {
            "Total_Service_Days_Per_Day_of_Week_Bar": fig1,
            "Service_Durations_Distribution_Histogram": fig2
        }
    }

def transit_service_exception_and_holiday_schedule_analysis(df):
    print("\n--- Transit Service Exception and Holiday Schedule Analysis ---")
    expected = {
        'ServiceID': ['ServiceID', 'ID', 'ServiceCode'],
        'Date': ['Date', 'ExceptionDate', 'HolidayDate'],
        'ExceptionType': ['ExceptionType', 'Type', 'AdjustmentType'], # e.g., ADDED, REMOVED, MODIFIED
        'Description': ['Description', 'Note', 'Reason']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'ExceptionType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_exceptions = len(df)
    most_common_exception_type = df['ExceptionType'].mode()[0]
    unique_exception_dates = df['Date'].nunique()
    
    print(f"Total Service Exceptions/Holidays: {total_exceptions}")
    print(f"Most Common Exception Type: {most_common_exception_type}")
    print(f"Unique Dates with Exceptions: {unique_exception_dates}")
    
    fig1 = px.histogram(df, x='ExceptionType', title='Distribution of Service Exception Types')
    fig1.show()
    
    exceptions_by_month = df['Date'].dt.to_period('M').value_counts().sort_index().reset_index(name='Count')
    exceptions_by_month.columns = ['Month', 'Count'] # Rename for Plotly
    fig2 = px.line(exceptions_by_month, x='Month', y='Count', title='Monthly Trend of Service Exceptions')
    fig2.show()

    return {
        "metrics": {
            "Total Service Exceptions/Holidays": total_exceptions,
            "Most Common Exception Type": most_common_exception_type,
            "Unique Dates with Exceptions": unique_exception_dates
        },
        "figures": {
            "Service_Exception_Types_Distribution_Histogram": fig1,
            "Monthly_Service_Exceptions_Trend_Line": fig2
        }
    }

def transit_fare_structure_analysis(df):
    print("\n--- Transit Fare Structure Analysis ---")
    expected = {
        'FareID': ['FareID', 'ID'],
        'Price': ['Price', 'FareAmount', 'Cost'],
        'Currency': ['Currency', 'CurrencyType'],
        'PaymentMethod': ['PaymentMethod', 'Method', 'PaymentOption'],
        'TransferPolicy': ['TransferPolicy', 'TransfersAllowed', 'TransferRule'] # e.g., 0, 1, 2+
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_fare_price = df['Price'].mean()
    most_common_currency = df['Currency'].mode()[0]
    unique_payment_methods = df['PaymentMethod'].nunique()
    
    print(f"Average Fare Price: ${avg_fare_price:.2f}")
    print(f"Most Common Currency: {most_common_currency}")
    print(f"Unique Payment Methods: {unique_payment_methods}")
    
    fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
    fig1.show()
    
    fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price Distribution by Payment Method')
    fig2.show()
    
    if 'TransferPolicy' in df.columns:
        fig3 = px.histogram(df, x='TransferPolicy', title='Distribution of Transfer Policies')
        fig3.show()

    return {
        "metrics": {
            "Average Fare Price": avg_fare_price,
            "Most Common Currency": most_common_currency,
            "Unique Payment Methods": unique_payment_methods
        },
        "figures": {
            "Fare_Prices_Distribution_Histogram": fig1,
            "Fare_Price_by_Payment_Method_Box": fig2,
            "Transfer_Policies_Distribution_Histogram": fig3 if 'TransferPolicy' in df.columns else None
        }
    }

def transit_fare_rule_and_zone_analysis(df):
    print("\n--- Transit Fare Rule and Zone Analysis ---")
    expected = {
        'FareID': ['FareID', 'ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'OriginZoneID': ['OriginZoneID', 'FromZone', 'Zone1'],
        'DestinationZoneID': ['DestinationZoneID', 'ToZone', 'Zone2'],
        'FarePrice': ['FarePrice', 'Price', 'Amount']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FarePrice'] = pd.to_numeric(df['FarePrice'], errors='coerce')
    df = df.dropna(subset=['FarePrice', 'OriginZoneID', 'DestinationZoneID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_fare_rules = len(df)
    unique_origin_zones = df['OriginZoneID'].nunique()
    avg_fare_price = df['FarePrice'].mean()
    
    print(f"Total Fare Rules Defined: {total_fare_rules}")
    print(f"Unique Origin Zones: {unique_origin_zones}")
    print(f"Average Fare Price (across rules): ${avg_fare_price:.2f}")
    
    fig1 = px.histogram(df, x='FarePrice', nbins=20, title='Distribution of Fare Prices per Rule')
    fig1.show()
    
    # Top 10 expensive zone-to-zone fares
    df['ZonePair'] = df['OriginZoneID'].astype(str) + ' to ' + df['DestinationZoneID'].astype(str)
    top_fares = df.sort_values('FarePrice', ascending=False).head(10)
    fig2 = px.bar(top_fares, x='ZonePair', y='FarePrice', title='Top 10 Most Expensive Zone-to-Zone Fares')
    fig2.show()
    
    if 'RouteID' in df.columns:
        fares_by_route = df.groupby('RouteID')['FarePrice'].mean().reset_index()
        fig3 = px.bar(fares_by_route.sort_values('FarePrice', ascending=False).head(20),
                      x='RouteID', y='FarePrice', title='Average Fare Price by Route (Top 20)')
        fig3.show()

    return {
        "metrics": {
            "Total Fare Rules Defined": total_fare_rules,
            "Unique Origin Zones": unique_origin_zones,
            "Average Fare Price": avg_fare_price
        },
        "figures": {
            "Fare_Prices_per_Rule_Distribution_Histogram": fig1,
            "Top_10_Most_Expensive_Zone_to_Zone_Fares_Bar": fig2,
            "Average_Fare_Price_by_Route_Bar": fig3 if 'RouteID' in df.columns else None
        }
    }

def transit_route_shape_and_path_geospatial_analysis(df):
    print("\n--- Transit Route Shape and Path Geospatial Analysis ---")
    expected = {
        'ShapeID': ['ShapeID', 'ID'],
        'ShapeLat': ['ShapeLat', 'Latitude', 'Lat'],
        'ShapeLon': ['ShapeLon', 'Longitude', 'Lon'],
        'ShapeSequence': ['ShapeSequence', 'Sequence', 'Order']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ShapeLat'] = pd.to_numeric(df['ShapeLat'], errors='coerce')
    df['ShapeLon'] = pd.to_numeric(df['ShapeLon'], errors='coerce')
    df['ShapeSequence'] = pd.to_numeric(df['ShapeSequence'], errors='coerce')
    df = df.dropna(subset=['ShapeLat', 'ShapeLon', 'ShapeSequence', 'ShapeID'])
    df = df.sort_values(by=['ShapeID', 'ShapeSequence']) # Ensure correct drawing order

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_shape_points = len(df)
    unique_route_shapes = df['ShapeID'].nunique()
    
    print(f"Total Geospatial Shape Points: {total_shape_points}")
    print(f"Unique Route Shapes: {unique_route_shapes}")
    
    # Plot all shapes on a map
    fig1 = px.line_mapbox(df, lat='ShapeLat', lon='ShapeLon', color='ShapeID', line_group='ShapeID',
                          zoom=9, title='Geospatial Paths of Transit Routes',
                          mapbox_style="carto-positron")
    fig1.show()
    
    # Plot a specific shape (e.g., the first one encountered)
    if unique_route_shapes > 0:
        sample_shape_id = df['ShapeID'].iloc[0]
        sample_shape_df = df[df['ShapeID'] == sample_shape_id]
        fig2 = px.line_mapbox(sample_shape_df, lat='ShapeLat', lon='ShapeLon',
                              zoom=12, title=f'Geospatial Path of Sample Route Shape: {sample_shape_id}',
                              mapbox_style="carto-positron")
        fig2.show()
    else:
        fig2 = None

    return {
        "metrics": {
            "Total Geospatial Shape Points": total_shape_points,
            "Unique Route Shapes": unique_route_shapes
        },
        "figures": {
            "Geospatial_Paths_of_Transit_Routes_Map": fig1,
            "Sample_Route_Shape_Geospatial_Path_Map": fig2
        }
    }

def transit_frequency_and_headway_analysis(df):
    print("\n--- Transit Frequency and Headway Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'StartTime': ['StartTime', 'DepartureTime', 'FirstDeparture'],
        'EndTime': ['EndTime', 'ArrivalTime', 'LastArrival'],
        'HeadwaySeconds': ['HeadwaySeconds', 'FrequencySeconds', 'IntervalSeconds']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['HeadwaySeconds'] = pd.to_numeric(df['HeadwaySeconds'], errors='coerce')
    df = df.dropna(subset=['HeadwaySeconds', 'RouteID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_headway_seconds = df['HeadwaySeconds'].mean()
    avg_headway_minutes = avg_headway_seconds / 60
    
    print(f"Average Headway: {avg_headway_minutes:.1f} minutes")
    
    fig1 = px.histogram(df, x=df['HeadwaySeconds']/60, nbins=30, title='Distribution of Headway (Minutes)')
    fig1.show()
    
    avg_headway_by_route = df.groupby('RouteID')['HeadwaySeconds'].mean().reset_index()
    avg_headway_by_route['HeadwayMinutes'] = avg_headway_by_route['HeadwaySeconds'] / 60
    
    fig2 = px.bar(avg_headway_by_route.sort_values('HeadwayMinutes', ascending=True).head(20),
                  x='RouteID', y='HeadwayMinutes', title='Top 20 Routes by Lowest Average Headway')
    fig2.show()
    
    fig3 = px.box(df, y=df['HeadwaySeconds']/60, title='Headway Distribution (Minutes)')
    fig3.show()

    return {
        "metrics": {
            "Average Headway (minutes)": avg_headway_minutes
        },
        "figures": {
            "Headway_Distribution_Histogram": fig1,
            "Top_20_Routes_by_Lowest_Avg_Headway_Bar": fig2,
            "Headway_Distribution_Box": fig3
        }
    }

def station_pathway_and_accessibility_analysis(df):
    print("\n--- Station Pathway and Accessibility Analysis ---")
    expected = {
        'FromStopID': ['FromStopID', 'OriginStopID'],
        'ToStopID': ['ToStopID', 'DestinationStopID'],
        'PathwayID': ['PathwayID', 'ID', 'PathID'],
        'PathwayMode': ['PathwayMode', 'Mode', 'Type'], # e.g., WALKWAY, STAIRS, ESCALATOR, ELEVATOR
        'IsAccessible': ['IsAccessible', 'Accessible', 'WheelchairAccessible']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['PathwayMode', 'IsAccessible'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_pathways = len(df)
    accessible_pathways_pct = (df['IsAccessible'] == 'Yes').mean() * 100
    most_common_pathway_mode = df['PathwayMode'].mode()[0]
    
    print(f"Total Pathways Defined: {total_pathways}")
    print(f"Percentage of Accessible Pathways: {accessible_pathways_pct:.2f}%")
    print(f"Most Common Pathway Mode: {most_common_pathway_mode}")
    
    fig1 = px.pie(df, names='IsAccessible', title='Accessibility of Station Pathways')
    fig1.show()
    
    fig2 = px.histogram(df, x='PathwayMode', color='IsAccessible', barmode='group',
                       title='Pathway Mode Distribution by Accessibility')
    fig2.show()
    
    if 'FromStopID' in df.columns and 'ToStopID' in df.columns:
        pathways_per_stop = df.groupby('FromStopID').size().reset_index(name='Count')
        fig3 = px.bar(pathways_per_stop.sort_values('Count', ascending=False).head(20),
                      x='FromStopID', y='Count', title='Stops with Most Outgoing Pathways (Top 20)')
        fig3.show()

    return {
        "metrics": {
            "Total Pathways Defined": total_pathways,
            "Accessible Pathways (%)": accessible_pathways_pct,
            "Most Common Pathway Mode": most_common_pathway_mode
        },
        "figures": {
            "Accessibility_of_Station_Pathways_Pie": fig1,
            "Pathway_Mode_Distribution_by_Accessibility_Histogram": fig2,
            "Stops_with_Most_Outgoing_Pathways_Bar": fig3 if 'FromStopID' in df.columns and 'ToStopID' in df.columns else None
        }
    }

def gtfs_feed_information_and_version_analysis(df):
    print("\n--- GTFS Feed Information and Version Analysis ---")
    expected = {
        'PublisherName': ['PublisherName', 'Name', 'FeedPublisher'],
        'PublisherURL': ['PublisherURL', 'URL', 'FeedURL'],
        'Lang': ['Lang', 'Language'],
        'FeedStartDate': ['FeedStartDate', 'StartDate', 'ValidFrom'],
        'FeedEndDate': ['FeedEndDate', 'EndDate', 'ValidUntil'],
        'FeedVersion': ['FeedVersion', 'Version', 'GTFSVersion']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FeedStartDate'] = pd.to_datetime(df['FeedStartDate'], errors='coerce')
    df['FeedEndDate'] = pd.to_datetime(df['FeedEndDate'], errors='coerce')
    df = df.dropna(subset=['PublisherName', 'PublisherURL', 'Lang', 'FeedStartDate', 'FeedEndDate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_feeds = len(df)
    most_common_language = df['Lang'].mode()[0]
    avg_feed_validity_days = (df['FeedEndDate'] - df['FeedStartDate']).dt.days.mean()
    
    print(f"Total GTFS Feeds: {total_feeds}")
    print(f"Most Common Language: {most_common_language}")
    print(f"Average Feed Validity Duration: {avg_feed_validity_days:.1f} days")
    
    fig1 = px.histogram(df, x='Lang', title='Distribution of Feed Languages')
    fig1.show()
    
    if 'FeedVersion' in df.columns:
        fig2 = px.histogram(df, x='FeedVersion', title='Distribution of GTFS Feed Versions')
        fig2.show()
    else:
        print("Note: 'FeedVersion' column not found.")
    
    if 'PublisherName' in df.columns:
        publisher_counts = df['PublisherName'].value_counts().reset_index()
        publisher_counts.columns = ['PublisherName', 'Count']
        fig3 = px.bar(publisher_counts.head(20), x='PublisherName', y='Count', title='Top 20 Feed Publishers')
        fig3.show()

    return {
        "metrics": {
            "Total GTFS Feeds": total_feeds,
            "Most Common Language": most_common_language,
            "Average Feed Validity Duration (days)": avg_feed_validity_days
        },
        "figures": {
            "Feed_Languages_Distribution_Histogram": fig1,
            "GTFS_Feed_Versions_Distribution_Histogram": fig2 if 'FeedVersion' in df.columns else None,
            "Top_20_Feed_Publishers_Bar": fig3 if 'PublisherName' in df.columns else None
        }
    }

def real_time_vehicle_position_and_trip_update_analysis(df):
    print("\n--- Real-Time Vehicle Position and Trip Update Analysis ---")
    expected = {
        'VehicleID': ['VehicleID', 'ID', 'Vehicle_ID'],
        'TripID': ['TripID', 'Trip_ID'],
        'Latitude': ['Latitude', 'Lat'],
        'Longitude': ['Longitude', 'Lon'],
        'Timestamp': ['Timestamp', 'Time', 'UpdateTime'],
        'DelaySeconds': ['DelaySeconds', 'Delay', 'ActualDelay']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df['DelaySeconds'] = pd.to_numeric(df['DelaySeconds'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'Timestamp', 'DelaySeconds', 'VehicleID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_updates = len(df)
    avg_delay_seconds = df['DelaySeconds'].mean()
    avg_delay_minutes = avg_delay_seconds / 60
    
    print(f"Total Real-Time Updates: {total_updates}")
    print(f"Average Delay: {avg_delay_minutes:.2f} minutes")
    
    fig1 = px.scatter_mapbox(df, lat='Latitude', lon='Longitude', color='DelaySeconds',
                             size='DelaySeconds', hover_name='VehicleID',
                             zoom=10, title='Vehicle Positions by Delay',
                             mapbox_style="carto-positron")
    fig1.show()
    
    fig2 = px.histogram(df, x=df['DelaySeconds']/60, nbins=30, title='Distribution of Delays (Minutes)')
    fig2.show()
    
    if 'TripID' in df.columns:
        delay_by_trip = df.groupby('TripID')['DelaySeconds'].mean().reset_index()
        fig3 = px.box(delay_by_trip, y=delay_by_trip['DelaySeconds']/60, title='Delay Distribution by Trip (Minutes)')
        fig3.show()

    return {
        "metrics": {
            "Total Real-Time Updates": total_updates,
            "Average Delay (minutes)": avg_delay_minutes
        },
        "figures": {
            "Vehicle_Positions_by_Delay_Map": fig1,
            "Delays_Distribution_Histogram": fig2,
            "Delay_Distribution_by_Trip_Box": fig3 if 'TripID' in df.columns else None
        }
    }

def extended_transit_route_attribute_analysis(df):
    print("\n--- Extended Transit Route Attribute Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'ID', 'Route_ID'],
        'RouteShortName': ['RouteShortName', 'ShortName'],
        'RouteLongName': ['RouteLongName', 'LongName'],
        'RouteType': ['RouteType', 'Type'],
        'RouteDesc': ['RouteDesc', 'Description', 'RouteDescription'],
        'RouteURL': ['RouteURL', 'URL'],
        'RouteColor': ['RouteColor', 'Color'],
        'RouteTextColor': ['RouteTextColor', 'TextColor']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['RouteID', 'RouteType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_routes = len(df)
    unique_route_types = df['RouteType'].nunique()
    
    print(f"Total Routes with Extended Attributes: {total_routes}")
    print(f"Unique Route Types: {unique_route_types}")
    
    fig1 = px.histogram(df, x='RouteType', title='Distribution of Route Types (Extended Attributes)')
    fig1.show()
    
    if 'RouteColor' in df.columns and not df['RouteColor'].dropna().empty:
        color_counts = df['RouteColor'].value_counts().reset_index()
        color_counts.columns = ['RouteColor', 'Count']
        fig2 = px.bar(color_counts.head(10), x='RouteColor', y='Count', title='Top 10 Most Common Route Colors')
        fig2.show()
    else:
        print("Note: 'RouteColor' column not found for color analysis.")

    if 'RouteURL' in df.columns:
        print("\nSample Route URLs:")
        for url in df['RouteURL'].head().tolist():
            print(f"- {url}")

    return {
        "metrics": {
            "Total Routes with Extended Attributes": total_routes,
            "Unique Route Types": unique_route_types
        },
        "figures": {
            "Route_Types_Distribution_Histogram_Extended": fig1,
            "Top_10_Most_Common_Route_Colors_Bar": fig2 if 'RouteColor' in df.columns else None
        }
    }

def transit_fare_zone_definition_analysis(df):
    print("\n--- Transit Fare Zone Definition Analysis ---")
    expected = {
        'ZoneID': ['ZoneID', 'ID', 'FareZoneID'],
        'ZoneName': ['ZoneName', 'Name', 'Description'],
        'ZoneCenterLat': ['ZoneCenterLat', 'Lat', 'Latitude'],
        'ZoneCenterLon': ['ZoneCenterLon', 'Lon', 'Longitude'],
        'ZoneType': ['ZoneType', 'Type', 'AreaType']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ZoneCenterLat'] = pd.to_numeric(df['ZoneCenterLat'], errors='coerce')
    df['ZoneCenterLon'] = pd.to_numeric(df['ZoneCenterLon'], errors='coerce')
    df = df.dropna(subset=['ZoneID', 'ZoneName', 'ZoneCenterLat', 'ZoneCenterLon'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_zones = len(df)
    unique_zone_types = df['ZoneType'].nunique() if 'ZoneType' in df.columns else 0
    
    print(f"Total Fare Zones Defined: {total_zones}")
    print(f"Unique Zone Types: {unique_zone_types}")
    
    fig1 = px.scatter_mapbox(df, lat='ZoneCenterLat', lon='ZoneCenterLon', hover_name='ZoneName', color='ZoneType' if 'ZoneType' in df.columns else None,
                             zoom=9, title='Geospatial Locations of Fare Zones',
                             mapbox_style="carto-positron")
    fig1.show()
    
    if 'ZoneType' in df.columns:
        fig2 = px.histogram(df, x='ZoneType', title='Distribution of Fare Zone Types')
        fig2.show()

    return {
        "metrics": {
            "Total Fare Zones Defined": total_zones,
            "Unique Zone Types": unique_zone_types
        },
        "figures": {
            "Fare_Zones_Geospatial_Map": fig1,
            "Fare_Zone_Types_Distribution_Histogram": fig2 if 'ZoneType' in df.columns else None
        }
    }

def multi_level_station_and_platform_information_analysis(df):
    print("\n--- Multi-Level Station and Platform Information Analysis ---")
    expected = {
        'StationID': ['StationID', 'ID', 'StopID'],
        'StationName': ['StationName', 'Name', 'StopName'],
        'LevelID': ['LevelID', 'Level_ID', 'FloorID'],
        'LevelName': ['LevelName', 'FloorName', 'LevelDescription'],
        'PlatformCount': ['PlatformCount', 'NumPlatforms', 'Platforms'],
        'HasElevator': ['HasElevator', 'ElevatorAccess', 'AccessibleElevator']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['PlatformCount'] = pd.to_numeric(df['PlatformCount'], errors='coerce')
    df['HasElevator'] = df['HasElevator'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['StationID', 'StationName', 'LevelID', 'PlatformCount'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_levels = len(df)
    total_platforms = df['PlatformCount'].sum()
    stations_with_elevators_pct = (df['HasElevator'] == 'Yes').mean() * 100 if 'HasElevator' in df.columns else np.nan
    
    print(f"Total Levels Defined: {total_levels}")
    print(f"Total Platforms Across All Levels: {total_platforms:,.0f}")
    if not np.isnan(stations_with_elevators_pct):
        print(f"Percentage of Levels with Elevator Access: {stations_with_elevators_pct:.2f}%")
    
    fig1 = px.histogram(df, x='PlatformCount', title='Distribution of Platform Counts per Level')
    fig1.show()
    
    if 'HasElevator' in df.columns:
        fig2 = px.pie(df, names='HasElevator', title='Elevator Availability Across Levels')
        fig2.show()

    stations_with_multiple_levels = df.groupby('StationID').filter(lambda x: x['LevelID'].nunique() > 1)
    if not stations_with_multiple_levels.empty:
        print("\nStations with multiple levels:")
        for station_id in stations_with_multiple_levels['StationID'].unique():
            station_data = df[df['StationID'] == station_id]
            print(f"- Station: {station_data['StationName'].iloc[0]} (ID: {station_id})")
            for index, row in station_data.iterrows():
                print(f"  Level: {row['LevelName']} (ID: {row['LevelID']}), Platforms: {row['PlatformCount']}, Elevator: {row.get('HasElevator', 'N/A')}")
    else:
        print("No stations with multiple levels found.")

    return {
        "metrics": {
            "Total Levels Defined": total_levels,
            "Total Platforms Across All Levels": total_platforms,
            "Percentage of Levels with Elevator Access": stations_with_elevators_pct
        },
        "figures": {
            "Platform_Counts_per_Level_Histogram": fig1,
            "Elevator_Availability_Pie": fig2 if 'HasElevator' in df.columns else None
        }
    }

def transit_trip_stop_timepoint_analysis(df):
    print("\n--- Transit Trip Stop Timepoint Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'StopID': ['StopID', 'Stop_ID'],
        'ArrivalTime': ['ArrivalTime', 'ScheduledArrival', 'Arrival'],
        'DepartureTime': ['DepartureTime', 'ScheduledDeparture', 'Departure'],
        'StopSequence': ['StopSequence', 'Sequence', 'Order'],
        'Timepoint': ['Timepoint', 'IsTimepoint', 'DesignatedTimepoint'] # Indicates if a stop is a timepoint
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
    df['Timepoint'] = df['Timepoint'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence', 'Timepoint'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_stop_times = len(df)
    total_timepoints = (df['Timepoint'] == 'Yes').sum()
    
    print(f"Total Stop Times Records: {total_stop_times}")
    print(f"Total Designated Timepoints: {total_timepoints}")
    print(f"Percentage of Timepoints: {total_timepoints / total_stop_times * 100:.2f}%")
    
    fig1 = px.pie(df, names='Timepoint', title='Distribution of Timepoints vs. Regular Stops')
    fig1.show()
    
    timepoints_per_trip = df[df['Timepoint'] == 'Yes'].groupby('TripID').size().reset_index(name='NumTimepoints')
    fig2 = px.histogram(timepoints_per_trip, x='NumTimepoints', title='Distribution of Number of Timepoints per Trip')
    fig2.show()
    
    # Calculate dwell time for timepoints
    df['DwellTimeSeconds'] = (pd.to_datetime(df['DepartureTime'].astype(str)) - pd.to_datetime(df['ArrivalTime'].astype(str))).dt.total_seconds()
    df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan)
    
    if not df[df['Timepoint'] == 'Yes']['DwellTimeSeconds'].dropna().empty:
        fig3 = px.box(df[df['Timepoint'] == 'Yes'], y='DwellTimeSeconds', title='Dwell Time Distribution at Timepoints (Seconds)')
        fig3.show()
    else:
        print("Note: No valid dwell times for timepoints found.")

    return {
        "metrics": {
            "Total Stop Times Records": total_stop_times,
            "Total Designated Timepoints": total_timepoints,
            "Percentage of Timepoints": total_timepoints / total_stop_times * 100
        },
        "figures": {
            "Timepoints_vs_Regular_Stops_Pie": fig1,
            "Num_Timepoints_per_Trip_Distribution_Histogram": fig2,
            "Dwell_Time_at_Timepoints_Box": fig3 if 'DwellTimeSeconds' in df.columns and not df[df['Timepoint'] == 'Yes']['DwellTimeSeconds'].dropna().empty else None
        }
    }

def transit_trip_details_and_accessibility_features_analysis(df):
    print("\n--- Transit Trip Details and Accessibility Features Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'ServiceID': ['ServiceID', 'Service_ID'],
        'TripHeadsign': ['TripHeadsign', 'Headsign', 'Destination'],
        'TripShortName': ['TripShortName', 'ShortName'],
        'DirectionID': ['DirectionID', 'Direction'],
        'WheelchairAccessible': ['WheelchairAccessible', 'Accessible', 'Wheelchair'],
        'BikesAllowed': ['BikesAllowed', 'BikeAccess', 'Bikes']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['DirectionID'] = pd.to_numeric(df['DirectionID'], errors='coerce')
    df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_trips = len(df)
    accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
    bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
    
    print(f"Total Trip Records: {total_trips}")
    print(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
    print(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
    
    fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
    fig1.show()
    
    fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
    fig2.show()
    
    fig3 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
    fig3.show()

    if 'TripHeadsign' in df.columns:
        headsign_counts = df['TripHeadsign'].value_counts().reset_index()
        headsign_counts.columns = ['TripHeadsign', 'Count']
        fig4 = px.bar(headsign_counts.head(20), x='TripHeadsign', y='Count', title='Top 20 Most Common Trip Headsigns')
        fig4.show()

    return {
        "metrics": {
            "Total Trip Records": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        },
        "figures": {
            "Wheelchair_Accessibility_Pie": fig1,
            "Bike_Accessibility_Pie": fig2,
            "Trip_Directions_Distribution_Histogram": fig3,
            "Top_20_Most_Common_Trip_Headsigns_Bar": fig4 if 'TripHeadsign' in df.columns else None
        }
    }

def transit_stop_and_station_location_analysis(df):
    print("\n--- Transit Stop and Station Location Analysis ---")
    expected = {
        'StopID': ['StopID', 'ID', 'Stop_ID'],
        'StopName': ['StopName', 'Name', 'LocationName'],
        'StopLat': ['StopLat', 'Latitude', 'Lat'],
        'StopLon': ['StopLon', 'Longitude', 'Lon'],
        'LocationType': ['LocationType', 'Type', 'StopType'], # e.g., stop, station, entrance
        'ParentStationID': ['ParentStationID', 'ParentStation', 'StationID'] # For stops that are part of a station
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StopLat'] = pd.to_numeric(df['StopLat'], errors='coerce')
    df['StopLon'] = pd.to_numeric(df['StopLon'], errors='coerce')
    df = df.dropna(subset=['StopLat', 'StopLon', 'StopName', 'LocationType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_locations = len(df)
    total_stations = df[df['LocationType'] == 'station'].shape[0] if 'LocationType' in df.columns else 0
    total_stops = df[df['LocationType'] == 'stop'].shape[0] if 'LocationType' in df.columns else 0
    
    print(f"Total Locations (Stops/Stations): {total_locations}")
    print(f"Total Stations: {total_stations}")
    print(f"Total Stops: {total_stops}")
    
    fig1 = px.scatter_mapbox(df, lat='StopLat', lon='StopLon', hover_name='StopName', color='LocationType',
                             zoom=9, title='Transit Stop and Station Locations',
                             mapbox_style="carto-positron")
    fig1.show()
    
    fig2 = px.histogram(df, x='LocationType', title='Distribution of Location Types (Stop/Station)')
    fig2.show()
    
    if 'ParentStationID' in df.columns and not df['ParentStationID'].dropna().empty:
        stops_per_station = df.groupby('ParentStationID').size().reset_index(name='NumStops')
        fig3 = px.histogram(stops_per_station, x='NumStops', title='Distribution of Number of Stops per Station')
        fig3.show()

    return {
        "metrics": {
            "Total Locations": total_locations,
            "Total Stations": total_stations,
            "Total Stops": total_stops
        },
        "figures": {
            "Transit_Stop_and_Station_Locations_Map": fig1,
            "Location_Types_Distribution_Histogram": fig2,
            "Num_Stops_per_Station_Distribution_Histogram": fig3 if 'ParentStationID' in df.columns else None
        }
    }

def public_transportation_route_details_analysis(df):
    print("\n--- Public Transportation Route Details Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'ID', 'Route_ID'],
        'RouteShortName': ['RouteShortName', 'ShortName', 'RouteNum'],
        'RouteLongName': ['RouteLongName', 'LongName', 'RouteDescription'],
        'RouteDesc': ['RouteDesc', 'Description'],
        'RouteType': ['RouteType', 'Type', 'TransitType'],
        'AgencyID': ['AgencyID', 'Agency_ID']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_routes = len(df)
    unique_route_types = df['RouteType'].nunique()
    
    print(f"Total Public Transportation Routes: {total_routes}")
    print(f"Unique Route Types: {unique_route_types}")
    
    fig1 = px.histogram(df, x='RouteType', title='Distribution of Public Transportation Route Types')
    fig1.show()
    
    if 'AgencyID' in df.columns:
        routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
        fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                      title='Number of Routes per Agency (Top 20)')
        fig2.show()
    else:
        fig2 = None

    if 'RouteLongName' in df.columns and 'RouteShortName' in df.columns:
        print("\nSample Route Definitions:")
        for index, row in df.head(10).iterrows():
            print(f"- {row['RouteShortName']}: {row['RouteLongName']} (Type: {row['RouteType']})")

    return {
        "metrics": {
            "Total Public Transportation Routes": total_routes,
            "Unique Route Types": unique_route_types
        },
        "figures": {
            "Public_Transportation_Route_Types_Distribution_Histogram": fig1,
            "Routes_per_Agency_Bar": fig2
        }
    }

def transportation_agency_contact_and_timezone_analysis(df):
    print("\n--- Transportation Agency Contact and Timezone Analysis ---")
    expected = {
        'AgencyID': ['AgencyID', 'ID'],
        'AgencyName': ['AgencyName', 'Name'],
        'AgencyURL': ['AgencyURL', 'URL'],
        'AgencyTimezone': ['AgencyTimezone', 'Timezone'],
        'AgencyLang': ['AgencyLang', 'Language'],
        'AgencyPhone': ['AgencyPhone', 'Phone', 'ContactNumber']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['AgencyID', 'AgencyName', 'AgencyTimezone'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_agencies = len(df)
    unique_timezones = df['AgencyTimezone'].nunique()
    most_common_timezone = df['AgencyTimezone'].mode()[0]
    
    print(f"Total Transportation Agencies: {total_agencies}")
    print(f"Unique Timezones: {unique_timezones}")
    print(f"Most Common Timezone: {most_common_timezone}")
    
    fig1 = px.histogram(df, x='AgencyTimezone', title='Distribution of Agency Timezones')
    fig1.show()
    
    if 'AgencyLang' in df.columns:
        fig2 = px.pie(df, names='AgencyLang', title='Distribution of Agency Languages')
        fig2.show()

    if 'AgencyPhone' in df.columns:
        agencies_with_phone = df['AgencyPhone'].notna().sum()
        print(f"Agencies with Phone Numbers: {agencies_with_phone} ({agencies_with_phone / total_agencies * 100:.2f}%)")

    print("\nSample Agency Contact Info:")
    if 'AgencyName' in df.columns and 'AgencyURL' in df.columns and 'AgencyPhone' in df.columns:
        for index, row in df.head(5).iterrows():
            print(f"- {row['AgencyName']} | URL: {row['AgencyURL']} | Phone: {row['AgencyPhone']} | Timezone: {row['AgencyTimezone']}")
    else:
        print("Not enough columns to display detailed contact info for sample agencies.")

    return {
        "metrics": {
            "Total Transportation Agencies": total_agencies,
            "Unique Timezones": unique_timezones,
            "Most Common Timezone": most_common_timezone
        },
        "figures": {
            "Agency_Timezones_Distribution_Histogram": fig1,
            "Agency_Languages_Distribution_Pie": fig2 if 'AgencyLang' in df.columns else None
        }
    }

def transit_trip_planning_and_route_shape_analysis(df):
    print("\n--- Transit Trip Planning and Route Shape Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'ShapeID': ['ShapeID', 'Shape_ID'],
        'DirectionID': ['DirectionID', 'Direction'],
        'BlockID': ['BlockID', 'Block_ID'],
        'TripHeadsign': ['TripHeadsign', 'Headsign']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['TripID', 'RouteID', 'ShapeID', 'DirectionID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_trips = len(df)
    unique_route_shapes_used = df['ShapeID'].nunique()
    
    print(f"Total Trips with Planning Data: {total_trips}")
    print(f"Unique Route Shapes Used: {unique_route_shapes_used}")
    
    fig1 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
    fig1.show()
    
    trips_per_shape = df.groupby('ShapeID').size().reset_index(name='TripCount')
    fig2 = px.bar(trips_per_shape.sort_values('TripCount', ascending=False).head(20),
                  x='ShapeID', y='TripCount', title='Top 20 Most Used Route Shapes by Trip Count')
    fig2.show()
    
    if 'TripHeadsign' in df.columns:
        headsign_counts = df['TripHeadsign'].value_counts().reset_index()
        headsign_counts.columns = ['TripHeadsign', 'Count']
        fig3 = px.bar(headsign_counts.head(20), x='TripHeadsign', y='Count', title='Top 20 Most Common Trip Headsigns')
        fig3.show()

    return {
        "metrics": {
            "Total Trips with Planning Data": total_trips,
            "Unique Route Shapes Used": unique_route_shapes_used
        },
        "figures": {
            "Trip_Directions_Distribution_Histogram": fig1,
            "Top_20_Most_Used_Route_Shapes_Bar": fig2,
            "Top_20_Most_Common_Trip_Headsigns_Bar": fig3 if 'TripHeadsign' in df.columns else None
        }
    }

def transit_service_schedule_definition(df):
    print("\n--- Transit Service Schedule Definition Analysis ---")
    expected = {
        'ServiceId': ['ServiceId', 'ID', 'Service_ID'],
        'RouteId': ['RouteId', 'Route_ID'],
        'TripId': ['TripId', 'Trip_ID'],
        'StartDate': ['StartDate', 'ServiceStartDate'],
        'EndDate': ['EndDate', 'ServiceEndDate'],
        'Monday': ['Monday', 'ServiceOnMonday'],
        'Tuesday': ['Tuesday', 'ServiceOnTuesday'],
        'Wednesday': ['Wednesday', 'ServiceOnWednesday'],
        'Thursday': ['Thursday', 'ServiceOnThursday'],
        'Friday': ['Friday', 'ServiceOnFriday'],
        'Saturday': ['Saturday', 'ServiceOnSaturday'],
        'Sunday': ['Sunday', 'ServiceOnSunday']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['StartDate'] = pd.to_datetime(df['StartDate'], errors='coerce')
    df['EndDate'] = pd.to_datetime(df['EndDate'], errors='coerce')
    
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
        df[day] = pd.to_numeric(df[day], errors='coerce').fillna(0).astype(bool)
    
    df = df.dropna(subset=['ServiceId', 'StartDate', 'EndDate'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_service_ids = len(df)
    avg_duration_days = (df['EndDate'] - df['StartDate']).dt.days.mean()
    
    print(f"Total Service Schedules Defined: {total_service_ids}")
    print(f"Average Schedule Duration: {avg_duration_days:.1f} days")
    
    service_day_counts = {
        day: df[day].sum() for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    }
    service_day_df = pd.DataFrame(service_day_counts.items(), columns=['Day', 'ServiceCount'])
    fig1 = px.bar(service_day_df, x='Day', y='ServiceCount', title='Number of Service Days per Weekday',
                  category_orders={"Day": ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    fig1.show()
    
    if 'RouteId' in df.columns:
        routes_per_service = df.groupby('ServiceId')['RouteId'].nunique().reset_index(name='NumRoutes')
        fig2 = px.histogram(routes_per_service, x='NumRoutes', title='Distribution of Number of Routes per Service ID')
        fig2.show()

    return {
        "metrics": {
            "Total Service Schedules Defined": total_service_ids,
            "Average Schedule Duration (days)": avg_duration_days
        },
        "figures": {
            "Service_Days_per_Weekday_Bar": fig1,
            "Num_Routes_per_Service_ID_Histogram": fig2 if 'RouteId' in df.columns else None
        }
    }

def stop_by_stop_transit_schedule_analysis(df):
    print("\n--- Stop-by-Stop Transit Schedule Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'StopID': ['StopID', 'Stop_ID'],
        'ArrivalTime': ['ArrivalTime', 'ScheduledArrival', 'Arrival'],
        'DepartureTime': ['DepartureTime', 'ScheduledDeparture', 'Departure'],
        'StopSequence': ['StopSequence', 'Sequence', 'Order'],
        'TravelTimeSeconds': ['TravelTimeSeconds', 'TimeBetweenStops', 'SegmentTime']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
    df['TravelTimeSeconds'] = pd.to_numeric(df['TravelTimeSeconds'], errors='coerce')
    df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_stop_times = len(df)
    avg_stops_per_trip = df.groupby('TripID').size().mean()
    
    print(f"Total Stop-by-Stop Schedule Records: {total_stop_times}")
    print(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
    
    fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences')
    fig1.show()
    
    if not df['TravelTimeSeconds'].dropna().empty:
        avg_travel_time_between_stops = df['TravelTimeSeconds'].mean()
        print(f"Average Travel Time Between Stops: {avg_travel_time_between_stops:.1f} seconds")
        fig2 = px.histogram(df, x='TravelTimeSeconds', nbins=30, title='Distribution of Travel Times Between Stops (Seconds)')
        fig2.show()
    else:
        fig2 = None
        print("Note: 'TravelTimeSeconds' column not available for analysis.")

    # Calculate dwell time if both arrival and departure are available and meaningful
    df['DwellTimeSeconds'] = (pd.to_datetime(df['DepartureTime'].astype(str)) - pd.to_datetime(df['ArrivalTime'].astype(str))).dt.total_seconds()
    df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan)
    if not df['DwellTimeSeconds'].dropna().empty:
        fig3 = px.box(df, y='DwellTimeSeconds', title='Dwell Time Distribution at Stops (Seconds)')
        fig3.show()
    else:
        fig3 = None
        print("Note: Dwell time calculation not possible or no valid dwell times found.")

    return {
        "metrics": {
            "Total Stop-by-Stop Schedule Records": total_stop_times,
            "Average Number of Stops per Trip": avg_stops_per_trip,
            "Average Travel Time Between Stops (seconds)": avg_travel_time_between_stops if 'TravelTimeSeconds' in df.columns and not df['TravelTimeSeconds'].dropna().empty else None
        },
        "figures": {
            "Stop_Sequences_Distribution_Histogram": fig1,
            "Travel_Times_Between_Stops_Histogram": fig2,
            "Dwell_Time_at_Stops_Box": fig3
        }
    }

def public_transport_agency_directory_analysis(df):
    print("\n--- Public Transport Agency Directory Analysis ---")
    expected = {
        'AgencyID': ['AgencyID', 'ID'],
        'AgencyName': ['AgencyName', 'Name'],
        'AgencyURL': ['AgencyURL', 'URL'],
        'AgencyTimezone': ['AgencyTimezone', 'Timezone'],
        'AgencyLang': ['AgencyLang', 'Language'],
        'AgencyPhone': ['AgencyPhone', 'Phone', 'ContactNumber']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['AgencyID', 'AgencyName', 'AgencyTimezone'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_agencies = len(df)
    unique_timezones = df['AgencyTimezone'].nunique()
    
    print(f"Total Public Transport Agencies: {total_agencies}")
    print(f"Unique Timezones: {unique_timezones}")
    
    fig1 = px.histogram(df, x='AgencyTimezone', title='Distribution of Agency Timezones')
    fig1.show()
    
    if 'AgencyLang' in df.columns:
        fig2 = px.pie(df, names='AgencyLang', title='Distribution of Agency Languages')
        fig2.show()

    print("\nSample Agency Details:")
    if 'AgencyURL' in df.columns and 'AgencyPhone' in df.columns:
        for index, row in df.head(5).iterrows():
            print(f"- {row['AgencyName']} (ID: {row['AgencyID']}) | URL: {row['AgencyURL']} | Phone: {row['AgencyPhone']}")
    else:
        print("Not enough columns to display detailed agency info.")

    return {
        "metrics": {
            "Total Public Transport Agencies": total_agencies,
            "Unique Timezones": unique_timezones
        },
        "figures": {
            "Agency_Timezones_Distribution_Histogram": fig1,
            "Agency_Languages_Distribution_Pie": fig2 if 'AgencyLang' in df.columns else None
        }
    }

def transit_fare_attribute_analysis(df):
    print("\n--- Transit Fare Attribute Analysis ---")
    expected = {
        'FareID': ['FareID', 'ID'],
        'Price': ['Price', 'FareAmount', 'Cost'],
        'Currency': ['Currency', 'CurrencyType'],
        'PaymentMethod': ['PaymentMethod', 'Method', 'PaymentOption'],
        'Transfers': ['Transfers', 'TransfersAllowed'],
        'TransferDuration': ['TransferDuration', 'TransferTimeSeconds', 'DurationSeconds']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['Transfers'] = pd.to_numeric(df['Transfers'], errors='coerce')
    df['TransferDuration'] = pd.to_numeric(df['TransferDuration'], errors='coerce')
    df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod', 'Transfers'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_price = df['Price'].mean()
    most_common_payment_method = df['PaymentMethod'].mode()[0]
    avg_transfer_duration_minutes = df['TransferDuration'].mean() / 60 if 'TransferDuration' in df.columns else np.nan
    
    print(f"Average Fare Price: ${avg_price:.2f}")
    print(f"Most Common Payment Method: {most_common_payment_method}")
    if not np.isnan(avg_transfer_duration_minutes):
        print(f"Average Transfer Duration: {avg_transfer_duration_minutes:.1f} minutes")
    
    fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
    fig1.show()
    
    fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price by Payment Method')
    fig2.show()
    
    fig3 = px.histogram(df, x='Transfers', title='Distribution of Allowed Transfers')
    fig3.show()

    if 'TransferDuration' in df.columns and not df['TransferDuration'].dropna().empty:
        fig4 = px.histogram(df, x=df['TransferDuration']/60, title='Distribution of Transfer Durations (Minutes)')
        fig4.show()

    return {
        "metrics": {
            "Average Fare Price": avg_price,
            "Most Common Payment Method": most_common_payment_method,
            "Average Transfer Duration (minutes)": avg_transfer_duration_minutes
        },
        "figures": {
            "Fare_Prices_Distribution_Histogram": fig1,
            "Fare_Price_by_Payment_Method_Box": fig2,
            "Allowed_Transfers_Distribution_Histogram": fig3,
            "Transfer_Durations_Distribution_Histogram": fig4 if 'TransferDuration' in df.columns else None
        }
    }

def inter_stop_transfer_path_analysis(df):
    print("\n--- Inter-Stop Transfer Path Analysis ---")
    expected = {
        'FromStopID': ['FromStopID', 'OriginStopID'],
        'ToStopID': ['ToStopID', 'DestinationStopID'],
        'PathwayType': ['PathwayType', 'Type', 'TransferType'], # e.g., walkway, elevator, stairs
        'TransferTimeSeconds': ['TransferTimeSeconds', 'DurationSeconds', 'TimeSeconds'],
        'MinTransferTime': ['MinTransferTime', 'MinimumTransferTime'],
        'IsAccessible': ['IsAccessible', 'Accessible', 'WheelchairAccessible']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TransferTimeSeconds'] = pd.to_numeric(df['TransferTimeSeconds'], errors='coerce')
    df['MinTransferTime'] = pd.to_numeric(df['MinTransferTime'], errors='coerce')
    df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['FromStopID', 'ToStopID', 'PathwayType', 'TransferTimeSeconds'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_paths = len(df)
    avg_transfer_time_seconds = df['TransferTimeSeconds'].mean()
    avg_transfer_time_minutes = avg_transfer_time_seconds / 60
    accessible_paths_pct = (df['IsAccessible'] == 'Yes').mean() * 100 if 'IsAccessible' in df.columns else np.nan
    
    print(f"Total Inter-Stop Transfer Paths: {total_paths}")
    print(f"Average Transfer Time: {avg_transfer_time_minutes:.1f} minutes")
    if not np.isnan(accessible_paths_pct):
        print(f"Percentage of Accessible Transfer Paths: {accessible_paths_pct:.2f}%")
    
    fig1 = px.histogram(df, x=df['TransferTimeSeconds']/60, nbins=30, title='Distribution of Transfer Times (Minutes)')
    fig1.show()
    
    fig2 = px.box(df, x='PathwayType', y=df['TransferTimeSeconds']/60, title='Transfer Time by Pathway Type (Minutes)')
    fig2.show()
    
    if 'IsAccessible' in df.columns:
        fig3 = px.pie(df, names='IsAccessible', title='Accessibility of Transfer Paths')
        fig3.show()

    return {
        "metrics": {
            "Total Inter-Stop Transfer Paths": total_paths,
            "Average Transfer Time (minutes)": avg_transfer_time_minutes,
            "Accessible Transfer Paths (%)": accessible_paths_pct
        },
        "figures": {
            "Transfer_Times_Distribution_Histogram": fig1,
            "Transfer_Time_by_Pathway_Type_Box": fig2,
            "Accessibility_of_Transfer_Paths_Pie": fig3 if 'IsAccessible' in df.columns else None
        }
    }

def geospatial_route_path_analysis(df):
    print("\n--- Geospatial Route Path Analysis ---")
    expected = {
        'ShapeID': ['ShapeID', 'ID'],
        'Latitude': ['Latitude', 'Lat', 'ShapeLat'],
        'Longitude': ['Longitude', 'Lon', 'ShapeLon'],
        'Sequence': ['Sequence', 'ShapeSequence', 'Order'],
        'RouteID': ['RouteID', 'Route_ID']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['Sequence'] = pd.to_numeric(df['Sequence'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude', 'Sequence', 'ShapeID'])
    df = df.sort_values(by=['ShapeID', 'Sequence'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_path_points = len(df)
    unique_route_shapes = df['ShapeID'].nunique()
    
    print(f"Total Geospatial Path Points: {total_path_points}")
    print(f"Unique Route Shapes: {unique_route_shapes}")
    
    fig1 = px.line_mapbox(df, lat='Latitude', lon='Longitude', color='ShapeID', line_group='ShapeID',
                          zoom=9, title='Geospatial Paths of Transit Routes',
                          mapbox_style="carto-positron")
    fig1.show()
    
    if unique_route_shapes > 0:
        sample_shape_id = df['ShapeID'].iloc[0]
        sample_shape_df = df[df['ShapeID'] == sample_shape_id]
        fig2 = px.line_mapbox(sample_shape_df, lat='Latitude', lon='Longitude',
                              zoom=12, title=f'Geospatial Path of Sample Route Shape: {sample_shape_id}',
                              mapbox_style="carto-positron")
        fig2.show()
    else:
        fig2 = None

    if 'RouteID' in df.columns:
        shapes_per_route = df.groupby('RouteID')['ShapeID'].nunique().reset_index(name='NumShapes')
        fig3 = px.histogram(shapes_per_route, x='NumShapes', title='Distribution of Number of Shapes per Route')
        fig3.show()

    return {
        "metrics": {
            "Total Geospatial Path Points": total_path_points,
            "Unique Route Shapes": unique_route_shapes
        },
        "figures": {
            "Geospatial_Paths_of_Transit_Routes_Map": fig1,
            "Sample_Route_Shape_Geospatial_Path_Map": fig2,
            "Num_Shapes_per_Route_Distribution_Histogram": fig3 if 'RouteID' in df.columns else None
        }
    }

def trip_frequency_and_service_interval_analysis(df):
    print("\n--- Trip Frequency and Service Interval Analysis ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'ServiceID': ['ServiceID', 'Service_ID'],
        'StartTime': ['StartTime', 'DepartureTime'],
        'EndTime': ['EndTime', 'ArrivalTime'],
        'FrequencySeconds': ['FrequencySeconds', 'HeadwaySeconds', 'IntervalSeconds']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['FrequencySeconds'] = pd.to_numeric(df['FrequencySeconds'], errors='coerce')
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S', errors='coerce').dt.time
    df = df.dropna(subset=['FrequencySeconds', 'RouteID', 'ServiceID'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_frequency_seconds = df['FrequencySeconds'].mean()
    avg_frequency_minutes = avg_frequency_seconds / 60
    
    print(f"Average Service Interval (Frequency): {avg_frequency_minutes:.1f} minutes")
    
    fig1 = px.histogram(df, x=df['FrequencySeconds']/60, nbins=30, title='Distribution of Service Intervals (Minutes)')
    fig1.show()
    
    avg_frequency_by_route = df.groupby('RouteID')['FrequencySeconds'].mean().reset_index()
    avg_frequency_by_route['FrequencyMinutes'] = avg_frequency_by_route['FrequencySeconds'] / 60
    fig2 = px.bar(avg_frequency_by_route.sort_values('FrequencyMinutes', ascending=True).head(20),
                  x='RouteID', y='FrequencyMinutes', title='Top 20 Routes by Lowest Average Service Interval')
    fig2.show()
    
    if 'ServiceID' in df.columns:
        fig3 = px.box(df, x='ServiceID', y=df['FrequencySeconds']/60, title='Service Interval by Service ID (Minutes)')
        fig3.show()

    return {
        "metrics": {
            "Average Service Interval (minutes)": avg_frequency_minutes
        },
        "figures": {
            "Service_Intervals_Distribution_Histogram": fig1,
            "Top_20_Routes_by_Lowest_Avg_Service_Interval_Bar": fig2,
            "Service_Interval_by_Service_ID_Box": fig3 if 'ServiceID' in df.columns else None
        }
    }

def fare_cost_and_transfer_policy_analysis(df):
    print("\n--- Fare Cost and Transfer Policy Analysis ---")
    expected = {
        'FareID': ['FareID', 'ID'],
        'Price': ['Price', 'FareAmount', 'Cost'],
        'Currency': ['Currency', 'CurrencyType'],
        'PaymentMethod': ['PaymentMethod', 'Method', 'PaymentOption'],
        'TransfersAllowed': ['TransfersAllowed', 'Transfers', 'NumTransfers'],
        'TransferDurationSeconds': ['TransferDurationSeconds', 'TransferTimeSeconds', 'DurationSeconds']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
    df['TransfersAllowed'] = pd.to_numeric(df['TransfersAllowed'], errors='coerce')
    df['TransferDurationSeconds'] = pd.to_numeric(df['TransferDurationSeconds'], errors='coerce')
    df = df.dropna(subset=['Price', 'Currency', 'PaymentMethod', 'TransfersAllowed'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_fare_price = df['Price'].mean()
    most_common_payment_method = df['PaymentMethod'].mode()[0]
    
    print(f"Average Fare Price: ${avg_fare_price:.2f}")
    print(f"Most Common Payment Method: {most_common_payment_method}")
    
    fig1 = px.histogram(df, x='Price', nbins=20, title='Distribution of Fare Prices')
    fig1.show()
    
    fig2 = px.box(df, x='PaymentMethod', y='Price', title='Fare Price by Payment Method')
    fig2.show()
    
    fig3 = px.histogram(df, x='TransfersAllowed', title='Distribution of Allowed Transfers')
    fig3.show()

    if 'TransferDurationSeconds' in df.columns and not df['TransferDurationSeconds'].dropna().empty:
        fig4 = px.histogram(df, x=df['TransferDurationSeconds']/60, title='Distribution of Transfer Durations (Minutes)')
        fig4.show()

    return {
        "metrics": {
            "Average Fare Price": avg_fare_price,
            "Most Common Payment Method": most_common_payment_method
        },
        "figures": {
            "Fare_Prices_Distribution_Histogram": fig1,
            "Fare_Price_by_Payment_Method_Box": fig2,
            "Allowed_Transfers_Distribution_Histogram": fig3,
            "Transfer_Durations_Distribution_Histogram": fig4 if 'TransferDurationSeconds' in df.columns else None
        }
    }

def trip_service_detail_and_distance_analysis(df):
    print("\n--- Trip Service Detail and Distance Analysis ---")
    expected = {
        'TripID': ['TripID', 'ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'ServiceID': ['ServiceID', 'Service_ID'],
        'TripDistanceMiles': ['TripDistanceMiles', 'DistanceMiles', 'Distance'],
        'TripDurationMinutes': ['TripDurationMinutes', 'DurationMinutes', 'TravelTime'],
        'WheelchairAccessible': ['WheelchairAccessible', 'Accessible'],
        'BikesAllowed': ['BikesAllowed', 'BikeAccess']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['TripDistanceMiles'] = pd.to_numeric(df['TripDistanceMiles'], errors='coerce')
    df['TripDurationMinutes'] = pd.to_numeric(df['TripDurationMinutes'], errors='coerce')
    df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['TripID', 'RouteID', 'TripDistanceMiles', 'TripDurationMinutes'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    avg_trip_distance = df['TripDistanceMiles'].mean()
    avg_trip_duration = df['TripDurationMinutes'].mean()
    accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100 if 'WheelchairAccessible' in df.columns else np.nan
    
    print(f"Average Trip Distance: {avg_trip_distance:.2f} miles")
    print(f"Average Trip Duration: {avg_trip_duration:.1f} minutes")
    if not np.isnan(accessible_trips_pct):
        print(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
    
    fig1 = px.scatter(df, x='TripDistanceMiles', y='TripDurationMinutes', hover_name='TripID',
                     title='Trip Duration vs. Distance')
    fig1.show()
    
    fig2 = px.histogram(df, x='TripDistanceMiles', nbins=20, title='Distribution of Trip Distances')
    fig2.show()
    
    if 'WheelchairAccessible' in df.columns:
        fig3 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
        fig3.show()

    return {
        "metrics": {
            "Average Trip Distance": avg_trip_distance,
            "Average Trip Duration": avg_trip_duration,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct
        },
        "figures": {
            "Trip_Duration_vs_Distance_Scatter": fig1,
            "Trip_Distances_Distribution_Histogram": fig2,
            "Wheelchair_Accessibility_of_Trips_Pie": fig3 if 'WheelchairAccessible' in df.columns else None
        }
    }

def transit_data_feed_publisher_information_analysis(df):
    print("\n--- Transit Data Feed Publisher Information Analysis ---")
    expected = {
        'PublisherName': ['PublisherName', 'Name', 'FeedPublisher'],
        'PublisherURL': ['PublisherURL', 'URL', 'FeedURL'],
        'PublisherLang': ['PublisherLang', 'Language', 'Lang'],
        'PublisherTimezone': ['PublisherTimezone', 'Timezone', 'FeedTimezone']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['PublisherName', 'PublisherURL', 'PublisherLang'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_publishers = len(df)
    unique_languages = df['PublisherLang'].nunique()
    most_common_language = df['PublisherLang'].mode()[0]
    
    print(f"Total Feed Publishers: {total_publishers}")
    print(f"Unique Languages Used by Publishers: {unique_languages}")
    print(f"Most Common Publisher Language: {most_common_language}")
    
    fig1 = px.histogram(df, x='PublisherLang', title='Distribution of Publisher Languages')
    fig1.show()
    
    if 'PublisherTimezone' in df.columns:
        fig2 = px.histogram(df, x='PublisherTimezone', title='Distribution of Publisher Timezones')
        fig2.show()

    print("\nSample Publisher Information:")
    if 'PublisherURL' in df.columns and 'PublisherTimezone' in df.columns:
        for index, row in df.head(5).iterrows():
            print(f"- {row['PublisherName']} | URL: {row['PublisherURL']} | Lang: {row['PublisherLang']} | Timezone: {row['PublisherTimezone']}")
    else:
        print("Not enough columns to display detailed publisher info.")

    return {
        "metrics": {
            "Total Feed Publishers": total_publishers,
            "Unique Languages Used by Publishers": unique_languages,
            "Most Common Publisher Language": most_common_language
        },
        "figures": {
            "Publisher_Languages_Distribution_Histogram": fig1,
            "Publisher_Timezones_Distribution_Histogram": fig2 if 'PublisherTimezone' in df.columns else None
        }
    }

def pedestrian_pathway_analysis_in_transit_stations(df):
    print("\n--- Pedestrian Pathway Analysis in Transit Stations ---")
    expected = {
        'PathwayID': ['PathwayID', 'ID'],
        'FromStopID': ['FromStopID', 'OriginStopID'],
        'ToStopID': ['ToStopID', 'DestinationStopID'],
        'PathwayMode': ['PathwayMode', 'Type', 'Mode'],
        'LengthMeters': ['LengthMeters', 'Length', 'Distance'],
        'IsAccessible': ['IsAccessible', 'Accessible', 'WheelchairAccessible']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['LengthMeters'] = pd.to_numeric(df['LengthMeters'], errors='coerce')
    df['IsAccessible'] = df['IsAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['PathwayMode', 'LengthMeters', 'IsAccessible'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_pathways = len(df)
    avg_pathway_length = df['LengthMeters'].mean()
    accessible_pathways_pct = (df['IsAccessible'] == 'Yes').mean() * 100
    
    print(f"Total Pedestrian Pathways: {total_pathways}")
    print(f"Average Pathway Length: {avg_pathway_length:.2f} meters")
    print(f"Percentage of Accessible Pathways: {accessible_pathways_pct:.2f}%")
    
    fig1 = px.histogram(df, x='PathwayMode', color='IsAccessible', barmode='group',
                       title='Pathway Mode Distribution by Accessibility')
    fig1.show()
    
    fig2 = px.box(df, x='PathwayMode', y='LengthMeters', title='Pathway Length Distribution by Mode')
    fig2.show()
    
    fig3 = px.pie(df, names='IsAccessible', title='Accessibility of Pedestrian Pathways')
    fig3.show()

    return {
        "metrics": {
            "Total Pedestrian Pathways": total_pathways,
            "Average Pathway Length (meters)": avg_pathway_length,
            "Accessible Pathways (%)": accessible_pathways_pct
        },
        "figures": {
            "Pathway_Mode_Distribution_by_Accessibility_Histogram": fig1,
            "Pathway_Length_Distribution_by_Mode_Box": fig2,
            "Accessibility_of_Pedestrian_Pathways_Pie": fig3
        }
    }

def transit_route_information_analysis(df):
    print("\n--- Transit Route Information Analysis ---")
    expected = {
        'RouteID': ['RouteID', 'ID', 'Route_ID'],
        'AgencyID': ['AgencyID', 'Agency_ID'],
        'RouteShortName': ['RouteShortName', 'ShortName', 'RouteNum'],
        'RouteLongName': ['RouteLongName', 'LongName', 'RouteDescription'],
        'RouteType': ['RouteType', 'Type', 'TransitType'],
        'RouteURL': ['RouteURL', 'URL'],
        'RouteColor': ['RouteColor', 'Color']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df = df.dropna(subset=['RouteID', 'RouteShortName', 'RouteType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_routes = len(df)
    unique_route_types = df['RouteType'].nunique()
    most_common_route_type = df['RouteType'].mode()[0]
    
    print(f"Total Transit Routes: {total_routes}")
    print(f"Unique Route Types: {unique_route_types}")
    print(f"Most Common Route Type: {most_common_route_type}")
    
    fig1 = px.histogram(df, x='RouteType', title='Distribution of Transit Route Types')
    fig1.show()
    
    routes_per_agency = df.groupby('AgencyID').size().reset_index(name='Count')
    fig2 = px.bar(routes_per_agency.sort_values('Count', ascending=False).head(20), x='AgencyID', y='Count',
                  title='Number of Routes per Agency (Top 20)')
    fig2.show()
    
    if 'RouteURL' in df.columns:
        print("\nSample Route URLs:")
        for url in df['RouteURL'].head().tolist():
            print(f"- {url}")

    return {
        "metrics": {
            "Total Transit Routes": total_routes,
            "Unique Route Types": unique_route_types,
            "Most Common Route Type": most_common_route_type
        },
        "figures": {
            "Transit_Route_Types_Distribution_Histogram": fig1,
            "Routes_per_Agency_Bar": fig2
        }
    }

def trip_accessibility_and_direction_analysis(df):
    print("\n--- Trip Accessibility and Direction Analysis ---")
    expected = {
        'TripID': ['TripID', 'ID'],
        'RouteID': ['RouteID', 'Route_ID'],
        'DirectionID': ['DirectionID', 'Direction'],
        'WheelchairAccessible': ['WheelchairAccessible', 'Accessible', 'Wheelchair'],
        'BikesAllowed': ['BikesAllowed', 'BikeAccess', 'Bikes']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['DirectionID'] = pd.to_numeric(df['DirectionID'], errors='coerce')
    df['WheelchairAccessible'] = df['WheelchairAccessible'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df['BikesAllowed'] = df['BikesAllowed'].astype(str).str.lower().map({'1': 'Yes', '0': 'No', 'yes': 'Yes', 'no': 'No', 'true': 'Yes', 'false': 'No'})
    df = df.dropna(subset=['TripID', 'RouteID', 'DirectionID', 'WheelchairAccessible', 'BikesAllowed'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_trips = len(df)
    accessible_trips_pct = (df['WheelchairAccessible'] == 'Yes').mean() * 100
    bikes_allowed_trips_pct = (df['BikesAllowed'] == 'Yes').mean() * 100
    
    print(f"Total Trips Analyzed: {total_trips}")
    print(f"Percentage of Wheelchair Accessible Trips: {accessible_trips_pct:.2f}%")
    print(f"Percentage of Trips Allowing Bikes: {bikes_allowed_trips_pct:.2f}%")
    
    fig1 = px.pie(df, names='WheelchairAccessible', title='Wheelchair Accessibility of Trips')
    fig1.show()
    
    fig2 = px.pie(df, names='BikesAllowed', title='Bike Accessibility of Trips')
    fig2.show()
    
    fig3 = px.histogram(df, x='DirectionID', title='Distribution of Trip Directions')
    fig3.show()

    return {
        "metrics": {
            "Total Trips Analyzed": total_trips,
            "Wheelchair Accessible Trips (%)": accessible_trips_pct,
            "Trips Allowing Bikes (%)": bikes_allowed_trips_pct
        },
        "figures": {
            "Wheelchair_Accessibility_Pie": fig1,
            "Bike_Accessibility_Pie": fig2,
            "Trip_Directions_Distribution_Histogram": fig3
        }
    }

def scheduled_stop_times_analysis_for_trips(df):
    print("\n--- Scheduled Stop Times Analysis for Trips ---")
    expected = {
        'TripID': ['TripID', 'Trip_ID'],
        'StopID': ['StopID', 'Stop_ID'],
        'ArrivalTime': ['ArrivalTime', 'ScheduledArrival', 'Arrival'],
        'DepartureTime': ['DepartureTime', 'ScheduledDeparture', 'Departure'],
        'StopSequence': ['StopSequence', 'Sequence', 'Order'],
        'PickupType': ['PickupType', 'PickupRule'],
        'DropOffType': ['DropOffType', 'DropOffRule']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['ArrivalTime'] = pd.to_datetime(df['ArrivalTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce').dt.time
    df['StopSequence'] = pd.to_numeric(df['StopSequence'], errors='coerce')
    df = df.dropna(subset=['TripID', 'StopID', 'ArrivalTime', 'DepartureTime', 'StopSequence'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_scheduled_stops = len(df)
    avg_stops_per_trip = df.groupby('TripID').size().mean()
    
    print(f"Total Scheduled Stop Times Records: {total_scheduled_stops}")
    print(f"Average Number of Stops per Trip: {avg_stops_per_trip:.1f}")
    
    fig1 = px.histogram(df, x='StopSequence', title='Distribution of Stop Sequences within Trips')
    fig1.show()
    
    if 'PickupType' in df.columns:
        fig2 = px.histogram(df, x='PickupType', title='Distribution of Pickup Types')
        fig2.show()
    
    if 'DropOffType' in df.columns:
        fig3 = px.histogram(df, x='DropOffType', title='Distribution of Drop-Off Types')
        fig3.show()

    # Calculate dwell time if both arrival and departure are available and meaningful
    df['DwellTimeSeconds'] = (pd.to_datetime(df['DepartureTime'].astype(str)) - pd.to_datetime(df['ArrivalTime'].astype(str))).dt.total_seconds()
    df['DwellTimeSeconds'] = df['DwellTimeSeconds'].apply(lambda x: x if x >= 0 else np.nan)
    if not df['DwellTimeSeconds'].dropna().empty:
        fig4 = px.box(df, y='DwellTimeSeconds', title='Dwell Time Distribution at Stops (Seconds)')
        fig4.show()

    return {
        "metrics": {
            "Total Scheduled Stop Times Records": total_scheduled_stops,
            "Average Number of Stops per Trip": avg_stops_per_trip
        },
        "figures": {
            "Stop_Sequences_Distribution_Histogram": fig1,
            "Pickup_Types_Distribution_Histogram": fig2 if 'PickupType' in df.columns else None,
            "Drop_Off_Types_Distribution_Histogram": fig3 if 'DropOffType' in df.columns else None,
            "Dwell_Time_at_Stops_Box": fig4 if 'DwellTimeSeconds' in df.columns and not df['DwellTimeSeconds'].dropna().empty else None
        }
    }

def special_service_dates_and_schedule_exception_analysis(df):
    print("\n--- Special Service Dates and Schedule Exception Analysis ---")
    expected = {
        'ServiceID': ['ServiceID', 'ID', 'Service_ID'],
        'Date': ['Date', 'ServiceDate', 'ExceptionDate'],
        'ExceptionType': ['ExceptionType', 'Type', 'AddedRemoved'], # 1 for added, 2 for removed
        'Description': ['Description', 'Note', 'Reason']
    }
    df, missing = check_and_rename_columns(df, expected)

    if missing:
        show_missing_columns_warning(missing, expected)
        return {"message": "Missing required columns, unable to perform specific analysis."}
    
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['ExceptionType'] = pd.to_numeric(df['ExceptionType'], errors='coerce')
    df = df.dropna(subset=['Date', 'ExceptionType'])

    if df.empty:
        print("No sufficient data after cleaning for this analysis.")
        return {"message": "No sufficient data after cleaning."}

    total_exceptions = len(df)
    added_services = (df['ExceptionType'] == 1).sum()
    removed_services = (df['ExceptionType'] == 2).sum()
    
    print(f"Total Special Service Dates/Exceptions: {total_exceptions}")
    print(f"Number of Added Services: {added_services}")
    print(f"Number of Removed Services: {removed_services}")
    
    fig1 = px.histogram(df, x='ExceptionType', title='Distribution of Exception Types (1: Added, 2: Removed)')
    fig1.show()
    
    exceptions_by_month = df['Date'].dt.to_period('M').value_counts().sort_index().reset_index(name='Count')
    exceptions_by_month.columns = ['Month', 'Count']
    fig2 = px.line(exceptions_by_month, x='Month', y='Count', title='Monthly Trend of Service Exceptions')
    fig2.show()

    if 'ServiceID' in df.columns:
        exceptions_per_service = df.groupby('ServiceID').size().reset_index(name='NumExceptions')
        fig3 = px.histogram(exceptions_per_service, x='NumExceptions', title='Distribution of Number of Exceptions per Service ID')
        fig3.show()

    return {
        "metrics": {
            "Total Special Service Dates/Exceptions": total_exceptions,
            "Added Services": added_services,
            "Removed Services": removed_services
        },
        "figures": {
            "Exception_Types_Distribution_Histogram": fig1,
            "Monthly_Service_Exceptions_Trend_Line": fig2,
            "Num_Exceptions_per_Service_ID_Histogram": fig3 if 'ServiceID' in df.columns else None
        }
    }

# --- Main execution logic for the console application ---

def main():
    print("🚛 Transportation Analytics Dashboard")
    file_path = input("Enter path to your transportation data file (csv or xlsx): ")
    encoding = input("Enter file encoding (utf-8, latin1, cp1252), or press Enter for utf-8: ")
    if not encoding:
        encoding = 'utf-8'
    
    df = load_data(file_path, encoding=encoding)
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print("Data loaded successfully!")
    
    analysis_options = [
        "fleet_analysis",
        "route_analysis",
        "driver_analysis",
        "fuel_analysis",
        "maintenance_analysis",
        "delivery_analysis",
        "cost_analysis",
        "safety_analysis",
        "commuter_transportation_mode_choice_analysis",
        "public_bus_performance_and_delay_analysis",
        "global_air_transport_passenger_trends_analysis",
        "airline_directory_and_operational_status_analysis",
        "public_transit_fare_and_journey_type_analysis",
        "regional_vehicle_registration_trend_analysis",
        "transportation_user_survey_response_analysis",
        "bus_route_schedule_analysis",
        "public_transit_station_ridership_analysis",
        "county-level_transportation_infrastructure_and_commute_analysis",
        "transit_agency_information_analysis",
        "public_transit_route_definition_analysis",
        "transit_trip_schedule_and_accessibility_analysis",
        "transit_stop_location_and_information_analysis",
        "transit_stop_time_and_sequence_analysis",
        "transit_service_calendar_analysis",
        "transit_service_exception_and_holiday_schedule_analysis",
        "transit_fare_structure_analysis",
        "transit_fare_rule_and_zone_analysis",
        "transit_route_shape_and_path_geospatial_analysis",
        "transit_frequency_and_headway_analysis",
        "station_pathway_and_accessibility_analysis",
        "gtfs_feed_information_and_version_analysis",
        "real-time_vehicle_position_and_trip_update_analysis",
        "extended_transit_route_attribute_analysis",
        "transit_fare_zone_definition_analysis",
        "multi-level_station_and_platform_information_analysis",
        "transit_trip_stop_timepoint_analysis",
        "transit_trip_details_and_accessibility_features_analysis",
        "transit_stop_and_station_location_analysis",
        "public_transportation_route_details_analysis",
        "transportation_agency_contact_and_timezone_analysis",
        "transit_trip_planning_and_route_shape_analysis",
        "transit_service_schedule_definition",
        "stop-by-stop_transit_schedule_analysis",
        "public_transport_agency_directory_analysis",
        "transit_fare_attribute_analysis",
        "inter-stop_transfer_path_analysis",
        "geospatial_route_path_analysis",
        "trip_frequency_and_service_interval_analysis",
        "fare_cost_and_transfer_policy_analysis",
        "trip_service_detail_and_distance_analysis",
        "transit_data_feed_publisher_information_analysis",
        "pedestrian_pathway_analysis_in_transit_stations",
        "transit_route_information_analysis",
        "trip_accessibility_and_direction_analysis",
        "scheduled_stop_times_analysis_for_trips",
        "special_service_dates_and_schedule_exception_analysis",
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

    # Execute analysis based on selection
    if selected == "fleet_analysis":
        fleet_analysis(df.copy())
    elif selected == "route_analysis":
        route_analysis(df.copy())
    elif selected == "driver_analysis":
        driver_analysis(df.copy())
    elif selected == "fuel_analysis":
        fuel_analysis(df.copy())
    elif selected == "maintenance_analysis":
        maintenance_analysis(df.copy())
    elif selected == "delivery_analysis":
        delivery_analysis(df.copy())
    elif selected == "cost_analysis":
        cost_analysis(df.copy())
    elif selected == "safety_analysis":
        safety_analysis(df.copy())
    elif selected == "commuter_transportation_mode_choice_analysis":
        commuter_transportation_mode_choice_analysis(df.copy())
    elif selected == "public_bus_performance_and_delay_analysis":
        public_bus_performance_and_delay_analysis(df.copy())
    elif selected == "global_air_transport_passenger_trends_analysis":
        global_air_transport_passenger_trends_analysis(df.copy())
    elif selected == "airline_directory_and_operational_status_analysis":
        airline_directory_and_operational_status_analysis(df.copy())
    elif selected == "public_transit_fare_and_journey_type_analysis":
        public_transit_fare_and_journey_type_analysis(df.copy())
    elif selected == "regional_vehicle_registration_trend_analysis":
        regional_vehicle_registration_trend_analysis(df.copy())
    elif selected == "transportation_user_survey_response_analysis":
        transportation_user_survey_response_analysis(df.copy())
    elif selected == "bus_route_schedule_analysis":
        bus_route_schedule_analysis(df.copy())
    elif selected == "public_transit_station_ridership_analysis":
        public_transit_station_ridership_analysis(df.copy())
    elif selected == "county-level_transportation_infrastructure_and_commute_analysis":
        county_level_transportation_infrastructure_and_commute_analysis(df.copy())
    elif selected == "transit_agency_information_analysis":
        transit_agency_information_analysis(df.copy())
    elif selected == "public_transit_route_definition_analysis":
        public_transit_route_definition_analysis(df.copy())
    elif selected == "transit_trip_schedule_and_accessibility_analysis":
        transit_trip_schedule_and_accessibility_analysis(df.copy())
    elif selected == "transit_stop_location_and_information_analysis":
        transit_stop_location_and_information_analysis(df.copy())
    elif selected == "transit_stop_time_and_sequence_analysis":
        transit_stop_time_and_sequence_analysis(df.copy())
    elif selected == "transit_service_calendar_analysis":
        transit_service_calendar_analysis(df.copy())
    elif selected == "transit_service_exception_and_holiday_schedule_analysis":
        transit_service_exception_and_holiday_schedule_analysis(df.copy())
    elif selected == "transit_fare_structure_analysis":
        transit_fare_structure_analysis(df.copy())
    elif selected == "transit_fare_rule_and_zone_analysis":
        transit_fare_rule_and_zone_analysis(df.copy())
    elif selected == "transit_route_shape_and_path_geospatial_analysis":
        transit_route_shape_and_path_geospatial_analysis(df.copy())
    elif selected == "transit_frequency_and_headway_analysis":
        transit_frequency_and_headway_analysis(df.copy())
    elif selected == "station_pathway_and_accessibility_analysis":
        station_pathway_and_accessibility_analysis(df.copy())
    elif selected == "gtfs_feed_information_and_version_analysis":
        gtfs_feed_information_and_version_analysis(df.copy())
    elif selected == "real-time_vehicle_position_and_trip_update_analysis":
        real_time_vehicle_position_and_trip_update_analysis(df.copy())
    elif selected == "extended_transit_route_attribute_analysis":
        extended_transit_route_attribute_analysis(df.copy())
    elif selected == "transit_fare_zone_definition_analysis":
        transit_fare_zone_definition_analysis(df.copy())
    elif selected == "multi-level_station_and_platform_information_analysis":
        multi_level_station_and_platform_information_analysis(df.copy())
    elif selected == "transit_trip_stop_timepoint_analysis":
        transit_trip_stop_timepoint_analysis(df.copy())
    elif selected == "transit_trip_details_and_accessibility_features_analysis":
        transit_trip_details_and_accessibility_features_analysis(df.copy())
    elif selected == "transit_stop_and_station_location_analysis":
        transit_stop_and_station_location_analysis(df.copy())
    elif selected == "public_transportation_route_details_analysis":
        public_transportation_route_details_analysis(df.copy())
    elif selected == "transportation_agency_contact_and_timezone_analysis":
        transportation_agency_contact_and_timezone_analysis(df.copy())
    elif selected == "transit_trip_planning_and_route_shape_analysis":
        transit_trip_planning_and_route_shape_analysis(df.copy())
    elif selected == "transit_service_schedule_definition":
        transit_service_schedule_definition(df.copy())
    elif selected == "stop-by-stop_transit_schedule_analysis":
        stop_by_stop_transit_schedule_analysis(df.copy())
    elif selected == "public_transport_agency_directory_analysis":
        public_transport_agency_directory_analysis(df.copy())
    elif selected == "transit_fare_attribute_analysis":
        transit_fare_attribute_analysis(df.copy())
    elif selected == "inter-stop_transfer_path_analysis":
        inter_stop_transfer_path_analysis(df.copy())
    elif selected == "geospatial_route_path_analysis":
        geospatial_route_path_analysis(df.copy())
    elif selected == "trip_frequency_and_service_interval_analysis":
        trip_frequency_and_service_interval_analysis(df.copy())
    elif selected == "fare_cost_and_transfer_policy_analysis":
        fare_cost_and_transfer_policy_analysis(df.copy())
    elif selected == "trip_service_detail_and_distance_analysis":
        trip_service_detail_and_distance_analysis(df.copy())
    elif selected == "transit_data_feed_publisher_information_analysis":
        transit_data_feed_publisher_information_analysis(df.copy())
    elif selected == "pedestrian_pathway_analysis_in_transit_stations":
        pedestrian_pathway_analysis_in_transit_stations(df.copy())
    elif selected == "transit_route_information_analysis":
        transit_route_information_analysis(df.copy())
    elif selected == "trip_accessibility_and_direction_analysis":
        trip_accessibility_and_direction_analysis(df.copy())
    elif selected == "scheduled_stop_times_analysis_for_trips":
        scheduled_stop_times_analysis_for_trips(df.copy())
    elif selected == "special_service_dates_and_schedule_exception_analysis":
        special_service_dates_and_schedule_exception_analysis(df.copy())
    elif selected == "General Insights":
        show_general_insights(df.copy())
    else:
        print(f"Analysis option '{selected}' not recognized or not implemented.")
        show_general_insights(df.copy())

if __name__ == "__main__":
    main()