import streamlit as st
import pandas as pd
import joblib
import pickle

# Define the app
def app():
    # Set the title
    st.title("Employee Attrition Prediction")

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Add input fields to each column
    with col1:
        business_travel = st.selectbox("Business Travel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
        age = st.slider("Age", 18, 65, 30)
        daily_rate = st.slider("Daily Rate", 100, 1500, 500)
        distance_from_home = st.slider("Distance From Home", 1, 30, 10)
        education = st.slider("Education", 1, 5, 3)
        environment_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
        hourly_rate = st.slider("Hourly Rate", 50, 100, 75)
        job_involvement = st.slider("Job Involvement", 1, 4, 2)
        job_level = st.slider("Job Level", 1, 5, 3)
        
    with col2:
        education_field = st.selectbox("Education Field", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
        monthly_income = st.slider("Monthly Income", 1000, 20000, 5000)
        monthly_rate = st.slider("Monthly Rate", 1000, 20000, 5000)
        num_companies_worked = st.slider("Num Companies Worked", 0, 10, 5)
        percent_salary_hike = st.slider("Percent Salary Hike", 0, 25, 10)
        performance_rating = st.slider("Performance Rating", 1, 5, 3)
        relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
        stock_option_level = st.slider("Stock Option Level", 0, 3, 1)

    with col3:
        job_role = st.selectbox("Job Role", ["Sales Executive", "Research Scientist", "Laboratory Technician", "Manufacturing Director", "Healthcare Representative", "Manager", "Sales Representative", "Research Director", "Human Resources"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        over_time = st.selectbox("Over Time", ["Yes", "No"])
        total_working_years = st.slider("Total Working Years", 0, 40, 20)
        training_times_last_year = st.slider("Training Times Last Year", 0, 10, 5)
        work_life_balance = st.slider("Work Life Balance", 1, 4, 2)
        years_at_company = st.slider("Years At Company", 0, 20, 10)
        years_in_current_role = st.slider("Years In Current Role", 0, 20, 10)
        years_since_last_promotion = st.slider("Years Since Last Promotion", 0, 20, 10)
        years_with_curr_manager = st.slider("Years With Curr Manager", 0, 20, 10)
        

    input_data = {
        "Age": age,
        "DailyRate": daily_rate,
        "DistanceFromHome": distance_from_home,
        "Education": education,
        "EnvironmentSatisfaction": environment_satisfaction,
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobSatisfaction": job_satisfaction,
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies_worked,
        "PercentSalaryHike": percent_salary_hike,
        "PerformanceRating": performance_rating,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option_level,
        "TotalWorkingYears": total_working_years,
        "TrainingTimesLastYear": training_times_last_year,
        "WorkLifeBalance": work_life_balance,
        "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "BusinessTravel": business_travel,
        "Department": department,
        "EducationField": education_field,
        "Gender": gender,
        "JobRole": job_role,
        "MaritalStatus": marital_status,
        "OverTime": over_time
    }

    
    # Load the preprocessor object
    # Load the preprocessor object
    preprocessor = joblib.load('./preprocessor.joblib') # preprocessor.pkl
    # Load the model
    # Load the preprocessor object
    model = joblib.load('./model.pkl') # preprocessor.pkl

    # Make predictions
    st.write("## Make Prediction")

    # Add a predict button
    if st.button("Predict"):
        input_df = pd.DataFrame(input_data, index=[0])
        #st.write(input_df.iloc[0])
        # apply the preprocessor
        input_df = preprocessor.transform(input_df)
        # predict
        prediction = model.predict(input_df)

        # display the results
        st.write("### Results")
        if prediction[0] == 1:
            st.warning("Possible attrition employee")
        else:
            st.success("No possible attrition employee")

if __name__ == "__main__":
    app()
