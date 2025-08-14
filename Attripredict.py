import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
data=pd.read_csv("C:/Users/Hp/Downloads/Employee-Attrition - Employee-Attrition.csv")
df=pd.DataFrame(data)


# Load model & preprocessors
with open("rf_attrition_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Exact feature names from training
feature_names = [
    'Age', 'BusinessTravel', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'JobInvolvement', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating',
    'TotalWorkingYears', 'TrainingTimesLastYear','WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]
# Streamlit UI

st.set_page_config(page_title="Dashboard Home", layout="wide")

menu = st.sidebar.selectbox("🚪Employee Attrition Analysis", ["Home","Predict Employee Attrition"])

# Main page of Employee Attrition
if menu=='Home':
    st.title("👨‍💼 Employee Insights Dashboard")
    # ---- Calculate KPIs ----
    total_employees = len(df)
    attrition_count = df['Attrition'].value_counts().get("Yes", 0)  # Count Yes
    attrition_rate = (attrition_count / total_employees) * 100
    avg_income = df['MonthlyIncome'].mean()
    avg_years = df['YearsAtCompany'].mean()

    # ---- Display in KPI cards ----
    st.title("📊 Employee Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Employees", total_employees)
    col2.metric("Attrition Rate (%)", f"{attrition_rate:.2f}")
    col3.metric("Avg Monthly Income", f"₹{avg_income:,.0f}")
    col4.metric("Avg Years at Company", f"{avg_years:.1f}")
    # Create tabs
    tab1, tab2 = st.tabs(["📊 Bar Charts", "🥧 Pie Charts"])

    # -------------------------
    # 📊 TAB 1: Bar Charts
    # -------------------------
    with tab1:
        st.subheader("Attrition Count by Job Role, Department, Education Field")

        def plot_attrition_bar(column_name, title):
         attrition_counts = df[df['Attrition'] == "Yes"][column_name].value_counts()

         fig, ax = plt.subplots()
         attrition_counts.plot(kind='bar', ax=ax, color='tomato', edgecolor='black')
         ax.set_title(title, fontsize=14)
         ax.set_xlabel(column_name, fontsize=12)
         ax.set_ylabel("Count", fontsize=12)
         plt.xticks(rotation=45, ha='right')

         st.pyplot(fig)

        plot_attrition_bar("JobRole", "Attrition Count by Job Role")
        plot_attrition_bar("Department", "Attrition Count by Department")
        plot_attrition_bar("EducationField", "Attrition Count by Education Field")

    # -------------------------
    # 🥧 TAB 2: Pie Charts
    # -------------------------
    with tab2:
     st.subheader("Attrition by Gender and Marital Status")

     def plot_attrition_pie(column_name, title):
        attrition_counts = df[df['Attrition'] == "Yes"][column_name].value_counts()

        fig, ax = plt.subplots()
        attrition_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90,
                              colors=['skyblue', 'lightgreen', 'orange'], ax=ax)
        ax.set_ylabel("")
        ax.set_title(title, fontsize=14)

        st.pyplot(fig)

     plot_attrition_pie("Gender", "Attrition by Gender")
     plot_attrition_pie("MaritalStatus", "Attrition by Marital Status")
    
elif menu=="Predict Employee Attrition":
    st.title("👨‍💼 Predict Employee Attrition")
    user_input = {}

    for feature in feature_names:
     if feature in label_encoders:  
        # Categorical feature → dropdown from training data
        options = list(label_encoders[feature].classes_)
        user_input[feature] = st.selectbox(f"{feature}:", options)

     else:  
        # Numeric feature → number input
        user_input[feature] = st.number_input(f"{feature}:", value=0)

    if st.button("Predict Attrition"):
       # Convert to DataFrame
        input_df = pd.DataFrame([user_input])

    # Encode categorical values to match model training
        for col, le in label_encoders.items():
         if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    # Scale numeric columns
         numeric_cols = input_df.select_dtypes(include=['int64', 'float64']).columns
         input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
         pred = model.predict(input_df)[0]
         proba = model.predict_proba(input_df)[0]

         st.write(f"**Prediction:** {'Attrition' if pred == 1 else 'No Attrition'}")
         st.write(f"**Probability of Attrition:** {proba[1]:.2f}")
  