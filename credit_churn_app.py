from datetime import date
import streamlit as st
import pandas as pd
import joblib
import time
import random


st.set_page_config(
    page_title="Credit card Churn Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header("💳 Credit Card Churn Predictor")
st.caption("Predicting the likelihood of credit card customer churn using machine learning.")
st.divider()

#categorical options
Gender = ["M", "F"]
Education_Level = ["Uneducated","High School","College","Graduate","Post-Graduate","Doctorate","Unknown"]
Marital_Status = ["Single","Married","Divorced","Unknown"]
Income_Category = ["Less than $40K","$40K - $60K","$60K - $80K","$80K - $120K","$120K +","Unknown"]
Card_Category = ["Blue","Silver","Gold","Platinum"]

st.sidebar.header("Input Customer Details")
# Demographic Information and Account Details
st.sidebar.subheader("Demographic and Account")
col1,col2  = st.sidebar.columns(2)
col1.selectbox("Gender", Gender,key='gender')
col2.number_input("Age", min_value=18, max_value=100, value=30, step=1,key='age')
st.sidebar.slider('Dependent Count', 0, 10, 0,key='dependent_count')
st.sidebar.selectbox("Education Level", Education_Level,key='education_level')
st.sidebar.selectbox("Marital Status", Marital_Status,key='marital_status')
st.sidebar.selectbox("Income Category", Income_Category,key='income_category')
st.sidebar.selectbox("Card Category", Card_Category,key='card_category')
st.sidebar.slider('Months on Book', 0,240, 12,key='months_on_book')
st.sidebar.slider('Total Relationship Count', 1,10,1,key='relationship_count')

# Activity and transaction details
st.sidebar.subheader("Activity and Transaction Details")
st.sidebar.slider('Months Inactive', 0, 12, 1,key='months_inactive')
st.sidebar.slider('Contacts Count', 0, 20, 1,key='contacts_count')
st.sidebar.number_input('Total Transiction Amount', min_value=0, max_value=200000, value=1000, step=100,key='trans_amount')
st.sidebar.number_input('Total Transiction Count', min_value=0, max_value=500, value=10, step=1,key='trans_count')
st.sidebar.number_input('Total Amount Change Q4-Q1', min_value=0.0, max_value=5.0, value=0.0, step=0.001,format="%0.3f",key='amount_change')
st.sidebar.number_input('Total Count Change Q4-Q1', min_value=0.0, max_value=5.0, value=0.0, step=0.001,format="%0.3f",key='count_change')

# Credit card usage details
st.sidebar.subheader("Credit Card Usage Details")
st.sidebar.number_input('Card Limit', min_value=100, max_value=100000, value=5000, step=100,key='card_limit')
st.sidebar.number_input('Total Revolving Balance', min_value=0, max_value=100000, value=1000, step=100,key='revolving_balance')
st.sidebar.number_input('Avg Open to Buy', min_value=0, max_value=100000, value=4000, step=100,key='avg_open_to_buy')
st.sidebar.number_input('Avg Utilization Ratio', min_value=0.0, max_value=1.0, value=0.3, step=0.01,key='util_ratio')




def new_features(data_df):
    data_df['Tenure_Age_ratio'] = data_df['Months_on_book'] /(data_df['Customer_Age'] *12)
    data_df['Credit_Age_ratio'] = data_df['Credit_Limit'] /(data_df['Customer_Age']*12)
    data_df['Utilization_per_Age'] = data_df['Avg_Utilization_Ratio'] /(data_df['Customer_Age']*12)
    
    data_df['Amount_per_credit'] = data_df['Total_Trans_Amt'] /(data_df['Credit_Limit'])
    data_df['Amount_count_per_credit'] = data_df['Total_Trans_Ct'] /(data_df['Credit_Limit'])
    
    return data_df

def model_load():
    try:
        model = joblib.load('models/best_model.pkl')
        pipeline = joblib.load('models/preprocess_pipeline.pkl')
        st.sidebar.success("Model and pipeline loaded successfully!")
        return model, pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, pipeline = model_load()
main_features=["Gender", "Customer_Age", "Dependent_count", "Education_Level",
               "Marital_Status", "Income_Category", "Card_Category",
               "Months_on_book", "Total_Relationship_Count",
               "Months_Inactive_12_mon", "Contacts_Count_12_mon",
               "Credit_Limit", "Total_Revolving_Bal", "Avg_Open_To_Buy",
               "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct",
               "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]
Engineered_features=["Tenure_Age_ratio", "Credit_Age_ratio", "Utilization_per_Age",
                     "Amount_per_credit", "Amount_count_per_credit"]
if st.sidebar.button("Predict Churn Risk",type='primary'):
    

    input_df = pd.DataFrame([{
    "Gender": st.session_state.gender,
    "Customer_Age": st.session_state.age,
    "Dependent_count": st.session_state.dependent_count,
    "Education_Level": st.session_state.education_level,
    "Marital_Status": st.session_state.marital_status,
    "Income_Category": st.session_state.income_category,
    "Card_Category": st.session_state.card_category,
    "Months_on_book": st.session_state.months_on_book,
    "Total_Relationship_Count": st.session_state.relationship_count,
    "Months_Inactive_12_mon": st.session_state.months_inactive,
    "Contacts_Count_12_mon": st.session_state.contacts_count,
    "Credit_Limit": st.session_state.card_limit,
    "Total_Revolving_Bal": st.session_state.revolving_balance,
    "Avg_Open_To_Buy": st.session_state.avg_open_to_buy,
    "Total_Amt_Chng_Q4_Q1": st.session_state.amount_change,
    "Total_Trans_Amt": st.session_state.trans_amount,
    "Total_Trans_Ct": st.session_state.trans_count,
    "Total_Ct_Chng_Q4_Q1": st.session_state.count_change,
    "Avg_Utilization_Ratio": st.session_state.util_ratio,
    
    }])

    input_df_new = new_features(input_df)
    
    processed_input = pipeline.transform(input_df_new)
    prediction = model.predict(processed_input)[0]
    proba = model.predict_proba(processed_input)[0][1]
    proba_percent = proba * 100


    st.subheader("Prediction Results ")


    if proba>= 0.75:
        risk_level = "🚨Critical Risk"
        alert_style = st.error
    elif proba>=0.5:
        risk_level = "⚠️Elevated Risk"
        alert_style = st.warning
    elif proba>=0.25:
        risk_level = "🛡️Moderate Risk"
        alert_style = st.info
    else:
        risk_level = "🟢Minimal Risk"
        alert_style = st.success

    alert_style(f"{risk_level} | probability of churn: {proba:.4f}")
    st.markdown(f"Model Assement : The profile is assessed to have a **{risk_level}** of credit card churn with a probability of **{proba_percent:.2f}%** based on the provided customer details.")

    st.divider()

    st.subheader("Feature Details")
    main_features_col, engineered_features_col = st.columns(2)
    with main_features_col:
        st.markdown("**Input Features**")
        df_main_display = input_df[main_features].T.rename_axis('Features').rename(columns={0:'Values'})
        df_main_display['Values'] = df_main_display['Values'].astype(str)
        st.dataframe(df_main_display)

    with engineered_features_col:
        st.markdown("**Engineered Features**")
        df_display = input_df_new[Engineered_features].round(4)
        df_display = df_display.T.rename_axis('Features').rename(columns={0:'Values'})
        df_display['Values'] = df_display['Values'].astype(str)
        st.dataframe(df_display)


else:
    st.info("⬅️ Please enter customer details in the sidebar and click on 'Predict Churn Risk'")







