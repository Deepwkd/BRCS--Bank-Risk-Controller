import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from PIL import Image
import torch
import yaml
import os
from pathlib import Path
import io

df = pd.read_csv("Final_new.csv")

# --------------------------------------------------Logo & details on top
st.set_page_config(page_title="Bank Risk Controller System",
                   layout="wide",
                   initial_sidebar_state="expanded")

# Function to safely convert to sqrt
def log_trans(value):
    try:
        return np.log(float(value))  # Conversion to float
    except (ValueError, TypeError):
        raise ValueError(f"Invalid input: {value}")

# Define occupation types in alphabetical order with corresponding numeric codeslabel_encoding
occupation = {
    'Accountants': 0, 'Cleaning staff': 1, 'Cooking staff': 2,
    'Core staff': 3, 'Drivers': 4, 'HR staff': 5, 'High skill tech staff': 6, 'IT staff': 7,
    'Laborers': 8, 'Low-skill Laborers': 9, 'Managers': 10, 'Medicine staff': 11, 'Private service staff': 12,
    'Realty agents': 13, 'Sales staff': 14, 'Secretaries': 15, 'Security staff': 16,
    'Waiters/barmen staff': 17
}

# Mapping for NAME_EDUCATION_TYPE
education = {'Secondary / secondary special': 4,
             'Higher education': 1,
             'Incomplete higher': 2,
             'Lower secondary': 3, 'Academic degree': 0}

# Mapping for Gender
Gender = {'M': 1, 'F': 0, 'XNA': 2}

Income = {'Working': 5, 'State servant': 3, 'Commercial associate': 0, 'Student': 4,
          'Pensioner': 2, 'Maternity leave': 1}

Reject_reason = {'XAP(X-Application Pending)': 7, 'LIMIT(Credit Limit Exceeded)': 2, 'SCO(Scope of Credit)': 3,
                 'HC(High Credit Risk)': 1, 'VERIF(Verification Failed)': 6, 'CLIENT(Client Request)': 0,
                 'SCOFR(Scope of Credit for Rejection)': 4, 'XNA(Not Applicable)': 8, 'SYSTEM(System Error)': 5}

status = {'Approved': 0.0, 'Canceled': 1.0, 'Refused': 2.0, 'Unused offer': 2.5}

Yield = {'low_normal': 3, 'middle': 4, 'XNA': 0, 'high': 1, 'low_action': 2}

with st.sidebar:
    st.image("images (1).png")

    opt = option_menu("Menu",
                      ["Home", 'Matrix Insights', 'EDA', 'Model Prediction', 'ML Sentiment Analysis', "Conclusion"],
                      icons=["house", "table", "bar-chart-line", "graph-up-arrow", "search", "binoculars", "exclamation-circle"],
                      menu_icon="cast",
                      default_index=0,
                      styles={"icon": {"color": "Yellow", "font-size": "20px"},
                              "nav-link": {"font-size": "15px", "text-align": "left", "margin": "-2px", "--hover-color": "blue"},
                              "nav-link-selected": {"background-color": "blue"}})

if opt == "Home":
    # Content for the Home option
    col, coll = st.columns([1, 4], gap="small")
    with col:
        st.write(" ")
    with coll:
        st.markdown("# BANK RISK CONTROLLER SYSTEM")
        st.write(" ")

    st.markdown("""
    ### <span style="color:blue;">OVERVIEW</span>
    The goal of this project is to develop a reliable predictive model 
    that effectively identifies customers at high risk of loan default. 
    This will allow the financial institution to proactively manage its credit portfolio, 
    implement targeted strategies, and ultimately minimize the likelihood of loan defaults.
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.markdown("""
        ### <span style="color:blue;">DOMAIN</span>
        Banking
        """, unsafe_allow_html=True)

        st.markdown("""
        ### <span style="color:blue;">TECHNOLOGIES USED</span>
        - Python  
        - Data Preprocessing  
        - EDA (Exploratory Data Analysis)  
        - Pandas  
        - Numpy  
        - Visualization  
        - Machine Learning - Classification Model  
        - Streamlit GUI  
        """, unsafe_allow_html=True)

    with col2:
        st.write(" ")

elif opt == "Matrix Insights":
    # Content for the Matrix Insights option
    st.markdown("""
    ### <span style="color:blue;">DataFrame and Matrix Insights</span>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### <span style="color:blue;">Model Performance</span>
    """, unsafe_allow_html=True)

    # Create DataFrame with performance metrics
    data = {
        "Algorithm": ["Decision Tree", "KNN", "Random Forest", "XGradientBoost"],
        "Accuracy": [95, 93, 94, 93],
        "Precision": [95, 93, 95, 1],
        "Recall": [95, 93, 95, 1],
        "F1 Score": [95, 93, 95, 1]
    }
    dff = pd.DataFrame(data)

    # Display DataFrame
    st.dataframe(dff)

    # Highlight the selected algorithm and accuracy
    st.markdown("""
    ## The Selected Algorithm is <span style="color:green;"><b>Decision Tree</b></span> and its Accuracy is <span style="color:green;"><b>95%</b></span>
    """, unsafe_allow_html=True)

elif opt == "EDA":
    st.subheader(":blue[Insights of Bank Risk Controller System]")

    col1, col2 = st.columns(2)

    # Function to plot skewed data using a boxplot
    def skewplot(df, column):
        fig, ax = plt.subplots(figsize=(5, 4))  # Set figure size
        sns.boxplot(x=df[column], ax=ax)  # Plot boxplot
        plt.tight_layout()  # Adjust layout
        st.pyplot(fig)  # Display the plot in Streamlit

    with col1:
        # Columns to visualize for skewness
        skewed_columns = [
            'NAME_EDUCATION_TYPE', 'NAME_CONTRACT_STATUS', 'CODE_REJECT_REASON',
            'CODE_GENDER', 'NAME_GOODS_CATEGORY', 'NAME_YIELD_GROUP',
            'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'REGION_RATING_CLIENT_W_CITY',
            'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_DECISION',
            'REG_CITY_NOT_WORK_CITY', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
            'DAYS_REGISTRATION', 'AMT_INCOME_TOTAL', 'TARGET'
        ]

        # Loop through skewed columns and display plots
        st.write("### Skewed Data Before Transformation")
        for column in skewed_columns:
            if column in df.columns:
                st.write(f"#### {column}")
                skewplot(df, column)
            else:
                st.write(f"Column {column} not found in the dataframe.")

    with col2:
        # Apply log transformation to reduce skewness
        df['DAYS_EMPLOYED_log'] = np.log1p(df['DAYS_EMPLOYED'].clip(lower=1))
        df['AMT_INCOME_TOTAL_log'] = np.log1p(df['AMT_INCOME_TOTAL'].clip(lower=1))
        df['DAYS_LAST_PHONE_CHANGE_log'] = np.log1p(df['DAYS_LAST_PHONE_CHANGE'].clip(lower=1))
        df['DAYS_ID_PUBLISH_log'] = np.log1p(df['DAYS_ID_PUBLISH'].clip(lower=1))
        df['DAYS_REGISTRATION_log'] = np.log1p(df['DAYS_REGISTRATION'].clip(lower=1))

        # Columns to visualize after log transformation
        transformed_columns = [
            'AMT_INCOME_TOTAL_log', 'DAYS_LAST_PHONE_CHANGE_log',
            'DAYS_ID_PUBLISH_log', 'DAYS_REGISTRATION_log', 'DAYS_EMPLOYED_log'
        ]

        # Loop through transformed columns and display plots
        st.write("### Skewed Data After Log Transformation")
        for column in transformed_columns:
            if column in df.columns:
                st.write(f"#### {column}")
                skewplot(df, column)
            else:
                st.write(f"Column {column} not found in the dataframe.")

elif opt == "Model Prediction":
    # Streamlit form for user inputs
    st.markdown(f'## :blue[Predicting Customers Default on Loans]')
    st.write(" ")

    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
        
        with col1:
            OCCUPATION_TYPE = st.selectbox("OCCUPATION TYPE", list(occupation.keys()), key='OCCUPATION_TYPE')
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education.keys()), key='EDUCATION_TYPE')
            NAME_INCOME_TYPE = st.selectbox("INCOME TYPE", list(Income.keys()), key='NAME_INCOME_TYPE')
            TOTAL_INCOME = st.number_input("TOTAL INCOME PA", key='TOTAL_INCOME', format="%.2f")
            CODE_REJECT_REASON = st.selectbox("CODE REJECTION REASON", list(Reject_reason.keys()), key='CODE_REJECT_REASON')
            NAME_CONTRACT_STATUS = st.selectbox("CONTRACT STATUS", list(status.keys()), key='NAME_CONTRACT_STATUS')
            NAME_YIELD_GROUP = st.selectbox("YIELD GROUP", list(Yield.keys()), key='NAME_YIELD_GROUP')

        with col2:
            CODE_GENDER = st.selectbox("CODE GENDER", list(Gender.keys()), key='CODE_GENDER')
            AGE = st.text_input("AGE", key="AGE")
            CLIENT_RATING = st.text_input("CLIENT RATING", key="CLIENT_RATING")
            DAYS_LAST_PHONE_CHANGE = st.number_input("PHONE CHANGE", key="DAYS_LAST_PHONE_CHANGE", format="%.2f")
            DAYS_ID_PUBLISH = st.number_input("DAYS ID PUBLISH", key="DAYS_ID_PUBLISH", format="%.1f")
            DAYS_REGISTRATION = st.number_input("DAYS REGISTRATION", key="DAYS_REGISTRATION", format="%.1f")
            DAYS_EMPLOYED_log = st.number_input("DAYS EMPLOYED", key='DAYS_EMPLOYED_log', format="%.5f")
        
        submit_button = st.form_submit_button(label="PREDICT STATUS")

        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #ADD8E6;
            color: green;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    flag = 0
    if submit_button:
        try:
            for i in [TOTAL_INCOME, DAYS_EMPLOYED_log, AGE, CLIENT_RATING, DAYS_LAST_PHONE_CHANGE, DAYS_ID_PUBLISH, DAYS_REGISTRATION]:             
                if i is None or i == '':
                    flag = 1
                    break
        except ValueError:
            flag = 1
        
        if flag == 1:
            st.write("Please enter a valid number. Fields cannot be empty.")
        if submit_button and flag == 1:
            if len(i) == 0:
                st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)

        if submit_button and flag == 0:
            try:
                # Encode categorical variables
                le = LabelEncoder()

                Occupation = occupation[OCCUPATION_TYPE]
                Education = education[EDUCATION_TYPE]
                Income_type = Income[NAME_INCOME_TYPE]
                Income_amt = int(TOTAL_INCOME)
                Days_employed = int(DAYS_EMPLOYED_log)  
                Reason = Reject_reason[CODE_REJECT_REASON]
                Status = status[NAME_CONTRACT_STATUS]
                yield_group = Yield[NAME_YIELD_GROUP]
                Genders = Gender[CODE_GENDER]
                Age  = int(AGE.strip())
                Rating = int(CLIENT_RATING.strip())
                Phone  = int(DAYS_LAST_PHONE_CHANGE)
                ID_Published = int(DAYS_ID_PUBLISH)
                Registration = int(DAYS_REGISTRATION)

                # Create sample array with encoded categorical variables
                sample = np.array([
                    [
                        Occupation,
                        Education,
                        Income_type,
                        Income_amt,
                        Reason,
                        Status,
                        yield_group,
                        Genders,
                        Age,
                        Rating,
                        Phone, 
                        ID_Published, 
                        Registration,
                        log_trans(Days_employed) 
                    ]
                ])

                with open("dtmodel.pkl", 'rb') as file:
                    Decision_tree = pickle.load(file)

                pred = Decision_tree.predict(sample)

                if pred == 1:
                    st.markdown(f' ## The status is: :red[Won\'t Repay]')
                else:
                    st.write(f' ## The status is: :green[Repay]')
            except ValueError as e:
                st.error(f"Error processing inputs: {e}")
                st.write("Please check your input values. Only numeric values are allowed.")

elif opt == "ML Sentiment Analysis":

    st.markdown(f"## :blue[ML Sentiment Analysis]")
    st.write("")
    st.write("")

    # Initialize the sentiment analyzer
    nltk.download('vader_lexicon')  # VADER (Valence Aware Dictionary and Sentiment Reasoner)
    sia = SentimentIntensityAnalyzer()

    # Create a function to analyze the sentiment
    def analyze_sentiment(text):
        sentiment = sia.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            return "Positive"
        elif sentiment['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Create a Streamlit app
    st.title("Sentiment Analysis App")

    # Get the user input
    text = st.text_input("Enter some text:")

    # Check if text is not empty
    if text:
        # Analyze the sentiment
        sentiment = analyze_sentiment(text)

        # Display the sentiment as a word
        st.write("Sentiment:", sentiment)

        # Get the sentiment scores
        sentiment = sia.polarity_scores(text)

        # Display the bar chart
        st.bar_chart({'Positive': sentiment['pos'], 'Negative': sentiment['neg'], 'Neutral': sentiment['neu']})

elif opt == "Conclusion":
    st.markdown(f"## :blue[Conclusion]")
    st.markdown(f"#### In the financial industry, Default occurs when a borrower fails to meet the legal obligations of a loan. The ML Model Streamlit App can accurately identify the customers who are likely to default on their loans based on their historical data.")
    st.markdown(f"#### It also provides deep insights on the features which can help predict the customers who are likely to default on their loans.")
    st.markdown(f"#### This will enable the financial institution to proactively manage their credit portfolio and ultimately reduce the risk of loan defaults.")
    st.write(" ")
