import streamlit as st
from COUNTRY_DASHBOARD import run_country_dashboard
from ATHLETE_DASHBOARD import run_athlete_dashboard

st.set_page_config(page_title="Olympics Dashboard", layout="wide")

st.sidebar.title("Olympics Dashboard")
option = st.sidebar.radio("Select Dashboard:", ["Country Medal Prediction", "Athlete Medal Prediction"])

if option == "Country Medal Prediction":
    run_country_dashboard()
elif option == "Athlete Medal Prediction":
    run_athlete_dashboard()
