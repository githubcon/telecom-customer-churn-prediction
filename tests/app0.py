# Load  the librairies
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

from flask import Flask, request, jsonify
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
import pickle
import joblib

# Create a title
st.title('Welcome to Churn Prediction Application Using Machine Learning Algorithms') 



# =========================
# LOAD MODELS
# =========================
forest = joblib.load("RandomForestClassifier.pkl")

knn = joblib.load("KNeighborsClassifier.pkl")

with open("XGBClassifier.pkl", "rb") as model:
    xgb = pickle.load(model)
    
with open("LogisticRegression.pkl", "rb") as logistic:
    log_reg = pickle.load(logistic)
    
    
with open("SVC.pkl", "rb") as f:
    svc = pickle.load(f)




# =========================
# LOAD datasets
# =========================
y_test = pd.read_csv("y_test.csv", sep=";", index_col = 0).values.ravel()
x_test = pd.read_csv("x_test.csv", sep=";", index_col=0)

scaler = joblib.load("scaler.pkl")
x_test = scaler.transform(x_test)



# =========================
# Sidebar model selection
# =========================
with st.sidebar:
    selected= option_menu(menu_title= "Which churn prediction model do you choose ?",   #required
    options= ["XGBClassifier", "RandomForestClassifier", "LogisticRegression", "SVC", "KNeighborsClassifier"],   #required
    icons= ["graph-up", "diagram-3" ,"percent", "bounding-box", "people" ] ,  #optional
    menu_icon= "cast",   #optional
    default_index= 0,   #optional
    styles={
            "container": {"padding": "5!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "20px"}, 
            "nav-link": {"font-size": "18px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "green"},
        })
    
    
    
    
# =========================
# Model Evaluation
# =========================
def evaluate_model(model, name):
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)

    st.sidebar.metric(f"{name} Accuracy", f"{accuracy:.2%}")
    st.sidebar.metric(f"{name} Precision", f"{precision:.2%}")

    # AUC only if model supports predict_proba
    if hasattr(model, "predict_proba"):
        
        y_proba = model.predict_proba(x_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_proba)
        
        st.sidebar.metric(f"{name} AUC", f"{auc:.2%}")

    else:
        st.sidebar.info(f"{name}: AUC not available")     

# ----------------------------
# Routing
# ----------------------------
if selected == "XGBClassifier":
    evaluate_model(xgb, "XGB")

elif selected == "RandomForestClassifier":
    evaluate_model(forest, "RF")

elif selected == "LogisticRegression":
    evaluate_model(log_reg, "LR")

elif selected == "SVC":
    evaluate_model(svc, "SVC")  
    
    
elif selected == "KNeighborsClassifier":
    evaluate_model(knn, "KNN")  
    
    
# =========================
#  Enter values of featture
# =========================
gender = st.slider("Enter your gender", 0, 1)
SeniorCitizen = st.slider("Enter your SeniorCitizen", 0, 1)
Partner = st.slider("Enter your Partner", 0, 1)
Dependents = st.slider("Enter your Dependents", 0, 1)
tenure = st.slider("Enter your tenure", 0.0, 120.0)
PhoneService = st.slider("Enter your PhoneService", 0, 1)
MultipleLines = st.slider("Enter your MultipleLines", 0, 2)
InternetService = st.slider("Enter your InternetService", 0, 2)

OnlineSecurity = st.slider("Enter your OnlineSecurity", 0, 2)
OnlineBackup = st.slider("Enter your OnlineBackup", 0, 2)
DeviceProtection = st.slider("Enter your DeviceProtection", 0, 2)
TechSupport = st.slider("Enter your TechSupport", 0, 2)
StreamingTV = st.slider("Enter your StreamingTV", 0, 2)
StreamingMovies = st.slider("Enter your StreamingMovies", 0, 2)
Contract = st.slider("Enter your Contract", 0, 2)
PaperlessBilling = st.slider("Enter your PaperlessBilling", 0, 1)

PaymentMethod = st.slider("Enter your PaymentMethod", 0, 3)
MonthlyCharges = st.slider("Enter your MonthlyCharges", 0.0, 1000000.00)
TotalCharges = st.slider("Enter your TotalCharges",  0.0, 1000000.00)
ChargePerMonthTenure = st.slider("Enter your ChargePerMonthTenure", 0.0, 100000.00)
NbServices = st.slider("Enter your NbServices", 0, 15)
LongTermContract = st.slider("Enter your LongTermContract", 0, 1)
FiberOptic = st.slider("Enter your FiberOptic", 0, 1)
AutoPayment = st.slider("Enter your AutoPayment", 0, 1)
tenureSegment = st.slider("Enter your tenureSegment", 0, 4)



# =========================
# Evaluate of prediction
# =========================
if(st.button('Predict Churn')):
    
    query = np.array([gender, SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity,                       OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
                MonthlyCharges, TotalCharges, ChargePerMonthTenure, NbServices, LongTermContract, FiberOptic, AutoPayment, tenureSegment])
    
    # On recupère un tableau et on le transforme en 1 ligne et 27 colonnes
    query = query.reshape(1,25)
    
    
     # --------------------------
    # XGBoost
    # --------------------------
    if selected == "XGBClassifier":

        prediction = xgb.predict(query)[0]
        probability = xgb.predict_proba(query)[0]
    
    
     # --------------------------
    # Random Forest
    # --------------------------
    elif selected == "RandomForestClassifier":

        prediction = forest.predict(query)[0]
        probability = forest.predict_proba(query)[0]

    
    # --------------------------
    # Logistic Regression
    # --------------------------
    elif selected == "LogisticRegression":

        prediction = log_reg.predict(query)[0]
        probability = log_reg.predict_proba(query)[0]

    # --------------------------
    # SVC
    # --------------------------
    elif selected == "SVC":

        prediction = svc.predict(query)[0]

        if hasattr(svc, "predict_proba"):
            probability = svc.predict_proba(query)[0]
        else:
            probability = None
            
    # --------------------------
    # KNeighborsClassifier
    # --------------------------
    elif selected == "KNeighborsClassifier":

        prediction = knn.predict(query)[0]

        if hasattr(knn, "predict_proba"):
            probability = knn.predict_proba(query)[0]
        else:
            probability = None       
            
            
 
    
      # --------------------------
    # Affichage résultat
    # --------------------------
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ Customer will churn")
    else:
        st.success("✅ Customer will not churn")



    # --------------------------
    # Probabilités
    # --------------------------
    if probability is not None:

        st.metric( "Probability of No Churn", f"{probability[0]:.2%}")

        st.metric("Probability of Churn", f"{probability[1]:.2%}")

    else:
        st.info(
            "Probability not available for this model."
        )


    

# streamlit run ./app0.py
