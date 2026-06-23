# =========================
# LOAD LIBRAIRIES
# =========================
import streamlit as st
from streamlit_option_menu import option_menu

import joblib
import pickle
import pandas as pd
import numpy as np

import shap #pip install shap

from sklearn.metrics import (accuracy_score, precision_score, roc_auc_score, confusion_matrix, roc_curve)

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Churn Dashboard", layout="wide")

st.title("📊 Churn Prediction Dashboard (Machine Learning)")


# =========================
# LOAD MODELS
# =========================
@st.cache_resource
def load_models():
    forest = joblib.load("RandomForestClassifier.pkl")

    knn = joblib.load("KNeighborsClassifier.pkl")

    with open("XGBClassifier.pkl", "rb") as f:
        xgb = pickle.load(f)

    with open("LogisticRegression.pkl", "rb") as f:
        log_reg = pickle.load(f)

    with open("SVC.pkl", "rb") as f:
        svc = pickle.load(f)

    return forest, knn, xgb, log_reg, svc

forest, knn, xgb, log_reg, svc = load_models()

models = {
    "XGBClassifier": xgb,
    "RandomForestClassifier": forest,
    "LogisticRegression": log_reg,
    "SVC": svc,
    "KNeighborsClassifier": knn
}


# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    y_test = pd.read_csv("y_test.csv", sep=";", index_col=0).values.ravel()
    x_test = pd.read_csv("x_test.csv", sep=";", index_col=0)
    data = pd.read_csv("dataviz.csv", sep=";", index_col=0) #Import dataset for Visualization
    return y_test, x_test, data

y_test, x_test, data = load_data()

scaler = joblib.load("scaler.pkl")
x_test_scaled = pd.DataFrame( scaler.transform(x_test), columns=x_test.columns, index=x_test.index)




# =========================
# SIDEBAR MENU
# =========================
with st.sidebar:
    selected = option_menu(
        "📌 Navigation",
        ["📊 Model Comparison", "📈 Single Model Analysis", "🔮 Prediction", "📉 Data Visualization"],
        icons=["bar-chart", "activity", "person", "pie-chart"],
        default_index=0
    )


# =========================
# FUNCTIONS
# =========================
def evaluate(model, name):
    y_pred = model.predict(x_test_scaled)

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred)
       
    }

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(x_test_scaled)[:, 1]
        metrics["AUC"] = roc_auc_score(y_test, y_proba)
    else:
        metrics["AUC"] = np.nan

    return metrics


def plot_confusion(model):
    y_pred = model.predict(x_test_scaled)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)


def plot_roc(model):
    if not hasattr(model, "predict_proba"):
        st.warning("ROC not available for this model")
        return

    y_proba = model.predict_proba(x_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    roc_auc = roc_auc_score(y_test, y_proba, average='macro')

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2%})')
    ax.plot([0, 1], [0, 1], "--")

    ax.set_title("Receiver Operating Characteristic")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    
    ax.legend(loc="lower right")

    st.pyplot(fig)
    

def shap_summary(model, X):

    st.subheader("🔍 SHAP Feature Importance (Global Explanation)")
    
    X_sample = X.sample(min(200, len(X)), random_state=42) # Pour éviter d'expliquer tout x_test_scaled, on utilise un échantillon
   
    if model.__class__.__name__ in ["RandomForestClassifier", "XGBClassifier"]: # Tree models (XGBoost, RandomForest) -> fast explainer
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        
        if isinstance(shap_values, list): # Au cas où SHAP retourne  une liste de tableaux
            shap_values = shap_values[1]
            
        
    elif model.__class__.__name__ == "LogisticRegression":
        
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

    else:
        explainer = shap.Explainer(model.predict_proba, X_sample)
        shap_values = explainer(X_sample)
        shap_values = shap_values[:, :, 1]

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display = 10)

    st.pyplot(plt.gcf())
    plt.clf()
    plt.close()

    

def shap_single_prediction(model, X, index=0):

    st.subheader("🎯 SHAP Explanation (Single Customer)")

    row = X.iloc[[index]]

    if model.__class__.__name__ in ["RandomForestClassifier", "XGBClassifier"]:

        explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(row, check_additivity=False)
        
        st.write("Shape :", np.array(shap_values).shape)

        if isinstance(shap_values, list): # Cas où shap_values renvoie une liste
            shap_values = shap_values[1]
            
        base_value = explainer.expected_value

        if isinstance(base_value, (list, np.ndarray)):
            base_value = base_value[1]

        plt.figure()

        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=row.iloc[0],
                feature_names=row.columns
            ),
            show=False
        )

        st.pyplot(plt.gcf())
        plt.clf()
        plt.close()
        

    else:

        if not hasattr(model, "predict_proba"):
            st.warning("SHAP explanation not available for this model")
            return
        
        background = X.sample(min(100, len(X)), random_state=42)

        explainer = shap.Explainer(model.predict_proba, background)
        shap_values = explainer(row)
     

        st.write("SHAP shape:", shap_values.values.shape)

        plt.figure()

        shap.plots.waterfall(
            shap_values[0, :, 1],
            show=False
        )

        st.pyplot(plt.gcf())
        plt.clf()
        plt.close()
    

# =========================
# PAGE 1 - COMPARISON
# =========================
if selected == "📊 Model Comparison":

    st.subheader("🏆 Model Performance Comparison")

    results = []

    for name, model in models.items():
        metrics = evaluate(model, name)
        metrics["Model"] = name
        results.append(metrics)

    df = pd.DataFrame(results).set_index("Model")
    df = df.sort_values("AUC", ascending=False) # Trier
    
    # Formatage en pourcentage
    df_display = df.copy()

    for col in ["Accuracy", "Precision", "AUC"]:
        df_display[col] = df_display[col].apply(
            lambda x: f"{x*100:.2f} %" if pd.notna(x) else "N/A"
        )

    st.dataframe(df_display, width=1000) # width="stretch" ou use_container_width=True



# =========================
# PAGE 2 - SINGLE MODEL
# =========================
elif selected == "📈 Single Model Analysis":

    model_name = st.selectbox("Choose a model", list(models.keys()))
    model = models[model_name]

    st.subheader(f"📌 Analysis: {model_name}")

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Metrics")
        metrics = evaluate(model, model_name)

        for k, v in metrics.items():
            st.metric(k, f"{v:.2%}" if not np.isnan(v) else "N/A")

    with col2:
        st.write("### Confusion Matrix")
        plot_confusion(model)

    st.write("### ROC Curve")
    plot_roc(model)
    
     # SHAP GLOBAL
    st.write("# 🧠 SHAP Global Explanation")
    shap_summary(model, x_test_scaled)
    
    
    # SHAP LOCAL 
    st.write("# 🎯 SHAP Local Explanation (Customer)")
    customer_id = st.slider("Select customer", min_value=0, max_value=len(x_test)-1, value=0)
    
    # Données du client
    customer = x_test_scaled.iloc[[customer_id]]
    
    st.write("### Customer Information")
    st.dataframe(customer) # Afficher les données du client sélectionné
    
    # Afficher également la prédiction du client sélectionné
    prediction = model.predict(customer)[0]

    if prediction == 1:
        st.error("⚠️ This customer is predicted to churn")
    else:
        st.success("✅ This customer is predicted to stay")
        
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(customer)[0, 1] # Si le modèle possède predict_proba
        st.metric("Churn Probability",f"{proba:.2%}")
        
  
    
    # SHAP
    #shap_single_prediction(model, x_test_scaled, index = customer_id)
    try:
        shap_single_prediction(model, x_test_scaled, index=customer_id)

    except Exception as e:
        st.error(str(e))


# =========================
# PAGE 3 - PREDICTION
# =========================
elif selected == "🔮 Prediction":

    st.subheader("🔮 Predict Customer Churn")

    model_name = st.selectbox("Select model", list(models.keys()))
    model = models[model_name]

    st.write("Enter customer features:")

    input_data = {}

    for col in x_test.columns:
        input_data[col] = st.number_input(col, value=float(x_test[col].mean()))

    if st.button("Predict"):
        input_df = pd.DataFrame([input_data])
        input_df_scaled = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        prediction = model.predict(input_df_scaled)[0]

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df_scaled)[0][1]
        else:
            proba = None

        if prediction == 1:
            st.error("⚠️ Customer WILL churn")
        else:
            st.success("✅ Customer will NOT churn")

        if proba is not None:
            st.info(f"Churn probability: {proba:.2%}")
            

# =========================
# PAGE 4 - Data Visualization
# =========================            
elif selected == "📉 Data Visualization":

    st.subheader("📊 Churn Data Exploration")
    
    col1, col2 = st.columns(2)
    
    
    # =====================
    # Pie Chart
    # =====================
    with col1:
        fig, ax = plt.subplots(figsize=(6,6))

        ax.pie(
            data["Churn"].value_counts(),
            labels=["No Churn", "Churn"],
            autopct="%1.2f%%",
            explode=[0.05, 0.1],
            colors=["green", "red"]
        )

        ax.set_title("Churn Distribution")

        st.pyplot(fig)
        
    # =====================
    # Correlation Matrix
    # =====================
    with col2:
        st.image(
            "correlation_matrix.png",
            caption="Correlation matrix",
            width=400

        ) # use_container_width=True
        
        
    # =====================
    # Contract vs Tenure
    # =====================
    df_plot = data.groupby(['Contract', 'Churn'])['tenure'].mean().reset_index()

    fig = px.bar(df_plot, x='Contract', y='tenure', color='Churn', barmode='group',
                 title='Tenure Mean Distribution by Contract and Churn',

        color_discrete_map={
            'No': '#00CC00',   # vert vif
            'Yes': '#FF0000'   # rouge vif 
        })

    fig.update_layout(
        xaxis_title="Contract Type", 
        yaxis_title="Average Tenure (Months)",
        legend_title="Churn",
        template="plotly_white",
        

        # Taille du titre principal
        title_font=dict(size=22),

        # Taille des titres d'axes
        xaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),

        yaxis=dict(
            title_font=dict(size=18),
            tickfont=dict(size=14)
        ),

        # Taille de la légende
        legend=dict(
            font=dict(size=14),
            title_font=dict(size=16)
        )
    
    )


    st.plotly_chart(fig, width="stretch")
    
    
    # =====================
    # TotalCharges vs Tenure
    # =====================
    
    fig = px.scatter(data, x="tenure", y ="TotalCharges" , color = "Churn", 
                     title='TotalCharges Distribution by Tenure and Churn',
                    color_discrete_map={'No': '#00CC00',   'Yes': '#FF0000' })

    fig.update_layout( 
    xaxis_title="Tenure (Months)", 
    yaxis_title="TotalCharges($)",
    legend_title="Churn",
    template="plotly_white")
    
    # Taille du titre principal
    title_font=dict(size=31),

    # Taille des titres d'axes
    xaxis=dict(
        title_font=dict(size=23),
        tickfont=dict(size=17)
    ),

    yaxis=dict(
        title_font=dict(size=23),
        tickfont=dict(size=17)
    ),

    # Taille de la légende
    legend=dict(
        font=dict(size=23),
        title_font=dict(size=17)
    )
    
    st.plotly_chart(fig, width="stretch")
    
    
    # =====================
    # TenureSegment
    # =====================
    fig, ax = plt.subplots(figsize=(10, 7))

    s = sns.countplot(data=data, x="tenureSegment", hue="Churn", palette="Set2")
    
    total = len(data)
    for p in s.patches:
        height = p.get_height()

        if height > 0:
            percentage = 100 * height / total

            s.annotate( f"{percentage:.1f}%", (p.get_x() + p.get_width()/2, height), ha="center", va="bottom", fontsize=10)

    ax.set_title("Distribution of TenureSegment by Churn", fontsize=17)
    ax.set_xlabel("Tenure Segment", fontsize=10)
    ax.set_ylabel("Number of Customers", fontsize=10)
    
    plt.tight_layout()
    
    st.pyplot(fig)
    
    
    # =====================
    # NbServices
    # =====================
    fig, ax = plt.subplots(figsize=(15, 10))

    s = sns.countplot(data=data, x="NbServices", hue="Churn", palette="tab10");

    total = len(data)
    for p in s.patches:
        height = p.get_height()

        if height > 0:
            percentage = 100 * height / total

            s.annotate(
                f'{percentage:.1f}%', (p.get_x() + p.get_width() / 2, height), ha='center', va='bottom')

    ax.set_title("Customers of Number Services by Churn", fontsize=20)
    ax.set_xlabel("Number of Services",  fontsize=15)
    ax.set_ylabel("Number of Customers",  fontsize=15)

    plt.tight_layout()

    st.pyplot(fig)
    
    
    
      
    # =====================
    # Data Preview 
    # =====================
    with st.expander("📋 Raw Data Preview", expanded=False):
        st.dataframe(
            data.head(100),
            width=1000
        )
            
            
# Anaconda Powershell Prompt
# streamlit run .\dashboard.py
