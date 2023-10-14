# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import shap
#import matplotlib.pyplot as plt
#import numpy as np
#import pyplot
#import matplotlib.pyplot as plt
#from PIL import Image

st.header("An AI Model for Predicting Postoperative In-Hospital Mortality in Geriatric Hip Fracture Patients")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Age = st.sidebar.slider("Age", 40, 80)
Numberofcommorbidity = st.sidebar.selectbox("Number of comorbidities", ("None", "Only one", "Two or above"))
ECOG = st.sidebar.selectbox("ECOG", ("1", "2", "3", "4"))
Surgicalsite3fj = st.sidebar.selectbox("Surgical site", ("Cervical and cervical thoracic", "Thoracic and thoracolumbar", "Lumbar and lumbosacral"))
Albumin = st.sidebar.slider("Preoperative albumin (g/L)", 30.0, 50.0)
TCHO = st.sidebar.slider("Total cholesterol (mmol/L)", 3.00, 6.00)
PT = st.sidebar.slider("Prothrombin time (seconds)", 8.0, 14.0)
Bilskyscore = st.sidebar.selectbox("Bilsky score", ("1", "2", "3"))
Preoperativeambulatory = st.sidebar.selectbox("Preoperative ambulatory status", ("Ability to walk", "Inability to walk"))

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Numberofcommorbidity, ECOG, Surgicalsite3fj, Bilskyscore, Preoperativeambulatory, Age, Albumin, TCHO, PT]],
                     columns=['Numberofcommorbidity', 'ECOG', 'Surgicalsite3fj', 'Bilskyscore', 'Preoperativeambulatory', 'Age', 'Albumin', 'TCHO', 'PT'])
    x = x.replace(["None", "Only one", "Two or above"], [0, 1, 2])
    x = x.replace(["1", "2", "3", "4"], [1, 2, 3, 4])
    x = x.replace(["Cervical and cervical thoracic", "Thoracic and thoracolumbar", "Lumbar and lumbosacral"], [1, 2, 3])
    x = x.replace(["1", "2", "3"], [1, 2, 3])
    x = x.replace(["Ability to walk", "Inability to walk"], [0, 1])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Probability of inability to walk after surgery: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.550:
        st.success(f"Risk group: low-risk group")
    else:
        st.success(f"Risk group: High-risk group")
    if prediction < 0.550:
        st.markdown(f"Management Measures for Low-risk Population: Geriatric hip fracture patients identified as low-risk for postoperative in-hospital mortality also require comprehensive postoperative management. Standard care protocols such as pain management, wound care, and mobilization should be followed consistently. Early mobilization protocols and physical therapy should be integrated into their care plan to facilitate optimal recovery and functional outcomes. Adequate nutrition support, psychological support, and regular follow-up appointments are important for healing and well-being. Patient education and engagement in the care plan ensure a clear understanding of postoperative instructions and adherence to medication regimens, contributing to a smooth transition to a home setting.")
    else:
        st.markdown(f"Management Measures for High-risk Population: For geriatric hip fracture patients identified as high-risk for postoperative in-hospital mortality, a proactive and individualized management approach is essential. Close monitoring of vital signs and regular pain assessment should be implemented to promptly identify any signs of deterioration. Early intervention by a multidisciplinary team, comprehensive medication management, and optimization are crucial to address potential complications and minimize adverse drug events. Specialized rehabilitation programs tailored to their specific needs and comorbidities can optimize functional recovery. Postoperative infection prevention strategies and regular communication among healthcare providers ensure seamless coordination and continuity of care.")

    st.subheader('Model explanation: contribution of each model predictor')
    star = pd.read_csv('X_train.csv', low_memory=False)
    y_train0 = pd.read_csv('y_train.csv', low_memory=False)
    data_train_X = star.loc[:, ["Age", "Numberofcommorbidity", "ECOG", "Surgicalsite3fj", "Albumin", "TCHO", "PT", "Bilskyscore", "Preoperativeambulatory"]]
    y_train = y_train0.Postoperativeambulatory
    model = rf_clf.fit(data_train_X, y_train)
    explainer = shap.Explainer(model)
    shap_value = explainer(x)
    #st.text(shap_value)

    shap.initjs()
    #image = shap.plots.force(shap_value)
    #image = shap.plots.bar(shap_value)

    shap.plots.waterfall(shap_value[0])
    st.pyplot(bbox_inches='tight')
    st.set_option('deprecation.showPyplotGlobalUse', False)



st.subheader('Model information')
st.markdown('The AI prediction model, developed using the eXGBoosting Machine (eXGBM) algorithm, demonstrated outstanding performance in predicting postoperative in-hospital mortality in geriatric hip fracture patients. It exhibited the highest scores in various evaluation metrics, including accuracy, precision, specificity, F1 score, Brier score, and log loss. With an AUC of 0.908, the model showcased excellent discrimination ability. Additionally, the model showed favorable calibration, indicating its accuracy in estimating risk levels. The comprehensive scoring system ranked the eXGBM model as the top-performing model, further validating its predictive capability. This AI model is freely accessible for research purposes, providing a valuable tool for enhancing clinical decision-making in managing geriatric hip fracture patients’ in-hospital mortality risk.')