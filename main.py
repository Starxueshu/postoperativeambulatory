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

st.header("Establishment and validation of an interactive artificial intelligence platform to predict postoperative ambulatory status in patients with metastatic spinal disease")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")
Age = st.sidebar.slider("Age", 40, 80)
#Numberofcommorbidity = st.sidebar.selectbox("Number of comorbidities", ("None", "Only one", "Two or above"))
ECOG = st.sidebar.selectbox("ECOG", ("1", "2", "3", "4"))
Surgicalsite3fj = st.sidebar.selectbox("Surgical site", ("Cervical and cervical thoracic", "Thoracic and thoracolumbar", "Lumbar and lumbosacral"))
Albumin = st.sidebar.slider("Preoperative albumin (g/L)", 30.0, 50.0)
TCHO = st.sidebar.slider("Total cholesterol (mmol/L)", 3.00, 6.00)
PT = st.sidebar.slider("Prothrombin time (seconds)", 8.0, 14.0)
Bilskyscore = st.sidebar.selectbox("Bilsky score", ("1", "2", "3"))
Preoperativeambulatory = st.sidebar.selectbox("Preoperative ambulatory status", ("Ability to walk", "Inability to walk"))

if st.button("Submit"):
    rf_clf = jl.load("ensemble_clf_final_round-softweight.pkl")
    x = pd.DataFrame([[ECOG, Surgicalsite3fj, Bilskyscore, Preoperativeambulatory, Age, Albumin, TCHO, PT]],
                     columns=['ECOG', 'Surgicalsite3fj', 'Bilskyscore', 'Preoperativeambulatory', 'Age', 'Albumin', 'TCHO', 'PT'])
    x = x.replace(["None", "Only one", "Two or above"], [0, 1, 2])
    x = x.replace(["1", "2", "3", "4"], [1, 2, 3, 4])
    x = x.replace(["Cervical and cervical thoracic", "Thoracic and thoracolumbar", "Lumbar and lumbosacral"], [1, 2, 3])
    x = x.replace(["1", "2", "3"], [1, 2, 3])
    x = x.replace(["Ability to walk", "Inability to walk"], [0, 1])


    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.success(f"Probability of inability to walk after surgery: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.381:
        st.success(f"Risk group: low-risk group")
    else:
        st.success(f"Risk group: High-risk group")

    st.subheader('Model explanation: contribution of each feature')
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

    if prediction < 0.381:
        st.success(f"Recommended Management Measures for the Low-risk Population: ")
        st.markdown(f"For the low-risk group, a more conservative approach may be taken. Resources can be allocated more efficiently, with interventions focused on minimizing complications. This group may require less intensive rehabilitation and follow-up, allowing resources to be redirected to patients who are in greater need.")
    else:
        st.success(f"Recommended Management Measures for the High-risk Population: ")
        st.markdown(f"For the high-risk group, it may be necessary to adopt aggressive interventions and closely monitor their progress. This may involve more extensive removal of metastatic tumors, early initiation of physical therapy, use of assistive devices, implementation of intensive rehabilitation programs, and proactive management of pain and complications. Additionally, frequent follow-up visits and regular assessments can help identify any deterioration in ambulatory status and allow for timely intervention.")

st.subheader('Model information')
st.markdown('To develop the human-machine interactive AI platform, we employed the extreme gradient boosting machine, which yielded an impressive area under the curve value of 0.911 (95%CI: 0.881-0.936). This freely accessible platform serves as a valuable tool for enhancing clinical decision-making in patients with metastatic spinal disease. However, it is important to note that clinical decision-making should not solely rely on the AI platform but should also incorporate the expertise of healthcare providers and take into account individual patient context.')
