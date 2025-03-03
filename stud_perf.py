import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler,LabelEncoder


def load_model():
    with open("student_lr_final_model.pkl","rb") as file:
        model,le,scaler=pickle.load(file)
        print(model,le,scaler)
    return model,le,scaler

def preprocessing_input_data(data,le,scaler):
    data["Extracurricular Activities"]=le.transform([data["Extracurricular Activities"]])[0]
    df=pd.DataFrame([data])
    df_transformed=scaler.transform(df)
    print(df_transformed)
    return df_transformed

def predict_data(data):
    model,le,scaler=load_model()
    processed_data=preprocessing_input_data(data,le,scaler)
    print(processed_data)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter the data to predict the student performance:")

    # Hours Studied	Previous Scores	Extracurricular Activities	Sleep Hours	Sample Question Papers Practiced
    hours_studied = st.number_input("Hours Studied", min_value=0,max_value=24, value=0)
    previous_scores = st.number_input("Previous Scores", min_value=0,max_value=100, value=0)
    extra_curricular_activities = st.selectbox("Extracurricular Activities",["Yes","No"])
    hours_sleep = st.number_input("Sleep Hours", min_value=0,max_value=24-hours_studied, value=0)
    sample_question_papers_practiced = st.number_input("Sample Question Papers Practiced", min_value=0,max_value=10, value=0)

    if st.button("Predict your Score"):
        user_data ={
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Extracurricular Activities": extra_curricular_activities,
            "Sleep Hours": hours_sleep,
            "Sample Question Papers Practiced": sample_question_papers_practiced
        }

        prediction=predict_data(user_data)
        st.success(f"Your expected score:{prediction}")
 
if __name__ == "__main__":
    main()
    
