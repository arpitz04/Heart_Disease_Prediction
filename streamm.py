import numpy as np
import pickle
import streamlit as st
import os

# Load the trained model using relative path (works locally and on Streamlit Cloud)
model_path = os.path.join(os.path.dirname(__file__), 'trained_model.sav')
loaded_model = pickle.load(open(model_path, 'rb'))

def heart_disease_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model.predict(input_data_reshaped)
    
    if prediction[0] == 0:
        return 'The person does not have heart disease'
    else:
        return 'The person has heart disease'

def main():
    st.title('Heart Disease Prediction Web App')
    
    age = st.text_input('Age')
    sex = st.text_input('Sex (1 = male; 0 = female)')
    cp = st.text_input('Chest Pain Type (0-3)')
    trtbps = st.text_input('Resting Blood Pressure (in mm Hg)')
    chol = st.text_input('Serum Cholesterol (in mg/dl)')
    fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)')
    restecg = st.text_input('Resting Electrocardiographic Results (0-2)')
    thalachh = st.text_input('Maximum Heart Rate Achieved')
    exng = st.text_input('Exercise Induced Angina (1 = yes; 0 = no)')
    oldpeak = st.text_input('ST Depression Induced by Exercise Relative to Rest')
    slp = st.text_input('Slope of the Peak Exercise ST Segment (0-2)')
    caa = st.text_input('Number of Major Vessels Colored by Fluoroscopy (0-3)')
    thall = st.text_input('Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)')

    diagnosis = ''
    
    if st.button('Heart Disease Test Result'):
        if '' in [age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]:
            st.error("Please fill in all the input fields.")
        else:
            try:
                data = [
                    int(age), int(sex), int(cp), int(trtbps), int(chol), int(fbs),
                    int(restecg), int(thalachh), int(exng), float(oldpeak),
                    int(slp), int(caa), int(thall)
                ]
                diagnosis = heart_disease_prediction(data)
                st.success(diagnosis)
            except ValueError:
                st.error("Please enter valid numeric values in all fields.")

if __name__ == '__main__':
    main()
