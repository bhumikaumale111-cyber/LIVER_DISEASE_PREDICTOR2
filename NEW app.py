import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the pre-trained model, scaler, and label encoder
@st.cache_resource
def load_resources():
    with open('best_liver_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)
    return model, scaler, label_encoder

model, scaler, label_encoder = load_resources()

st.title('Liver Disease Prediction App')
st.write('Enter the patient details below to predict the likelihood of liver disease.')

# Input fields for user data
age = st.slider('Age', 19, 77, 40)

sex_options = {'Female': 0, 'Male': 1}
sex_display = st.selectbox('Sex', list(sex_options.keys()))
sex = sex_options[sex_display]

albumin = st.number_input('Albumin (g/dL)', min_value=14.9, max_value=82.2, value=40.0, step=0.1)
alkaline_phosphatase = st.number_input('Alkaline Phosphatase (U/L)', min_value=11.3, max_value=416.6, value=70.0, step=0.1)
alanine_aminotransferase = st.number_input('Alanine Aminotransferase (U/L)', min_value=0.9, max_value=325.3, value=25.0, step=0.1)
aspartate_aminotransferase = st.number_input('Aspartate Aminotransferase (U/L)', min_value=10.6, max_value=324.0, value=30.0, step=0.1)
bilirubin = st.number_input('Bilirubin (mg/dL)', min_value=0.8, max_value=254.0, value=10.0, step=0.1)
cholinesterase = st.number_input('Cholinesterase (U/L)', min_value=1.42, max_value=16.41, value=8.0, step=0.01)
cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=1.43, max_value=9.67, value=5.0, step=0.01)
creatinina = st.number_input('Creatinine (mg/dL)', min_value=8.0, max_value=1079.1, value=80.0, step=0.1)
gamma_glutamyl_transferase = st.number_input('Gamma-Glutamyl Transferase (U/L)', min_value=4.5, max_value=650.9, value=40.0, step=0.1)
protein = st.number_input('Protein (g/dL)', min_value=44.8, max_value=90.0, value=70.0, step=0.1)

# Create a DataFrame from inputs
input_data = pd.DataFrame([{
    'age': age,
    'sex': sex,
    'albumin': albumin,
    'alkaline_phosphatase': alkaline_phosphatase,
    'alanine_aminotransferase': alanine_aminotransferase,
    'aspartate_aminotransferase': aspartate_aminotransferase,
    'bilirubin': bilirubin,
    'cholinesterase': cholinesterase,
    'cholesterol': cholesterol,
    'creatinina': creatinina,
    'gamma_glutamyl_transferase': gamma_glutamyl_transferase,
    'protein': protein
}])

# Order columns consistently with training data
# (Assuming X_train had columns in this order after preprocessing in the notebook)
ordered_columns = ['age', 'sex', 'albumin', 'alkaline_phosphatase',
                   'alanine_aminotransferase', 'aspartate_aminotransferase', 'bilirubin',
                   'cholinesterase', 'cholesterol', 'creatinina',
                   'gamma_glutamyl_transferase', 'protein']

input_data = input_data[ordered_columns]

# Preprocess the input data
# Scale numerical features
scaled_input_data = scaler.transform(input_data)

# Make prediction
if st.button('Predict'):
    prediction_proba = model.predict_proba(scaled_input_data)[0]
    prediction_class = model.predict(scaled_input_data)[0]

    # Decode the predicted class using the label encoder
    predicted_category = label_encoder.inverse_transform([prediction_class])[0]

    st.subheader('Prediction Results:')
    st.write(f"Predicted Liver Disease Category: **{predicted_category}**")
    st.write("Prediction Probabilities:")

    # Display probabilities for all classes
    class_names = label_encoder.inverse_transform(np.arange(len(label_encoder.classes_)))
    for i, prob in enumerate(prediction_proba):
        st.write(f"- {class_names[i]}: {prob:.2f}")

    st.markdown("--- Say something about the predicted category here. ---")
