import streamlit as st
import streamlit.components.v1 as stc

from sklearn import model_selection
import streamlit as st
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import tensorflow as tf

# Load the ANN model from file
model = tf.keras.models.load_model('app/ann.h5')
scaler = pickle.load(open('app/scaler.pkl', 'rb'))
    
# Function to make predictions
def predict(preprocessed_data,age):
    # Make predictions using the loaded model
    predictions = model.predict([[preprocessed_data[0][0],preprocessed_data[0][1],preprocessed_data[0][2],preprocessed_data[0][3],preprocessed_data[0][4],preprocessed_data[0][5],preprocessed_data[0][6],preprocessed_data[0][7], age]])
    return predictions

# Main function
def main():
    # Set the title and description of your web app
    st.title("ANN Prediction App")
    st.write("Enter the input data to get predictions.")

    # Get user input
    sinovac_1st_dose = st.number_input("Sinovac 1st dose", value=0)
    sinovac_2nd_dose = st.number_input("Sinovac 2nd dose", value=0)
    sinovac_3rd_dose = st.number_input("Sinovac 3rd dose", value=0)
    sinovac_4rd_dose = st.number_input('Sinovac 4th dose', value=0)
    biontech_1st_dose = st.number_input("BioNTech 1st dose", value=0)
    biontech_2nd_dose = st.number_input("BioNTech 2nd dose", value=0)
    biontech_3rd_dose = st.number_input("BioNTech 3rd dose", value=0)
    biontech_4rd_dose = st.number_input("BioNTech 4rd dose", value=0)
    age = st.number_input("Age Group", value=0)
    
    
    # Preprocess the input data
    preprocessed_data = scaler.transform([[sinovac_1st_dose,sinovac_2nd_dose,sinovac_3rd_dose,sinovac_4rd_dose,biontech_1st_dose,biontech_2nd_dose,biontech_3rd_dose,biontech_4rd_dose]])
    
    # Make predictions
    if st.button("Predict"):
        predictions = predict(preprocessed_data,age)
        st.write("Predictions:", predictions)

# Run the app
if __name__ == '__main__':
    main()
