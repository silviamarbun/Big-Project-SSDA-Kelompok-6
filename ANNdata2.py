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
model = tf.keras.models.load_model('modelanndata2.h5')
scaler = pickle.load(open('scalerdata2.pkl', 'rb'))
    
# Function to make predictions
def predict(preprocessed_data,location):
    # Make predictions using the loaded model
    predictions = model.predict([[preprocessed_data[0][0],preprocessed_data[0][1],preprocessed_data[0][2],preprocessed_data[0][3],preprocessed_data[0][4],preprocessed_data[0][5],preprocessed_data[0][6],
                                  preprocessed_data[0][7], location]])
    return predictions

# Main function
def main():
    # Set the title and description of your web app
    st.markdown('<h1 style="font-size: 32px;">Artificial Neural Network (ANN) Prediction APP</h1>', unsafe_allow_html=True)

    st.write("""
        <h2>Deployment Data 2 Kelompok 6</h2>
        <h3>With US State Covid-19 Vaccination Dataset</h3>
    """, unsafe_allow_html=True)
    st.image("us_vacpict.png", caption="Pict Credit to Bloomberg.com ", use_column_width=True)

    st.write("Enter the input data to get predictions.")


    # Get user input
    col1, col2 = st.columns(2)
    
    with col1:
        Total_vaccinations = st.number_input("Total_vaccination", value=0.0)
        Total_distributed = st.number_input("total_distributed", value=0.0)
        People_vaccinated = st.number_input("people_vaccinated", value=0.0)
        People_fully_vaccinated_per_hundred = st.number_input("people_fully_vaccinated_per_hundred", value=0.0)
        Total_vaccinations_per_hundred = st.number_input("total_vaccinations_per_hundred", value=0.0)
        People_fully_vaccinated = st.number_input("people_fully_vaccinated", value=0.0)
        People_vaccinated_per_hundred = st.number_input('people_vaccinated_per_hundred', value=0.0)
        
    with col2:
        Distributed_per_hundred = st.number_input('distributed_per_hundred', value=0.0)
        Daily_vaccinations_raw = st.number_input('daily_vaccinations_raw', value=0.0)
        Daily_vaccinations = st.number_input('daily_vaccinations', value=0.0)
        Daily_vaccinations_per_million = st.number_input('daily_vaccinations_per_million', value=0.0)
        Share_doses_used = st.number_input("share_doses_used", value=0.0)
        Total_boosters = st.number_input("total_boosters", value=0.0)
        Total_boosters_per_hundred = st.number_input("total_boosters_per_hundred", value=0.0)
    
    Location = st.number_input("location", value=0.0)
    
    
    # Preprocess the input data
    preprocessed_data = scaler.transform([[Total_vaccinations, Total_distributed, People_vaccinated, People_fully_vaccinated_per_hundred, Total_vaccinations_per_hundred, People_fully_vaccinated, People_vaccinated_per_hundred, Distributed_per_hundred, Daily_vaccinations_raw, Daily_vaccinations, Daily_vaccinations_per_million, Share_doses_used, Total_boosters, Total_boosters_per_hundred]])
    
    # Make predictions
    if st.button("Predict"):
        predictions = predict(preprocessed_data,Location)
        st.write("Predictions:", predictions)

# Run the app
if __name__ == '__main__':
    main()
