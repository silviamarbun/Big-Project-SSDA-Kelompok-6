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
model = tf.keras.models.load_model('annmodel2new.h5')
scaler = pickle.load(open('scaler2.pkl', 'rb'))
    
# Function to make predictions
def predict(preprocessed_data,location):
    # Make predictions using the loaded model
    predictions = model.predict([[preprocessed_data[0][0],preprocessed_data[0][1],preprocessed_data[0][2],preprocessed_data[0][3],preprocessed_data[0][4],preprocessed_data[0][5],preprocessed_data[0][6],
                                  preprocessed_data[0][7], location]])
    return predictions

# Menampilkan foto dengan frame bulat
st.sidebar.image('kel6.png', use_column_width=True, caption='Kelompok 6 Kelas A')

# Menampilkan nama kelompok
st.sidebar.markdown('Keterangan Variabel Location:')
st.sidebar.markdown('- Alabama = 0')
st.sidebar.markdown('- Alaska = 1')
st.sidebar.markdown('- American Samoa = 2')
st.sidebar.markdown('- Arizona = 3')
st.sidebar.markdown('- Arkansas = 4')
st.sidebar.markdown('- Bureau of Prisons = 5')
st.sidebar.markdown('- California = 6')
st.sidebar.markdown('- Colorado = 7')
st.sidebar.markdown('- Connecticut = 8')
st.sidebar.markdown('- Delaware = 9')
st.sidebar.markdown('- Dept of Defense = 10')
st.sidebar.markdown('- District of Columbia = 11')
st.sidebar.markdown('- Federated States of Micronesia = 12')
st.sidebar.markdown('- Florida = 13')
st.sidebar.markdown('- Georgia = 14')
st.sidebar.markdown('- Guam = 15')
st.sidebar.markdown('- Hawaii = 16')
st.sidebar.markdown('- Idaho = 17')
st.sidebar.markdown('- Illinois = 18')
st.sidebar.markdown('- Indian Health Svc = 19')
st.sidebar.markdown('- Indiana = 20')
st.sidebar.markdown('- Iowa = 21')
st.sidebar.markdown('- Kansas = 22')
st.sidebar.markdown('- Kentucky = 23')
st.sidebar.markdown('- Long Term Care = 24')
st.sidebar.markdown('- Louisiana = 25')
st.sidebar.markdown('- Maine = 26')
st.sidebar.markdown('- Marshall Islands = 27')
st.sidebar.markdown('- Maryland = 28')
st.sidebar.markdown('- Massachusetts = 29')
st.sidebar.markdown('- Michigan = 30')
st.sidebar.markdown('- Minnesota = 31')
st.sidebar.markdown('- Mississippi = 32')
st.sidebar.markdown('- Missouri = 33')
st.sidebar.markdown('- Montana = 34')
st.sidebar.markdown('- Nebraska = 35')
st.sidebar.markdown('- Nevada = 36')
st.sidebar.markdown('- New Hampshire = 37')
st.sidebar.markdown('- New Jersey = 38')
st.sidebar.markdown('- New Mexico = 39')
st.sidebar.markdown('- New York State = 40')
st.sidebar.markdown('- North Carolina = 41')
st.sidebar.markdown('- North Dakota = 42')
st.sidebar.markdown('- Northern Mariana Islands = 43')
st.sidebar.markdown('- Ohio = 44')
st.sidebar.markdown('- Oklahoma = 45')
st.sidebar.markdown('- Oregon = 46')
st.sidebar.markdown('- Pensylvania = 47')
st.sidebar.markdown('- Puerto Rico = 48')
st.sidebar.markdown('- Replubic of Palau = 49')
st.sidebar.markdown('- Rhode Island = 50')
st.sidebar.markdown('- South Carolina = 51')
st.sidebar.markdown('- South Dakota = 52')
st.sidebar.markdown('- Tennessee = 53')
st.sidebar.markdown('- Texas = 54')
st.sidebar.markdown('- United States = 55')
st.sidebar.markdown('- Utah = 56')
st.sidebar.markdown('- Vermont = 57')
st.sidebar.markdown('- Veterans Health = 58')
st.sidebar.markdown('- Virgin Islands = 59')
st.sidebar.markdown('- Virginia = 60')
st.sidebar.markdown('- Washington = 61')
st.sidebar.markdown('- West Virginia = 62')
st.sidebar.markdown('- Wisconsin = 63')
st.sidebar.markdown('- Wyoming = 64')

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
    
    Location = st.number_input("location", value=0)
    
    
    # Preprocess the input data
    preprocessed_data = scaler.transform([[Total_vaccinations, Total_distributed, People_vaccinated, People_fully_vaccinated_per_hundred, Total_vaccinations_per_hundred, People_fully_vaccinated, People_vaccinated_per_hundred, Distributed_per_hundred, Daily_vaccinations_raw, Daily_vaccinations, Daily_vaccinations_per_million, Share_doses_used, Total_boosters, Total_boosters_per_hundred]])
    
    # Make predictions
    if st.button("Predict"):
        predictions = predict(preprocessed_data,Location)
        st.write("Predictions:", predictions)

# Run the app
if __name__ == '__main__':
    main()
