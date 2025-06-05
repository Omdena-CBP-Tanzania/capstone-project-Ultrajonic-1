import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load data and model
df = pd.read_csv('data/cleaned/cleaned_climate_data.csv')
model = joblib.load('models/climate_rf_model.pkl')

# Title and introduction
st.title("Tanzania Climate Analysis and Prediction")
st.markdown("""
This app provides insights into Tanzania's climate data, including temperature and rainfall trends, and uses a machine learning model to predict average temperatures.
""")

if not df.empty:
    st.header('Exploratory Data Analysis')

    # Display basic data info
    st.write('### Raw Climate Data')
    st.dataframe(df.head())


# Visualizations
st.subheader("Temperature Trends")
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Average_Temperature_C'], label='Average Temperature (°C)')
ax.set_xlabel('Year')
ax.set_ylabel('Average Temperature (°C)')
ax.legend()
st.pyplot(fig)

st.subheader("Rainfall Trends")
fig, ax = plt.subplots()
ax.plot(df['Year'], df['Total_Rainfall_mm'], label='Total Rainfall (mm)', color='green')
ax.set_xlabel('Year')
ax.set_ylabel('Total Rainfall (mm)')
ax.legend()
st.pyplot(fig)

# Input fields for prediction future Temperature
st.subheader("Predict Average Temperature")
year = st.number_input('Enter Year', min_value=2000, max_value=2100, value=2000)
month = st.number_input('Enter Month', min_value=1, max_value=12, value=1)

if st.button('Predict Temperature'):
    if model is not None:
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month]
        })

        # Make the prediction
        predicted_temp = model.predict(input_data)[0]
        st.success(f'Predicted Temperature for {month}/{year}: {predicted_temp:.2f} °C')