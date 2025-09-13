import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Main App Logic ---

def train_and_save_model():
    """Trains the model and saves it along with the scalers and encoder."""
    try:
        # Load the datas
        data = pd.read_csv("GREENSKILL ai data.csv")
    except FileNotFoundError:
        st.error("Error: The 'GREENSKILL ai data.csv' file was not found. Please ensure it's in the same directory.")
        return None, None, None, None

    # Preprocessing
    scaler = StandardScaler()
    data[['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']] = scaler.fit_transform(
        data[['LV ActivePower (kW)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)', 'Wind Direction (°)']]
    )

    encoder = LabelEncoder()
    data['Date/Time'] = encoder.fit_transform(data['Date/Time'])

    # Split data
    X = data[['Date/Time', 'Wind Direction (°)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)']]
    y = data['LV ActivePower (kW)']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    #Save components
    joblib.dump(model, 'Greenskill_AI_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(encoder, 'encoder.pkl')

    return model, scaler, encoder, mse, r2

@st.cache_data
def load_model_and_components():
    """Loads the pre-trained model and components."""
    try:
        model = joblib.load('Greenskill_AI_model.pkl')
        scaler = joblib.load('scaler.pkl')
        encoder = joblib.load('encoder.pkl')
        return model, scaler, encoder
    except FileNotFoundError:
        return None, None, None

# --- Streamlit UI ---

st.title("Wind Turbine Power Output Prediction 🌬")
st.markdown("This application predicts the *LV ActivePower (kW)* based on wind and atmospheric conditions.")
st.write("---")

# Check if model components exist, if not, train and save them
model, scaler, encoder = load_model_and_components()
if model is None:
    st.info("Training a new model. This may take a moment...")
    model, scaler, encoder, mse, r2 = train_and_save_model()
    if model:
        st.success("Model trained and saved successfully!")
        st.write(f"*Model Performance*")
        st.write(f"Mean Squared Error: *{mse:.2f}*")
        st.write(f"R-squared: *{r2:.2f}*")
        st.write("---")
    else:
        st.stop()

st.header("Predict Power Output")

# Input fields for user
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=25.0, value=7.0, step=0.1)
wind_direction = st.number_input("Wind Direction (°)", min_value=0.0, max_value=360.0, value=200.0, step=1.0)
theoretical_power = st.number_input("Theoretical Power Curve (KWh)", min_value=0.0, value=500.0, step=10.0)

# The 'Date/Time' column was encoded in the notebook, so we need to explain how to input it
st.info("The 'Date/Time' field is a numerical representation. For a new prediction, please enter a unique number. A value between 0 and 50529 is a good start as this represents the range of values in our training data.")
date_time = st.number_input("Date/Time (numeric)", min_value=0, value=0, step=1)

if st.button("Predict Power Output"):
    # Create a DataFrame for prediction
    new_data = pd.DataFrame([[date_time, wind_direction, wind_speed, theoretical_power]],
                            columns=['Date/Time', 'Wind Direction (°)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)'])

    # Scale the input features using the pre-trained scaler
    new_data[['Wind Direction (°)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)']] = scaler.transform(
        new_data[['Wind Direction (°)', 'Wind Speed (m/s)', 'Theoretical_Power_Curve (KWh)']]
    )

    # Make prediction
    scaled_prediction = model.predict(new_data)

    # Inverse transform the prediction to get the actual power output
    # This requires a DataFrame with the same columns as the original data
    # We create a dummy DataFrame to perform the inverse_transform
    dummy_df = pd.DataFrame([[0, scaled_prediction[0], 0, 0, 0]], columns=scaler.feature_names_in_)
    actual_prediction = scaler.inverse_transform(dummy_df)[0][1]


    st.success(f"*Predicted LV ActivePower: {actual_prediction:.2f} kW*")



