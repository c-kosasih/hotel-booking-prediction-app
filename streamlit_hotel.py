import streamlit as st
import joblib
import pandas as pd

# Load model dan encoders
model = joblib.load("savedPickle/rf_model.pkl")
encoders = joblib.load("savedPickle/encoders.pkl")

# Fungsi untuk form input pengguna
def user_input_form():
    with st.form("hotel_booking_form"):
        booking_id = st.text_input("Booking ID")
        no_of_adults = st.number_input("Number of Adults", min_value=0, value=1)
        no_of_children = st.number_input("Number of Children", min_value=0, value=0)
        no_of_weekend_nights = st.number_input("Number of Weekend Nights", min_value=0, value=0)
        no_of_week_nights = st.number_input("Number of Week Nights", min_value=0, value=1)
        type_of_meal_plan = st.selectbox("Meal Plan", ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
        required_car_parking_space = st.selectbox("Required Car Parking Space", [0, 1])
        room_type_reserved = st.selectbox("Room Type Reserved", ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
        lead_time = st.number_input("Lead Time (Days)", min_value=0, value=1)
        arrival_year = st.number_input("Arrival Year", min_value=2022, value=2025)
        arrival_month = st.number_input("Arrival Month", min_value=1, max_value=12, value=5)
        arrival_date = st.number_input("Arrival Date", min_value=1, max_value=31, value=15)
        market_segment_type = st.selectbox("Market Segment Type", ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
        repeated_guest = st.selectbox("Repeated Guest", [0, 1])
        no_of_previous_cancellations = st.number_input("Number of Previous Cancellations", min_value=0, value=0)
        no_of_previous_bookings_not_canceled = st.number_input("Number of Previous Bookings Not Canceled", min_value=0, value=0)
        avg_price_per_room = st.number_input("Average Price per Room (EUR)", min_value=0.0, value=100.0)
        no_of_special_requests = st.number_input("Number of Special Requests", min_value=0, value=0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        return {
            "booking_id": booking_id,
            "no_of_adults": no_of_adults,
            "no_of_children": no_of_children,
            "no_of_weekend_nights": no_of_weekend_nights,
            "no_of_week_nights": no_of_week_nights,
            "type_of_meal_plan": type_of_meal_plan,
            "required_car_parking_space": required_car_parking_space,
            "room_type_reserved": room_type_reserved,
            "lead_time": lead_time,
            "arrival_year": arrival_year,
            "arrival_month": arrival_month,
            "arrival_date": arrival_date,
            "market_segment_type": market_segment_type,
            "repeated_guest": repeated_guest,
            "no_of_previous_cancellations": no_of_previous_cancellations,
            "no_of_previous_bookings_not_canceled": no_of_previous_bookings_not_canceled,
            "avg_price_per_room": avg_price_per_room,
            "no_of_special_requests": no_of_special_requests
        }
    else:
        return None

# Fungsi untuk prediksi
def predict_booking_status(input_dict):
    df = pd.DataFrame([input_dict])

    # Menggunakan OrdinalEncoder untuk fitur ordinal
    df[["type_of_meal_plan", "market_segment_type"]] = encoders.transform(
        df[["type_of_meal_plan", "market_segment_type"]]
    )

    # Menggunakan LabelEncoder untuk fitur kategorikal
    for col, le in encoders.items():
        df[col] = le.transform(df[col])

    # Prediksi status pemesanan
    pred = model.predict(df)[0]
    return pred

# Fungsi utama aplikasi Streamlit
def main():
    st.title("🏨 Hotel Booking Cancellation Prediction App")
    st.write("Input your booking data below to predict whether your booking will be canceled.")

    user_data = user_input_form()

    if user_data:
        result = predict_booking_status(user_data)
        if result == 1:
            st.success("YOUR BOOKING IS CANCELED")
        else: 
            st.error("YOUR BOOKING IS NOT CANCELED")

if __name__ == "__main__":
    main()
