import pandas as pd
import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

class HotelBookingModel:
    def __init__(self, path, seed=42):
        self.path = path
        self.SEED = seed
        self.data = None
        self.encoders = {
            'type_of_meal_plan': {'Not Selected': 0, 'Meal Plan 1': 1, 'Meal Plan 2': 2, 'Meal Plan 3': 3},
            'room_type_reserved': {'Room_Type 1': 1, 'Room_Type 2': 2, 'Room_Type 3': 3,
                                   'Room_Type 4': 4, 'Room_Type 5': 5, 'Room_Type 6': 6, 'Room_Type 7': 7},
            'market_segment_type': {'Online': 0, 'Offline': 1, 'Corporate': 2,
                                    'Complementary': 3, 'Aviation': 4},
            'booking_status': {'Not_Canceled': 0, 'Canceled': 1} 
        }
        self.model = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    def load_and_clean_data(self):
        self.data = pd.read_csv(self.path)
        self.data = self.data.dropna().drop_duplicates()
        self.data.drop('Booking_ID', axis=1, inplace=True)
        self.data['total_no_of_nights'] = (
            self.data['no_of_weekend_nights'] + self.data['no_of_week_nights']
        )
        for col, mapping in self.encoders.items():
            self.data[col] = self.data[col].replace(mapping)

    def split_data(self):
        X = self.data.drop('booking_status', axis=1)
        y = self.data['booking_status']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed
        )

    def train_model(self):
        self.model = RandomForestClassifier(random_state=self.seed)
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        return self.model.predict(self.X_test)

    def save_model(self, model_path='rf_model.pkl', enc_path='encoders.pkl'):
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        with open(enc_path, 'wb') as f:
            pickle.dump(self.encoders, f)


model = HotelBookingModel('Dataset_B_hotel.csv', seed=42)
model.load_and_clean_data()
model.split_data()
model.train_model()
predictions = model.predict()
model.save_model()
