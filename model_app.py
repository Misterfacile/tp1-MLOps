import joblib
import streamlit as st
import pandas as pd
from pydantic import BaseModel, validator

def load_model():
    
    model = joblib.load('./regression.joblib')
    return model

def predict_model(input_data):
    model = load_model()
    return model.predict(input_data)
    

def load_dashboard():
    house_size = st.number_input(
        "Insert size of the house", value=None, placeholder="Type a number...")
    
    house_bedroom = st.number_input(
        "Insert number of bedroom in the house", value=None, placeholder="Type a number...")

    house_garden = st.number_input(
        "Insert 1 to want a garden, else 0", value=None, placeholder="Type 1 or 0", min_value=0, max_value=1)

    df_predict = pd.DataFrame({
        'size' : [house_size],
        'nb_rooms' : [house_bedroom],
        'garden' : [house_garden]
    })

    st.write("House Size", house_size)
    st.write("House Bedroom", house_bedroom)
    st.write("House Garden", house_garden)

    model = load_model()
    try : 
        predictions_price = model.predict(df_predict)
        st.write("Predictions Price", predictions_price)
    except Exception as e:
        st.write("Error : No data or data incorrect", e)    