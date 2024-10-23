import joblib
import streamlit as st
import pandas as pd
from pydantic import BaseModel, validator
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

def load_model():
    
    model = joblib.load('./regression.joblib')
    return model

def load_bert_model():
    model = DistilBertForSequenceClassification.from_pretrained('./bert_regression_model')
    tokenizer = DistilBertTokenizer.from_pretrained('./bert_regression_model')
    return model, tokenizer

def predict_model(input_data):
    model = load_model()
    return model.predict(input_data)

def predict_bert_model(size, rooms, garden):
    model, tokenizer = load_bert_model()
    text = f"This house is {size} square meters, has {rooms} bedrooms, and {'has' if garden == 1 else 'does not have'} a garden."
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_price = outputs.logits.item()
    return predicted_price

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
        
def load_dashboard_bert_model():
    
    model, tokenizer = load_bert_model()
    
    house_size = st.number_input(
        "Insert size of the house", value=None, placeholder="Type a number...")
    
    house_bedroom = st.number_input(
        "Insert number of bedroom in the house", value=None, placeholder="Type a number...")

    house_garden = st.number_input(
        "Insert 1 to want a garden, else 0", value=None, placeholder="Type 1 or 0", min_value=0, max_value=1)
    
    text = f"This house is {house_size} square meters, has {house_bedroom} bedrooms, and {'has' if house_garden == 1 else 'does not have'} a garden."
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    
    try : 
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_price = outputs.logits.item()
            st.write("Predictions Price", predicted_price)
    except Exception as e:
        st.write("Error : No data or data incorrect", e)

def main():
    st.title("House Price Prediction Dashboard")
    
    # You can toggle between different models using a radio button
    model_choice = st.radio("Choose model to use:", ("Classic Regression Model", "BERT Model"))
    
    if model_choice == "Classic Regression Model":
        load_dashboard()
    elif model_choice == "BERT Model":
        load_dashboard_bert_model()

if __name__ == "__main__":
    main()