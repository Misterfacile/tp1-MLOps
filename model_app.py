import joblib
import streamlit as st
import pandas as pd
import torch
import train_model

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.preprocessing import MinMaxScaler

def load_model():
    model = joblib.load('./regression.joblib')
    return model

def load_bert_model():
    model = train_model.DistilBertForRegression()
    model.load_state_dict(torch.load('./bert_regression_model/model.pth'))
    model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('./bert_regression_model/tokenizer/')
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
    
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Inverse the Scaler to obtain the real result
    df = pd.read_csv("./houses.csv")
    y = df['price'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaler.fit(y)
    
    try : 
        with torch.no_grad():
            predicted_price = model(input_ids, attention_mask)
        predicted_price = predicted_price.item()
        predicted_price_actual = scaler.inverse_transform([[predicted_price]])[0][0]
        st.write("Predictions Price", predicted_price_actual)
    except Exception as e:
        st.write("Error : No data or data incorrect", e)

def main():
    st.title("House Price Prediction Dashboard")
    model_choice = st.radio("Choose model to use:", ("Classic Regression Model", "BERT Model"))
    
    if model_choice == "Classic Regression Model":
        load_dashboard()
    elif model_choice == "BERT Model":
        load_dashboard_bert_model()

if __name__ == "__main__":
    main()