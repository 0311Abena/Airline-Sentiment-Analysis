import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load the trained model
model = load_model("nlp_model.h5")

# load the saved tokenizer used during training
with open("tokenizer.pkl", "rb") as tk:
    tokenizer = pickle.load(tk)

# Define the function to preprocess the user text input
def preprocess_text(text):
    # Tokenize the text
    tokens = tokenizer.texts_to_sequences([text])

    # Pad the sequence to a fixed length
    padded_tokens = pad_sequences(tokens, maxlen = 100)
    return padded_tokens[0]

# create the title of the app
st.title(" Sentiment Analysis App for Virgin America Airline")

# Create a text input widget for the user input
user_input = st.text_area("Enter your review for sentiment analysis", " ")

# Create a button to trigger the sentiment analysis
if st.button("Predict Sentiment"):
    # preprocess the user input
    preprocessed_input = preprocess_text(user_input)

    # Make prediction using the loaded model
    prediction = model.predict(np.array([preprocessed_input]))
    st.write(prediction)
    # sentiment = "Negative" if prediction[0][0]> 0.5 else "Positive"

    # Map predicted class index to sentiment label
    sentiments = ["Negative", "Neutral", "Positive"]
    predicted_class_index = np.argmax(prediction)
    sentiment = sentiments[predicted_class_index]

    # Display the sentiment
    st.write(f"## Sentiment: {sentiment}")
