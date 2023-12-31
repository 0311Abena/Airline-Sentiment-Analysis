from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from flask import Flask, request, render_template

# create a flask app
app = Flask(__name__)


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

@app.route("/", methods = ["GET", "POST"])
def predict():
    sentiment = ""

    if request.method == "POST":
        user_input = request.form["user_input"]
        # Preprocess the user input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction using the loaded model
        prediction = model.predict(np.array([preprocessed_input]))

        # # Calculate the average prediction value
        # average_prediction = prediction.mean()

        # Map predicted class index to sentiment label
        sentiment = ["Negative", "Neutral", "Positive"]
        predicted_class_index = np.argmax(prediction)
        sentiment = sentiment[predicted_class_index]

        # # Make a decision based on the average prediction
        # if average_prediction > 0.5:
        #     sentiment = "Negative"
        # else:
        #     sentiment = "Positive"
    return render_template("index.html", sentiment = sentiment)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)