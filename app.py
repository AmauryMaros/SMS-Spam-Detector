import streamlit as st
import pandas as pd
import random
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

with open('models/spam_detector.pkl', 'rb') as model_file:
  model = pickle.load(model_file)
with open('models/vectorizer.pkl', 'rb') as model_file:
  vectorizer = pickle.load(model_file)

data = pd.read_csv('spam_clean.csv', encoding = 'Windows-1252')
X, Y = data['text'], data['label']

pages = st.sidebar.radio("SMS SPAM DETECTOR", options = ["Raw data", "Detector"])

if pages == "Raw data":

    st.title("Raw data")
    button = st.button("Generate")

    if button : 
        index = random.randint(0, len(X))
        st.text_area(label = "",value = X[index])
        st.subheader("Results")
        st.write(Y[index])

if pages == "Detector":
    st.title("Spam SMS Detector")
    text_input = [st.text_area("", placeholder="enter a text here")] # we put a list here because the model needs an iterable data to make his prediction
    button = st.button("Spam SMS detector")

    if button :
        if text_input == [""]:
            st.write("Enter a text")
        else :
            result = model.predict(vectorizer.transform(text_input))
            if result == 1:
               st.write("It's a spam !")
            else :
               st.write("It's not a spam.")