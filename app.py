import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

port_stem = PorterStemmer()


def stemming(content):
    stemmingcontent = re.sub('[^a-zA-Z]', ' ', content)
    stemmingcontent = stemmingcontent.lower()
    stemmingcontent = stemmingcontent.split()
    stemmingcontent = [port_stem.stem(word) for word in stemmingcontent if word not in stopwords.words('english')]
    stemmingcontent = ' '.join(stemmingcontent)
    return stemmingcontent

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def predict(text):
    text = stemming(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)
    if prediction[0] == 1:
        return "It's a Spam Mail"
    else:
        return "It's NOT A SPAM Mail"


st.title("Spam Mail Detector")
st.write("Enter the text of an email to check if it is spam or not.")

user_input = st.text_area("Enter Email Text", "")

if st.button("Predict"):
    if user_input.strip():
        result = predict(user_input)
        st.success(result)
    else:
        st.warning("Please enter some text to analyze.")
