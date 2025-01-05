import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

port_stem = PorterStemmer() 

def stemming(content):
    stemmingcontent = re.sub('[^a-zA-Z]',' ',content)
    stemmingcontent = stemmingcontent.lower()
    stemmingcontent = stemmingcontent.split()
    stemmingcontent = [port_stem.stem(word) for word in stemmingcontent if not word in stopwords.words('english')] 
    stemmingcontent = ' '.join(stemmingcontent)
    return stemmingcontent

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def predict(text):
    text = stemming(text)
    text = vectorizer.transform([text])
    prediction = model.predict(text)
    if (prediction[0] == 1):
        return "It's a Spam Mail"
    else:
        return "It's NOT A SPAM Mail"	
    
print(predict(""))