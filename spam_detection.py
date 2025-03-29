import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

nltk.download('punkt')
nltk.download('stopwords')

data = pd.read_csv('spam.csv', encoding='latin-1')
data = data[['v1', 'v2']] 
data.columns = ['label', 'message']
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
def preprocess_text(text):
    text = text.lower() 
    text = re.sub(r'[^a-zA-Z\s]', '', text) 
    tokens = word_tokenize(text) 
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words] 
    return ' '.join(tokens)


data['message'] = data['message'].apply(preprocess_text)


X = data['message']
y = data['label']

tfidf = TfidfVectorizer(max_features=3000)
X_tfidf = tfidf.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", report)

with open('results.txt', 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)