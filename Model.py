import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
import re

# Load data from CSV file
data = pd.read_csv('data.csv')

data['combined_text'] = data['combined_text'].astype(str)
data['combined_text'] = data['name'] + ' ' + data['email'] + ' ' + data['content']

# Separate features and labels
X = data['combined_text']
y = data['spam_label']

# Read custom words from text file and ignore duplicates
with open('custom_words.txt', 'r') as file:
    custom_words_set = set(file.read().splitlines())

custom_words = list(custom_words_set)
tfidf_vectorizer = TfidfVectorizer(max_features=1000, vocabulary=custom_words)
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Train Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_tfidf, y)

# Function to predict spam or not spam
def predict_spam(name, email, content):
    combined_input = name + ' ' + email + ' ' + content
    # Check if any custom words are present in the combined input text
    custom_words_present = any(word in combined_input.lower() for word in custom_words)
    if custom_words_present:
        X_input_tfidf = tfidf_vectorizer.transform([combined_input])
        predicted_label = classifier.predict(X_input_tfidf)
        return predicted_label[0]
    else:
        return 0

def is_valid_email(email):
    return bool(re.match(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", email))

# Streamlit app
st.set_page_config(layout="wide")

def get_email(email):
    if email and not is_valid_email(email):
        st.error("Invalid Email ID!")

#  navigation bar
nav_option = st.sidebar.radio("Menu", ["Home", "Model", "Dataset"])

if nav_option == "Home":
    st.title('Model Details')

    with open("README.md", "r") as readme_file:
        readme_content = readme_file.read()
        st.markdown(readme_content)

elif nav_option == "Model":
    st.title('Email Spam Classification')
    st.empty()

    name = st.text_input('Enter your Name:')
    email = st.text_input('Enter  Email-ID:')
    get_email(email)  
    content = st.text_area('Enter Email content:', height=200)

    # Predict spam or not spam
    if st.button('Predict'):
        if name == "" or email == "" or content == "":
            st.error("Please fill in all input fields.")
        else:
            # Generate random 10-digit mobile number
            mobile_number = ''.join(random.choices('0123456789', k=10))

            predicted_label = predict_spam(name, email, content)

            if predicted_label == 1:
                st.error("This email is classified as spam.")
            else:
                st.success("This email is not spam.")

            ct=name+' '+email+' '+content
            new_data = {'User': [len(data) + 1], 'name': [name], 'mobile': [mobile_number], 'email': [email], 'content': [content], 'spam_label': [predicted_label],'combined_text':[ct]}
            data = pd.concat([data, pd.DataFrame(new_data)], ignore_index=True)
            data.to_csv('data.csv', index=False)

elif nav_option == "Dataset":
    st.subheader("Prediction Dataset")
    st.write(data)
