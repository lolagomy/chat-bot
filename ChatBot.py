import nltk
import random
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
# import spacy
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')

data = pd.read_csv('Samsung Dialog.txt', sep = ':', header = None)
cust = data.loc[data[0] == 'Customer']
sales = data.loc[data[0] == 'Sales Agent']

sales = sales[1].reset_index(drop = True)
cust = cust[1].reset_index(drop = True)

new_data = pd.DataFrame()
new_data['Question'] = cust
new_data['Answers'] = sales

def preprocess_text(text):

  sentences = nltk.sent_tokenize(text)


  preprocessed_sentences = []
  for sentence in sentences:
      tokens = [lemmatizer.lemmatize(word.lower())for word in nltk.word_tokenize(sentence) if word.isalnum()]


      preprocessed_sentence = ' '.join(tokens)
      preprocessed_sentences.append(preprocessed_sentence)
  
  return ' '.join(preprocessed_sentences)


new_data['Tokenized Questions'] = new_data['Question'].apply(preprocess_text)

xtrain = new_data['Tokenized Questions'].to_list()

tfidf_vectorizer = TfidfVectorizer()
corpus = tfidf_vectorizer.fit_transform(xtrain)



bot_greeting = ['Hello User! Do you have any questions?',
                'Hey  you! tell me what you want',
                'I am like a genie in a bottle. Hit me with your question.',
                'Hi! how can i help you today?']

bot_farewell= ['Thanks for your usage... bye.',
              'I hope you had a good experience.',
              'Have a great day and keep enjoying Samsung.']


human_greeting= ['hi','hello', 'good day', 'hey', 'hola' ]

human_exit = ['thank you', 'thanks', 'bye bye', 'goodbye', 'quit']


import random
random_greeting = random.choice(bot_greeting)
random_farewell = random.choice(bot_farewell) 

#------------------------STEAMLIT--------------------------

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ORGANIZATION CHATBOT</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by LOLA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)

st.header('Project Background Information', divider = True)
st.write('Organizations need efficient communication solutions to handle high demands. Traditional models often lead to delays and inefficiencies.Chatbots, powered by AI, automate responses, enhance communication, and reduce costs. This study explores their impact on efficiency and workload reduction')

st.markdown("<br>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html = True)
col1, col2 = st.columns(2)
col2.image('pngwing.com.png')

userPrompt = st.chat_input('Ask Your Question')
if userPrompt:
   col1.chat_message("ai").write(userPrompt)

   userPrompt = userPrompt.lower()
   if userPrompt in human_greeting:
      col1.chat_message("human").write(random_greeting)
   elif userPrompt in human_exit:
      col1.chat_message("human").write(random_farewell)
   else:
      proUserinput = preprocess_text(userPrompt)
      vect_user = tfidf_vectorizer.transform([proUserinput])
      similarity_scores = cosine_similarity(vect_user, corpus)
      most_similar_index = np.argmax(similarity_scores)
      col1.chat_message("human").write(new_data['Answers'].iloc[most_similar_index])
