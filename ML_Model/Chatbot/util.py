# import tensorflow as tf
# import tflearn
# import joblib
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import json
# import tflearn

import pickle

intents=json.load(open('intents.json'))
data=pickle.load(open('training_data','rb'))
words=data['words']
model=load('./model.tflearn')
def clean_sentence(sentence):
    sentence=nltk.word_tokenize(sentence)
    sentence_words=[stemmer.stem(w.lower()) for w in sentence]
    return sentence_words

def bag_of_words(sentences):
    matched=[]

    words_convert=[0]*len(words)
    sentence=clean_sentence(sentences)
    for w in sentence:
        for i,s in enumerate(words):
            if w==s:
                matched.append(w)
                words_convert[i]=1
    return words_convert


def classify(sentence):
    THRESOLD=0.30
    result=model.predict([bag_of_words(sentence)])[0]
    result=[[i,r] for i,r in enumerate(result) if r>THRESOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return result

def response(sentence):
    result=classify(sentence)
    return result
