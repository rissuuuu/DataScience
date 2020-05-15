
import tflearn
import tensorflow as tf
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import json
# import tflearn
import random
import pickle

intents=json.load(open('intents_chat.json'))
data=pickle.load(open('training_data','rb'))
words=data['words']

tf.reset_default_graph()

net=tflearn.input_data(shape=[None,len(data['train_x'][0])])
net=tflearn.fully_connected(net,10)
net=tflearn.fully_connected(net,10)
net=tflearn.fully_connected(net,len(data['train_y'][0]),activation='softmax')
net=tflearn.regression(net)
# defining model and setting up tensorboard
model=tflearn.DNN(net,tensorboard_dir='tflearn_logs')
# load our saved model
model.load('./model.tflearn')

print(model)
def clean_sentence(sentence):
    sentence=nltk.word_tokenize(sentence)
    sentence_words=[stemmer.stem(w.lower()) for w in sentence]
    print(sentence_words)
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
    return result[0]

def response(sentence):
    matched=" "
    result=classify(sentence)
    for i,j in enumerate(data['classes']):
        if i==result[0]:
            matched=j
    if result:
        while(result):
            for i in intents['intents']:
                    if i['tag']==matched:
                        return(random.choice(i['responses']))
            result.pop(0)
