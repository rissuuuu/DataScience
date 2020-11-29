import pandas as pd
import nltk
import pickle
import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
stemmer=PorterStemmer()

df=pd.read_csv('df_clean.csv')
df.drop('Unnamed: 0',axis=1,inplace=True)

passive=pickle.load(open('passive.pk','rb'))
vectorizer=pickle.load(open('vectorizer.pk','rb'))
X_=pickle.load(open('data.pk','rb'))
def cleantext(txt):
    txt=txt.split() #spliting all text into single words like nltk.word_tokenize()
    txt=[stemmer.stem(i.lower()) for i in txt if i not in stopwords.words('english') ] #removing stop words
    txt=' '.join(txt)    #again joining the words to make a sentence
    txt=re.sub(r'[^A-Za-z0-9]',' ',txt)   #removing all punctuations and special symbols 
    txt=' '.join(txt.split()) #splitting texts because some of the sentences were having more than 3,4 times and rejoining
                            #them
    return txt #returning the final cleaned text.
    
    
    
def return_tfidf_data(text):
    return vectorizer.transform([text])    #It returns tfidf vectorized output for the text that we want to predict


def return_vectorized_data(text):
    text=cleantext(text)                #Firstly, we clean the text to be predicted and pass the text to above 
                                        #tfidf function.
    #print('Asked Question: ',text)
    #print('-----------------------------')
    text=return_tfidf_data(text)     #Finally we get the vector for particular sentence
    return text.toarray()

    
def predict_label(temp,model):              #This function is used to predict the vectorized data with the output label
    prediction=model.predict(temp)[0]  #This will return predicted label of index 0 so as to smooth process.
    return prediction
    
def return_response(temp_df):
    most_relevant=temp_df.copy()
    relevant_index=most_relevant.index[0] #This will try to give the index of data with highest score.
    response_list=df.iloc[relevant_index,:]['Answer'] #This will store list of reponses separated by comma
    
    try:   #This function is checking either a string can be splitted into other items or not using comma.
        responses=df.iloc[relevant_index,:]['Answer'].split(',')
        return(np.random.choice(responses))
        #This will give a random response corrosponding to particular question 
    except:
        return(responses) #This will run if it is a single string.
    
#For finding relavant questions that matches the given question using TFIDF 
def find_relevant(temp,items):
    prediction=predict_label(temp,passive) #Predicting label for test data
    index=df[df['Label']==prediction].index  #finding index of dataset with the predicted label so as to compare with

    cos_sim=[cosine_similarity(j.reshape(1,-1),temp.reshape(1,-1)) for j in X_[index]] #This is vectorized data for TFIDF
    cos_sim=np.array(cos_sim)
    cos_sim=cos_sim.flatten() #Creating a numpy array and flattening to a 1d list.
    temp_df=df.iloc[index,:].copy() #This returns the dataset with the predicted labels only.
    temp_df['cosine_sim']=cos_sim #Store the cosine similarity for particular input data.
    temp_df=temp_df.sort_values('cosine_sim',ascending=False) #Finding out top similarity scores.
    return(temp_df[['Question','cosine_sim']].head(items)),return_response(temp_df) 
    #this will return top questions that are matched with the test data with particular limit specified by items.
    #and also will try to give response to that question.
def response(text):

    temp=return_vectorized_data(text)
    returned_df,relevant_answer=find_relevant(temp,3)
    return returned_df,relevant_answer
        
print(response('How to check product warranty?'))   
    
    
    
    
    
    
