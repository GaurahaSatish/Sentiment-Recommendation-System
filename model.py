#!/usr/bin/env python
# coding: utf-8

# ## Final Sentiment and Recommendation Model with  with the code to deploy the end-to-end project using Flask and Heroku

# In[1]:


# Supress Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import pickle as pkl
from nltk.tokenize import word_tokenize
import xgboost


# ## import all pickle files
# #### xbg.pkl - sentiment analysis XGBoost model pickle file
# #### tfidf.pkl - tfidf vectorizer 
# #### transform.pkl - this pickle file after text cleaning
# #### user_recommendation.pkl - user based recommendation model

# In[3]:


xgb        = pkl.load(open('models/Xgbooster.pkl','rb'))
tfidf      = pkl.load(open('models/tfidf_vctr.pkl','rb'))
transform  = pkl.load(open('dataset/transformer.pkl','rb'))
user_recom = pkl.load(open('models/user_recommendation_sys.pkl','rb'))


# In[4]:

 # Sentiment model to Filtered out the top 5 products with the highest percentage of positive reviews
def sentiment(recom_prod):
    df = transform[transform.name.isin(recom_prod)]
    tfidf_vectorizer_vectors = tfidf.transform(df['text'])
    pred_data = xgb.predict(tfidf_vectorizer_vectors)
    predictions = [round(value) for value in pred_data]
    df['predicted'] = predictions
    output_data = df[df['predicted']==1][['name',    'brand','categories']].drop_duplicates()[:5].reset_index(drop=True)
    
    return output_data


# In[5]:

# First user recommendation to get top 20 product recommondation.
def recommendation(user_input):
    recom = user_recom.loc[user_input.lower()].sort_values(ascending=False)[0:20].index
    return recom


# In[10]:


# Take the user ID as input.
#user_input = input("Enter your user name")
#print(user_input)


# In[ ]:




