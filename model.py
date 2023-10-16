#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv("FinalBalancedDataset.csv")


# In[3]:


data.info()


# In[4]:


data.head(5)


# In[5]:


data = data.drop("Unnamed: 0", axis=1)


# In[6]:


data.head(5)


# In[7]:


data['Toxicity'].value_counts()


# In[8]:


import nltk
nltk.download('punkt')
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from nltk import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords as nltk_stopwords
from nltk.corpus import wordnet


# ## Lemmatizer
# 1. Leaves
# 2. Leafs
# Leaf

# ## Text pre-processing

# In[9]:


wordnet_lemmatizer = WordNetLemmatizer()


# In[10]:


import re


# In[11]:


def prepare_text(text):
    def get_wordnet_pos(treebank_tag):
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    text = re.sub(r'[^a-zA-Z\']', ' ', text)
    text = text.split()
    text = ' '.join(text)
    text = word_tokenize(text)
    text = pos_tag(text)
    lemma = []
    for i in text: lemma.append(wordnet_lemmatizer.lemmatize(i[0], pos = get_wordnet_pos(i[1])))
    lemma = ' '.join(lemma)
    return lemma


# In[12]:


data['clean_tweets'] = data['tweet'].apply(lambda x: prepare_text(x))


# In[13]:


data.head(5)


# ## Tfidf for features

# In[14]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve


# In[15]:


corpus = data['clean_tweets'].values.astype('U')


# In[16]:


stopwords = set(nltk_stopwords.words('english'))


# In[17]:


count_tf_idf = TfidfVectorizer(stop_words = stopwords)
tf_idf = count_tf_idf.fit_transform(corpus)


# In[18]:


import pickle


# In[19]:


pickle.dump(count_tf_idf, open("tf_idf.pkt", "wb"))


# In[20]:


tf_idf_train, tf_idf_test, target_train, target_test = train_test_split(
    tf_idf, data['Toxicity'], test_size = 0.8, random_state= 42, shuffle=True
)


# ## Create a Binary Classification Model

# In[21]:


model_bayes = MultinomialNB()


# In[22]:


model_bayes = model_bayes.fit(tf_idf_train, target_train)


# In[23]:


y_pred_proba = model_bayes.predict_proba(tf_idf_test)[::, 1]


# In[24]:


y_pred_proba


# In[25]:


fpr, tpr, _ = roc_curve(target_test, y_pred_proba)


# In[26]:


final_roc_auc = roc_auc_score(target_test, y_pred_proba)


# In[27]:


final_roc_auc


# In[28]:


test_text = "I hate you moron"
test_tfidf = count_tf_idf.transform([test_text])
display(model_bayes.predict_proba(test_tfidf))
display(model_bayes.predict(test_tfidf))


# In[30]:


test_text = "you look ugly"
test_tfidf = count_tf_idf.transform([test_text])
display(model_bayes.predict_proba(test_tfidf))
display(model_bayes.predict(test_tfidf))


# In[31]:


test_text = "you are fat"
test_tfidf = count_tf_idf.transform([test_text])
display(model_bayes.predict_proba(test_tfidf))
display(model_bayes.predict(test_tfidf))

pickle.dump(model_bayes, open("model.pkt", "wb"))