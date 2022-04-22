from __future__ import print_function

#!pip install -r requirements.txt
import pandas as pd
import numpy as np
import seaborn as sns

#for text pre-processing
import re, string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

#for model-building
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix, roc_curve, auc

#bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# importing the LIME libraries
import lime
import sklearn.ensemble
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer

#read the data
df_train=pd.read_csv(r"C:\Users\dcolu\OneDrive\Documents\nlp-getting-started/train.csv")
df_test=pd.read_csv(r"C:\Users\dcolu\OneDrive\Documents\nlp-getting-started/test.csv")

#print(df_train)

#convert to lowercase, strip and remove punctuations
def preprocess(text):
    text = text.lower() 
    text=text.strip()  
    text=re.compile('<.*?>').sub('', text) 
    text = re.compile('[%s]' % re.escape(string.punctuation)).sub(' ', text)  
    text = re.sub('\s+', ' ', text)  
    text = re.sub(r'\[[0-9]*\]',' ',text) 
    text=re.sub(r'[^\w\s]', '', str(text).lower().strip())
    text = re.sub(r'\d',' ',text) 
    text = re.sub(r'\s+',' ',text)
    text=' '.join([i for i in text.split() if i not in stopwords.words('english')])
    return text

#lemmatization: doing things properly with the use of a vocabulary and morphological analysis of words, normally aiming to remove inflectional endings only and to return the base or dictionary form of a word

#initialize the lemmatizer
wl = WordNetLemmatizer()
#function to map NTLK position tags
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

#tokeinze the sentence
def lemmatizer(string):
    word_pos_tags = nltk.pos_tag(word_tokenize(string)) # Get position tags
    a=[wl.lemmatize(tag[0], get_wordnet_pos(tag[1])) for idx, tag in enumerate(word_pos_tags)] # Map the position tag and lemmatize the word/token
    return " ".join(a)
def finalpreprocess(text):
    return lemmatizer(preprocess(text))
df_train['cleaned_text'] = df_train['text'].apply(lambda x: finalpreprocess(x))

#SPLITTING THE TRAINING DATASET INTO TRAINING AND VALIDATION
#tf=TfidfVectorizer(strip_accents = 'ascii', stop_words='english')

X_train, X_val, y_train, y_val = train_test_split(df_train["cleaned_text"],df_train["target"],test_size=0.2, shuffle=True)
#X_train, X_val, y_train, y_val = train_test_split(df_train["cleaned_text"],df_train["target"])

#print(df_train)
#print(df_train['target'])

#TF-IDF
# Convert x_train to vector
tfidf_vectorizer = TfidfVectorizer(use_idf=True)
X_train_vectors_tfidf = tfidf_vectorizer.fit_transform(X_train) 
X_val_vectors_tfidf = tfidf_vectorizer.transform(X_val)
#model
model=RandomForestClassifier(n_estimators = 100, random_state = 10)
model.fit(X_train_vectors_tfidf, y_train) 
#Predict y value for test dataset
y_pred = model.predict(X_val_vectors_tfidf)
y_prob = model.predict_proba(X_val_vectors_tfidf)[:,1]
#print(classification_report(y_val,y_pred))
#print('Confusion Matrix:',confusion_matrix(y_val, y_pred))

fpr, tpr, thresholds = roc_curve(y_val, y_prob)
roc_auc = auc(fpr, tpr)
#print('AUC:', roc_auc)

#LIME installation
# converting the vectoriser and model into a pipeline
# this is necessary as LIME takes a model pipeline as an input
c = make_pipeline(tfidf_vectorizer, model)

# saving a list of strings version of the X_test object
ls_X_test= list(X_val)
# saving the class names in a dictionary to increase interpretability
class_names = {0: 'non-disaster', 1:'disaster'}

# create the LIME explainer
# add the class names for interpretability
LIME_explainer = LimeTextExplainer(class_names=class_names)

# choose a random single prediction
idx = 15
# explain the chosen prediction 
# use the probability results of the logistic regression
# can also add num_features parameter to reduce the number of features explained
LIME_exp = LIME_explainer.explain_instance(ls_X_test[idx], c.predict_proba)
# print results
print('Document id: %d' % idx)
print('Tweet: ', ls_X_test[idx])
print('Probability disaster =', c.predict_proba([ls_X_test[idx]]).round(3)[0,1])
print('True class: %s' % class_names.get(list(y_val)[idx]))

# print class names to show what classes the viz refers to
#print("1 = disaster class, 0 = non-disaster class")
# show the explainability results with highlighted text
#LIME_exp.show_in_notebook(text=True)