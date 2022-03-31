# importing library
import pandas as pd
import matplotlib.pyplot as pl
import string
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics


#from transformers import AutoModelForSequenceClassification, AutoTokenizer

#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = BiomimicryTagger.load_from_checkpoint(
    '/Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt',
    n_classes=len(LABEL_COLUMNS)
)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#dont use path
#path: /Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt