#import dependencies
import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import pytorch_lightning as pl

from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
#BERT_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
import torch.nn as nn



#from transformer import LABEL_COLUMNS, BiomimicryTagger ## from brandon's code

#from transformers import BiomimicryTagger ##this is from other site, from huggingface library

tokenizer = AutoTokenizer.load_from_checkpoint("scibert-top29-aws-epoch46-122221.ckpt")

LABEL_COLUMNS = ['label']
model = AutoModelForSequenceClassification.load_from_checkpoint("scibert-top29-aws-epoch46-122221.ckpt", n_classes=len(LABEL_COLUMNS))

class_names = ['distribute liquids','neither','respond to signals']

def predictor(texts):
  outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
  probas = F.softmax(outputs.logits).detach().numpy()
  return probas
# define softmax function
# def softmax(x):
#     e_x = np.exp(x - np.max(x))
#     return e_x / e_x.sum(axis=0)

# define prediction function
# def predict_probs(texts):
#     predictions = model.predict(texts)
#     x = np.array(list(predictions)[1])
#     return np.apply_along_axis(softmax, 1, x)

explainer = LimeTextExplainer(class_names=class_names)

text = '''
the pupils and optical systems of gecko eyes the nocturnal helmet gecko tarentola chazaliae discriminates colors in dim moonlight when humans are color blind the sensitivity of the helmet gecko eye has been calculated to be  times higher than human cone vision at the color vision threshold the optics and the large cones of the gecko are important reasons why they can use color vision at low light intensities using photorefractometry and an adapted laboratory hartmann shack wavefront sensor of high resolution we also show that the optical system of the helmet gecko has distinct concentric zones of different refractive powers a so called multifocal optical system the intraspecific variation is large but in most of the individuals studied the zones differed by  diopters this is of the same magnitude as needed to focus light of the wavelength range to which gecko photoreceptors are most sensitive we compare the optical system of the helmet gecko to that of the diurnal day gecko phelsuma madagascariensis grandis the optical system of the day gecko shows no signs of distinct concentric zones and is thereby monofocal
'''

str_to_predict = 'text'
exp = explainer.explain_instance(str_to_predict, predictor, num_features=20, num_samples=2000)
exp.show_in_notebook(text=str_to_predict)
# explain instance with LIME

#exp = explainer.explain_instance(text, predict_probs, num_features=6)
















#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
#model = BiomimicryTagger.load_from_checkpoint(
#    '/Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt',
#   n_classes=len(LABEL_COLUMNS)
#)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#dont use path
#path: /Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt