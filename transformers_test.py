import pandas as pd
import numpy as np
from biomimicry import BiomimicryDataModule, BiomimicryDataset, BiomimicryTagger

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00",
    "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
rcParams['figure.figsize'] = 12, 8

pl.seed_everything(RANDOM_SEED)


# Read in the cleaned data
data_prefix = 'https://raw.githubusercontent.com/nasa-petal/search-engine/main/data/'
df = pd.read_csv(data_prefix + 'cleaned_leaves.csv')

# Drop all non-feature columns
non_feat = ['y', 'text']
df.drop(non_feat, axis=1, inplace=True)

# Drop all labels with < 20 papers
LABEL_COLUMNS = df.columns.tolist()[:-1]
df.drop([col for col, val in df[LABEL_COLUMNS].sum().iteritems()
        if val < 25], axis=1, inplace=True)
dropcols = ['protect_from_animals', 'coordinate_by_self-organization', 'maintain_biodiversity',
    'compete_within/between_species', 'cooperate_within/between_species']
df.drop(dropcols, axis=1, inplace=True)

# df = df[df.columns[df[LABEL_COLUMNS].sum()>3]]
print(df.shape)
df.head()


LABEL_COLUMNS = df.columns.tolist()[:-1]

biom = df[df[LABEL_COLUMNS].sum(axis=1) > 0]
nonbiom = df[df[LABEL_COLUMNS].sum(axis=1) == 0]

# remove all non-biomimicry papers from dataset.
df = biom





X_train, y_train, X_test, y_test = iterative_train_test_split_dataframe(
    X=df[['text_raw']], y=df[LABEL_COLUMNS], test_size=0.15)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_test, y_test], axis=1)
print(train_df.shape, val_df.shape)

'''
X_train_val, y_train_val, X_test, y_test = iterative_train_test_split_dataframe(
    X=df[['text_raw']], y=df[LABEL_COLUMNS], test_size = 0.1)
test_df = pd.concat([X_test, y_test], axis=1)
X_train, y_train, X_val, y_val = iterative_train_test_split_dataframe(
    X=X_train_val, y=y_train_val, test_size = 0.13)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
# train_df, val_df = train_test_split(df, test_size=0.1)
train_df.shape, val_df.shape, test_df.shape
'''

BERT_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
MAX_TOKEN_COUNT = 512

N_EPOCHS = 100
BATCH_SIZE = 12  # smaller is better.

data_module = BiomimicryDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

dummy_model = nn.Linear(2, 1)

optimizer = AdamW(params=dummy_model.parameters(), lr=0.001)

warmup_steps = 20
total_training_steps = 100

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=warmup_steps,
  num_training_steps=total_training_steps
)

learning_rate_history = []

for step in range(total_training_steps):
    optimizer.step()
    scheduler.step()
    learning_rate_history.append(optimizer.param_groups[0]['lr'])

steps_per_epoch=len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5
warmup_steps, total_training_steps

model = BiomimicryTagger(
  n_classes=len(LABEL_COLUMNS),
  n_warmup_steps=warmup_steps,
  n_training_steps=total_training_steps 
)

checkpoint_callback = ModelCheckpoint(
  dirpath="checkpoints",
  filename="best-checkpoint",
  save_top_k=1, # save top 5 or 10.
  verbose=True,
  monitor="val_loss",
  mode="min"
)

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)

trainer = pl.Trainer(
  checkpoint_callback=checkpoint_callback,
  callbacks=[early_stopping_callback],
  max_epochs=N_EPOCHS,
  gpus=0, #changed from 1 to 0
  progress_bar_refresh_rate=30
)

#from transformers import AutoModelForSequenceClassification, AutoTokenizer

#model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = BiomimicryTagger.load_from_checkpoint(
    '/Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt',
    n_classes=len(LABEL_COLUMNS)
)
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#dont use path
#path: /Users/dolungwe/Documents/PeTaL/scibert-top29-aws-epoch46-122221.ckpt

from transformers_interpret import SequenceClassificationExplainer
cls_explainer = SequenceClassificationExplainer(
    model,
    tokenizer)