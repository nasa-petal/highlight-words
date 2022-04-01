pip install transformers-interpret

!pip install -r requirements.txt

!nvidia-smi

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1, auroc, precision_recall_curve, average_precision
#from pytorch_lightning.metrics.functional import accuracy, f1, auroc
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format='retina'

RANDOM_SEED = 42

sns.set(style='whitegrid', palette='muted', font_scale=1.2)
HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
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
df.drop([col for col, val in df[LABEL_COLUMNS].sum().iteritems() if val < 25], axis=1, inplace=True)
dropcols = ['protect_from_animals', 'coordinate_by_self-organization', 'maintain_biodiversity', 'compete_within/between_species', 'cooperate_within/between_species']
df.drop(dropcols, axis=1, inplace=True)

#df = df[df.columns[df[LABEL_COLUMNS].sum()>3]]
print(df.shape)
df.head()

LABEL_COLUMNS = df.columns.tolist()[:-1]

biom = df[df[LABEL_COLUMNS].sum(axis=1) > 0]
nonbiom = df[df[LABEL_COLUMNS].sum(axis=1) == 0]

# remove all non-biomimicry papers from dataset.
df = biom


from skmultilearn.model_selection import iterative_train_test_split


def iterative_train_test_split_dataframe(X, y, test_size):
    df_index = np.expand_dims(X.index.to_numpy(), axis=1)
    df_index_y = np.expand_dims(y.index.to_numpy(), axis=1)
    X_train, y_train, X_test, y_test = iterative_train_test_split(df_index, df_index_y, test_size = test_size)
    X_train = X.loc[X_train[:,0]]
    X_test = X.loc[X_test[:,0]]
    y_train = y.loc[y_train[:,0]]
    y_test = y.loc[y_test[:,0]]
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = iterative_train_test_split_dataframe(X=df[['text_raw']], y=df[LABEL_COLUMNS], test_size = 0.15)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_test, y_test], axis=1)
print(train_df.shape, val_df.shape)

'''
X_train_val, y_train_val, X_test, y_test = iterative_train_test_split_dataframe(X=df[['text_raw']], y=df[LABEL_COLUMNS], test_size = 0.1)
test_df = pd.concat([X_test, y_test], axis=1)
X_train, y_train, X_val, y_val = iterative_train_test_split_dataframe(X=X_train_val, y=y_train_val, test_size = 0.13)
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
#train_df, val_df = train_test_split(df, test_size=0.1)
train_df.shape, val_df.shape, test_df.shape
'''


BERT_MODEL_NAME = 'allenai/scibert_scivocab_uncased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)


MAX_TOKEN_COUNT = 512

class BiomimicryDataset(Dataset):

  def __init__(
    self, 
    data: pd.DataFrame, 
    tokenizer: BertTokenizer, 
    max_token_len: int = 128
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    text = data_row.text_raw
    labels = data_row[LABEL_COLUMNS]

    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      text=text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor(labels)
    )

class BiomimicryDataModule(pl.LightningDataModule):

  def __init__(self, train_df, test_df, tokenizer, batch_size=8, max_token_len=128):
    super().__init__()
    self.batch_size = batch_size
    self.train_df = train_df
    self.test_df = test_df
    self.tokenizer = tokenizer
    self.max_token_len = max_token_len

  def setup(self, stage=None):
    self.train_dataset = BiomimicryDataset(
      self.train_df,
      self.tokenizer,
      self.max_token_len
    )

    self.test_dataset = BiomimicryDataset(
      self.test_df,
      self.tokenizer,
      self.max_token_len
    )

  def train_dataloader(self):
    return DataLoader(
      self.train_dataset,
      batch_size=self.batch_size,
      shuffle=True,
      num_workers=2
    )

  def val_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )

  def test_dataloader(self):
    return DataLoader(
      self.test_dataset,
      batch_size=self.batch_size,
      num_workers=2
    )
N_EPOCHS = 100
BATCH_SIZE = 12 # smaller is better.

data_module = BiomimicryDataModule(
  train_df,
  val_df,
  tokenizer,
  batch_size=BATCH_SIZE,
  max_token_len=MAX_TOKEN_COUNT
)

class BiomimicryTagger(pl.LightningModule):

  def __init__(self, n_classes: int, n_training_steps=None, n_warmup_steps=None):
    super().__init__()
    self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
    self.n_training_steps = n_training_steps
    self.n_warmup_steps = n_warmup_steps
    self.criterion = nn.BCELoss()

  def forward(self, input_ids, attention_mask, labels=None):
    output = self.bert(input_ids, attention_mask=attention_mask)
    output = self.classifier(output.pooler_output)
    output = torch.sigmoid(output)    
    loss = 0
    if labels is not None:
        loss = self.criterion(output, labels)
    return loss, output

  def training_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("train_loss", loss, prog_bar=True, logger=True)
    return {"loss": loss, "predictions": outputs, "labels": labels}

  def validation_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("val_loss", loss, prog_bar=True, logger=True)
    return loss

  def test_step(self, batch, batch_idx):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    loss, outputs = self(input_ids, attention_mask, labels)
    self.log("test_loss", loss, prog_bar=True, logger=True)
    return loss

  def training_epoch_end(self, outputs):
    
    labels = []
    predictions = []
    for output in outputs:
      for out_labels in output["labels"].detach().cpu():
        labels.append(out_labels)
      for out_predictions in output["predictions"].detach().cpu():
        predictions.append(out_predictions)

    labels = torch.stack(labels).int()
    predictions = torch.stack(predictions)

    for i, name in enumerate(LABEL_COLUMNS):
      class_roc_auc = auroc(predictions[:, i],  labels[:, i])
      #class_ap = average_precision(predictions[:, i], labels[:, i])
      self.logger.experiment.add_scalar(f"{name}_roc_auc/Train", class_roc_auc, self.current_epoch)
      #self.logger.experiment.add_scalar(f"{name}_ap/Train", class_ap, self.current_epoch)


  def configure_optimizers(self):

    optimizer = AdamW(self.parameters(), lr=2e-5)

    scheduler = get_linear_schedule_with_warmup(
      optimizer,
      num_warmup_steps=self.n_warmup_steps,
      num_training_steps=self.n_training_steps
    )

    return dict(
      optimizer=optimizer,
      lr_scheduler=dict(
        scheduler=scheduler,
        interval='step'
      )
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

!rm -rf lightning_logs/
!rm -rf checkpoints/

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

