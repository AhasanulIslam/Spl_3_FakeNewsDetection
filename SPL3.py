# coding: utf8
import time
import tkinter
from collections import defaultdict
from tkinter.tix import IMAGETEXT

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm
from transformers import (BertForSequenceClassification, BertModel,
                          BertTokenizer)
from wordcloud import STOPWORDS, WordCloud

# import seaborn as sns

auth = pd.read_csv("D://SPL3//Authentic-48K.csv")
fake = pd.read_csv("D://SPL3//Fake-1K.csv")

fake.head()


#Analysis Words

from bnlp import BasicTokenizer

tokenizer = BasicTokenizer()

# s = """আমি বাংলায় গান গাই
#     আমি বাংলার গান গাই।
#     আমি আমার আমিকে চিরদিন
#     এই বাংলায় খুঁজে পাই।"""

# print(tokenizer.tokenize(s))

def create_corpus(texts):
    corpus=[]

    for txt in texts:
      tokens = tokenizer.tokenize(txt)
      corpus.extend(tokens)

    return corpus

# auth_corpus = create_corpus(auth.headline[:1000])
# print("Total auth tokens in 1000", len(auth_corpus))

# fake_corpus = create_corpus(fake.headline[:1000])
# print("Total auth tokens in 1000", len(fake_corpus))

from bnlp.corpus import digits, punctuations, stopwords


def filters(corpus):
  res = []
  for i in corpus:
    if i in stopwords:
      continue

    if i in punctuations + '‘' + '’':
      continue

    if i in digits:
      continue

    res.append(i)

  return res

# auth_corpus_filtered = filters(auth_corpus)
# fake_corpus_filtered = filters(fake_corpus)

def get_top_words(corpus):
  dic = defaultdict(int)

  for word in corpus:
      dic[word] +=1

  top = sorted(dic.items(), key=lambda x:x[1],reverse=True)
  x,y=zip(*top)
  return x, y

#Train and Test split:

class NewsDatasets(Dataset):
    def __init__(self, data, max_length=100):
        self.data = data
        
        self.config = {
            "max_length": max_length,
            "padding": "max_length",
            "return_tensors": "pt",
            "truncation": True,
            "add_special_tokens": True
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        value = self.data.iloc[idx]
        return value['head']+"[SEP]"+value['con'] , value['label']

from bnlp import BasicTokenizer
from bnlp.corpus import digits, letters, punctuations, stopwords

btokenizer = BasicTokenizer()

def clean_text(text):
    tokens = btokenizer.tokenize(text)
    filtered = []
    for i in tokens:
        if i in stopwords:
            continue
    
        if i in punctuations + '‘' + '’':
            continue
    
        filtered.append(i)
    
    return " ".join(filtered)

#Define Model

class NewsBert(nn.Module):

    def __init__(self, bert):
        super(NewsBert, self).__init__()

        self.bert = bert

        # dropout layer
        self.dropout = nn.Dropout(0.2)

        # relu activation function
        self.relu = nn.ReLU()

        # dense layer 1
        self.fc1 = nn.Linear(768, 128)

        # dense layer 2 (Output layer)
        self.fc2 = nn.Linear(128, 2)  

    # define the forward pass
    def forward(self, input_ids, token_type_ids, attention_mask):
        # pass the inputs to the model
        out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        x = self.fc1(out[1])
        x = self.relu(x)
        # output layer
        x = self.fc2(self.dropout(x))
        
        return x

bert_model_name = "sagorsarker/bangla-bert-base"
bert = BertModel.from_pretrained(bert_model_name)
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NewsBert(bert)
model.to(device)

from torch.optim.lr_scheduler import StepLR

optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)

#testing

model.load_state_dict(torch.load("D://SPL3//news_model_epoch1_91.pth", map_location = device))

def predict(heading, content):
        # heading = 'জুম্মার নামাজে সবচেয়ে বেশি মসজিদে যায় নোয়াখালীর ছেলেরা !' 
        # content = "এক গবেষণা থেকে জানা গেছে, বাংলাদেশের অন্যান্য সব জেলার চেয়ে নোয়াখালী জেলার তরুণরাই জুম্মার নামাজ পড়তে সবচেয়ে বেশি মসজিদে যান। যা কিনা শতকরার হিসাবে অন্যান্য সব জেলার চাইতে কয়েকশ গুণ পরিমাণে বেশি !জুম্মার নামাজে নোয়াখালীর মানুষদের এরকম উপস্থিতির হার দেখেই স্পষ্ঠ ধারণা পাওয়া যায়, এদের ইমানি শক্তি ভালো এবং চারিত্রিক দিক থেকে এরা হয় সৎ।এই জরীপে জুম্মার নামাজে নোয়াখালীর মুসল্লির হার  দেখা যায়৭৫%। এর পরপরই তালিকায় রয়েছে ঢাকা ৬৫%, চট্টগ্রাম ৪০%, রাজশাহী-রংপুর ৩৮% এবং বরিশাল মাত্র ১১%  !যদিও অনেকে মনে করছেন, নিমকি-জিলাপির লোভেই নোয়াখালীর বাসিন্দারা শুক্রবারে নামাজে গিয়ে থাকেন।"
  global result
  print(heading)
  data = [[heading, content]]
    
  # Create the pandas DataFrame
  input_df = pd.DataFrame(data, columns=['head', 'con'])


  input_df['head'] = input_df['head'].apply(clean_text)
  input_df['con'] = input_df['con'].apply(clean_text)
  input_df['label'] = 6

  input_data = NewsDatasets(input_df)
  # input_data

  input_dataloader = DataLoader(input_data, batch_size=1)

  tokenizer_config = {
      "max_length": 100,
      "padding": "max_length",
      "return_tensors": "pt",
      "truncation": True,
      "add_special_tokens": True
  }

  for x in input_dataloader:
    text, label = x
    inputs = tokenizer.batch_encode_plus(
            text, **tokenizer_config
        )
    input_ids = inputs['input_ids'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
  #   labels = labels.to(device)

    # move things to model
    output = model(token_type_ids=token_type_ids, input_ids=input_ids, attention_mask=attention_mask)
    preds = output.detach().cpu().numpy()
    print(preds)
    preds = np.argmax(preds, axis = 1)
    if preds[0]==0:
      print("fake")
      result = "fake"
      LF2.config(text = result)
    else:
      print("Authentic")
      result = "authentic"
      LF2.config(text = result)
    # all_preds.extend(preds)
    # all_labels.extend(labels.cpu().numpy())

# predict("এক্সচভক্সচভক্সচভ", "এক্সবচব ভক্সচভভক্সচভভসদসদ")

result = ''
from tkinter import *

from PIL import Image, ImageTk

root=Tk()

def on_resize(event):
    # resize the background image to the size of label
    image = bgimg.resize((event.width, event.height), Image.ANTIALIAS)
    # update the image of the label
    l.image = ImageTk.PhotoImage(image)
    l.config(image=l.image)


root.geometry("600x700")
  
bgimg = Image.open('new5.jpg') # load the background image
l = Label(root)
l.place(x=0, y=0, relwidth=1, relheight=1) # make label l to fit the parent window always
l.bind('<Configure>', on_resize) # on_resize will be executed whenever label l is resized

LF2=Label(root, text="", font=("Arial Black", 30), background= "#4C7FA9")
LF2.grid(row=0, column=1)


text1 = Text(root, height=3, width=60, borderwidth=1, relief="solid")
scroll = Scrollbar(root)
text1.configure(yscrollcommand=scroll.set)
text1.grid(row=1, column=1)


L1 = Label( text="Heading:", font = 60)
L1.grid(row=1, column=0)

LF2=Label(root, text="", font=("Arial Black", 30), background= "#4C7FA9")
LF2.grid(row=2, column=1)


# def C1():
#     E1.insert(0, "#")
#     L2= Label(LF, text=E1.get())
#     L2.pack()

L1 = Label( text="Content:", font = 60, )
L1.grid(row=3, column=0)

text = Text(root, height=8, width=60, borderwidth=1, relief="solid")
scroll = Scrollbar(root)
text.configure(yscrollcommand=scroll.set)
text.grid(row=3, column=1)

# def C1():
#     E2.insert(0, "#")
#     L2= Label(LF, text=E1.get())
#     L2.pack()

# action_with_arg = predict(E1.get(), E2.get())

B1=Button(root, text="Submit", font=40, command=lambda: predict(text1.get("1.0",'end-1c'), text.get("1.0",'end-1c')))
B1.grid(row=5, column=1)




LF2=Label(root, text="", font=("Arial Black", 30), background= "#4C7FA9")
LF2.grid(row=4, column=1)

LF2=Label(root, text="", font=("Arial Black", 30), background= "#4C7FA9")
LF2.grid(row=6, column=1)

LF2=Label(root, text=result, font=("Arial Black", 30), background= "#4C7FA9")
LF2.grid(row=7, column=1)

B1=Button(root, text="Exit", font=40, command=root.destroy)
B1.grid(row=8, column=1)

# LF2.pack()

print(11111, result)
root.mainloop()
