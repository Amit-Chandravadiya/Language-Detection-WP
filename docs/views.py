from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from django.contrib import messages
import joblib
import sklearn
import re 
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
# Create your views here.

# loding Tf-IDf model from joblib file
cwd=os.getcwd()
loc=os.path.join(cwd,'docs/finalmodel.joblib')
model=joblib.load(loc)

# using pandas and label encoding 
loc2=os.path.join(cwd,'Language Detection.csv')
datasets = pd.read_csv(loc2)

y = datasets['Language']
x=datasets['Text']

lbe=LabelEncoder()
z=lbe.fit_transform(y)
# creating a list for appending the preprocessed text
data_list = []
# iterating through all the text
for text in x:
       # removing the symbols and numbers
        text = re.sub(r'[!@#$(),n"%^*?:;~`"0-9]', '', text)
        re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', '', text)
        text = re.sub(r'[[]]', '', text)
        # converting the text to lower case
        text = text.lower()
        # appending to data_list
        data_list.append(text)
# creating vectorizer
tf=TfidfVectorizer(ngram_range=(1,3), analyzer='char')
t=tf.fit_transform(data_list)

def predict(text,request):
     x = tf.transform([text]).toarray() # converting text to bag of words model (Vector)
     lang = model.predict(x) # predicting the language
     lang = lbe.inverse_transform(lang) # finding the language corresponding the the predicted value
     print("The langauge is in",lang[0])
     print("The langauge is in",lang[0])
     print("The langauge is in",lang[0])
     messages.error(request,f'This is written in {lang[0]} !')

def home(request):
    return render(request,"base.html")

def Text_Rec(request):
    if request.method == 'POST':
        print(request.POST)
        data=request.POST.get('txt')
        text = re.sub(r'[!@#$(),n"%^*?&:;~`"0-9]', '',data)
        re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', '', text)
        text = text.lower()
        predict(text,request)
        return render(request,'base.html')
    else:
        return render(request,'base.html')
      