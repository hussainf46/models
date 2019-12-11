import pandas as pd
import numpy as np
import matplotlib.pyplot as pl

'''dataset_tain = pd.read_excel('Final_Train')'''
dataset = pd.read_csv('Restaurant_Reviews.tsv',delimiter = '\t', quoting = 3)

#cleaning the dataset 
import re 
import nltk
#nltk.download('stopwords') # stopwords is a list which consist of all the unwamted words 
from nltk.corpus import stopwords 
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1000):
	review = re.sub('[^a-zA-Z]',' ', dataset['Review'][i]) #removing all unwanted symbols and notations 
	review = review.lower() #lowercasung weach and every word 
	review = review.split() #splitting the words for stemming  
	ps = PorterStemmer()
	review = [ps.stem(word) for word in review if not word  in set(stopwords.words('english'))] #this will remove all unwanted word from the dataset for eg (the ,this ,in for ,...etc )
	#stemming also helps to derive the original word for eg (loved is being replaced by  love ) it basically makes the word smpler 
	review = ' '.join(review)
	corpus.append(review)
	
#apllying bag words model to  the dataset 
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values

#didviding inti test set and train set 
from sklearn.cross_validation import train_test_split
xtrain,xtest ,ytrain, ytest = train_test_split(x ,y ,test_size = 0.20 , random_state= 0)

#tarining the model 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(xtrain,ytrain)

#prediction 
y_pred = classifier.predict(xtest)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest,y_pred)
