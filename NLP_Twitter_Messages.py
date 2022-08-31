import spacy
import pandas as pd
import string
import spacy
import random
import seaborn as sns
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

df_training = pd.read_csv('Train50.csv', delimiter=';')

'''Let's take a look at the dataset'''

# PRINT 1

'''The sentiment data is important because we'll used it for predictions. So it's important to understand it.
In this case 0 means negative emotions and 1 means positive emotions. Also we're checking if the dataset
is unbalanced and drop some non important data from the dataset'''

#  todo CLEANING DATA ------------------------

df_training.drop(columns=['id', 'tweet_date', 'query_used'], inplace=True)
sns.countplot(df_training['sentiment'], label='Count')
plt.show()

# PRINT 2

'''In this case, the training and test are in different files, so let's import the test dataset and apply the same
conditions we applied in the training dataset. The data is balanced, just like the training data'''

df_test = pd.read_csv('Test.csv', delimiter=';')
df_test.drop(columns=['id', 'tweet_date', 'query_used'], inplace=True)

'''Now I'll take a look at NaN data'''

print(df_training.isnull().sum())
print(df_test.isnull().sum())

# PRINT 3

'''Now I'm creating a function so we can transform upper case into lowercase letters, remove user names, whitespaces
and emoticons, add some stopwords, apply lemmatization and remove some punctuations'''

#  todo PRE PROCESSING ------------------------

nlp = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS
punct = string.punctuation


def pre_processing(text):
    # Lower case
    text = text.lower()
    # Username
    text = re.sub(r'@[A-Za-z0-9$-_@.&+]+', ' ', text)
    # URLs
    text = re.sub(r'https?://[A-Za-z0-9./]+', ' ', text)
    # Whitespaces
    text = re.sub(r' +', ' ', text)
    # Emoticons
    emoticon_list = {':)': 'emocaopositiva', ':d': 'emocaopositiva', ':(': 'emocaonegativa'}
    for emoticon in emoticon_list:
        text = text.replace(emoticon, emoticon_list[emoticon])
    # Lemmatization
    document = nlp(text)
    list = []
    for token in document:
        list.append(token.lemma_)
    # Stopwords
    list = [word for word in list if word not in stop_words and word not in punct]
    list = ' '.join([str(element) for element in list if not element.isdigit()])
    return list


test_t = 'Estava muito feliz naquele dia, :) e acabei comprando uma pizza no https://www.pizzasdna.com.br @pizzasdna'

test_text = pre_processing('Estava muito feliz naquele dia, :) e acabei comprando uma pizza'
                           ' no https://www.pizzasdna.com.br @pizzasdna')

'''Now I'm checking if everything is ok with the pre processing'''

print(test_t)
print(test_text)

# PRINT 4

'''And apply the function to our training and test dataset and save it, because this text pre processing takes
some time our datasets will look like that at the'''

#  todo CHECKPOINT SAVE ------------------------

df_training['tweet_text'] = df_training['tweet_text'].apply(pre_processing)
df_test['tweet_text'] = df_test['tweet_text'].apply(pre_processing)

df_training.to_csv('twitter_training_pre.csv')
df_test.to_csv('twitter_test_pre.csv')

df_training = pd.read_csv('twitter_training_pre.csv')
df_test = pd.read_csv('twitter_test_pre.csv')

# PRINT 5

'''Now I'm taking the new datasets to Google Colab so I can use a differente version from Spacy, for this model
I'll install the spacy version 2.2.3 fix the class we want to predict using 0 and 1 for Ture and False and
creating an example, at the end of this process we should have a list with the 50 thousand entries in the
training dataset'''

example = [["este trabalho é agradável", {"POSITIVO": True, "NEGATIVO": False}],
           ["este lugar continua assustador", {"POSITIVO": False, "NEGATIVO": True}]]

df_training_final = []
for text, emotion in zip(df_training['tweet_text'], df_training['sentiment']):
    if emotion == 1:
        dic = ({'POSITIVO': True, 'NEGATIVO': False})
    elif emotion == 0:
        dic = ({'POSITIVO': False, 'NEGATIVO': True})

    df_training_final.append([text, dic.copy()])

# PRINT 6

'''Now we can create the classifier'''

model = spacy.blank('pt')
categories = model.create_pipe("textcat")
categories.add_label("POSITIVO")
categories.add_label("NEGATIVO")
model.add_pipe(categories)

'''Now for the model part, where I'm training based on different aspects of the text, this concept is very
much like nlp for deep learning'''

historic = []
model.begin_training()
for epoch in range(5):
    random.shuffle(df_training_final)
    losses = {}
    for batch in spacy.util.minibatch(df_training_final, 512):
        text = [model(texto) for texto, entities in batch]
        annotations = [{'cats': entities} for texto, entities in batch]
        model.update([text, annotations], losses=losses)
        historic.append(losses)
    if epoch % 2 == 0:
        print(losses)

historic_loss = []
for i in historic:
    historic_loss.append(i.get('textcat'))

historic_loss = np.array(historic_loss)

'''Let's check the historic loss and plot it so we can understand it better'''

plt.plot(historic_loss)
plt.title('Error progression')
plt.xlabel('Batches')
plt.ylabel('Error')
plt.show()

# PRINT 7

'''I'm also saving the model now because we only need this part to be done in Google Colab, I'll carry the
model still on colab but I'm saving just in case'''

model.to_disk('model')
model_loaded = spacy.load('model')

'''Now let's see the results for positive and negative sentences and see how the model react to it'''

positive_text = 'eu gosto muito de você'
positive_text = model_loaded(positive_text)
print(model_loaded(positive_text).cats)

negative_text = 'estou meio chateado'
pre_dict = model_loaded(negative_text)
print(model_loaded(negative_text).cats)

# PRINT 8

'''Now I'll submit the test dataset to the model and see how much accuracy we have for the model answers'''

predictions = []
for text in df_test['tweet_text']:
    prev = model_loaded(text)
    predictions.append(prev.cats)

prediction_end = []
for pred in predictions:
    if pred['POSITIVO'] > pred['NEGATIVO']:
        prediction_end.append(1)
    else:
        prediction_end.append(0)

final_prediction = np.array(prediction_end)

real_answers = df_test['sentiment'].values

print(accuracy_score(real_answers, final_prediction))

# PRINT 9

'''As we can see we had a smaller accuracy, but still pretty high, let's check the confusion matrix
to check the answers'''

cm = confusion_matrix(real_answers, final_prediction)
sns.heatmap(cm, annot=True)
plt.show()

# PRINT 10

'''Now we can see the sentiment towards any hashtag or person towards a company or an event and see if
the company should be using some trends and to understand the general sentiment with clouds or doing a
classification for sentiment'''
