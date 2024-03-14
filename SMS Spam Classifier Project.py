#!/usr/bin/env python
# coding: utf-8

# In[1]:


#---------------------------------------------------------------------------------------------------------
# NAME: ELIYA CHRISTOPHER NANDI            ID: 901481493              FINAL PROJECT: SMS SPAM CLASSIFIER
#---------------------------------------------------------------------------------------------------------


# In[2]:


import numpy as np
import pandas as pd
from IPython.display import display, HTML

df = pd.read_csv('Spam SMS Data', header=None, sep='\t',names=['Label', 'Message'])
display(HTML(df.to_html(index=True)))


# In[3]:


df.shape


# In[4]:


df.columns


# In[5]:


df.dtypes


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.info()


# In[9]:


display(HTML(df.describe(include='object').to_html() + '<hr>'))


# In[10]:


# Replace the values in the 'Label' column with 0 for 'ham' and 1 for 'spam'
df['Label'] = df['Label'].replace({'ham': 0, 'spam': 1})
df.head()


# In[11]:


df.tail()


# In[12]:


# Finding total no of ham and spam messages
df['Label'] = df['Label'].replace({'ham': 0, 'spam': 1})

ham_count = df[df['Label'] == 0].shape[0]
spam_count = df[df['Label'] == 1].shape[0]

print("Total Number of Ham Messages:", ham_count)
print("Total Number of Spam Messages:", spam_count)


# In[13]:


# Visualizing the distribution of messages in the dataset  based on their label (spam or ham),using a pie chart. 
# The pie chart shows the percentage of messages for each label,making it easy to see that the dataset is imbalanced 
# (i.e., there are far more ham messages than spam messages).

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# Loading data from file
df = pd.read_csv('Spam SMS Data', sep='\t', names=['Label', 'Message'])

# Creating a pie chart of message count by label
count_by_label = df['Label'].value_counts()
plt.pie(count_by_label, labels=count_by_label.index, autopct='%1.1f%%')
plt.title('Distribution of Spam vs. Ham Messages')
plt.show()


# In[14]:


import numpy as np
import pandas as pd
from IPython.display import display, HTML
df = pd.read_csv('Spam SMS Data', header=None, sep='\t',names=['Label', 'Message'])

# Creating feature contains_currency_symbol
def currency(b):
    currency_symbols = ['€', '$', '¥', '£', '₹']
    for a in currency_symbols:
        if a in b:
            return 1
    return 0

df['contains_currency_symbol'] = df['Message'].apply(currency)


# In[15]:


df.head()


# In[16]:


# Countplot for contains_currency_symbol
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6,6))
g = sns.countplot(x='contains_currency_symbol', data=df, hue='Label')
p = plt.title('Countplot for SMS messages with currency symbol')
p = plt.xlabel('Does SMS contain currency symbol?')
p = plt.ylabel('Count')
p = plt.legend(labels=['Ham', 'Spam'], loc=9)


# In[17]:


# Creating feature contains_number
def numbers(x):
    for i in x:
        if ord(i)>=48 and ord(i)<=57:
            return 1
    return 0

df['Contains_Number'] = df['Message'].apply(numbers)


# In[18]:


df.head()


# In[19]:


# Visualizing Number Usage in SMS Messages
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.figure(figsize=(6,6))
g = sns.countplot(x='Contains_Number', data=df, hue='Label')
p = plt.title('Visualizing Number Usage in SMS Messages')
p = plt.xlabel('Does SMS contain number?')
p = plt.ylabel('Count')
p = plt.legend(labels=['Ham', 'Spam'], loc=9)


# In[20]:


# MODEL BULDING AND EVALUATION


# In[21]:


import nltk
import re

# Download required NLTK resources
nltk.download(['stopwords', 'wordnet'])

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set up stopwords and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

corpus = []
wnl = WordNetLemmatizer()

# Iterate over each sms message
for sms_string in list(df.Message):
    # Remove non-alphabetic characters from the sms message
    Message = re.sub('[^a-zA-Z]', ' ', sms_string)
    # Convert the message to lowercase
    Message = Message.lower()
    # Split the message into individual words
    words = Message.split()
    # Filter out stopwords from the words
    filtered_words = [word for word in words if word not in set(stopwords.words('english'))]
    # Lemmatize the words
    lemmatized_words = [wnl.lemmatize(word) for word in filtered_words]
    # Join the lemmatized words back into a cleaned message
    cleaned_message = ' '.join(lemmatized_words)
    # Append the cleaned message to the corpus
    corpus.append(cleaned_message)

# Print the first five elements of the corpus
print(corpus[0:5])


# In[22]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Creating the Bag of Words model
tfidf = TfidfVectorizer(max_features=500)
vectors = tfidf.fit_transform(corpus).toarray()
feature_names = tfidf.get_feature_names_out()

# Extracting independent and dependent variables from the dataset
X = pd.DataFrame(vectors, columns=feature_names)
y = df['Label']


from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[23]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Fitting Naive Bayes to the Training set
mnb = MultinomialNB()
scorer = make_scorer(f1_score, pos_label='spam')

cv_scores = cross_val_score(mnb, X, y, scoring=scorer, cv=10)

average_f1_score = round(cv_scores.mean(), 3)
standard_deviation = round(cv_scores.std(), 3)

print('Average F1-Score for MNB model= {}'.format(average_f1_score))
print('Standard Deviation= {}'.format(standard_deviation))


# In[24]:


# Classification Report for MNB model

mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Predict using the trained model
y_pred = mnb.predict(X_test)

# Generate and print the classification report
report = classification_report(y_test, y_pred)
print('=========================================================')
print('Classification Report for MNB Model')
print('=========================================================')
print(report)


# In[25]:


# Multinomial Naive Bayes Model Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
axis_labels = ['Ham', 'Spam']
g = sns.heatmap(data=cm, annot=True, cmap="Oranges",linewidths=2, linecolor='grey', xticklabels=axis_labels, yticklabels=axis_labels, fmt='g', cbar_kws={"shrink": 0.5})
p = plt.xlabel('Actual values')
p = plt.ylabel('Predicted values')
p = plt.title('Multinomial Naive Bayes Model Confusion Matrix')


# In[26]:


# Using a decision tree classifier to perform cross-validation and calculate the F1-score for a specific class,
# in this case, the 'spam' class.

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score

# Fitting Decision Tree to the Training set
dt = DecisionTreeClassifier()
# Create a custom scorer for calculating the F1-score
custom_scorer = make_scorer(f1_score, pos_label='spam')

# Perform cross-validation and calculate the F1-score for the 'spam' class
cv = cross_val_score(dt, X, y, scoring=custom_scorer, cv=10)

# Finally, we print the average F1-score and the standard deviation 
# to evaluate the performance of the decision tree model in classifying the 'spam' class

print('Average F1-Score for Decision Tree model: {}'.format(round(cv.mean(), 3)))
print('Standard Deviation: {}'.format(round(cv.std(), 3)))


# In[27]:


# Classification Report for Decision Tree Model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print('============================================================')
print('Classification Report for Decision Tree Model')
print('============================================================')
print(classification_report(y_test, y_pred))


# In[28]:


# Confusion Matrix of Decision Tree Model
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
axis_labels = ['Ham', 'Spam']
g = sns.heatmap(data=cm, annot=True, cmap="viridis",linewidths=2, linecolor='brown', xticklabels=axis_labels, yticklabels=axis_labels, fmt='g', cbar_kws={"shrink": 0.5})
p = plt.xlabel('Actual values')
p = plt.ylabel('Predicted values')
p = plt.title(' Decision Tree Model Confusion Matrix')


# In[29]:


# Random Forest Classifier with Cross-Validated F1-Score Evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score
import pandas as pd

# Convert X and y to pandas DataFrames
X_df = pd.DataFrame(X)
y_df = pd.Series(y)

# Create a Random Forest classifier with 10 estimators
rf = RandomForestClassifier(n_estimators=10)

# Create a scorer using make_scorer
scorer = make_scorer(f1_score, pos_label='spam')

# Perform cross-validation and calculate F1 score
cv_scores = cross_val_score(rf, X_df, y_df, scoring=scorer, cv=10)

# Calculate average F1 score and standard deviation
avg_f1_score = cv_scores.mean()
std_deviation = cv_scores.std()

# Print the results
print('Average F1-Score for Random Forest Model= {:.3f}'.format(avg_f1_score))
print('Standard Deviation= {:.3f}'.format(std_deviation))


# In[30]:


#  Comprehensive Classification Report for Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Classification report for Random Forest model
# The n_estimators parameter specifies the number of decision trees to include in the random forest.
#Increasing the number of estimators can lead to better performance, up to a certain point

rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print('============================================================')
print('Comprehensive Classification Report for Random Forest Model')
print('============================================================')
print(classification_report(y_test, y_pred))


# In[31]:


# Confusion Matrix of Random Forest Model
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
axis_labels = ['Ham', 'Spam']
g = sns.heatmap(data=cm, annot=True, cmap="magma",linewidths=2, linecolor='green', xticklabels=axis_labels, yticklabels=axis_labels, fmt='g', cbar_kws={"shrink": 0.5})
p = plt.xlabel('Actual values')
p = plt.ylabel('Predicted values')
p = plt.title('Random Forest Model Confusion Matrix ')


# In[32]:


# PREDICTION OF WHETHER THE MESSAGE IS SPAM OR HAM


# In[33]:


# USING RANDOM FOREST CLASSIFIER MODEL


# In[34]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data=pd.read_csv('Spam SMS Data', header=None, sep='\t',names=['Label', 'Message'])

# Split the data into features and labels
X = data['Message']
y = data['Label']

# Convert the text into numerical features
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
# Train the classifier
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.3}".format(accuracy))

# Example usage: Predict if a given message is spam or not
# Here i am checking 5 messages which are put in a list, obtained from the corpus 
# to see whether the model will make correct prediction.

sample_message =  [
    "As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune",
    "WINNER!! As a valued network customer you have been selected to receivea å£900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.",
    "I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today.",
    "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
    "Filthy stories and GIRLS waiting for your"
]
sample_message = vectorizer.transform(sample_message)
prediction = rf_classifier.predict(sample_message)
# The output of predictions is given in a list form showing the outcome of all 5 messages checked
print("Prediction:", prediction)


# In[35]:


#-----------------THANK YOU, IT HAS BEEN A GOOD EXPERIENCE ACCOMPLISHING THIS PROJECT-------------------------------------------

