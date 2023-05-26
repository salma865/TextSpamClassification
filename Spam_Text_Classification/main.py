# import the necessary libraries
import nltk
import string
import pandas as pd
import re
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

# Loading data
df = pd.read_csv("data_spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
# print("Data before preprocessing")
# print(df.v2)

# Preprocessing
# convert text to lowercase
df.v2 = df.v2.apply(lambda x: x.lower())


# remove_punctuation
def remove_punctuation(text):
    puncfree = "".join([i for i in text if i not in string.punctuation])
    return puncfree


df.v2 = df.v2.apply(lambda x: remove_punctuation(x))


# Remove numbers
def remove_numbers(text):
    result = re.sub(r'\d+', '', text)
    return result


df.v2 = df.v2.apply(lambda x: remove_numbers(x))


# remove whitespace from text
def remove_whitespace(text):
    return " ".join(text.split())


df.v2 = df.v2.apply(lambda x: remove_whitespace(x))


# tokenizer
def tokenization(text):
    tk = WhitespaceTokenizer()
    return tk.tokenize(text)


df.v2 = df.v2.apply(lambda x: tokenization(x))

# Stop words
stopwords = nltk.corpus.stopwords.words('english')


def remove_stopwords(text):
    output = [i for i in text if i not in stopwords]
    return output


df.v2 = df.v2.apply(lambda x: remove_stopwords(x))


# Stemming
porter_stemmer = PorterStemmer()


def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text


df.v2 = df.v2.apply(lambda x: stemming(x))

# lemmatization
df_string = df.v2.astype(str)
lemmatizer = WordNetLemmatizer()
wordnet_map = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}


def lemmatize_words(text):
    pos_tagged_text = nltk.pos_tag(text.split())
    return " ".join([lemmatizer.lemmatize(word, wordnet_map.get(pos[0], wordnet.NOUN)) for word, pos in pos_tagged_text])


df_string = df_string.apply(lambda text: lemmatize_words(text))
# print("Data after preprocessing")
# print(df.v2)
#######################################################################################################################

# Feature generation
myvar_list = df["v2"].tolist()
vectorizer = CountVectorizer()
bag_of_words = vectorizer.fit_transform(df_string)
# print(bag_of_words.toarray())
A_sparse = sparse.csr_matrix(bag_of_words)
similarities_sparse = cosine_similarity(A_sparse, dense_output=False)
# print('pairwise sparse output:\n {}\n'.format(similarities_sparse))

#######################################################################################################################

# Classification
Y = pd.get_dummies(df.v1)
Y = Y.iloc[:, 1].values
X_train, X_test, y_train, y_test = train_test_split(bag_of_words, Y, test_size=0.30, random_state=42)

# Decision Tree               ------>> accuracy 96.8%
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
DT_y_pred = DT_model.predict(X_test)
print("Decision tree model")
print(confusion_matrix(y_test, DT_y_pred), accuracy_score(y_test, DT_y_pred)*100, precision_score(y_test, DT_y_pred), recall_score(y_test, DT_y_pred))
print("*******************")

# Random Forest Classifier    ------>> accuracy 97.18%     --->> Best classifier
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
RF_y_pred = RF_model.predict(X_test)
print("Random Forest model")
print(confusion_matrix(y_test, RF_y_pred), accuracy_score(y_test, RF_y_pred)*100, precision_score(y_test, RF_y_pred), recall_score(y_test, RF_y_pred))
print("*******************")

# Naive Bayes classifier      ------>> accuracy 97.12% 
NB_model = MultinomialNB()
NB_model.fit(X_train, y_train)
NB_y_pred3 = NB_model.predict(X_test)
print("Naive Bayes model")
print(confusion_matrix(y_test, NB_y_pred3), accuracy_score(y_test, NB_y_pred3)*100, precision_score(y_test, NB_y_pred3), recall_score(y_test, NB_y_pred3))
