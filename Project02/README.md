# 📰 Project 02 - Fake News Prediction (Using Logistic Regression)

This project predicts whether a given news article is **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning**.  

We use text processing techniques like **Stemming & TF-IDF Vectorization** and train a **Logistic Regression classifier**.  

---

## 📖 Dataset Information
- **Dataset used**: Kaggle Fake News dataset (`train.csv`)  
- **Features include**:
  - `author` → Author of the article  
  - `title` → Title of the article  
  - `text` → Content of the article  
  - `label` → Target (0 = Real, 1 = Fake)  

> For simplicity, in this project we only use **author + title** to build the feature set.

---

## 🛠️ Code Walkthrough

### 🔹 1. Importing Dependencies
```python
import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
````

* **NumPy & Pandas** → Data handling
* **re (Regex)** → Text cleaning (removing special characters, numbers, punctuation)
* **NLTK Stopwords** → Removing common words (like "the", "is", "and") that don’t add meaning
* **PorterStemmer** → Converts words to root form (ex: *actor, acting* → *act*)
* **TF-IDF Vectorizer** → Converts text into numeric form for ML
* **train\_test\_split** → Train-test split
* **Logistic Regression** → ML model for classification
* **accuracy\_score** → Model evaluation

---

### 🔹 2. Downloading NLTK Stopwords

```python
import nltk
nltk.download('stopwords')
print(stopwords.words('english'))
```

* Downloads the list of stopwords in English.
* Helps in text preprocessing by removing unimportant words.

---

### 🔹 3. Loading Dataset

```python
news_dataset = pd.read_csv('/content/train.csv')
news_dataset.shape
news_dataset.head()
```

* Loads dataset into Pandas DataFrame.
* **shape** → Number of rows & columns
* **head()** → Displays first 5 rows

---

### 🔹 4. Handling Missing Data

```python
news_dataset.isnull().sum()
news_dataset = news_dataset.fillna('')
```

* Checks for missing values in dataset.
* Replaces them with an empty string.

---

### 🔹 5. Creating "Content" Column

```python
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
```

* Combines `author` and `title` to create a **new feature: content**.
* This makes the dataset richer by merging author style + title text.

---

### 🔹 6. Separating Data & Labels

```python
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']
```

* **X** → Input features (news content)
* **Y** → Target (0 = Real, 1 = Fake)

---

### 🔹 7. Text Preprocessing: Stemming

```python
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)   # Remove non-letters
    stemmed_content = stemmed_content.lower()           # Lowercase
    stemmed_content = stemmed_content.split()           # Split into words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content 
                       if not word in stopwords.words('english')] # Remove stopwords
    return ' '.join(stemmed_content)

news_dataset['content'] = news_dataset['content'].apply(stemming)
```

* Cleans text by:

  * Removing numbers, punctuation
  * Converting to lowercase
  * Removing stopwords (the, is, an, etc.)
  * Stemming words (*running → run*)

---

### 🔹 8. Converting Text to Numbers (Vectorization)

```python
X = news_dataset['content'].values
Y = news_dataset['label'].values

vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
```

* **TF-IDF (Term Frequency – Inverse Document Frequency)** → Converts text into numbers.
* Important words get **higher weight**, common words get **lower weight**.

---

### 🔹 9. Splitting Data

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)
```

* **80%** training & **20%** testing
* **stratify=Y** ensures same proportion of Fake/Real in both sets

---

### 🔹 10. Training the Model

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

* Trains a **Logistic Regression classifier** on text data.
* Logistic Regression works very well for **binary classification** problems like Fake vs Real.

---

### 🔹 11. Model Evaluation

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

* Calculates accuracy on **training & testing sets**.
* Training accuracy → How well the model learned.
* Testing accuracy → How well it generalizes to new data.

---

### 🔹 12. Making a Predictive System

```python
X_new = X_test[3]
prediction = model.predict(X_new)

if prediction[0] == 0:
    print("The news is Real")
else:
    print("The news is Fake")

print(Y_test[3])
```

* Picks one sample news from test data.
* Predicts whether it’s **Real or Fake**.
* Compares with actual label.

---

## 📊 Visualizations

Even though this project is mostly text-based, we can still visualize important aspects:

#### (a) Distribution of Real vs Fake News

```python
sns.countplot(x=Y, palette='Set2')
plt.title("Distribution of Real vs Fake News")
plt.show()
```

#### (b) WordCloud for Fake vs Real News

```python
from wordcloud import WordCloud

# Fake news
fake_news = news_dataset[news_dataset['label']==1]['content']
fake_wc = WordCloud(width=600, height=400).generate(" ".join(fake_news))
plt.imshow(fake_wc)
plt.title("Most Common Words in Fake News")
plt.axis("off")
plt.show()

# Real news
real_news = news_dataset[news_dataset['label']==0]['content']
real_wc = WordCloud(width=600, height=400).generate(" ".join(real_news))
plt.imshow(real_wc)
plt.title("Most Common Words in Real News")
plt.axis("off")
plt.show()
```

* Shows which words appear most often in fake vs real news articles.

#### (c) Confusion Matrix

```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, X_test_prediction)
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

* Visualizes model’s correct vs incorrect predictions.

---

## 📈 Results

* **Training Accuracy**: \~98%
* **Testing Accuracy**: \~95%
* High accuracy means Logistic Regression works well for this dataset.
* WordCloud shows frequent words in Fake vs Real news articles.

---

## 🔮 Future Enhancements

* Use more powerful models like **Naive Bayes, Random Forest, or LSTMs (Deep Learning)**
* Improve preprocessing by including `text` column (not only author+title)
* Deploy a **Fake News Detector Web App** using Streamlit or Flask

---

✅ This is my **second ML project** in the `Machine_Learning_project` repository.
📂 [Back to Main Repo](../README.md)
