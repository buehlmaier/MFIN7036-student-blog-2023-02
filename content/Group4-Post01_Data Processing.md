---
Title: Data processing of articles from SeekingAlpha (by "Group 4")
Date: 2023-03-17 19:29:39
Category: Progress Report
---

By Group 4

We have collected articles about health care industry 
from Seeking Alpha. In order to better analyze the data, we need to preprocess it.

## Merge data

After read in all the data files, we need to merge them.

```python
data = pd.concat([data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14])
data = data.reset_index(drop=True)
```

## Remove null values and duplicates

There may be some null values and duplicate articles which are useless, so we need to remove them.

```python
data = data.dropna(how='any')
data = data.drop_duplicates(subset=['text'])
```

## Remove useless symbols and letters

In some articles, there are URLs and users’ names that are also useless for sentiment analysis, and we use Regex to remove them. 

```python
import re
remove_handles = lambda x: re.sub('@[^\s]+','', x) #remove strings like @abc
remove_urls = lambda x: re.sub('http[^\s]+','', x) #remove strings like http:
remove_hashtags = lambda x: re.sub('#[^\s]*','',x) #remove strings like #abc
remove_num = lambda x: re.sub(r'[0-9]+','',x)  #remove numbers
data['text']= data['text'].apply(remove_handles)
data['text'] = data['text'].apply(remove_urls)
data['text'] = data['text'].apply(remove_hashtags)
data['text'] = data['text'].apply(remove_num)
```

## Remove stop words

In natural language processing, useless words (data) are called stop words. Stop words are common words (e.g. "the", "a", "an", "in") that search engines have programmed to ignore.
We don't want these words to take up space in our database or take up valuable processing time. To do this, we can easily delete them by storing a list of words that we want to stop using. NLTK (Natural Language Toolkit) in python has a list of stop words stored in 16 different languages. They can be found in the nltk_data directory

```python
#remove stopwords and lower letters
from nltk.corpus import stopwords

def preprocess (doc) :
return [w for w in doc.lower().split() if w not in stopwords.words('english')]

data['text'] = data['text'].apply(preprocess)

data['text']=data['text'].apply(lambda x:' '.join(x))
```










