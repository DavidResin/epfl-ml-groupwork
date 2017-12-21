# Text Sentiment Classification

Twitter and other social medias can be good indicators of a population’s emotions, making so that tweets - due to their format of short texts - are a good dataset to try sentiment analysis techniques upon. Thus, our project focused on finding a combination of Machine Learning techniques to be able to predict what a user’s sentiment is just from the text.
To test these out we were provided with a dataset of tweets which had previously happy or sad smileys.
Our approach to solve this problem was to first clean up the data using usual NLP used in data science. Then, we implemented several methods to create predictions that would be as accurate as possible. Thus, we tried to apply Fast Text, a Random Tree Classifier, TF-IDF using two different classifiers and a Convolutional Neural Network.
As a result, we found that Fast Text and CNN are the best prediction techniques for our dataset.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

You will need to have several libraries installed, for example pandas and numpy should be installed on your computer. Other libraries include sklearn, fasttext, tensorflow, pickle and keras.
You will also need Jupyter Notebook.

```
pip install tensorflow
pip install keras
```

To install Fast Text, you will need to follow the steps described here:

```
https://github.com/facebookresearch/fastText
```

You will also need to download the nltk library at least once by using the

```
nltk.download()
```

command. Note that this command can be found in the process_tweets notebook.

### Installing

Once you have cloned the code, you will see that a few files are missing. This is because a lot of the files are too big to be directly on the git. Those file are the full positive and negative tweets that were given with the project.

```
\twitter-datasets\train_neg_full.txt
\twitter-datasets\train_pos_full.txt
```

You should also download this zip, as it contains the glove embeddings used in CNN and Random Trees. This file should be put in a new glove folder.

```
https://nlp.stanford.edu/data/glove.twitter.27B.zip
```

 This file should be put in a new glove folder.

```
\project2\glove\glove.twitter.27B.200d.txt
```

After this, process_tweets notebook should always be run first as it is where the data is created or processed, unless specified otherwise.

## Running the code

You should run the notebooks in the following order:

```
process_tweets.ipynb
ml_tf_idf.ipynb
ml_fasttext.ipynb
CNN_notebook.ipynb
ml_random_forest.ipynb
```

You should run the notebooks in the following order:

```
process_tweets.ipynb
ml_tf_idf.ipynb
ml_fasttext.ipynb
CNN_notebook.ipynb
ml_random_forest.ipynb
```

If you only want to test out the best submission, you should run:
```
Keras/run.py
```
This script might take a long time to run depending on your computation power.
