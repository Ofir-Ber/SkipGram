# SkipGram

The goal is to implement skip-gram with negative-sampling from scratch and do not rely on any libraries beyond numpy, scikit-learn, spacy. 


Skip-gram Word2Vec is an architecture for computing word embeddings. Instead of using surrounding words to predict the center word, as with CBow Word2Vec, Skip-gram Word2Vec uses the central word to predict the surrounding words.

The skip-gram objective function sums the log probabilities of the surrounding  words to the left and right of the target word  to produce the following objective:

![image](https://user-images.githubusercontent.com/98849886/178115151-698fd8b2-b86d-487d-856e-b38ca3d9aa30.png)

