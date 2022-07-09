# SkipGram

The goal is to implement skip-gram with negative-sampling from scratch and do not rely on any libraries beyond numpy, scikit-learn, spacy. 


Skip-gram Word2Vec is an architecture for computing word embeddings. Instead of using surrounding words to predict the center word, as with CBow Word2Vec, Skip-gram Word2Vec uses the central word to predict the surrounding words.

The skip-gram objective function sums the log probabilities of the surrounding  words to the left and right of the target word  to produce the following objective:

![image](https://user-images.githubusercontent.com/98849886/178115151-698fd8b2-b86d-487d-856e-b38ca3d9aa30.png)


# Text Preprocessing:

1. Stop words removal – Removed the stop words using the nltk corpus of english stopwords
2. Text Cleaning - First, as python is a case sensitive language, we put every word in lowercase, 
then we did some classic processing such as removing:
- Punctuation
- Digits
- Hyphens
- ‘s and ‘t
- ‘empty’ words (words composed of only one letter) 
3. Stemming - Moreover, to make our training data denser and reduce the size of the dictionary 
(number of words used in the corpus), we applied the stemming method – we used the stem
function from the nltk package to transform the words to their root form.

# Skip Gram Implementation:

• Functions -
1) Unigram distribution -
The idea behind unigram distribution was that the probability of picking a word should be equal to 
the number of times this word appears in the corpus raised to the power ¾ , divided by the total 
number of words in the corpus also raised to the power ¾ as it is done in the original paper.
Our function generates a large table that contains all the words repeated as many times as their 
distribution. We will then randomly sample the negative words from this unigram table.
2) Word frequency -
Filtering stopwords, words below a minimum count and words that are not in vocab.
3) Word to ID mapping -
Preparing Word to ID input mapping vector
4) Positive words -
Words that actually appear within the context window of the center word. A word vector of a center 
word will be more similar to a positive word than of randomly drawn negative words because words 
appearing together have strong correlation. 
5) Negative sampling
The negative sampling is performed by randomly picking word from the unigram table taking into 
account that the sampled word is neither part of the context nor the center word with the length of 
the negative word set predefined as a hyperparameter.

• Training -
1) Forward Propagation: Computing hidden (projection) layer
2) Forward Propagation: Sigmoid output layer
3) Backward Propagation: Prediction Error
4) Backward Propagation: Computing ∇ Winput
5) Backward Propagation: Computing ∇ Woutput
6) Backward Propagation: Updating Weight Matrices

• Similarity –
For the computation of the similarity, we chose the cosine similarity which is defined as the cosine of the 
angle between two sequences of numbers or the dot product of the vectors divided by the product of their 
lengths.
Given the fact that we are in a positive space, it returns a number between [0 ; 1] indicating the similarity
(the higher the more similar the words are)

# Hyper-parameter Tuning:
We performed hyperparameter tuning to find the setup to improve the accuracy 
our model. This process has been done using a trial-and-error approach based on the reported 
correlation between the computed and the actual similarity figures.
1. Learning rate
In the Skip-Gram algorithm the learning rate is a hyper-parameter that determines the step size 
of the gradient descent at each epoch while moving toward the minimum of the loss function.
Over the project, we tried different values for the learning rate ranging from 0.001 to 0.5, and 
the value that led to the best result was 0.1.
2. Embeddings
We tried to see the impact of the number of embedded words on the accuracy of the model. For 
this purpose, we tried the different values ranging from 50 to 1000.
It appears that increasing the number of embedded words has a substantial impact and 
improved the correlation of between the actual and the computed similarity up to a certain 
value. The reason is that good-sized weight matrices allow the model to better capture the 
relationships between words.
For the rest of the assignment, we decided to keep the value of 500.
3. Minimum count
The minimum count parameter defines threshold for a word to be included in the vocabulary. 
Therefore, this hyperparameter has an important impact on the computation time of our 
algorithm.
We wanted to test its effect on the accuracy of our model, to do so we tried different values 
ranging from 2 to 10.
All else being equal, the minimum word frequency that led to the best result on the test set was 3.
4. Window size
The hyperparameter window size or ‘winSize’ defines the number of tokens inside the window. 
Again, we experiment with different values ranging from 2 to 8 and we found that the optimal 
window size was 7.
5. Negative rate
The negative rate hyperparameter defines the number of words that will be picked for the 
negative sampling. 
The optimal value we found for the negative rate was 5.

