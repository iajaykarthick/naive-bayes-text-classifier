import gradio as gr
import string
import re
import pickle
import huggingface_hub

import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords



def clean_review(review):
    review = review.lower()
    review = re.sub(r"http\S+|www.\S+", "", review)
    review = re.sub(r"<[^>]*>", "", review)
    review = review.replace(".", " ")

    review = "".join([c for c in review if c not in string.punctuation])
    review = " ".join([word for word in re.split('\W+', review) 
                               if word not in stopwords.words('english')])
    wn = nltk.WordNetLemmatizer()    
    review = " ".join([wn.lemmatize(word, 'r') for word in re.split('\W+', review)])

    return review

def find_occurrence(frequency, word, label):
    n = 0 
    if (word, label) in frequency:
        n = frequency[(word, label)]
    
    return n

def classify_text(freqs, logprior, text):
    loglikelihood = {}
    p_w_pos = {}
    p_w_neg = {}

    # calculate V, the number of unique words in the vocabulary
    vocab = set([word for word, label in freqs.keys()])
    V = len(vocab)

    #calculate num_pos and num_neg - the total number of positive and negative words for all documents
    num_pos = num_neg = 0
    for word, label in freqs.keys():
        # if the label is positive (greater than zero)
        if label > 0:
            
            # Increment the number of positive words by the count for this (word, label) pair
            num_pos += freqs[(word, label)]

        # else, the label is negative
        else:
            
            # increment the number of negative words by the count for this (word,label) pair
            num_neg += freqs[(word, label)]

            
            
    # process the review to get a list of words
    word_l = clean_review(text).split()
    
    # initialize probability to zero
    total_prob = 0

    # add the logprior
    total_prob += logprior
    
    # For each word in the vocabulary...
    for word in word_l:
        # get the positive and negative frequency of the word
        freq_pos = find_occurrence(freqs, word, 1)
        freq_neg = find_occurrence(freqs, word, 0)

        # calculate the probability that each word is positive, and negative
        p_w_pos[word] = (freq_pos + 1) / (num_pos + V)
        p_w_neg[word] = (freq_neg + 1) / (num_neg + V)
        
        if freq_pos + freq_neg > 0:
            # calculate the log likelihood of the word
            loglikelihood[word] = np.log(p_w_pos[word] / p_w_neg[word])
            # add the log likelihood of that word to the probability
            total_prob += loglikelihood[word]
        else:
            loglikelihood[word] = ''
        
    if total_prob > 0:
        total_prob = 1
    else:
        total_prob = 0

    return total_prob
    
model_path = huggingface_hub.hf_hub_download("ajaykarthick/naive-bayes-review-classify-model", "naive-bayes-text-classifier-model")

model_params = pickle.load(open(model_path, mode='rb'))
freqs = model_params['freqs_dict']
logprior = model_params['logprior']


def greet(name):
    total_prob = classify_text(freqs, logprior, name)
    print(name, str(total_prob))
    return 'POSITIVE' if total_prob == 0 else 'NEGATIVE'

iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch(share=True)