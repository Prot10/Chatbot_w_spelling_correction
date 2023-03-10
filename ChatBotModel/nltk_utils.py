import numpy as np
import nltk
import os
from spellchecker import SpellChecker
# nltk.download('punkt')


stemmer = nltk.stem.snowball.ItalianStemmer()


def tokenize1(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)


def tokenize2(sentence):
    """
    split sentence into array of words/tokens
    a token can be a word or punctuation character, or number
    """
    path = 'Files/final_data.json'
    spell = SpellChecker(language=None)
    spell.word_frequency.load_dictionary(path)
    tokens = nltk.word_tokenize(sentence)
    return [spell.correction(token) for token in tokens]


def stem(word):
    """
    stemming = find the root form of the word
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1
    return bag

