# import nltk
# #from nltk.stem.porter import PorterStemmer
# #stemmer = PorterStemmer()
# from nltk.stem.snowball import SnowballStemmer
# stemmer = SnowballStemmer("italian")
#
# def tokenize(sentence):
#     """
#     split sentence into array of words/tokens
#     a token can be a word or punctuation character, or number
#     """
#     return nltk.word_tokenize(sentence)
#
#
# def stem(word):
#     """
#     stemming = find the root form of the word
#     examples:
#     words = ["organize", "organizes", "organizing"]
#     words = [stem(w) for w in words]
#     -> ["organ", "organ", "organ"]
#     """
#     return stemmer.stem(word.lower())
#
# words = ["giocare", "Giocando", "giocoso", "allineatori", "Attachments"]
# prova = [stem(w) for w in words]
# print(prova)

###### QUESTO FA COSE
import os
from spellchecker import SpellChecker

# get the current path
current_dir = os.getcwd()
# get the path of the father folder
parent_dir = os.path.dirname(current_dir)
# set the wd to the father's folder
os.chdir(parent_dir)

path = f'{os.getcwd()}/Files/final_data.json'
spell = SpellChecker(language=None)
spell.word_frequency.load_dictionary(path)
new_path = f'{os.getcwd()}/Files/italian.gz'
# spell.export(new_path, gzipped=True)

word = "ciae"

if spell.correction(word) != word:
    print(f'{word} è scritto in modo sbagliato, il corretto è {spell.correction(word)}')
else:
    print(f'{word} è scritto correttamente')


# from collections import defaultdict
# import nltk
# nltk.download('punkt')
#
# remove = ["u", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"]
#
# def process_chunk(chunk, word_counts):
#     chunk = chunk.replace("'", " ")
#     for word in nltk.word_tokenize(chunk.lower()):
#         if word.isalnum() and word not in remove:
#             word = word.strip().lower()
#             word_counts[word] += 1
#     print(word_counts)
#
# word_counts = defaultdict(int)
# stringa = "Fu tuttavia la forma ciao a fare fortuna e nel secolo successivo si diffuse in tutta la Penisola.[3][4].Un'etimologia analoga ha il saluto informale servus diffuso nell'Europa centrale. l'altro città,via città"
# process_chunk(stringa, word_counts)



