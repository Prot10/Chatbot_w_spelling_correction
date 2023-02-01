import json
import nltk
from collections import defaultdict
import os



to_remove = ["u", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "q", "r", "s", "t", "v", "w", "x", "y", "z"]



def process_chunk(chunk, word_counts):
    for words in chunk:
        words = words.replace("'", " ")
        for word in nltk.word_tokenize(words.lower()):
            if word.isalpha() and word not in to_remove:
                word = word.strip().lower()
                word_counts[word] += 1



def chunk_reader(file, chunk_size=1024):
    chunk = []
    for line in file:
        chunk.extend(line.split())
        if len(chunk) >= chunk_size:
            yield chunk
            chunk = []
    yield chunk



def main(file_name, name):
    word_counts = defaultdict(int)
    path = f'{os.getcwd()}/Files/{file_name}'
    with open(path, 'r') as f:
        f.seek(0)
        for i, chunk in enumerate(chunk_reader(f)):
            process_chunk(chunk, word_counts)
    save_to = f'{os.getcwd()}/Files/{name}.json'
    with open(save_to, 'w') as f:
        json.dump(word_counts, f)



# get the current path
current_dir = os.getcwd()
# get the path of the father folder
parent_dir = os.path.dirname(current_dir)
# set the wd to the father's folder
os.chdir(parent_dir)



# create the dictionary from the file it.txt that contains script of films dialogues
main('it.txt', 'word_counts')

# create the dictionary from the file sorridi_website.txt that contains all the text from the sorridi website
main("sorridi_website.txt", 'sorridi_website')
