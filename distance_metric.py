from gensim.models import Word2Vec
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained model
model = Word2Vec.load("path/to/word2vec/model")

def sentence_to_vec(sentence, model):
    words = sentence.split()
    vecs = [model[word] for word in words if word in model.vocab]
    return np.mean(vecs, axis=0)

def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def compute_distance(sentence1, sentence2, model):
    vec1 = sentence_to_vec(sentence1, model)
    vec2 = sentence_to_vec(sentence2, model)
    similarity = compute_similarity(vec1, vec2)
    return 1 - similarity

sentence1 = "Your example sentence here."
sentence2 = "Another example sentence here."
distance = compute_distance(sentence1, sentence2, model)
print(f"The semantic distance between the sentences is: {distance}")
