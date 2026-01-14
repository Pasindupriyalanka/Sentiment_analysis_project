import numpy as np
import pandas as pd
import re
import string
from pathlib import Path
import pickle

from nltk.stem import PorterStemmer
ps = PorterStemmer()

BASE_DIR = Path(__file__).resolve().parent

def _find_existing(base: Path, candidates):
    for p in candidates:
        candidate = base / p
        if candidate.exists():
            return candidate
    return None

candidates = [
    BASE_DIR / 'static' / 'model' / 'model.pickle',
    BASE_DIR / 'static' / 'model' / 'modle.pickle',  # accept the typo
    BASE_DIR / 'static' / 'model' / 'product' / 'model.pickle'
]
model_path = next((p for p in candidates if p.exists()), None)
if model_path is None:
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Model file not found. Tried: {tried}")
with model_path.open('rb') as f:
    model = pickle.load(f)

# stopwords
sw_candidates = [
    Path('static') / 'model' / 'corpora' / 'stopwords' / 'english',
    Path('static') / 'model' / 'corpora' / 'stopwords' / 'english.txt'
]
sw_path = _find_existing(BASE_DIR, sw_candidates)
if sw_path is None:
    sw = []  # fallback; or use nltk.corpus.stopwords if available
else:
    sw = sw_path.read_text().splitlines()

# vocabulary
vocab_candidates = [
    Path('static') / 'model' / 'vocabulary.txt',
    Path('model') / 'vocabulary.txt'
]
vocab_path = _find_existing(BASE_DIR, vocab_candidates)
if vocab_path is None:
    raise FileNotFoundError(f"vocabulary.txt not found. Tried: {', '.join(str(BASE_DIR / p) for p in vocab_candidates)}")
vocab = pd.read_csv(vocab_path, header=None)
tokens = vocab[0].tolist()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

def preprocessing(text):   
     data = pd.DataFrame([text], columns=['tweet'])
     data["tweet"] = data["tweet"].apply(lambda x: " ".join(x.lower() for x in x.split()))
     data["tweet"] = data["tweet"].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*', '', x, flags=re.MULTILINE).split()))
     data["tweet"] = data["tweet"].apply(remove_punctuations)
     data["tweet"] = data["tweet"].str.replace(r'\d+', '', regex=True)
     data["tweet"] = data["tweet"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
     data["tweet"] = data["tweet"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
     return data["tweet"]

def vectorizer(ds):
    vectorized_lst = []
    
    for sentence in ds:
        sentence_lst = np.zeros(len(tokens))
        
        for i in range(len(tokens)):
            if tokens[i] in sentence.split():
                sentence_lst[i] = 1
                
        vectorized_lst.append(sentence_lst)
        
    vectorized_lst_new = np.asarray(vectorized_lst, dtype=np.float32)
    return vectorized_lst_new

def get_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    if prediction == 1:
        return 'negative'
    else:
        return 'positive'