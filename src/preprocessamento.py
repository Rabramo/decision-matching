import re
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer

def limpar_texto(texto: str) -> str:
    t = str(texto).lower()
    t = re.sub(r"http\S+|www\.[^\s]+", " ", t)
    t = re.sub(r"\d+", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def preprocessar_textos(textos: List[str]) -> List[str]:
    return [limpar_texto(t) for t in textos]

def vetorizar_tfidf(
    textos: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> Tuple['scipy.sparse.csr_matrix', TfidfVectorizer]:
    vetorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    matriz = vetorizer.fit_transform(textos)
    return matriz, vetorizer

