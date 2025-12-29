import re
import contractions
import numpy as np
import torch
import threading
import queue



def load_glove_embeddings(glove_file_path="glove.6B.300d.txt"):
    """Loads GloVe embeddings into a dictionary.
    
    Args:
        glove_file_path (str): path to the GloVe file (default: "glove.6B.300d.txt")
                
    Returns:
        glove_embeddings (Dict[str, np.array]): dictionary containing word embeddings
    """
    glove_embeddings = {}
    
    with open(glove_file_path, "r", encoding="utf-8") as f:
        # Each line containing a word and its embedding
        for line in f:
            values = line.strip().split()
            glove_embeddings[values[0]] = np.array(values[1:], dtype=np.float32)

    return glove_embeddings

def load_glove_from_numpy(npy_file = 'glove_embeddings.npy', vocab_file='glove_vocab.txt'):
    """ 
    Load GloVe embeddings from NumPy file and vocabulary file.
    
    Args:
        npy_file (str): the path to the NumPy file containing GloVe embeddings
        vocab_file (str): the path to the vocabulary file
    
    Returns:
        word_to_index (Dict[str, int]): the dictionary mapping words to indices in the embeddings array
        embeddings (np.array): the GloVe embeddings
    """
    embeddings = np.load(npy_file)
    
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab = f.read().splitlines()
    
    word_to_index = {word: i for i, word in enumerate(vocab)}
    return word_to_index, embeddings

def _get_word_embedding(word, word_to_index=None, glove_embeddings=None):
    """
    Retrieve the GloVe embedding for a given word.

    Parameters:
    - word (str): The word to look up.
    - word_to_index (dict): Dictionary mapping words to indices in the embeddings array.
    - glove_embeddings (numpy.ndarray): Preloaded NumPy array of GloVe embeddings.

    Returns:
    - numpy.ndarray: The word's embedding vector, or a zero vector if the word is not found.
    """
    if word_to_index is None or glove_embeddings is None:
        word_to_index, glove_embeddings = load_glove_from_numpy('glove_embeddings.npy', 'glove_vocab.txt')

    idx = word_to_index.get(word)
    if idx is not None:
        return glove_embeddings[idx]
    else:
        return np.zeros(len(glove_embeddings[0]), dtype=np.float32)

def _get_word_embedding_from_txt(word, embedding_dict):
    """
    Get the word embedding of the given word
    
    Args:
        word (str): the word to get the embedding for
        embedding_dict (Dict[str, np.array]): the dictionary containing word embeddings
        
    Returns:
        embedding (np.array): the word embedding of the given word
    """
    if word in embedding_dict:
        return embedding_dict[word]
    else:
        return np.zeros(len(embedding_dict["the"]))
    
def _get_word_emb_in_sent(sentence, word_to_index=None, glove_embeddings=None):
    """ 
    Get the word embeddings of the words in the given sentence
    
    Args:
        sentence (str): the input sentence
        word_to_index (Dict[str, int]): the dictionary mapping words to indices in the embeddings array
        glove_embeddings (np.array): the GloVe embeddings
    
    Returns:
        embeddings (List[np.array]): the word embeddings of the words in the sentence
    """
    if word_to_index is None or glove_embeddings is None:
        word_to_index, glove_embeddings = load_glove_from_numpy('glove_embeddings.npy', 'glove_vocab.txt')
        
    words = word_tokenize(sentence)
    embeddings = [_get_word_embedding(word, word_to_index, glove_embeddings) for word in words]
    
    return embeddings, words

def extract_sentences(text):
    """ 
    Extract the sentences from the given text
    
    Args:
        text (str): the input text
    
    Returns:
        sentences (List[str]): the list of sentences in the text
    """
    ABBREVIATIONS = {"mr.", "mrs.", "ms.", "dr.", "prof.", "inc.", "vs.", "etc.", "jr.", "sr.", "st.", "co.", "ltd.", "gov.", "fig.", "ed.", "est."}
    
    sentences = []
    sentence = []
    
    # Spliting based on `.`, `!`, `?`
    splits = re.split(r'([.!?])', text)

    for i in range(0, len(splits) - 1, 2):  
        part, delimiter = splits[i].strip(), splits[i + 1]

        if not part:
            continue

        # Checking if the last word in the sentence is an abbreviation
        words = part.lower().split()
        
        # If the last word is an abbreviation, append it to the next part
        if words and words[-1]+delimiter in ABBREVIATIONS:
            sentence.append(part + delimiter)
        else:
            sentence.append(part + delimiter)
            sentences.append(" ".join(sentence).strip())
            sentence = []

    # any remaining sentence
    if sentence:
        sentences.append(" ".join(sentence).strip())

    return sentences
    
def word_tokenize(text):
    """ 
    Tokenize the given text into words
    
    Args:
        text (str): the input text
        
    Returns:
        tokens (List[str]): the list of tokens in the text
    """
    # Expanding the contractions in the text ("it's" -> "it is", "you're" -> "you are")
    text = contractions.fix(text)

    text = text.lower()
    # Spliting the words while keeping punctuation meaningful
    tokens = re.findall(r"\b\w+['’]?\w*|\S", text)

    return tokens

def word_frequencies(text):
    """ 
    Get the frequency of each word in the given text
    
    Args:
        text (str): the input text
    
    Returns:
        frequencies (Dict[str, int]): the frequency of each word in the text
    """
    
    # Tokenizing the text into words
    tokens = word_tokenize(text)
    frequencies = {}
    
    # Counting the frequency of each word
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1
        
    # Normalizing the frequencies
    sum_frequencies = sum(frequencies.values())
    frequencies = {token: freq/sum_frequencies for token, freq in frequencies.items()}
    
    return frequencies

def _compute_svd(X, n_components=1):
    """Compute Singular Value Decomposition (SVD) and extract the first singular vector.
    
    Args:
        X (List[List[float]]): The input matrix
        n_components (int): The number of components to extract (default: 1)
        
    Returns:
        np.array: The first singular vector
    """
    if X.shape[0] == 1:  # If there's only one vector, return a zero principal component
        return np.zeros((X.shape[1], n_components))

    X_mean = np.mean(X, axis=0) 
    X_centered = X - X_mean  # Center data

    # Covariance matrix
    covariance_matrix = np.dot(X_centered.T, X_centered) / (X.shape[0] - 1)

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sorting eigenvectors by largest eigenvalue
    sorted_indices = np.argsort(eigenvalues)[::-1]
    principal_components = eigenvectors[:, sorted_indices]

    return principal_components[:, :n_components]  # First principal component

def _sif_embedding(word_embeddings, words, word_freq):
    """Compute the SIF embedding for a sentence.
    
    Args:
        sentence (str): The input sentence
        word_freq (Dict[str, int]): The frequency of each word in the text
        embedding_dict (Dict[str, np.array]): The dictionary containing word embeddings
    
    Returns:
        torch.Tensor: The SIF embedding of the sentence
    """
    a = 1e-3  # Hyperparameter for SIF weights

    # SIF weights
    weights = np.array([a / (a + word_freq.get(word, 1e-5)) for word in words])
    weighted_embeddings = word_embeddings * weights[:, np.newaxis]
    sentence_embedding = np.mean(weighted_embeddings, axis=0)

    # Compute first singular vector
    u = _compute_svd(np.array(word_embeddings).reshape(1, -1))  # First singular vector

    # Ensure u is a valid 1D vector
    if not any(u):  # If SVD failed or vector is empty, return sentence embedding as is
        sif_embedding = sentence_embedding
    else:
        # Correct singular vector subtraction (element-wise)
        sif_embedding = sentence_embedding - np.dot(np.dot(sentence_embedding, u), u)

        
    return torch.tensor(sif_embedding, dtype=torch.float32) 

def get_one_sentence_embedding(sentence, word_freq=None, word_to_index=None, glove_embeddings=None):
    """Computes the SIF embedding for a sentence.
    
    Args:
        sentence (str): The input sentence
        word_freq (Dict[str, int]): The frequency of each word in the text
        word_to_index (Dict[str, int]): Dictionary mapping words to indices in the embeddings array
        glove_embeddings (numpy.ndarray): Preloaded NumPy array of GloVe embeddings
        
    Returns:
        torch.Tensor: The SIF embedding of the sentence
    """    
    if word_freq is None:
        word_freq = word_frequencies(sentence)
        
    # if emebedding_dict is not provided, load the GloVe embeddingss
    if glove_embeddings is None:
        word_to_index, glove_embeddings = load_glove_from_numpy('glove_embeddings.npy', 'glove_vocab.txt')
        
    word_embeddings, words = _get_word_emb_in_sent(sentence, word_to_index, glove_embeddings)
    if len(word_embeddings) == 0:
        return torch.zeros(len(glove_embeddings[0]))  # Returning zero array if no words found

    return _sif_embedding(word_embeddings, words, word_freq)

def get_text_embeddings(text, word_to_index=None, glove_embeddings=None):
    """ 
    Get the embeddings for the sentences in the given text using multithreading.
    
    Args:
        text (str): the input text
        embedding_dict (Dict[str, np.array]): the dictionary containing word embeddings
        
    Returns:
        embeddings (torch.Tensor): the embeddings of the sentences in the text
    """
    
    # If embedding_dict is not provided, load the GloVe embeddings
    if glove_embeddings is None:
        word_to_index, glove_embeddings = load_glove_from_numpy('glove_embeddings.npy', 'glove_vocab.txt')
    
    # Extracting the sentences and word frequencies from the text
    sentences = extract_sentences(text)
    word_freqs = word_frequencies(text)
    
    result_queue = queue.Queue()
    
    def worker(sentence, index):
        embedding = get_one_sentence_embedding(sentence, word_freqs, word_to_index, glove_embeddings)
        result_queue.put((index, embedding))

    # threads
    threads = []
    for i, sentence in enumerate(sentences):
        thread = threading.Thread(target=worker, args=(sentence, i))
        threads.append(thread)
        thread.start()
        
        if len(threads) >= 4:  # Limit to 4 concurrent threads
            for thread in threads:
                thread.join()
            threads.clear()

    # Waiting for all threads to complete
    for thread in threads:
        thread.join()

    # Correcting the order
    embeddings = [None] * len(sentences)
    while not result_queue.empty():
        index, embedding = result_queue.get()
        embeddings[index] = embedding

    return embeddings

def compute_mahalanobis_distance(sentence_embedding, paragraph_embeddings):
    """Computes the Mahalanobis distance between a sentence and a paragraph.
    
    Args:
        sentence_embedding: The SIF embedding of the sentence (NumPy array or Tensor).
        paragraph_embeddings: The matrix of sentence embeddings in the paragraph (list or NumPy array).
        
    Returns:
        float: The Mahalanobis distance.
    """
    paragraph_embeddings = np.array(paragraph_embeddings)

    if paragraph_embeddings.shape[0] == 0:
        return float('inf')
    
    if paragraph_embeddings.shape[0] == 1:  
        # If there's only one sentence in the paragraph, use Euclidean distance instead
        return np.linalg.norm(sentence_embedding - paragraph_embeddings[0], ord=2)

    X = np.array(paragraph_embeddings)
    x = np.array(sentence_embedding)

    mean_vector = np.mean(X, axis=0)
    
    if X.shape[0] <= X.shape[1]:  # If fewer samples than features, covariance might be singular
        cov_matrix = np.cov(X, rowvar=False, bias=True)
    else:
        cov_matrix = np.cov(X, rowvar=False)

    cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
    cov_inv = np.linalg.pinv(cov_matrix)

    # Compute Mahalanobis distance
    diff = x - mean_vector
    mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, cov_inv), diff.T))

    return mahalanobis_distance


def cosine_similarity(vec1, vec2):
    """ 
    Compute cosine similarity between two vectors.
    
    Args:
        vec1 (torch.Tensor): First vector.
        vec2 (torch.Tensor): Second vector.
    
    Returns:
        float: Cosine similarity score.
    """
    
    if not isinstance(vec1, np.ndarray):
        vec1 = np.array(vec1)
    if not isinstance(vec2, np.ndarray):
        vec2 = np.array(vec2)
        
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0  # Avoid division by zero
    
    return dot_product / (norm1 * norm2)
