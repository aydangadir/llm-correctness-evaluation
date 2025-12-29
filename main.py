import torch
import torch.nn as nn
import torchcrf
import pickle
import requests
from bs4 import BeautifulSoup
import re
import SentEmb
from nltk.stem import WordNetLemmatizer
import numpy as np
import time
import threading
import queue
from transformers import DistilBertTokenizerFast, DistilBertModel
from llama_cpp import Llama
import nltk

nltk.download('wordnet')

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

STOPWORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'but', 'by', 'for', 'if', 'in', 'into', 
    'is', 'it', 'no', 'not', 'of', 'on', 'or', 'such', 'that', 'the', 'their', 'then', 
    'there', 'these', 'they', 'this', 'to', 'was', 'will', 'with'
}

def get_raw_txt(question):
      model_path = "models/llama-2-7b.Q4_K_M.gguf"

      llm = Llama(model_path=model_path, verbose=False)
      output = llm(
            question, # Prompt
            max_tokens=128, # Generate up to 32 tokens
            stop=["Q:", "\n"], # Stop generating just before the model would generate a new question
            echo=True # Echo the prompt back in the output
      )
      return output['choices'][0]['text']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model class for Named Entity Recognition
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim=100, hidden_dim=128, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=2, 
                            bidirectional=True, batch_first=True, dropout=dropout)
        
        self.fc = nn.Linear(hidden_dim, label_size)
        self.crf = torchcrf.CRF(label_size, batch_first=True)

    def forward(self, x, mask):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        emissions = self.fc(x)
        return emissions, mask
    
    def loss(self, x, tags, mask):
        emissions, mask = self.forward(x, mask)
        tags = torch.where(tags == -100, torch.tensor(0, device=tags.device), tags)

        return -self.crf(emissions, tags, mask=mask, reduction='mean')

    def predict(self, x, mask):
        emissions, mask = self.forward(x, mask)
        return self.crf.decode(emissions, mask)

class BLANC(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", freeze_bert=True, dropout_rate=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size  

        self.dropout = nn.Dropout(dropout_rate)

        # Answer-span prediction heads (start & end positions)
        self.qa_start = nn.Linear(hidden_size, 1)
        self.qa_end = nn.Linear(hidden_size, 1)

        # Context prediction head (binary classification per token)
        self.context_head = nn.Linear(hidden_size, 1)

        # Optionally freezing the BERT's weights (but did not use it during actually training, otherwise the performance was poor)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False  

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)  # Applying dropout

        # Predicting start & end positions
        start_logits = self.qa_start(sequence_output).squeeze(-1)
        end_logits = self.qa_end(sequence_output).squeeze(-1)

        # Predicting context word probabilities (sigmoid activation applied later)
        context_logits = self.context_head(sequence_output).squeeze(-1)

        return start_logits, end_logits, context_logits

def load_ner_model():
    """ 
    Load the Named Entity Recognition (NER) model and vocabulary.
    
    Returns:
        model (BiLSTM_CRF): The loaded NER model
        word_vocab (Dict[str, int]): Word-to-index mapping
        id2label (Dict[int, str]): Index-to-label mapping
    """
    model_path = "bilstm_crf_ner_few-nerd10.pth"
    vocab_path = "vocab_few-nerd10.pkl"

    # Loading the vocabulary
    with open(vocab_path, "rb") as f:
        vocab_data = pickle.load(f)

    word_vocab = vocab_data["word_vocab"]
    id2label = vocab_data["label_vocab"]

    # Load the model
    model = BiLSTM_CRF(vocab_size=len(word_vocab), label_size=len(id2label)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model, word_vocab, id2label

def load_qa_model():
    """ 
    Load the Extractive Question Answering (QA) model.
    
    Returns:
        model (BLANC): The loaded QA model
    """
    model_path = "qa_squad2_final.pth"
    
    # Load the model
    model = BLANC()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model

def predict_ner(sentence, max_length=128):
    """ 
    Predict NER entities in a sentence using a trained model.
    
    Args:
        sentence (str): Input sentence
        max_length (int): Maximum length of the input sequence
        
    Returns:
        entities (List[Tuple[str, str]]): List of tuples containing the entity text and the entity type
    """
        
    model, word_vocab, id2label = load_ner_model()
    
    tokens = SentEmb.word_tokenize(sentence)
    input_ids = [word_vocab.get(token, word_vocab["<UNK>"]) for token in tokens]
    
    # Padding or truncating the input sequence
    if len(input_ids) < max_length:
        input_ids += [word_vocab["<PAD>"]] * (max_length - len(input_ids))
    else:
        input_ids = input_ids[:max_length]
    
    input_tensor = torch.tensor([input_ids])
    
    mask = (input_tensor != word_vocab["<PAD>"]).long()
    
    # Predicting the entities
    with torch.no_grad():
        output = model(input_tensor, mask)
    
    if isinstance(output, tuple):
        output = output[0]  # Extracting logits from tuple
    
    # Converting the predicted labels to entity text
    predicted_labels = torch.argmax(output, dim=2).squeeze(0).tolist()
    entity_predictions = [id2label[label] for label in predicted_labels[:len(tokens)]]
    
    # Merging consecutive entities of the same type
    merged_entities = []
    current_entity = None
    for token, entity in zip(tokens, entity_predictions):
        if entity != 'O':
            if current_entity and current_entity[1] == entity:
                current_entity[0] += " " + token
            else:
                if current_entity:
                    merged_entities.append(tuple(current_entity))
                current_entity = [token, entity]
        else:
            if current_entity:
                merged_entities.append(tuple(current_entity))
                current_entity = None
    if current_entity:
        merged_entities.append(tuple(current_entity))
    
    return merged_entities


def _check_disambiguation(soup, entity):
    """Check if the Wikipedia page is a disambiguation page.
    
    Args:
        soup (BeautifulSoup): Parsed Wikipedia page
        entity (str): Entity name
    
    Returns:
        is_disambiguation (bool): True if the page is a disambiguation page
        matching_links (List[Tuple[str, str]]): Relevant links found on the page
    """
    
    if not isinstance(soup, BeautifulSoup):  # Ensuring that the given soup is not a string
        return False, None

    # Checking if the page is a disambiguation page
    page_text = soup.get_text().lower()
    is_disambiguation = "disambiguation page" in page_text

    # Finding all the links on the page
    links = soup.find_all('a', href=True)
    matching_links = []

    # Iterating through the links to find the links to different wikipedia with same entity name
    for link in links:
        link_text = link.text.strip()
        href = link['href']

        # Wikipedia internal links and checking if the entity name is in the link text
        if href.startswith("/wiki/") and any(word in link_text for word in entity.split()) and "Special:PrefixIndex" not in href and "Disambiguation" not in href:
            full_link = f"https://en.wikipedia.org{href}"
            matching_links.append((link_text, full_link))

    # If the page is a disambiguation or multiple relevant links are found, return them
    if is_disambiguation and matching_links:
        return True, matching_links
    else:
        return False, None

def _get_page(url):
    """Fetch the Wikipedia page and return the parsed HTML content.
    
    Args:
        url (str): URL of the Wikipedia page
        
    Returns:
        Two cases:
            - page_soup (BeautifulSoup): Parsed Wikipedia page (If fectched succesfully and if the page exists)
            - error (str): Error message 
    """
    
    # Fetching the page
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    # Fecthing unsuccesful
    if response.status_code != 200:
        return f"Error: Unable to fetch the page (Status Code: {response.status_code})"

    page_soup = BeautifulSoup(response.text, 'html.parser')
    
    # Checking if the page exists    
    if "wikipedia does not have an article with this exact name" not in page_soup.text.lower():
        return page_soup
    else:
        return "Error: Wikipedia page does not exist."
    
def get_wikipedia_links(entity):
    """Get the Wikipedia page links for the given entity.

    Args:
        entity (str): Entity name

    Returns:
        List[Tuple[str, str]]: List of tuples containing the entity name and the Wikipedia page URL
    """
    entity = entity.capitalize()
    entity_ = entity.replace(" ", "_")  # Format entity for Wikipedia URL
    
    # Checking two links 
    url_entity = f"https://en.wikipedia.org/wiki/{entity_}" # The entity's url
    url_entity_disambiguation = f"https://en.wikipedia.org/wiki/{entity_}_(disambiguation)" # The entity's disambiguation url

    # Firstly checking the disambiguation page
    url_entity_disambiguation_soup = _get_page(url_entity_disambiguation)
    is_disambiguation = None
    matching_links = None

    if isinstance(url_entity_disambiguation_soup, BeautifulSoup):  # If is returned as a soup and not a string (which is an error message in this case)
        is_disambiguation, matching_links = _check_disambiguation(url_entity_disambiguation_soup, entity)
        
    if not is_disambiguation: # If the assumed disambiguation page does not exist or the page is not a disambiguation page
        url_entity_soup = _get_page(url_entity)
        if isinstance(url_entity_soup, str):  # In case of an error ruturn None
            return [(entity, None)]
        
        # Checking for disambiguation on the main page as well, because in some pages (for example James Craig), the disambiguation link is itself the main page
        is_disambiguation, matching_links = _check_disambiguation(url_entity_soup, entity)

    if not is_disambiguation:
        return [(entity, url_entity)] 
    else:
        return matching_links
    
def get_wikipedia_summary(url):
    """ 
    Fetch the Wikipedia page content and extract the summary.
    The summary is defined as the first section of the page content.
    
    Args:
        - url (str): URL of the Wikipedia page
    
    Returns:
        - str (str): Summary of the Wikipedia page
    """
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Unable to fetch the page (Status Code: {response.status_code})"

    # Parsing the main content in wikipedia page
    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find("h1", {"class": "firstHeading"}).text
    soup = soup.find("main", {"class": "mw-body"})
    
    # Only getting the content that is until the h2 (soo the first section between the title and second heading would be considered as summary)
    soup_str = str(soup)
    h2_match = re.search(r'<h2', soup_str)
    if h2_match:
        soup = BeautifulSoup(soup_str[:h2_match.start()], 'html.parser')
    
    # Extracting all the paragraphs until the second heading as the summary  
    paragraphs = soup.find_all('p')
    summary = ""
    for para in paragraphs:
        if para.text.strip():  # Ignore empty paragraphs
            summary += para.text.strip()

    return title + preprocess_text_custom(summary)

def get_wikipedia_content(url):
    """ 
    Fetch the Wikipedia page content.
    
    Args:
        url (str): URL of the Wikipedia page
    
    Returns:
        str: Content of the Wikipedia page
    """
    
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        return f"Error: Unable to fetch the page (Status Code: {response.status_code})"

    soup = BeautifulSoup(response.text, 'html.parser')
    title = soup.find("h1", {"class": "firstHeading"}).text
    content = soup.find("main", {"class": "mw-body"})
    
    paragraphs = content.find_all('p')
    content_text = ""
    for para in paragraphs:
        if para.text.strip():
            content_text += para.text.strip() + "\n\n"
    
    return title + preprocess_text_custom(content_text)

def preprocess_text_custom(text):
    """Converts text to lowercase, removes extra spaces, removes stopwords, and lemmatizes tokens.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Preprocessed text
    """
    text = text.lower()

    # Remove extra spaces and references (e.g., [1], [2], etc.)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\[\d+\]', '', text) 

    tokens = SentEmb.word_tokenize(text)

    # Removing stopwords 
    tokens = [word for word in tokens if word not in STOPWORDS]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return ' '.join(tokens)

def jaccard_similarity(text1, text2):
    """ 
    Compute Jaccard similarity between two texts.

    Args:
        text1 (str): First text (e.g., question).
        text2 (str): Second text (e.g., Wikipedia summary).

    Returns:
        float: Jaccard similarity score (between 0 and 1).
    """
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union != 0 else 0  # Avoid division by zero
    
def get_relative_wikipedia_page(question, entity):
    """ 
    Get the Wikipedia page for the given entity and extract the summary.
    
    Args:
        question (str): Question text
        entity (str): Entity name
    
    Returns:
        Tuple[str, str]: Tuple containing the entity name and the Wikipedia page content
    """
    
    links = get_wikipedia_links(entity)
    if not links:
        return None
    
    # if no disambiguation page is found, return the content of the main page
    if len(links) == 1:
        return links[0][1] if links[0][1] else None
        
    wiki_summary_embeddings = []
    for ent, url in links:
        summary = get_wikipedia_summary(url)
        wiki_summary_embeddings.append({
            "url": url,
            "summary": summary
        })

    max_priority_score = -float('inf')  # Higher is better (Jaccard similarity)
    most_similar_url = None

    for wiki_summary in wiki_summary_embeddings:
        page_url = wiki_summary["url"]
        page_summary = wiki_summary.get("summary", "")  # Get the summary text
        num_words = len(page_summary.split())  # Count words in the summary

        # Compute Jaccard similarity
        similarity = jaccard_similarity(question, page_summary)

        # Prioritization 
        priority_score = similarity 

        # Adding for the main Wikipedia page score, when the title is the same as the entity
        page_title = page_url.split("/")[-1].replace("_", " ")  
        if page_title.lower() == entity.lower():  
            priority_score += 0.5 

        # Removing some points if very short summaries (possible meaning that not much information is available)
        if num_words < 50:  
            priority_score -= 0.3

        # chosing the page with the highest priority score
        if priority_score > max_priority_score:
            max_priority_score = priority_score
            most_similar_url = page_url
        
    return most_similar_url

def get_entity_wikipedia_content(question):
    """ 
    Extract the entity from the question and fetch the Wikipedia page content.
    
    Args:
        question (str): Question text
    
    Returns:
        Dict[str, Dict]: Dictionary containing the entity name and its Wikipedia contents
    """
    
    entities = predict_ner(question)
    if not entities:
        return "Error: No entities found in the question."
    
    entities = list(set([ent[0] for ent in entities]))
    
    wikipedia_content = {}
    for ent in entities:
        wiki_url = get_relative_wikipedia_page(question, ent)
        content = get_wikipedia_content(wiki_url) if wiki_url else None
        if content:
            wikipedia_content[ent] = {
                "url": wiki_url,
                "content": content
            }

    return wikipedia_content

def chunk_sentences(sentences, chunk_size=3, overlap=1):
    """ 
    Create chunks of sentences with overlapping parts.

    Args:
        sentences (List[str]): List of sentences from the text.
        chunk_size (int): Number of sentences per chunk.
        overlap (int): Number of overlapping sentences between chunks.

    Returns:
        List[str]: List of text chunks.
    """
    chunks = []
    for i in range(0, len(sentences) - chunk_size + 1, chunk_size - overlap):
        chunk = " ".join(sentences[i:i + chunk_size])
        chunks.append(chunk)
    
    # Handle any remaining sentences
    if len(sentences) % chunk_size != 0 and len(sentences) > chunk_size:
        chunks.append(" ".join(sentences[-chunk_size:]))  # Ensure last chunk is included
    
    return chunks

def get_chunk_embeddings(text, word_to_index=None, glove_embeddings=None):
    """ 
    Compute embeddings for sentence chunks.

    Args:
        text (str): The full text.
        word_to_index (Dict[str, int]): Word-to-index mapping for embeddings.
        glove_embeddings (numpy.ndarray): Preloaded GloVe embeddings.

    Returns:
        List[torch.Tensor]: List of chunk embeddings.
        List[str]: List of chunk texts.
    """
    startime = time.time()
    sentences = SentEmb.extract_sentences(text)
    chunks = chunk_sentences(sentences)

    result_queue = queue.Queue()
    
    def worker(chunk, index):
        embedding = SentEmb.get_one_sentence_embedding(chunk, word_to_index, glove_embeddings)
        result_queue.put((index, embedding))
    
    # Creating and managing threads
    threads = []
    for i, chunk in enumerate(chunks):
        thread = threading.Thread(target=worker, args=(chunk, i))
        threads.append(thread)
        thread.start()
        
        if len(threads) >= 4:  # Limit to 4 concurrent threads
            for thread in threads:
                thread.join()
            threads.clear()
    
    # Ensure remaining threads are joined
    for thread in threads:
        thread.join()
    
    # Correcting the order
    chunk_embeddings = [None] * len(chunks)
    for i in range(len(chunks)):
        index, embedding = result_queue.get()
        chunk_embeddings[index] = embedding
    
    chunk_embeddings = np.vstack(chunk_embeddings)
    
    return chunk_embeddings, chunks

def retrieve_best_chunk(question, text, word_to_index=None, glove_embeddings=None):
    """ 
    Find the most relevant chunk based on cosine similarity using multiprocessing.
    
    Args:
        question (str): The input question.
        text (str): The full text from Wikipedia.
        word_to_index (Dict[str, int]): Word-to-index mapping for embeddings.
        glove_embeddings (numpy.ndarray): Preloaded GloVe embeddings.

    Returns:
        str: The most relevant chunk.
    """
    # Getting the embeddings for chunks
    chunk_embeddings, chunks = get_chunk_embeddings(text, word_to_index, glove_embeddings)
    question_embedding = SentEmb.get_one_sentence_embedding(question, word_to_index=word_to_index, glove_embeddings=glove_embeddings)

    similarities = [SentEmb.cosine_similarity(question_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
    # Getting the chunk with the highest similarity score
    best_chunk_index = np.argmax(similarities)
    
    return chunks[best_chunk_index]

def get_answer_from_wiki_content(question, wikipedia_content):
    """ 
    Get the answer to the question from the Wikipedia content.
    
    Args:
        question (str): The input question.
    
    Returns:
        str: The answer to the question.
    """    
    if isinstance(wikipedia_content, str):
        return retrieve_best_chunk(question, wikipedia_content)
    
    all_content = " ".join([data["content"] for data in wikipedia_content.values() if data["content"]])
    return retrieve_best_chunk(question, all_content)

def detecting_yes_no(answer):
    """ 
    Detect if the answer is a Yes/No answer.
    
    Args:
        answer (str): The answer text.
        
    Returns:
        Tuple[bool, Optional[str]]: Tuple containing a boolean indicating if the answer is a Yes/No answer and the answer type.
    """
    
    # Get the embeddings for "yes" and "no"
    answer_embedding = SentEmb.get_one_sentence_embedding(answer)
    yes_embedding = SentEmb._get_word_embedding("yes")
    no_embedding = SentEmb._get_word_embedding("no")
    
    # cosine similarity between the answer and "yes" and "no"
    similarity_yes = SentEmb.cosine_similarity(answer_embedding, yes_embedding)
    similarity_no = SentEmb.cosine_similarity(answer_embedding, no_embedding)
    
    # Checking if the similarity is greater than 0.7
    if similarity_yes > similarity_no and similarity_yes > 0.7:
        return True, "Yes"
    elif similarity_no > similarity_yes and similarity_no > 0.7:
        return True, "No"
    else:
        return False, None
    
def predict_answer(model, question, context):
    """ 
    Predict the answer span in the context given the question.
    
    Args:
        model (BLANC): The QA model.
        question (str): The input question.
        context (str): The input context.
        
    Returns:
        str: The predicted answer span.
    """
    # ensure the model is in evaluation mode
    model.eval()
    
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True).to(device) # tokenization
    start_logits, end_logits, context_logits = model(inputs["input_ids"], inputs["attention_mask"])

    # Get predicted answer span
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    # Extracting the predicted answer
    answer_tokens = inputs["input_ids"][0, start_idx:end_idx + 1]
    predicted_answer = tokenizer.decode(answer_tokens)

    return predicted_answer

def check_yes_no_similarity(question, predicted_answer, ground_truth):
    """
    Compute negation-aware similarity score.
    
    Adjust similarity based on negation presence.
    """
    negation_words = {"no", "not", "never", "none", "nobody"}
    
    def is_negative(text):
        words = SentEmb.word_tokenize(text) # lowers, contraction handling, splitting into tokens
        return any(word in negation_words for word in words)
    
    predicted_text = f"{question} {predicted_answer}"
    ground_truth_text = f"{question} {ground_truth}"
    
    pred_emb = SentEmb.get_one_sentence_embedding(predicted_text)
    gt_emb = SentEmb.get_one_sentence_embedding(ground_truth_text)
    
    similarity = SentEmb.cosine_similarity(pred_emb, gt_emb)
    
    # Adjusting similarity if one is negative and the other is not
    if is_negative(predicted_answer) != is_negative(ground_truth):
        similarity = 1 - similarity  # Flip similarity for opposite polarity
    
    return similarity

def fact_checking(question, answer, ground_truth):
    """ 
    Check if the answer to the question is correct.
    
    Args:
        question (str): The input question.
        answer (str): The predicted answer.
        
    Returns:
        str: The result of the fact-checking.
    """
    
    model = load_qa_model()
    # ground_truth = get_answer_from_wiki_content(question)
    extracted_groud_truth = SentEmb.get_one_sentence_embedding(predict_answer(model, question, ground_truth))
    
    if not ground_truth:
        return "Error: Ground truth answer not found."
    
    # Check if the answer is a Yes/No answer
    is_yes_no, answer_type = detecting_yes_no(answer)
    
    if not is_yes_no:
        # Check if the extracted answer is in the ground truth
        extracted_answer = SentEmb.get_one_sentence_embedding(predict_answer(model, question, answer))
        
        if SentEmb.cosine_similarity(extracted_answer, extracted_groud_truth) > 0.7:
            return "Correct"
        else:
            return "Incorrect"
        
    else:
        if check_yes_no_similarity(question, answer, ground_truth) > 0.7:
            return "Correct"
        else:
            return "Incorrect"

def processing_question(question):
    """ 
    Process the input question and generate answers in the required format.
    
    Args:
        question (str): Input question
        
    Returns:   
        Dict: Dictionary containing the raw response, extracted answer, correctness, and entities
    """
    
    raw_answer = get_raw_txt(question)
    entites = get_entity_wikipedia_content(question+raw_answer)
    
    entitiiiis = [(key, value["url"]) for key, value in entites.items()]
    
    extractive_qa = load_qa_model()
    answer = predict_answer(extractive_qa, question, raw_answer)
    
    ground_truth = get_answer_from_wiki_content(question, entites)
    correctness = fact_checking(question, answer, ground_truth)
    
    response = {
        "raw_response_from_llm": raw_answer,
        "entites": entitiiiis,
        "extracted_answer": answer,
        "correctness": correctness,
    }
    
    return response

def process_questions(input_file, output_file):
    """ 
    Process the input file containing questions and generate answers in the required format.
    
    Args:
        input_file (str): Path to the input file.
        output_file (str): Path to the output file.
        
    Returns:
        None
    """
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            print(line)
            line = line.strip()
            if not line:
                continue
            
            # question ID and question text
            parts = line.split('\t', 1)
            if len(parts) != 2:
                continue
            
            question_id, question = parts

            response = processing_question(question)
            
            # Extracting the response fields
            raw_response = response.get('raw_response_from_llm', '')
            extracted_answer = response.get('extracted_answer', '')
            correctness = response.get('correctness', '')
            entities = response.get('entites', [])
            
            # Writing according to the ourput format
            outfile.write(f'{question_id}\tR"{raw_response}"\n')
            outfile.write(f'{question_id}\tA"{extracted_answer}"\n')
            outfile.write(f'{question_id}\tC"{correctness.lower()}"\n')
            
            for entity, link in entities:
                outfile.write(f'{question_id}\tE"{str(entity).capitalize()}"\t"{link}"\n')

provided_input = "input.txt"
output_file = "output.txt"
process_questions(provided_input, output_file)