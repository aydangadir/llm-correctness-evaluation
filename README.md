# README

## Introduction
This project presents how a system processes the raw outputs of large language models, to return the correctness of the raw outputs of Large Language models (LLMs). For this I have trained two models: one for Named Entity Recognition (NER) using a BiLSTM-CRF architecture, and another for extractive question answering (QA) using DistilBERT. First, the system identifies key entities in the text and retrieves relevant Wikipedia urls and scrapes its content using web scraping techniques, and then it splits the Wikipedia content into smaller chunks and retrieves the particular chunk that potentially has the factual answer to the question. Then, the QA model extracts the answer from a raw response of the LLM and then the program checks whether the raw answer (yes/no or an entity) is correct by comparing it with the ground truth.

All the details can be found in the <a href=https://github.com/aydangadir/llm-correctness-evaluation/blob/main/Web_Data_Processing_Systems%20Report.pdf>pdf file</a>.

## Instructions to Run the Code

### 1. Place the Training Model States
Ensure that the pre-trained model states are stored in this folder before running the code.
See this <a href="https://vunl-my.sharepoint.com/:f:/g/personal/a_gadirzada_student_vu_nl/EgU3KBJwmz5Gh6anfOMz2AUBNHnwjs2opg97d-8oSObYGA?e=WjYXWG">url</a>.

### 2. Specify the Input File
Update the file path of `input.txt` at the very end of the `main.py` file.

### 3. Install Requirements
Install the necessary dependencies using the following command:
```bash
pip3 install -r requirements.txt
```

### 4. Running the main.py 
```bash
python3 main.py
```

This will generate output.txt in the required format.
