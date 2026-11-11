"""preprocessing"""
import json
import math
import os
import re
from collections import Counter, defaultdict
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import zipfile
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))


#tokenization method
def cleantext(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class preprocess:
    def __init__(self):
        #file path
        
        self.papers = {}
        self.paragraphs = {}
        
        #length of documents
        
        #average length of documents
        
        #the inverted index
    def load_json_files(self, data_path):
        # Case 1: ZIP file
        if zipfile.is_zipfile(data_path):
            with zipfile.ZipFile(data_path, 'r') as zip_file:
                for filename in zip_file.namelist():
                    if filename.endswith('.json'):
                        with zip_file.open(filename) as f:
                            data = json.load(f)

                            if isinstance(data, list):
                                for paper in data:
                                    yield paper
                            else:
                                yield data

        # Case 2: Directory containing JSON files
        elif os.path.isdir(data_path):
            for filename in os.listdir(data_path):
                if filename.endswith('.json'):
                    filepath = os.path.join(data_path, filename)

                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                        if isinstance(data, list):
                            for paper in data:
                                yield paper
                        else:
                            yield data

        # Case 3: Single JSON file
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

                if isinstance(data, list):
                    for paper in data:
                        yield paper
                else:
                    yield data    
    #loading papers 
    def load_paper(self, data_path):
        
        

        self.papers = {}
        # docid as key, dict "term: frequency" as value
        

        for doc in self.load_json_files(data_path):
            doc_id = str(doc["paper_id"])
            # For results representation
            title = cleantext(doc.get("title", ""))
            abstract = cleantext(doc.get("abstract", ""))
            text = title + " " + abstract
            self.papers[doc_id] = {
                "title": title,
                "text": text
            }
            
            

        

        print(f"indexed {len(self.papers)} papers")
        return self.papers

    #loading paragraphs
    def load_paragraph(self, data_path):
        

        
        
        self.paragraphs = {}

        for paper in self.load_json_files(data_path):
            paper_id = str(paper["paper_id"])
            for paragraph_id, text in paper.get("paragraphs", {}).items():
                paragraph_id = str(paragraph_id)
                self.paragraphs[paragraph_id] = {
                    "paper_id": paper_id,
                    "text": cleantext(text)
                }
                
                
        
        

        
        print(f"indexed {len(self.paragraphs)} paragraphs")
    
 
             

    
        
