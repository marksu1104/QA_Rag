import os
import re
import string
import json
import argparse
import pickle
import chromadb
from tqdm import tqdm
import pdfplumber
import fitz
from paddleocr import PaddleOCR
from typing import Any, Tuple, Dict, List, Optional

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger

import logging
logging.disable(logging.DEBUG)
logging.disable(logging.WARNING)


class VectorDatabase:
    def __init__(self, 
                 source_path: str = "./reference", 
                 db_path: str = "./database/chroma", 
                 docstore_path: str = "./database/docstore", 
                 pkl_path: str = "./database/raw_pkl",
                 stop_words_path: str = "./stopwords.txt"):
        """
        Initialize the VectorDatabase class with paths and models.
        """
        print("< VectorDatabase initialized >")
        self.source_path = source_path
        self.db_path = db_path
        self.docstore_path = docstore_path
        self.pkl_path = pkl_path
        self.stop_words_path = stop_words_path
        os.makedirs(self.docstore_path, exist_ok=True)
        
        self.my_embedding = HuggingFaceEmbedding(model_name="TencentBAC/Conan-embedding-v1")
        self.chroma_persist_client = chromadb.PersistentClient(db_path)
        
        try:
            self.ocr = PaddleOCR(use_angle_cls=True, lang='chinese_cht', use_gpu=True)
            print("  - PaddleOCR initialized successfully")
        except Exception as e:
            print(f"Error initializing PaddleOCR: {str(e)}")
            raise
        
        self.ws_driver = CkipWordSegmenter(model="bert-base", device=0)
        self.pos_driver = CkipPosTagger(model="albert-base", device=0)
        
        self.corpus_dict_insurance, self.corpus_dict_insurance_clean = self.load_data(
            os.path.join(self.source_path, 'insurance'),
            os.path.join(self.pkl_path, 'insurance.pkl'),
            category='insurance'
        )
            
        self.corpus_dict_finance, self.corpus_dict_finance_clean = self.load_data(
            os.path.join(self.source_path, 'finance'),
            os.path.join(self.pkl_path, 'finance.pkl'),
            category='finance'
        )
            
        self.corpus_dict_faq, self.corpus_dict_faq_clean = self.load_data(
            os.path.join(self.source_path, 'faq/pid_map_content.json'),
            os.path.join(self.pkl_path, 'faq.pkl'),
            category='faq'
        ) 

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Normalize text by removing special characters, URLs, and extra whitespace.
        """
        text = text.lower()
        text = re.sub(r'[.+?]', ' ', text)  
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        text = re.sub(r'<.+?>', ' ', text)
        text = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', text)
        text = re.sub(r'\n', '', text)
        text = re.sub(r'\w+\d\w+', ' ', text)
        text = ' '.join(text.split())  
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """
        Remove stopwords from text.
        """
        if len(text) == 0:
            return text
        splitted_text = self.ws_driver.__call__([text], show_progress=False)
        with open(self.stop_words_path, 'r', encoding='utf-8') as f:
            stopwords = {line.strip() for line in f}
        return ' '.join([word for word in splitted_text[0] if word not in stopwords])
    
    def clean(self, text: str, key: Any) -> str:
        """
        Clean text by normalizing and removing unwanted parts.
        """
        short_sentence = []
        stop_pos = set(['Nep', 'Nh'])  # Stop POS tags
        
        text = self.normalize_text(text)
        
        if len(text) == 0:
            return text
        
        sentence_ws = self.ws_driver.__call__([text], show_progress=False)
        sentence_pos = self.pos_driver.__call__(sentence_ws, show_progress=False)

        for word_ws, word_pos in zip(sentence_ws[0], sentence_pos[0]):
            is_N_or_V = word_pos.startswith("N") or word_pos.startswith("V")
            is_not_stop_pos = word_pos not in stop_pos
            is_not_one_char = len(word_ws) > 1
            if is_N_or_V and is_not_stop_pos and is_not_one_char:
                short_sentence.append(word_ws)
        return " ".join(short_sentence)
    
    def initialize_process(self, chunk_size: int = 256, chunk_overlap: int = 200):
        """
        Initialize and process data into ChromaDB.
        """
        print("  - loading data into ChromaDB ")
        
        if os.path.exists(self.db_path):
            for root, dirs, files in os.walk(self.db_path, topdown=False):
                for name in files:
                    if name != "chroma.sqlite3":
                        os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            
        existing_collections = self.chroma_persist_client.list_collections()
        existing_collections_name = [collection.name for collection in existing_collections]
        
        for category in ['faq', 'insurance', 'finance']:
            print(f"     - loading [{category}] ...")
            if category in existing_collections_name:
                self.delete_database(category)
            self.insert_database(
                getattr(self, f'corpus_dict_{category}'),
                getattr(self, f'corpus_dict_{category}_clean'),
                category,
                chunk_size,
                chunk_overlap
            )
            

    def load_data(self, category_path: str, corpus_dict_file: str, category: Optional[str] = None) -> Tuple[Dict[int, str], Dict[int, str]]:
        """
        Load data from files and process it.
        """
        corpus_dict_clean = {} 
        if os.path.exists(corpus_dict_file):
            with open(corpus_dict_file, 'rb') as file:
                corpus_dict = pickle.load(file)
        else:
            if category == 'faq':
                with open(category_path, 'rb') as f_s:
                    key_to_source_dict = json.load(f_s)
                    key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
                    corpus_dict = {key: str(value) for key, value in key_to_source_dict.items()}
            else:
                corpus_dict = {}
                masked_file_ls = os.listdir(category_path)
                new_files = [file for file in masked_file_ls if int(file.replace('.pdf', '')) not in corpus_dict]
                if new_files:
                    print(f"  - loading PDF ...")
                    for file in tqdm(new_files, desc="Loading new files"):
                        file_id = int(file.replace('.pdf', ''))
                        corpus_dict[file_id] = self.read_pdf(os.path.join(category_path, file))
                        
            with open(corpus_dict_file, 'wb') as file:
                pickle.dump(corpus_dict, file)  

        if os.path.exists(os.path.join(self.pkl_path, f'{category}_clean.pkl')):
            with open(os.path.join(self.pkl_path, f'{category}_clean.pkl'), 'rb') as file:
                corpus_dict_clean = pickle.load(file)
        else:
            corpus_dict_clean = {key: self.clean(str(value), key) for key, value in corpus_dict.items()}
            with open(os.path.join(self.pkl_path, f'{category}_clean.pkl'), 'wb') as file:
                pickle.dump(corpus_dict_clean, file)        

        return corpus_dict, corpus_dict_clean
    
    def process_image(self, image_path: str) -> str:
        """
        Process image using OCR to extract text.
        """
        try:
            result = self.ocr.ocr(image_path, cls=True)
            text = []
            for line in result:
                for word_info in line:
                    text.append(word_info[1][0])
                    
            return " ".join(text)
        except Exception as e:
            return " "

    def read_pdf(self, pdf_loc: str, page_infos: Optional[List[int]] = None) -> str:
        """
        Read and extract text from PDF.
        """
        with pdfplumber.open(pdf_loc) as pdf:
            pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
            file_text = ''.join(page.extract_text() for page in pages if page.extract_text())
            
        if len(file_text) == 0:
            text_content = []
            doc = fitz.open(pdf_loc)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_results = self.process_image(image_bytes)
                        text_content.append(f"{image_results}\n")
                    except Exception as e:
                        continue
            doc.close()
            return ' '.join(text_content)
        else:
            return file_text

    def insert_database(self, corpus_dict: Dict[int, str], corpus_dict_clean: Dict[int, str], category: str, chunk_size: int = 256, chunk_overlap: int = 200):
        """
        Insert data into ChromaDB.
        """
        documents = [
            Document(
                text=text,
                id_=f"doc_id_{id}",
                metadata={"category": category, "pid": id}
            ) for id, text in corpus_dict.items()
        ]

        chroma_collection = self.chroma_persist_client.get_or_create_collection(category)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir=os.path.join(self.db_path, category))

        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        vector_index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.my_embedding,
            storage_context=storage_context
        )

        storage_context = StorageContext.from_defaults(docstore=docstore)
        storage_context.docstore.persist(os.path.join(self.docstore_path, f'{category}.json'))
        
        documents2 = [
            Document(
                text=text,
                id_=f"doc_id_{id}",
                metadata={"category": category, "pid": id}
            ) for id, text in corpus_dict_clean.items()
        ]
        splitter2 = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes2 = splitter2.get_nodes_from_documents(documents2)
        chroma_collection2 = self.chroma_persist_client.get_or_create_collection(f'{category}_clean')
        vector_store2 = ChromaVectorStore(chroma_collection=chroma_collection2, persist_dir=os.path.join(self.db_path, category))
        storage_context2 = StorageContext.from_defaults(vector_store=vector_store2)
        vector_index_2 = VectorStoreIndex(
            nodes=nodes2,
            embed_model=self.my_embedding,
            storage_context=storage_context2
        )

        docstore2 = SimpleDocumentStore()
        docstore2.add_documents(nodes2)
        storage_context3 = StorageContext.from_defaults(docstore=docstore2)
        storage_context3.docstore.persist(os.path.join(self.docstore_path, f'{category}_clean.json'))

        return vector_index

    def load_database(self, category: str) -> VectorStoreIndex:
        """
        Load existing collection and create vector store.
        """
        chroma_collection = self.chroma_persist_client.get_or_create_collection(category)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        docstore = SimpleDocumentStore.from_persist_path(
            os.path.join(self.docstore_path, f'{category}.json')
        )
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            vector_store=vector_store
        )

        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.my_embedding
        )
    
    def load_database_clean(self, category: str) -> VectorStoreIndex:
        """
        Load existing cleaned collection and create vector store.
        """
        chroma_collection = self.chroma_persist_client.get_or_create_collection(f'{category}_clean')
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        docstore = SimpleDocumentStore.from_persist_path(
            os.path.join(self.docstore_path, f'{category}_clean.json')
        )
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            vector_store=vector_store
        )

        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.my_embedding
        )

    def delete_database(self, category: str):
        """
        Delete a collection from ChromaDB.
        """
        self.chroma_persist_client.delete_collection(category)
        print(f"        ... collection [{category}] deleted.")

    def get_vector_index(self, category: str) -> VectorStoreIndex:
        """
        Get vector index for a category.
        """
        return self.load_database(category)
    
    def get_vector_clean_index(self, category: str) -> VectorStoreIndex:
        """
        Get cleaned vector index for a category.
        """
        return self.load_database_clean(category)
    
    def get_text(self, category: str) -> Dict[int, str]:
        """
        Get text data for a category.
        """
        return getattr(self, f'corpus_dict_{category}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to read questions')
    parser.add_argument('--source_path', type=str, required=True, help='Path to read reference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output formatted answers')

    args = parser.parse_args()
    
    db = VectorDatabase(source_path=args.source_path)
    db.initialize_process()
