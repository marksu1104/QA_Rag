import os
import json
import argparse
import pickle
import chromadb
from tqdm import tqdm
import pdfplumber

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext, get_response_synthesizer
from llama_index.core import DocumentSummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.indices.document_summary import DocumentSummaryIndexLLMRetriever

from typing import Any, Callable, Dict, List, Optional, Union, cast
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
import torch
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)


class VectorDatabase:
    def __init__(self, source_path="./reference", db_path="./database/chroma", docstore_path="./database/docstore", pkl_path="./database"):
        
        print("< VectorDatabase initialized > ")
        # Initialize paths
        self.source_path = source_path
        self.db_path = db_path
        self.docstore_path = docstore_path
        self.pkl_path = pkl_path
        os.makedirs(self.docstore_path, exist_ok=True)
        
        # Setup embedding model
        self.my_embedding = HuggingFaceEmbedding(
            model_name="TencentBAC/Conan-embedding-v1"
        )
        
        # self.model = HuggingFaceEmbedding(
        #     model_name="taide/Llama3-TAIDE-LX-8B-Chat-Alpha1"
        # )
        
        # Initialize ChromaDB client
        self.chroma_persist_client = chromadb.PersistentClient(db_path)
        
        # Load different types of data
        self.corpus_dict_insurance = self.load_data(
            os.path.join(self.source_path, 'insurance'),
            os.path.join(self.pkl_path, 'insurance.pkl')
        )
        self.corpus_dict_finance = self.load_data(
            os.path.join(self.source_path, 'finance'),
            os.path.join(self.pkl_path, 'finance.pkl')
        )
        
        # Load and process FAQ data
        faq_path = os.path.join(self.source_path, 'faq/pid_map_content.json')
        with open(faq_path, 'rb') as f_s:
            key_to_source_dict = json.load(f_s)
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
            self.corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items()}
            
        
    def initialize_process(self, chunk_size=256, chunk_overlap=200):
        print("  - loading data into ChromaDB ")
        
        # clear existing database
        if os.path.exists(self.db_path):
            for root, dirs, files in os.walk(self.db_path, topdown=False):
                for name in files:
                    if name != "chroma.sqlite3":
                        os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            
        # Check and update collections
        existing_collections = self.chroma_persist_client.list_collections()
        existing_collections_name = [collection.name for collection in existing_collections]
        
        # Process each category       
        for category in ['faq', 'insurance', 'finance']:
            print(f"     - loading [{category}] ...")
            if category in existing_collections_name:
                self.delete_database(category)
            self.insert_database(
                getattr(self, f'corpus_dict_{category}'),
                category,
                chunk_size,
                chunk_overlap
            )
            

    def load_data(self, category_path, corpus_dict_file):
        # Load existing corpus if available
        if os.path.exists(corpus_dict_file):
            with open(corpus_dict_file, 'rb') as file:
                return pickle.load(file)

        corpus_dict = {}
        masked_file_ls = os.listdir(category_path)
        new_files = [file for file in masked_file_ls if int(file.replace('.pdf', '')) not in corpus_dict]

        # Process new PDF files
        if new_files:
            print(f"  - loading PDF ...")
            for file in tqdm(new_files, desc="Loading new files"):
                file_id = int(file.replace('.pdf', ''))
                corpus_dict[file_id] = self.read_pdf(os.path.join(category_path, file))

            # Save processed corpus
            with open(corpus_dict_file, 'wb') as file:
                pickle.dump(corpus_dict, file)

        return corpus_dict

    def read_pdf(self, pdf_loc, page_infos=None):
        with pdfplumber.open(pdf_loc) as pdf:
            pages = pdf.pages[page_infos[0]:page_infos[1]] if page_infos else pdf.pages
            return ''.join(page.extract_text() for page in pages if page.extract_text())

    def insert_database(self, corpus_dict, category, chunk_size=256, chunk_overlap=200):
        # Prepare documents
        documents = [
            Document(
                text=text,
                id_=f"doc_id_{id}",
                metadata={"category": category, "pid": id}
            ) for id, text in corpus_dict.items()
        ]

        # Setup ChromaDB collection and vector store
        chroma_collection = self.chroma_persist_client.get_or_create_collection(category)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection, persist_dir=os.path.join(self.db_path, category))

        # Process documents into nodes
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        nodes = splitter.get_nodes_from_documents(documents)

        # Setup storage context and document store
        docstore = SimpleDocumentStore()
        docstore.add_documents(nodes)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Create and save vector index
        vector_index = VectorStoreIndex(
            nodes=nodes,
            embed_model=self.my_embedding,
            storage_context=storage_context
        )

        # Persist document store
        storage_context = StorageContext.from_defaults(docstore=docstore)
        storage_context.docstore.persist(os.path.join(self.docstore_path, f'{category}.json'))

        return vector_index

    def load_database(self, category):
        # Load existing collection and create vector store
        chroma_collection = self.chroma_persist_client.get_or_create_collection(category)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # Load document store and create storage context
        docstore = SimpleDocumentStore.from_persist_path(
            os.path.join(self.docstore_path, f'{category}.json')
        )
        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            vector_store=vector_store
        )

        # Create vector index
        return VectorStoreIndex(
            nodes=[],
            storage_context=storage_context,
            embed_model=self.my_embedding
        )
        
    def summary_index(self, corpus_dict=123, category='finance', chunk_size=256, chunk_overlap=200):
        corpus_dict=self.corpus_dict_finance
        corpus_dict = {k: v for k, v in corpus_dict.items() if int(k) in [351, 900, 1021]}
        # Prepare documents
        documents = [
            Document(
                text=text,
                id_=f"doc_id_{id}",
                metadata={"category": category, "pid": id}
            ) for id, text in corpus_dict.items()
        ]

        # Process documents into nodes
        splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # response_synthesizer = get_response_synthesizer(
        #     llm=self.model ,response_mode="tree_summarize", use_async=True
        # )
        doc_summary_index = DocumentSummaryIndex.from_documents(
            documents=documents,
            # llm=OurLLM(),
            transformations=[splitter],
            # response_synthesizer=response_synthesizer,
            show_progress=True,
            embed_model=self.my_embedding
        )
        

        # Setup storage context and document store
        doc_summary_index.storage_context.persist(self.docstore_path, f'{category}_summary.json')

        return doc_summary_index
        

    def delete_database(self, category):
        self.chroma_persist_client.delete_collection(category)
        print(f"        ... collection [{category}] deleted.")

    def get_vector_index(self, category):
        return self.load_database(category)
    
    def get_text(self, category):
        return getattr(self, f'corpus_dict_{category}')


# # quantization_config = BitsAndBytesConfig(
# #     load_in_4bit=True,
# #     bnb_4bit_compute_dtype=torch.float16,
# #     bnb_4bit_quant_type="nf4",
# #     bnb_4bit_use_double_quant=True,
# # )
# model_name = "Llama3-TAIDE-LX-8B-Chat-Alpha1"
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, low_cpu_mem_usage=True, device_map="cuda", torch_dtype=torch.bfloat16).eval()
# #自定义本地模型
# class OurLLM(CustomLLM):
#     context_window: int = 4096
#     num_output: int = 1024
#     model_name_: str = "custom"
 
#     @property
#     def metadata(self) -> LLMMetadata:
#         """Get LLM metadata."""
#         return LLMMetadata(
#             context_window=self.context_window,
#             num_output=self.num_output,
#             model_name_=self.model_name,
#         )
 
#     @llm_completion_callback()
#     def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
#         text, history = model.chat(tokenizer, prompt, history=[], temperature=0.1)
#         return CompletionResponse(text=text)
 
#     @llm_completion_callback()
#     def stream_complete(
#             self, prompt: str, **kwargs: Any
#     ) -> CompletionResponseGen:
#         raise NotImplementedError()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to read questions')
    parser.add_argument('--source_path', type=str, required=True, help='Path to read reference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output formatted answers')

    args = parser.parse_args()
    
    db = VectorDatabase(source_path=args.source_path)
    VectorDatabase.initialize_process(source_path=args.source_path)



