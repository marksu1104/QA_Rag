import os
import json
import argparse
import pickle
import chromadb
from tqdm import tqdm
import pdfplumber

from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, Document, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore


class VectorDatabase:
    def __init__(self, source_path="./reference", db_path="./chroma", docstore_path="./docstore"):
        # Initialize paths
        self.source_path = source_path
        self.db_path = db_path
        self.docstore_path = docstore_path
        os.makedirs(self.docstore_path, exist_ok=True)
        
        # Setup embedding model
        self.my_embedding = HuggingFaceEmbedding(
            model_name="TencentBAC/Conan-embedding-v1"
        )
        
        # Initialize ChromaDB client
        self.chroma_persist_client = chromadb.PersistentClient(db_path)
        
        # Load different types of data
        self.corpus_dict_insurance = self.load_data(
            os.path.join(self.source_path, 'insurance'),
            os.path.join(self.source_path, 'insurance.pkl')
        )
        self.corpus_dict_finance = self.load_data(
            os.path.join(self.source_path, 'finance'),
            os.path.join(self.source_path, 'finance.pkl')
        )
        
        # Load and process FAQ data
        faq_path = os.path.join(self.source_path, 'faq/pid_map_content.json')
        with open(faq_path, 'rb') as f_s:
            key_to_source_dict = json.load(f_s)
            key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}
            self.corpus_dict_faq = {key: str(value) for key, value in key_to_source_dict.items()}
        
        print("\n< VectorDatabase initialized > ")

    def initialize_process(self, chunk_size=256, chunk_overlap=200):
        print("  - loading data into ChromaDB ")
        
        # Check and update collections
        existing_collections = self.chroma_persist_client.list_collections()
        existing_collections_name = [collection.name for collection in existing_collections]
        
        # Process each category
        for category in ['faq', 'insurance', 'finance']:
            if category in existing_collections_name:
                self.delete_database(category)
            self.insert_database(
                getattr(self, f'corpus_dict_{category}'),
                category,
                chunk_size,
                chunk_overlap
            )
            print(f"     - loading {category} done.")

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

    def insert_database(self, corpus_dict, category, chunk_size=256, chunk_overlap=100):
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
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

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
            storage_context=storage_context,
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

    def delete_database(self, category):
        self.chroma_persist_client.delete_collection(category)
        print(f"     - collection {category} deleted.")

    def get_vector_index(self, category):
        return self.load_database(category)
    
    def get_text(self, category):
        return getattr(self, f'corpus_dict_{category}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some paths and files.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to read questions')
    parser.add_argument('--source_path', type=str, required=True, help='Path to read reference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to output formatted answers')

    args = parser.parse_args()
    
    db = VectorDatabase(source_path=args.source_path)
    VectorDatabase.initialize_process(source_path=args.source_path)
