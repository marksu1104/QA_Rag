import json
import time
import argparse
from typing import Any, Callable, Dict, List, Optional, cast

import jieba  # Used for Chinese text segmentation
from rank_bm25 import BM25Okapi  # BM25 algorithm for document retrieval
from tqdm import tqdm

from llama_index.core.schema import BaseNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.retrievers.bm25 import BM25Retriever

from Utils.vector_database import VectorDatabase
from Utils.bm25_chinese_retriever import ChineseBM25Retriever


class Retriever:
    def __init__(self, source_path='./reference', question_path='./dataset/preliminary/questions_example.json', output_path='output.json'):
        """
        Initialize the Retriever with paths for sources, questions, and output.
        """
        self.source_path = source_path
        self.question_path = question_path
        self.output_path = output_path

        # Initialize the vector database
        self.vector_db = VectorDatabase(source_path)
        print("< Retriever initialized > ")

        # Initialize the embedding model
        self.my_embedding = HuggingFaceEmbedding(
            model_name="TencentBAC/Conan-embedding-v1"  # Options: "TencentBAC/Conan-embedding-v1", "sensenova/piccolo-base-zh"
        )    

    def vector_retrieve(self, qs, category, source_list, top_k=1):
        """
        Retrieve documents using vector similarity.
        """
        vector_index = self.vector_db.get_vector_index(category)
        
        # Get all documents from the vector index
        docstore = vector_index.docstore
        nodes = cast(List[BaseNode], list(docstore.docs.values()))
        num_vector_index = len(nodes)

        # Define metadata filters
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="category", value=category, operator=FilterOperator.EQ),
                MetadataFilter(key="pid", value=source_list, operator=FilterOperator.IN)
            ]
        )

        # Initialize the vector index retriever
        retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=num_vector_index,  # Retrieve all possible results
            filters=filters
        )

        # Retrieve results
        results = retriever.retrieve(qs)

        # Map file IDs to their scores
        file_scores = {}
        for result in results:
            file_id = result.metadata['pid']
            if file_id in source_list:
                score = result.score
                if file_id in file_scores:
                    file_scores[file_id] = max(file_scores[file_id], score)
                else:
                    file_scores[file_id] = score

        # Extract scores in the order of source_list
        scores = [file_scores[int(file_id)] for file_id in source_list]

        # Sort file IDs by their scores in descending order and get the highest score ID
        sorted_file_scores = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        # max_score_id = sorted_file_scores[0][0]
        id = [file_id for file_id, _ in sorted_file_scores[:top_k]]

        return id, scores
    
    def bm25_retrieve(self, qs, category, source_list, top_k=1):
        """
        Retrieve documents using vector similarity.
        """
        vector_index = self.vector_db.get_vector_index(category)
        
        # Get all documents from the vector index
        docstore = vector_index.docstore
        nodes = cast(List[BaseNode], list(docstore.docs.values()))
        num_vector_index = len(nodes)

        # Initialize the vector index retriever
        retriever = ChineseBM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=num_vector_index,  # Retrieve all possible results
        )

        # Retrieve results
        results = retriever.retrieve(qs)
        
        # Map file IDs to their scores
        file_scores = {}
        for result in results:
            file_id = result.metadata['pid']
            if file_id in source_list:
                score = result.score
                if file_id in file_scores:
                    file_scores[file_id] = max(file_scores[file_id], score)
                else:
                    file_scores[file_id] = score

        # Extract scores in the order of source_list
        scores = [file_scores[int(file_id)] for file_id in source_list]

        # Sort file IDs by their scores in descending order and get the highest score ID
        sorted_file_scores = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        # max_score_id = sorted_file_scores[0][0]
        id = [file_id for file_id, _ in sorted_file_scores[:top_k]]
        return id, scores


    def bm25_file_retrieve(self, qs, category, source_list, top_k=1):
        """
        Retrieve documents using BM25 algorithm.
        """
        corpus_dict = self.vector_db.get_text(category)
        filtered_corpus = [corpus_dict[int(file)] for file in source_list]
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = list(jieba.cut_for_search(qs))
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=top_k)
        bm25_scores = bm25.get_scores(tokenized_query)
        res = [key for key, value in corpus_dict.items() if value in ans]
        return res, bm25_scores

    def bm25_vector_retrieve(self, qs, category, source_list, top_k=1, k=60):
        """
        Combine BM25 and vector retrieval scores using Reciprocal Rank Fusion (RRF).
        """
        bm25_retrieved, bm25_scores = self.bm25_retrieve(qs, category, source_list, top_k=len(source_list))
        vector_retrieved, vector_scores = self.vector_retrieve(qs, category, source_list, top_k=100000)

        # Normalize scores
        def normalize_scores(scores):
            min_score = min(scores)
            max_score = max(scores)
            if max_score == min_score:
                return [1.0] * len(scores)
            return [(s - min_score) / (max_score - min_score) for s in scores]

        bm25_scores_norm = normalize_scores(bm25_scores)
        vector_scores_norm = normalize_scores(vector_scores)

        # Calculate RRF scores
        rrf_scores = {}
        for idx, (b_score, v_score) in enumerate(zip(bm25_scores_norm, vector_scores_norm)):
            file_id = int(source_list[idx])
            rrf_score = (1 / (k + (1 - b_score))) + (1 / (k + (1 - v_score)))
            rrf_scores[file_id] = rrf_score

        # Sort and return top_k results
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        if not sorted_results:
            return None

        return [r[0] for r in sorted_results[:top_k]]
    
    def bm25_then_vector_retrieve(self, qs, category, source_list):
        """
        First use BM25 to filter candidates, then apply vector retrieval.
        """
        # First get top candidates using BM25
        top_k = min(4, len(source_list))
        bm25_retrieved, bm25_scores = self.bm25_retrieve(qs, category, source_list, top_k=top_k)
        
        # Filter source list to only include top BM25 candidates
        filtered_source_list = [int(b) for b in bm25_retrieved]
        
        # Then do vector retrieval on filtered candidates
        vector_retrieved, vector_scores = self.vector_retrieve(qs, category, filtered_source_list)
        
        return vector_retrieved, vector_scores

    def process_questions(self, method='Vector'):
        """
        Process questions and retrieve answers using the specified method.
        """
        answer_dict = {"answers": []}
        with open(self.question_path, 'rb') as f:
            qs_ref = json.load(f)


        for q_dict in qs_ref['questions']:
            category = q_dict['category']
            query = q_dict['query']
            source = q_dict['source']
            qid = q_dict['qid']
                     
            if method == 'BM25f':
                retrieved, score = self.bm25_file_retrieve(query, category, source)
            elif method == 'Vector':
                retrieved, score = self.vector_retrieve(query, category, source)
            elif method == 'BM25_Vector':
                retrieved = self.bm25_vector_retrieve(query, category, source)
            elif method == 'BM25':
                retrieved, score = self.bm25_retrieve(query, category, source)    
            elif method == 'Vector_filter_BM25':
                retrieved, score = self.bm25_then_vector_retrieve(query, category, source)    
            else:
                raise ValueError("Invalid retrieval method")

            answer_dict['answers'].append({"qid": qid, "retrieve": retrieved[0]})

        with open(self.output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
            
            
        print(f"  - Answers saved to {self.output_path} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process paths and files for retrieval.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions file')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the reference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output answers')

    args = parser.parse_args()

    retriever = Retriever(args.source_path, args.question_path, args.output_path)
    retriever.process_questions(method='Vector')
