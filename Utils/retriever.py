import json
import time
import argparse
from typing import Any, Tuple, Dict, List, Optional, cast
from scipy import stats

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

from concurrent.futures import ThreadPoolExecutor
import logging


class Retriever:
    def __init__(self, 
                 source_path: str = './reference', 
                 question_path: str = './dataset/preliminary/questions_example.json', 
                 output_path: str = 'output.json'):
        """
        Initialize the Retriever with paths for sources, questions, and output.
        """
        self._source_path = source_path
        self._question_path = question_path
        self._output_path = output_path

        # Initialize the vector database
        self._vector_db = VectorDatabase(source_path)
        print("< Retriever initialized > ")

        # Initialize the embedding model
        self._embedding_model = HuggingFaceEmbedding(
            model_name="TencentBAC/Conan-embedding-v1"  # Options: "TencentBAC/Conan-embedding-v1", "sensenova/piccolo-base-zh"
        )    

    def vector_retrieve(self, 
                        qs: str, 
                        category: str, 
                        source_list: List[str], 
                        top_k: int = 1, 
                        text_type: str = 'chunk') -> Tuple[List[int], List[float]]:
        """
        Retrieve documents using vector similarity.
        """
        if text_type == 'chunk':
            vector_index = self._vector_db.get_vector_index(category)
        else:
            vector_index = self._vector_db.get_vector_clean_index(category)
        
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
        # Add any missing source_list IDs with score 0
        for file_id in source_list:
            if int(file_id) not in file_scores:
                file_scores[int(file_id)] = 0

        # Extract scores in the order of source_list
        scores = [file_scores[int(file_id)] for file_id in source_list]

        # Sort file IDs by their scores in descending order and get the highest score ID
        sorted_file_scores = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        id = [file_id for file_id, _ in sorted_file_scores[:top_k]]
        return id, scores
    
    def bm25_retrieve(self, 
                      qs: str, 
                      category: str, 
                      source_list: List[str], 
                      top_k: int = 1, 
                      text_type: str = 'chunk') -> Tuple[List[int], List[float]]:
        """
        Retrieve documents using BM25 algorithm.
        """
        if text_type == 'chunk':
            vector_index = self._vector_db.get_vector_index(category)
        else:
            vector_index = self._vector_db.get_vector_clean_index(category)
        
        # Get all documents from the vector index
        docstore = vector_index.docstore
        nodes = cast(List[BaseNode], list(docstore.docs.values()))
        num_vector_index = len(nodes)

        # Initialize the vector index retriever
        retriever = ChineseBM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=num_vector_index,  # Retrieve all possible results
            source_list=source_list
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
        # Add any missing source_list IDs with score 0
        for file_id in source_list:
            if int(file_id) not in file_scores:
                file_scores[int(file_id)] = 0

        # Extract scores in the order of source_list
        scores = [file_scores[int(file_id)] for file_id in source_list]

        # Sort file IDs by their scores in descending order and get the highest score ID
        sorted_file_scores = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)
        id = [file_id for file_id, _ in sorted_file_scores[:top_k]]
        return id, scores

    def original_retrieve(self, 
                          qs: str, 
                          category: str, 
                          source_list: List[str], 
                          top_k: int = 1) -> Tuple[List[int], List[float]]:
        """
        Retrieve documents using BM25 algorithm.
        """
        corpus_dict = self._vector_db.get_text(category)
        filtered_corpus = [corpus_dict[int(file)] for file in source_list]
        tokenized_corpus = [list(jieba.cut_for_search(doc)) for doc in filtered_corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = list(jieba.cut_for_search(qs))
        ans = bm25.get_top_n(tokenized_query, list(filtered_corpus), n=top_k)
        bm25_scores = bm25.get_scores(tokenized_query)
        res = [key for key, value in corpus_dict.items() if value in ans]
        return res, bm25_scores

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores"""
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(s - min_score) / (max_score - min_score) for s in scores]
        
    def bm25_vector_retrieve(self, 
                             qs: str, 
                             category: str, 
                             source_list: List[str], 
                             top_k: int = 1, 
                             k: int = 60) -> Optional[List[int]]:
        """
        Combine BM25 and vector retrieval scores using Reciprocal Rank Fusion (RRF).
        """
        bm25_retrieved, bm25_scores, vector_retrieved, vector_scores = self.bm25_vector_retrieve_parallel(qs, category, source_list)

        bm25_scores_norm = self.normalize_scores(bm25_scores)
        vector_scores_norm = self.normalize_scores(vector_scores)

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

    def relative_score_fusion(self, 
                              qs: str, 
                              category: str, 
                              source_list: List[str], 
                              top_k: int = 1, 
                              alpha: float = 0.5) -> Optional[List[int]]:
        """
        Combine BM25 and vector retrieval scores using Relative Score Fusion (RSF).
        """
        bm25_retrieved, bm25_scores, vector_retrieved, vector_scores = \
            self.bm25_vector_retrieve_parallel(qs, category, source_list)

        # Normalize scores
        bm25_norm = self.normalize_scores(bm25_scores)
        vector_norm = self.normalize_scores(vector_scores)

        # Calculate RSF scores
        rsf_scores = {}
        for idx, (b_score, v_score) in enumerate(zip(bm25_norm, vector_norm)):
            file_id = int(source_list[idx])
            rsf_score = alpha * v_score + (1 - alpha) * b_score
            rsf_scores[file_id] = rsf_score

        # Sort and return top_k results
        sorted_results = sorted(rsf_scores.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_results[:top_k]] if sorted_results else None
    
    def relative_score_clean_fusion(self, 
                                    qs: str, 
                                    category: str, 
                                    source_list: List[str], 
                                    top_k: int = 1, 
                                    alpha: float = 0.8) -> Optional[List[int]]:
        """
        Combine BM25 and vector retrieval scores using Relative Score Fusion (RSF) with clean text.
        """
        bm25_retrieved, bm25_scores, vector_retrieved, vector_scores = \
            self.bm25_vector_retrieve_parallel(qs, category, source_list, text_type='clean')

        # Normalize scores
        bm25_norm = self.normalize_scores(bm25_scores)
        vector_norm = self.normalize_scores(vector_scores)

        # Calculate RSF scores
        rsf_scores = {}
        for idx, (b_score, v_score) in enumerate(zip(bm25_norm, vector_norm)):
            file_id = int(source_list[idx])
            rsf_score = alpha * v_score + (1 - alpha) * b_score
            rsf_scores[file_id] = rsf_score

        # Sort and return top_k results
        sorted_results = sorted(rsf_scores.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_results[:top_k]] if sorted_results else None

    def distribution_score_fusion(self, 
                                  qs: str, 
                                  category: str, 
                                  source_list: List[str], 
                                  top_k: int = 1) -> Optional[List[int]]:
        """
        Combine BM25 and vector retrieval scores using Distribution-Based Score Fusion (DBSF).
        """
        bm25_retrieved, bm25_scores, vector_retrieved, vector_scores = \
            self.bm25_vector_retrieve_parallel(qs, category, source_list)

        # Z-score normalization
        bm25_z = stats.zscore(bm25_scores)
        vector_z = stats.zscore(vector_scores)

        # Calculate DBSF scores
        dbsf_scores = {}
        for idx, (b_z, v_z) in enumerate(zip(bm25_z, vector_z)):
            file_id = int(source_list[idx])
            dbsf_score = max(b_z, v_z)
            dbsf_scores[file_id] = dbsf_score

        # Sort and return top_k results
        sorted_results = sorted(dbsf_scores.items(), key=lambda x: x[1], reverse=True)
        return [r[0] for r in sorted_results[:top_k]] if sorted_results else None
    
    def bm25_vector_retrieve_parallel(self, 
                                      qs: str, 
                                      category: str, 
                                      source_list: List[str], 
                                      text_type: str = 'chunk') -> Tuple[List[int], List[float], List[int], List[float]]:
        """
        Perform BM25 and vector retrieval in parallel.
        """
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Two futures for BM25 and vector retrieval
            future_bm25 = executor.submit(
                self.bm25_retrieve,
                qs, 
                category, 
                source_list,
                top_k=len(source_list),
                text_type=text_type
            )
            
            future_vector = executor.submit(
                self.vector_retrieve,
                qs,
                category,
                source_list,
                top_k=len(source_list)
            )
            
            try:
                # Wait and get results
                bm25_retrieved, bm25_scores = future_bm25.result()
                vector_retrieved, vector_scores = future_vector.result()
                
                return (bm25_retrieved, bm25_scores, 
                        vector_retrieved, vector_scores)
                        
            except Exception as e:
                logging.error(f"Error during parallel retrieval: {str(e)}")
                return [], [], [], []

    def process_questions(self, 
                          method: str = 'Vector', 
                          k: int = 60, 
                          weight: float = 0.8, 
                          text_type: str = 'chunk'):
        """
        Process questions and retrieve answers using the specified method.
        """
        answer_dict = {"answers": []}
        with open(self._question_path, 'rb') as f:
            qs_ref = json.load(f)

        for q_dict in qs_ref['questions']:
            category = q_dict['category']
            query = q_dict['query']
            source = q_dict['source']
            qid = q_dict['qid']
                     
            if method == 'original':
                retrieved, score = self.original_retrieve(query, category, source)
            elif method == 'Vector':
                retrieved, score = self.vector_retrieve(query, category, source, text_type=text_type)
            elif method == 'BM25_Vector_rrf':
                retrieved = self.bm25_vector_retrieve(query, category, source, k=k)
            elif method == 'BM25':
                retrieved, score = self.bm25_retrieve(query, category, source, text_type=text_type)           
            elif method == 'relative_fusion':
                retrieved = self.relative_score_fusion(query, category, source, alpha=weight)  
            elif method == 'distribution_fusion':
                retrieved = self.distribution_score_fusion(query, category, source)      
            elif method == 'final':
                retrieved = self.relative_score_clean_fusion(query, category, source)    
            else:
                raise ValueError("Invalid retrieval method")

            answer_dict['answers'].append({"qid": qid, "retrieve": int(retrieved[0])})

        with open(self._output_path, 'w', encoding='utf8') as f:
            json.dump(answer_dict, f, ensure_ascii=False, indent=4)
            
        print(f"  - Answers saved to {self._output_path} ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process paths and files for retrieval.')
    parser.add_argument('--question_path', type=str, required=True, help='Path to the questions file')
    parser.add_argument('--source_path', type=str, required=True, help='Path to the reference data')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output answers')

    args = parser.parse_args()

    retriever = Retriever(args.source_path, args.question_path, args.output_path)
    retriever.process_questions(method='Vector')
