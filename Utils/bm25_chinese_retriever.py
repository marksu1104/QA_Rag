import logging

from typing import Any, Callable, Dict, List, Optional, Union, cast
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.constants import DEFAULT_SIMILARITY_TOP_K
from llama_index.core.indices.vector_store.base import VectorStoreIndex
from llama_index.core.schema import BaseNode, IndexNode, NodeWithScore, QueryBundle
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.vector_stores.utils import (
    node_to_metadata_dict,
    metadata_dict_to_node,
)
import bm25s
import jieba
from llama_index.retrievers.bm25 import BM25Retriever
from bm25s.tokenization import Tokenized
from tqdm.auto import tqdm
import hashlib, pickle, os

logger = logging.getLogger(__name__)

DEFAULT_PERSIST_ARGS = {"similarity_top_k": "similarity_top_k", "_verbose": "verbose"}
DEFAULT_PERSIST_FILENAME = "retriever.json"

class ChineseBM25Retriever(BM25Retriever):
    """A BM25 retriever that uses the BM25 algorithm to retrieve nodes.

    Args:
        nodes (List[BaseNode], optional): The nodes to index. If not provided, an existing BM25 object must be passed.
        similarity_top_k (int, optional): The number of results to return. Defaults to DEFAULT_SIMILARITY_TOP_K.
        callback_manager (CallbackManager, optional): The callback manager to use. Defaults to None.
        objects (List[IndexNode], optional): The objects to retrieve. Defaults to None.
        object_map (dict, optional): A map of object IDs to nodes. Defaults to None.
        verbose (bool, optional): Whether to show progress. Defaults to False.
    """
    def __init__(
            self,
            nodes: Optional[List[BaseNode]] = None,
            similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
            callback_manager: Optional[CallbackManager] = None,
            objects: Optional[List[IndexNode]] = None,
            object_map: Optional[dict] = None,
            verbose: bool = False,
    ) -> None:

        super().__init__(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            callback_manager=callback_manager,
            objects=objects,
            object_map=object_map,
            verbose=verbose,
        )
        
        
        text, corpus = self._prepare_text_and_corpus(nodes)
        text_hash = hashlib.md5("".join(text).encode('utf-8')).hexdigest()
        cache_file = f"./database/bm25_cache_{text_hash}.pkl"
        corpus_tokens = self.get_corpus_tokens(text, cache_file)
        
        self.bm25 = bm25s.BM25()
        self.bm25.corpus = corpus
        self.bm25.index(corpus_tokens, show_progress=False)
    
    
    @classmethod
    def from_defaults(
        cls,
        index: Optional[VectorStoreIndex] = None,
        nodes: Optional[List[BaseNode]] = None,
        docstore: Optional[BaseDocumentStore] = None,
        similarity_top_k: int = DEFAULT_SIMILARITY_TOP_K,
        verbose: bool = False,
        tokenizer: Optional[Callable[[str], List[str]]] = None,
    ) -> "BM25Retriever":

        if not any([index, nodes, docstore]):
            raise ValueError("Please pass exactly one of index, nodes, or docstore.")

        if index is not None:
            docstore = index.docstore

        if docstore is not None:
            nodes = cast(List[BaseNode], list(docstore.docs.values()))

        assert nodes is not None, "Please pass exactly one of index, nodes, or docstore."

        return cls(
            nodes=nodes,
            similarity_top_k=similarity_top_k,
            verbose=verbose,
        )
    
    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        query = query_bundle.query_str

        tokenized_query = self._tokenize_fun(query, show_progress=self._verbose)

        indexes, scores = self.bm25.retrieve(
            tokenized_query, k=self.similarity_top_k, show_progress=self._verbose
        )

        indexes = indexes[0]
        scores = scores[0]

        nodes: List[NodeWithScore] = []
        for idx, score in zip(indexes, scores):
            if isinstance(idx, dict):
                node = metadata_dict_to_node(idx)
            else:
                node_dict = self.corpus[int(idx)]
                node = metadata_dict_to_node(node_dict)
            nodes.append(NodeWithScore(node=node, score=float(score)))

        return nodes
    
    def _tokenize_fun(
        self,
        texts,
        return_ids: bool = True,
        show_progress: bool = False,
        leave: bool = False,
    ) -> Union[List[List[str]], Tokenized]:
        
        if isinstance(texts, str):
            texts = [texts]

        corpus_ids, token_to_index = self._build_token_index(texts, show_progress, leave)

        unique_tokens = list(token_to_index.keys())
        vocab_dict = token_to_index

        if return_ids:
            return Tokenized(ids=corpus_ids, vocab=vocab_dict)
        else:
            reverse_dict = unique_tokens
            for i, token_ids in corpus_ids:
                corpus_ids[i] = [reverse_dict[token_id] for token_id in token_ids]

            return corpus_ids

    def _build_token_index(self, texts, show_progress, leave):
        corpus_ids = []
        token_to_index = {}

        for text in texts:
            splitted = jieba.lcut(text)
            doc_ids = []

            for token in splitted:
                if token not in token_to_index:
                    token_to_index[token] = len(token_to_index)

                token_id = token_to_index[token]
                doc_ids.append(token_id)

            corpus_ids.append(doc_ids)

        return corpus_ids, token_to_index

    def _prepare_text_and_corpus(self, nodes):
        text = [node.get_content() for node in nodes]
        corpus = [node_to_metadata_dict(node) for node in nodes]
        return text, corpus
    
    def get_corpus_tokens(self, text, path: str):
        
        if os.path.exists(path):
            with open(path, 'rb') as f:
                corpus_tokens = pickle.load(f)
        else:
            corpus_tokens = self._tokenize_fun(text)
            with open(path, 'wb') as f:
                pickle.dump(corpus_tokens, f)
        
        return corpus_tokens
