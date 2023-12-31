import datetime
import uuid
from dataclasses import dataclass, field
from typing import Set
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from scipy.spatial.distance import cosine


class RetrievalMemory:
    def __init__(self, persistent_db_path="./retrieval_memory", embedding_model_name="all-MiniLM-L6-v2",
                 collection_name="retrieval_memory_collection"):
        self.next_memory_id = 0
        self.client = chromadb.PersistentClient(path=persistent_db_path)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model_name)
        self.collection = self.client.get_or_create_collection(name=collection_name,
                                                               embedding_function=self.sentence_transformer_ef)

    def add_memory(self, description: str, date: datetime.datetime = datetime.datetime.now(), importance: float = 1.0):
        """Add a memory with a given description and importance to the memory stream."""
        mem = [description]
        ids = [str(self.generate_unique_id())]
        metadata = {'memory_id': ids[0], 'memory': description, 'importance': importance, 'creation_timestamp': date,
                    'last_access_timestamp': date}
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)
        self.next_memory_id += 1

    def retrieve_memories(self, query: str, k, date=datetime.datetime.now(), alpha_recency=1, alpha_relevance=1,
                          alpha_importance=1):
        query_embedding = self.sentence_transformer_ef([query])
        query_result = self.collection.query(query_embedding, n_results=k * 4, include=["metadatas", "documents",
                                                                                        "distances"])  # Increase candidate pool size
        if len(query_result['metadatas']) == 0:
            return []
        # Step 2: Apply scoring to the candidate memories
        scores = [
            self.compute_memory_score(metadata, query_embedding, date, alpha_recency, alpha_relevance, alpha_importance)
            for metadata in query_result['metadatas']]

        # Normalize and select top k memories based on scores
        normalized_scores = self.normalize_scores(np.array(scores))
        top_indices = self.get_top_indices(normalized_scores, k)
        retrieved_memories = [query_result['metadatas'][i] for i in top_indices]

        # Update last access time
        for memory in retrieved_memories:
            memory = self.update_last_access(memory, date)
            self.collection.upsert(memory['memory_id'], metadatas=memory)
        return retrieved_memories

    @staticmethod
    def generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id

    def compute_memory_score(self, metadata, query_embedding, date, alpha_recency, alpha_relevance, alpha_importance):
        recency = self.compute_recency(metadata, date)
        relevance = self.compute_relevance(metadata['embedding'], query_embedding)
        importance = metadata['importance']
        return alpha_recency * recency + alpha_relevance * relevance + alpha_importance * importance

    @staticmethod
    def update_last_access(metadata, date):
        metadata['last_access_timestamp'].last_access_timestamp = date
        return metadata

    @staticmethod
    def compute_recency(metadata, date):
        decay_factor = 0.99
        time_diff = date - metadata['last_access_timestamp']
        hours_diff = time_diff.total_seconds() / 3600
        recency = decay_factor ** hours_diff
        return recency

    @staticmethod
    def compute_relevance(memory_embedding, query_embedding):
        relevance = 1 - cosine(memory_embedding, query_embedding)
        return relevance

    @staticmethod
    def normalize_scores(scores):
        min_score, max_score = np.min(scores), np.max(scores)
        if min_score == max_score:
            return np.zeros_like(scores)
        return (scores - min_score) / (max_score - min_score)

    @staticmethod
    def get_top_indices(scores, k):
        return scores.argsort()[-k:][::-1]
