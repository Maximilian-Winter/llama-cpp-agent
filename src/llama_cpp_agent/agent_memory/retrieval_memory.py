import datetime
import uuid

import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from scipy.spatial.distance import cosine


class RetrievalMemory:
    def __init__(
        self,
        persistent_db_path="./retrieval_memory",
        embedding_model_name="BAAI/bge-small-en-v1.5",
        collection_name="retrieval_memory_collection",
        decay_factor=0.99,
    ):
        self.client = chromadb.PersistentClient(path=persistent_db_path)
        self.sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.sentence_transformer_ef
        )
        self.decay_factor = decay_factor

    def add_memory(
        self,
        description: str,
        date: datetime.datetime = datetime.datetime.now(),
        importance: float = 1.0,
    ):
        """Add a memory with a given description and importance to the memory stream."""
        mem = [description]
        ids = [str(self.generate_unique_id())]
        metadata = {
            "memory_id": ids[0],
            "memory": description,
            "importance": importance,
            "creation_timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
            "last_access_timestamp": date.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def retrieve_memories(
        self,
        query: str,
        k,
        date=datetime.datetime.now(),
        alpha_recency=1,
        alpha_relevance=1,
        alpha_importance=1,
    ):
        query_embedding = self.sentence_transformer_ef([query])
        query_result = self.collection.query(
            query_embedding,
            n_results=k * 4,
            include=["metadatas", "embeddings", "documents", "distances"],
        )  # Increase candidate pool size
        if len(query_result["metadatas"][0]) == 0:
            return []
        # Step 2: Apply scoring to the candidate memories
        scores = []
        for index in range(len(query_result["metadatas"][0])):
            scores.append(
                self.compute_memory_score(
                    query_result["metadatas"][0][index],
                    query_result["embeddings"][0][index],
                    query_embedding,
                    date,
                    alpha_recency,
                    alpha_relevance,
                    alpha_importance,
                )
            )

        # Normalize and select top k memories based on scores
        normalized_scores = self.normalize_scores(np.array(scores))
        top_indices = self.get_top_indices(normalized_scores, k)
        retrieved_memories = [query_result["metadatas"][0][i] for i in top_indices]

        # Update last access time
        for memory in retrieved_memories:
            memory = self.update_last_access(memory, date)
            self.collection.upsert(
                ids=memory["memory_id"], documents=[memory["memory"]], metadatas=memory
            )
        return retrieved_memories

    @staticmethod
    def generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id

    def compute_memory_score(
        self,
        metadata,
        memory_embedding,
        query_embedding,
        date,
        alpha_recency,
        alpha_relevance,
        alpha_importance,
    ):
        recency = self.compute_recency(metadata, date)
        relevance = self.compute_relevance(memory_embedding, query_embedding)
        importance = metadata["importance"]
        return (
            alpha_recency * recency
            + alpha_relevance * relevance
            + alpha_importance * importance
        )

    @staticmethod
    def update_last_access(metadata, date):
        metadata["last_access_timestamp"] = date.strftime("%Y-%m-%d %H:%M:%S")
        return metadata

    def compute_recency(self, metadata, date):
        decay_factor = self.decay_factor
        time_diff = date - datetime.datetime.strptime(
            metadata["last_access_timestamp"], "%Y-%m-%d %H:%M:%S"
        )
        hours_diff = time_diff.total_seconds() / 3600
        recency = decay_factor**hours_diff
        return recency

    @staticmethod
    def compute_relevance(memory_embedding, query_embedding):
        relevance = 1 - cosine(memory_embedding, query_embedding[0])
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
