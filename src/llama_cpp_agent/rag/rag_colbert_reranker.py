import uuid
import chromadb
from chromadb.utils import embedding_functions
from ragatouille import RAGPretrainedModel


class RAGColbertReranker:
    """
    Represents a chromadb vector database with a Colbert reranker.
    """

    def __init__(
        self,
        persistent_db_path="./retrieval_memory",
        embedding_model_name="BAAI/bge-small-en-v1.5",
        collection_name="retrieval_memory_collection",
        persistent: bool = True,
    ):
        self.RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
        if persistent:
            self.client = chromadb.PersistentClient(path=persistent_db_path)
        else:
            self.client = chromadb.EphemeralClient()
        self.sentence_transformer_ef = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model_name
            )
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.sentence_transformer_ef
        )

    def add_document(self, document: str, metadata: dict = None):
        """Add a memory with a given description and importance to the memory stream."""
        mem = [document]
        ids = [str(self.generate_unique_id())]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def add_documents(self, documents: list[str], metadata: dict = None):
        """Add a memory with a given description and importance to the memory stream."""
        mem = documents
        ids = [str(self.generate_unique_id()) for _ in range(len(documents))]
        self.collection.add(documents=mem, metadatas=metadata, ids=ids)

    def retrieve_documents(self, query: str, k):
        query_embedding = self.sentence_transformer_ef([query])
        query_result = self.collection.query(
            query_embedding,
            n_results=k,
            include=["metadatas", "embeddings", "documents", "distances"],
        )
        documents = []
        for doc in query_result["documents"][0]:
            documents.append(doc)
        results = self.RAG.rerank(query=query, documents=documents, k=k)
        return results

    @staticmethod
    def generate_unique_id():
        unique_id = str(uuid.uuid4())
        return unique_id
