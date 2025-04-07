import numpy as np
from collections import defaultdict
from typing import List, Tuple, Callable
from aimakerspace.openai_utils.embedding import EmbeddingModel
import asyncio

### Cosine Similarity ###
def cosine_similarity(vector_a: np.array, vector_b: np.array) -> float:
    """Computes the cosine similarity between two vectors."""
    dot_product = np.dot(vector_a, vector_b)
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)
    return dot_product / (norm_a * norm_b)

###  euclidean distance ###
def euclidean_distance(vector_a: np.array, vector_b: np.array) -> float:
    return np.linalg.norm(vector_a - vector_b)

### manhattan distance ###
### write a function that takes two vectors (np arrays) and calculates the manhattan distance between them
def manhattan_distance(vector_a: np.array, vector_b: np.array) -> float:
    return np.sum(np.abs(vector_a - vector_b))  

class VectorDatabase:
    def __init__(self, embedding_model: EmbeddingModel = None):
        self.vectors = defaultdict(np.array)
        self.embedding_model = embedding_model or EmbeddingModel()

    
    def insert(self, key: str, vector: np.array) -> None:
        self.vectors[key] = vector

    def search(
        self,
        query_vector: np.array,
        k: int,
        distance_measure: Callable = cosine_similarity,
    ) -> List[Tuple[str, float]]:
        scores = [
            (key, distance_measure(query_vector, vector))
            for key, vector in self.vectors.items()
        ]
        return sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    def search_by_text(
        self,
        query_text: str,
        k: int,
        distance_measure: Callable = cosine_similarity,
        return_as_text: bool = False,
    ) -> List[Tuple[str, float]]:
        query_vector = self.embedding_model.get_embedding(query_text)
        results = self.search(query_vector, k, distance_measure)
        return [result[0] for result in results] if return_as_text else results

    def retrieve_from_key(self, key: str) -> np.array:
        return self.vectors.get(key, None)

    async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
        """
        Asynchronously builds a vector database from a list of text documents.
        
        This method performs the following operations:
        1. Asynchronously generates embeddings for all provided text documents
        2. Associates each text with its corresponding embedding vector
        3. Inserts each text-embedding pair into the vector database
        
        Using async processing allows for efficient batch processing of embeddings,
        which is particularly valuable when working with large document collections.
        
        Parameters
        ----------
        list_of_text : List[str]
            A list of text documents to be embedded and stored in the database.
            Each text will be used as the key for its corresponding embedding vector.
            
        Returns
        -------
        VectorDatabase
            The current instance of the VectorDatabase with the new vectors added,
            allowing for method chaining.
            
        Examples
        --------
        >>> vector_db = VectorDatabase()
        >>> texts = ["Document 1 content", "Document 2 content", "Document 3 content"]
        >>> vector_db = asyncio.run(vector_db.abuild_from_list(texts))
        >>> # Now database contains embeddings for all documents
        """
        embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
        for text, embedding in zip(list_of_text, embeddings):
            self.insert(text, np.array(embedding))
        return self


if __name__ == "__main__":
    list_of_text = [
        "I like to eat broccoli and bananas.",
        "I ate a banana and spinach smoothie for breakfast.",
        "Chinchillas and kittens are cute.",
        "My sister adopted a kitten yesterday.",
        "Look at this cute hamster munching on a piece of broccoli.",
    ]

    vector_db = VectorDatabase()
    vector_db = asyncio.run(vector_db.abuild_from_list(list_of_text))
    k = 2

    searched_vector = vector_db.search_by_text("I think fruit is awesome!", k=k)
    print(f"Closest {k} vector(s):", searched_vector)

    retrieved_vector = vector_db.retrieve_from_key(
        "I like to eat broccoli and bananas."
    )
    print("Retrieved vector:", retrieved_vector)

    relevant_texts = vector_db.search_by_text(
        "I think fruit is awesome!", k=k, return_as_text=True
    )
    print(f"Closest {k} text(s):", relevant_texts)
