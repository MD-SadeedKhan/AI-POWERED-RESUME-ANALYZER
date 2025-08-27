import faiss
import numpy as np
import pickle
import os
from loguru import logger

class FaissStore:
    def __init__(self, dim=384, index_path="vectorstore/resume_index.faiss", metadata_path="vectorstore/resume_metadata.pkl"):
        self.dim = dim
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []  # List of resume IDs
        # Load existing index and metadata if available
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            self.index = faiss.read_index(index_path)
            with open(metadata_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Loaded FAISS index from {index_path} with {len(self.metadata)} entries")

    def add(self, vector, resume_id):
        """Add a vector to FAISS index and save resume_id in metadata."""
        self.index.add(np.array([vector]).astype("float32"))
        self.metadata.append(resume_id)
        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Added vector for resume_id {resume_id} to FAISS index")

    def search(self, query_vector, k=3):
        """Search FAISS index for top k similar vectors."""
        D, I = self.index.search(np.array([query_vector]).astype("float32"), k)
        results = [self.metadata[i] for i in I[0] if i != -1 and i < len(self.metadata)]
        logger.info(f"FAISS search returned {len(results)} resume IDs: {results}")
        return results