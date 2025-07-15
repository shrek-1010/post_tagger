# ---------------------------------------------
# Credits:
# - Some logic and inspiration for this project were assisted by Claude AI.
# - Additional guidance and code patterns were adapted from a YouTube tutorial.
# ---------------------------------------------
import pickle
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT
import faiss
import re
import os

class PostModelManager:
    """
    Complete model package for post duplicate detection and tagging system.
    Includes pre-computed embeddings and models for faster inference.
    """
    
    def __init__(self):
        # Models
        self.sentence_model = None
        self.keybert_model = None
        
        # Pre-computed data
        self.posts_dataset = None
        self.corpus_embeddings = None
        self.faiss_index = None
        self.corpus_texts = None
        
        # Configuration
        self.config = None
        
        # Metadata
        self.metadata = {
            'model_version': '1.0',
            'sentence_model_name': 'all-MiniLM-L6-v2',
            'embedding_dim': None,
            'num_posts': None
        }
    
    def build_from_files(self, 
                        posts_file='posts_dataset.json',
                        config_file='config.json'):
        """Build the model package from your existing files."""
        
        print("Loading configuration...")
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        print("Loading posts dataset...")
        with open(posts_file, 'r') as f:
            self.posts_dataset = json.load(f)
        
        print("Initializing models...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.keybert_model = KeyBERT()
        
        print("Pre-computing embeddings...")
        self._build_embeddings_index()
        
        print("Setting metadata...")
        self.metadata['embedding_dim'] = self.corpus_embeddings.shape[1]
        self.metadata['num_posts'] = len(self.posts_dataset)
        
        print("Model package built successfully!")
    
    def _build_embeddings_index(self):
        """Pre-compute embeddings and build FAISS index."""
        # Prepare corpus
        self.corpus_texts = [
            post['title'] + ' ' + post['body'] 
            for post in self.posts_dataset
        ]
        
        # Compute embeddings
        self.corpus_embeddings = self.sentence_model.encode(
            self.corpus_texts, 
            convert_to_numpy=True
        )
        
        # Build FAISS index
        dim = self.corpus_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        
        # Normalize and add to index
        faiss.normalize_L2(self.corpus_embeddings)
        self.faiss_index.add(self.corpus_embeddings)
    
    def check_duplicate_post(self, input_post):
        """
        Check if input post is duplicate using pre-computed embeddings.
        
        Args:
            input_post: Dict with 'title' and 'body' keys
            
        Returns:
            Dict with duplicate detection results matching API format
        """
        # Encode new post
        new_text = input_post['title'] + ' ' + input_post['body']
        new_vec = self.sentence_model.encode([new_text], convert_to_numpy=True)
        faiss.normalize_L2(new_vec)
        
        # Search in pre-built index
        D, I = self.faiss_index.search(new_vec, k=1)
        max_idx = int(I[0][0])
        
        # Convert L2 distance to cosine similarity
        max_score = 1 - D[0][0] / 2
        threshold = self.config.get('similarity_threshold', 0.8)
        
        if max_score > threshold:
            return {
                'is_duplicate': True,
                'similar_post_id': self.posts_dataset[max_idx]['post_id'],
                'similarity_score': float(max_score),
                'status': 'blocked'
            }
        else:
            return {
                'is_duplicate': False,
                'similar_post_id': None,
                'similarity_score': float(max_score),
                'status': 'allowed'
            }
    
    def clean_text(self, text):
        
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def generate_post_tags(self, post_body):
        """Generate tags for a post using KeyBERT."""
        min_tag_score = self.config.get('min_tag_score', 0.2)
        max_tag_count = self.config.get('max_tag_count', 6)
        min_tag_count = self.config.get('min_tag_count', 3)
        
        cleaned = self.clean_text(post_body)
        keywords = self.keybert_model.extract_keywords(
            cleaned, 
            keyphrase_ngram_range=(1, 3), 
            stop_words='english', 
            top_n=max_tag_count
        )
        
        tags = [kw for kw, score in keywords if score >= min_tag_score]
        tags = [t.lower() for t in tags if len(t.split()) <= 4]
        
        # Ensure minimum tag count
        if len(tags) < min_tag_count:
            tags += [kw for kw, _ in keywords if kw not in tags][:min_tag_count - len(tags)]
        
        return tags[:max_tag_count]
    
    def tag_match_score(self, generated_tags, human_tags):
        """Calculate tag match score."""
        generated = set([t.lower() for t in generated_tags])
        human = set([t.lower() for t in human_tags])
        
        if not human:
            return 0.0
            
        match_count = len(generated & human)
        percent_match = match_count / len(human)
        return percent_match
    
    def is_tag_match_sufficient(self, generated_tags, human_tags, threshold=0.8):
        """Check if tag match is sufficient."""
        return self.tag_match_score(generated_tags, human_tags) >= threshold
    
    def update_dataset(self, new_posts):
        """Update dataset and rebuild index with new posts."""
        print("Adding new posts to dataset...")
        self.posts_dataset.extend(new_posts)
        
        print("Rebuilding embeddings index...")
        self._build_embeddings_index()
        
        print("Updating metadata...")
        self.metadata['num_posts'] = len(self.posts_dataset)
        
        print(f"Dataset updated! Now contains {len(self.posts_dataset)} posts.")
    
    def get_stats(self):
        """Get model package statistics."""
        return {
            'metadata': self.metadata,
            'config': self.config,
            'dataset_size': len(self.posts_dataset),
            'embedding_shape': self.corpus_embeddings.shape,
            'index_size': self.faiss_index.ntotal
        }
    
    def save_model(self, filepath='post_model.pkl'):
        """Save the complete model package with better error handling."""
        print(f"Saving model package to {filepath}...")
        
        # Create backup if file exists
        if os.path.exists(filepath):
            backup_path = filepath + '.backup'
            if os.path.exists(backup_path):
                os.remove(backup_path)
            os.rename(filepath, backup_path)
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            print("Model saved successfully!")
        except Exception as e:
            print(f"Error saving model: {e}")
            # Restore backup if save failed
            if os.path.exists(filepath + '.backup'):
                os.rename(filepath + '.backup', filepath)
            raise
    
    @staticmethod
    def load_model(filepath='post_model.pkl'):
        """Load the complete model package with better error handling."""
        print(f"Loading model package from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        try:
            with open(filepath, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    @staticmethod
    def create_and_save_model(posts_file='posts_dataset.json', 
                             config_file='config.json',
                             model_file='post_model.pkl'):
        """Convenience method to create and save model in one step."""
        model = PostModelManager()
        model.build_from_files(posts_file, config_file)
        model.save_model(model_file)
        return model

# Usage Example
if __name__ == "__main__":
    # Create and build model package
    model_manager = PostModelManager()
    model_manager.build_from_files('posts_dataset.json', 'config.json')
    
    # Save the model
    model_manager.save_model('post_model.pkl')
    
    # Example usage after loading
    loaded_model = PostModelManager.load_model('post_model.pkl')
    
    # Test duplicate checking
    test_post = {
        'title': 'Binary search trees explained',
        'body': 'Can someone explain how binary search trees work?'
    }
    result = loaded_model.check_duplicate_post(test_post)
    print("Duplicate check:", result)
    
    # Test tag generation
    tags = loaded_model.generate_post_tags(test_post['body'])
    print("Generated tags:", tags)
    
    # Get model stats
    stats = loaded_model.get_stats()
    print("Model stats:", stats)