import modal
import os
import sqlite3
import uuid
import time
import json
import base64
import torch
import numpy as np
import requests
import logging
import pickle
import pandas as pd
from typing import Optional, Dict, Any, List
from collections import Counter
from PIL import Image
from io import BytesIO
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from rerankers import Reranker
import traceback

from fasthtml.common import *
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse, Response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Define app
app = modal.App("bee_classifier_rag")

# Constants and directories
DATA_DIR = "/data"
RESULTS_FOLDER = "/data/classification_results"
DB_PATH = "/data/bee_classifier.db"
STATUS_DIR = "/data/status"
TEMP_UPLOAD_DIR = "/data/temp_uploads"
PDF_IMAGES_DIR = "/data/pdf_images"
HEATMAP_DIR = "/data/heatmaps"
TEMPLATES_DIR = "/data/templates"

# Claude API constants
CLAUDE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"

# Insect categories for classification
INSECT_CATEGORIES = [
    "Bumblebees", 
    "Solitary bees",
    "Honeybee", 
    "Wasps",
    "Hoverflies", 
    "Butterflies & Moths",
    "Beetles (>3mm)",
    "Small insects (<3mm)",
    "Other insects",
    "Other flies"
]

# Create custom image with all dependencies - FIXED for NumPy issue
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "requests",
        "python-fasthtml==0.12.0",
        "numpy==1.23.5",  # Specify a compatible version that won't have the _core issue
        "pandas",
        "Pillow",
        "matplotlib",
        "rerankers",
        "rank-bm25",
        "nltk",
        "sentence-transformers"
    )
)

# Look up data volume for storing results
try:
    bee_volume = modal.Volume.lookup("bee_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    bee_volume = modal.Volume.persisted("bee_volume")

# Base prompt template for Claude with context
CLASSIFICATION_PROMPT = """
You are an expert entomologist specializing in insect identification. Your task is to analyze the 
provided image and classify the insect(s) visible.

Please categorize the insect into one of these categories:
{categories}

{context_text}

{additional_instructions}

Format your response as follows:
- Main Category: [the most likely category from the list]
- Confidence: [High, Medium, or Low]
- Description: [brief description of what you see]
{format_instructions}

IMPORTANT: Just provide the formatted response above with no additional explanation or apology.
"""

# Batch prompt template for Claude with context
BATCH_PROMPT = """
You are an expert entomologist specializing in insect identification. Your task is to analyze {count} 
images of insects and classify each one.

For EACH image, categorize the insect into one of these categories:
{categories}

{context_text}

{additional_instructions}

Format your response as follows, with a separate analysis for each image:

IMAGE 1:
- Main Category: [the most likely category from the list]
- Confidence: [High, Medium, or Low]
- Description: [brief description of what you see]
{format_instructions}

IMAGE 2:
- Main Category: [the most likely category from the list]
- Confidence: [High, Medium, or Low]
- Description: [brief description of what you see]
{format_instructions}

And so on for each image...

IMPORTANT: Provide a separate, clearly labeled analysis for each image using the format above.
"""

# Global variables for RAG
colpali_embeddings = None
df = None
page_images = {}
bm25_index = None
tokenized_docs = None

# Setup database for classification results with feedback support
def setup_database(db_path: str):
    """Initialize SQLite database for classification results"""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    
    # Enable WAL mode for better concurrency
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA synchronous=NORMAL;")
    
    # Create tables for both single and batch results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id TEXT PRIMARY KEY,
            category TEXT NOT NULL,
            confidence TEXT NOT NULL,
            description TEXT NOT NULL,
            additional_details TEXT,
            status TEXT DEFAULT 'generated',
            feedback TEXT DEFAULT NULL, 
            context_source TEXT DEFAULT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Add a table for batch results
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_results (
            batch_id TEXT PRIMARY KEY,
            result_count INTEGER NOT NULL,
            results TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    return conn

# Function to save results to file
def save_results_file(result_id, result_content):
    """Save classification results to a file"""
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    result_file = os.path.join(RESULTS_FOLDER, f"{result_id}.json")
    result_data = {
        "id": result_id,
        "result": result_content,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    try:
        with open(result_file, "w") as f:
            json.dump(result_data, f)
        print(f"✅ Saved result file for ID: {result_id}")
        return True
    except Exception as e:
        print(f"⚠️ Error saving result file: {e}")
        return False
    

    # Get classification statistics for dashboard
def get_classification_stats():
    """Query the database to get statistics about insect classifications"""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=30.0)
        cursor = conn.cursor()
        
        # Get overall counts by category from single results
        cursor.execute("""
            SELECT category, COUNT(*) as count 
            FROM results 
            GROUP BY category 
            ORDER BY count DESC
        """)
        category_counts = cursor.fetchall()
        
        # Get total classifications
        cursor.execute("SELECT COUNT(*) FROM results")
        total_single = cursor.fetchone()[0] or 0
        
        # Get batch results count
        cursor.execute("SELECT SUM(result_count) FROM batch_results")
        total_batch_result = cursor.fetchone()[0]
        total_batch = total_batch_result if total_batch_result is not None else 0
        
        # Get confidence levels distribution
        cursor.execute("""
            SELECT confidence, COUNT(*) as count 
            FROM results 
            GROUP BY confidence 
            ORDER BY count DESC
        """)
        confidence_counts = cursor.fetchall()
        
        # Get feedback statistics
        cursor.execute("""
            SELECT feedback, COUNT(*) as count 
            FROM results 
            WHERE feedback IS NOT NULL
            GROUP BY feedback
        """)
        feedback_counts = cursor.fetchall()
        
        # Get recent classifications (last 10)
        cursor.execute("""
            SELECT id, category, confidence, feedback, created_at, context_source 
            FROM results 
            ORDER BY created_at DESC 
            LIMIT 10
        """)
        recent_classifications = cursor.fetchall()
        
        # Get statistics from batch results
        batch_categories = Counter()
        batch_feedback = Counter()
        
        cursor.execute("SELECT results FROM batch_results ORDER BY created_at DESC LIMIT 50")
        batch_results = cursor.fetchall()
        
        for batch in batch_results:
            if batch[0]:
                try:
                    results_data = json.loads(batch[0])
                    for item in results_data:
                        category = item.get('category', 'Unknown')
                        batch_categories[category] += 1
                        
                        # Count feedback from batch results
                        feedback = item.get('feedback')
                        if feedback:
                            batch_feedback[feedback] += 1
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue
        
        # Combine single and batch category statistics
        combined_categories = {}
        for category, count in category_counts:
            combined_categories[category] = count
        
        for category, count in batch_categories.items():
            if category in combined_categories:
                combined_categories[category] += count
            else:
                combined_categories[category] = count
                
        # Create combined category counts list
        combined_category_counts = [(category, count) for category, count in combined_categories.items()]
        combined_category_counts.sort(key=lambda x: x[1], reverse=True)
        
        # Combine feedback statistics
        combined_feedback = {}
        for feedback, count in feedback_counts:
            combined_feedback[feedback] = count
            
        for feedback, count in batch_feedback.items():
            if feedback in combined_feedback:
                combined_feedback[feedback] += count
            else:
                combined_feedback[feedback] = count
                
        combined_feedback_counts = [(feedback, count) for feedback, count in combined_feedback.items()]
        
        # Get count of classifications by date
        cursor.execute("""
            SELECT DATE(created_at) as date, COUNT(*) as count 
            FROM results 
            GROUP BY DATE(created_at) 
            ORDER BY date DESC 
            LIMIT 14
        """)
        daily_counts = cursor.fetchall()
        
        # Get most commonly referenced context sources
        cursor.execute("""
            SELECT context_source, COUNT(*) as count 
            FROM results 
            WHERE context_source IS NOT NULL
            GROUP BY context_source 
            ORDER BY count DESC 
            LIMIT 10
        """)
        context_counts = cursor.fetchall() 
        
        conn.close()
        
        return {
            "category_counts": category_counts,
            "combined_category_counts": combined_category_counts,
            "confidence_counts": confidence_counts,
            "feedback_counts": combined_feedback_counts,
            "recent_classifications": recent_classifications,
            "daily_counts": daily_counts,
            "context_counts": context_counts,
            "total_single": total_single,
            "total_batch": total_batch,
            "total": total_single + total_batch
        }
        
    except Exception as e:
        print(f"Error getting classification stats: {e}")
        traceback.print_exc()
        return {
            "category_counts": [],
            "combined_category_counts": [],
            "confidence_counts": [],
            "feedback_counts": [],
            "recent_classifications": [],
            "daily_counts": [],
            "context_counts": [],
            "total_single": 0,
            "total_batch": 0,
            "total": 0
        }
    

# RAG-related functions
def load_rag_data():
    """Load all data needed for document retrieval"""
    global colpali_embeddings, df, page_images, bm25_index, tokenized_docs
    
    # Path definitions for RAG data
    COLPALI_EMBEDDINGS_PATH = os.path.join(DATA_DIR, "colpali_embeddings.pkl")
    DATA_PICKLE_PATH = os.path.join(DATA_DIR, "data.pkl")
    PDF_PAGE_IMAGES_PATH = os.path.join(DATA_DIR, "pdf_page_image_paths.pkl")
    BM25_INDEX_PATH = os.path.join(DATA_DIR, "bm25_index.pkl")
    TOKENIZED_PARAGRAPHS_PATH = os.path.join(DATA_DIR, "tokenized_paragraphs.pkl")
    
    # Load data frame with metadata
    if os.path.exists(DATA_PICKLE_PATH):
        try:
            df = pd.read_pickle(DATA_PICKLE_PATH)
            logging.info(f"Loaded DataFrame with {len(df)} documents")
        except Exception as e:
            logging.error(f"Error loading DataFrame: {e}")
            df = pd.DataFrame(columns=["filename", "page", "paragraph_size", "text", "image_key", "full_path"])
    else:
        logging.error(f"DataFrame not found at {DATA_PICKLE_PATH}")
        df = pd.DataFrame(columns=["filename", "page", "paragraph_size", "text", "image_key", "full_path"])
    
    # Load image paths
    if os.path.exists(PDF_PAGE_IMAGES_PATH):
        try:
            with open(PDF_PAGE_IMAGES_PATH, "rb") as f:
                page_images = pickle.load(f)
            logging.info(f"Loaded {len(page_images)} image paths")
        except Exception as e:
            logging.error(f"Error loading image paths: {e}")
            page_images = {}
    else:
        logging.error(f"Image paths file not found at {PDF_PAGE_IMAGES_PATH}")
        page_images = {}
    
    # Load ColPali embeddings
    if os.path.exists(COLPALI_EMBEDDINGS_PATH):
        try:
            with open(COLPALI_EMBEDDINGS_PATH, "rb") as f:
                colpali_embeddings = pickle.load(f)
            logging.info(f"Loaded {len(colpali_embeddings)} ColPali embeddings")
        except Exception as e:
            logging.error(f"Error loading ColPali embeddings: {e}")
            colpali_embeddings = None
    else:
        logging.error(f"ColPali embeddings not found at {COLPALI_EMBEDDINGS_PATH}")
        colpali_embeddings = None
    
    # Load BM25 index
    try:
        if os.path.exists(BM25_INDEX_PATH) and os.path.exists(TOKENIZED_PARAGRAPHS_PATH):
            with open(BM25_INDEX_PATH, "rb") as f:
                bm25_index = pickle.load(f)
            with open(TOKENIZED_PARAGRAPHS_PATH, "rb") as f:
                tokenized_docs = pickle.load(f)
            logging.info("Loaded BM25 index successfully")
        else:
            logging.warning("BM25 index not found, will create if needed")
    except Exception as e:
        logging.error(f"Error loading BM25 index: {e}")
        bm25_index = None
        tokenized_docs = None
        
    # Create directories if they don't exist
    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
    os.makedirs(HEATMAP_DIR, exist_ok=True)
    os.makedirs(PDF_IMAGES_DIR, exist_ok=True)
    os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Retrieve relevant documents for RAG
async def retrieve_relevant_documents(query, top_k=5):
    """Retrieve most relevant documents using embeddings and BM25"""
    global colpali_embeddings, df, bm25_index, tokenized_docs
    
    if colpali_embeddings is None or df is None or len(df) == 0:
        logging.error("No documents or embeddings available for retrieval")
        return [], []
        
    retrieved_paragraphs = []
    top_sources_data = []
    
    # First try using sentence-transformers for vector search
    try:
        from sentence_transformers import SentenceTransformer, util
        
        # Initialize sentence-transformer model if needed for vector search
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Convert colpali embeddings to tensors for similarity comparison
        # Assuming colpali_embeddings is a list of numpy arrays
        document_embeddings = [torch.tensor(emb) for emb in colpali_embeddings]
        
        # Calculate similarities
        similarities = []
        for idx, doc_emb in enumerate(document_embeddings):
            # Ensure embeddings have same dimensions
            if len(doc_emb.shape) > 1 and doc_emb.shape[0] > 1:
                # If document has multiple vectors, take max similarity
                # Reshape to match dimensions for comparison
                doc_emb_reshaped = doc_emb.reshape(-1, doc_emb.shape[-1])
                sim = util.pytorch_cos_sim(query_embedding, doc_emb_reshaped).max().item()
            else:
                # Single vector case
                sim = util.pytorch_cos_sim(query_embedding, doc_emb).item()
            
            similarities.append((idx, sim))
        
        # Sort by similarity score
        similarities.sort(key=lambda x: x[1], reverse=True)
        vector_top_indices = [idx for idx, _ in similarities[:top_k]]
        
        # Try BM25 keyword search if available
        keyword_top_indices = []
        bm25_scores = None
        if bm25_index is not None and tokenized_docs is not None:
            try:
                # Tokenize query and get BM25 scores
                tokenized_query = word_tokenize(query.lower())
                bm25_scores = bm25_index.get_scores(tokenized_query)
                keyword_top_indices = np.argsort(bm25_scores)[-top_k:][::-1].tolist()
            except Exception as e:
                logging.error(f"Error in BM25 scoring: {e}")
        
        # Create BM25 index if it doesn't exist yet
        elif df is not None and len(df) > 0:
            try:
                logging.info("Building BM25 index from document texts")
                # Tokenize all documents
                tokenized_docs = []
                for _, row in df.iterrows():
                    tokenized_docs.append(word_tokenize(row['text'].lower()))
                
                # Create BM25 index
                bm25_index = BM25Okapi(tokenized_docs)
                
                # Get scores for the current query
                tokenized_query = word_tokenize(query.lower())
                bm25_scores = bm25_index.get_scores(tokenized_query)
                keyword_top_indices = np.argsort(bm25_scores)[-top_k:][::-1].tolist()
                
                # Save the index and tokenized documents for future use
                os.makedirs(DATA_DIR, exist_ok=True)
                with open(os.path.join(DATA_DIR, "bm25_index.pkl"), "wb") as f:
                    pickle.dump(bm25_index, f)
                with open(os.path.join(DATA_DIR, "tokenized_paragraphs.pkl"), "wb") as f:
                    pickle.dump(tokenized_docs, f)
                    
                logging.info("Created and saved BM25 index")
            except Exception as e:
                logging.error(f"Error creating BM25 index: {e}")
        
        # Combine results (hybrid retrieval)
        all_indices = list(set(vector_top_indices + keyword_top_indices))
        
        # Get data for reranking
        docs_for_reranking = []
        doc_indices = []
        
        for idx in all_indices:
            if idx < len(df):
                # Get document info
                filename = df.iloc[idx]['filename']
                page_num = df.iloc[idx]['page']
                image_key = df.iloc[idx]['image_key']
                text = df.iloc[idx]['text']
                
                # Get vector score
                vector_score = 0.0
                for v_idx, score in similarities:
                    if v_idx == idx:
                        vector_score = score
                        break
                
                # Get keyword score (if available)
                keyword_score = 0.0
                if bm25_scores is not None and len(bm25_scores) > idx:
                    keyword_score = float(bm25_scores[idx] / max(bm25_scores) if max(bm25_scores) > 0 else 0)
                
                # Combine scores (weighted)
                alpha = 0.7  # Weight for vector search
                combined_score = alpha * vector_score + (1 - alpha) * keyword_score
                
                # Store for reranking
                docs_for_reranking.append(text)
                doc_indices.append(idx)
                
                # Add to results
                retrieved_paragraphs.append(text)
                top_sources_data.append({
                    'filename': filename,
                    'page': page_num,
                    'score': combined_score,
                    'vector_score': vector_score,
                    'keyword_score': keyword_score,
                    'image_key': image_key,
                    'idx': idx
                })
        
        # Rerank results if we have documents
        if docs_for_reranking:
            try:
                # Use a cross-encoder reranker
                ranker = Reranker('cross-encoder/ms-marco-MiniLM-L-6-v2', model_type="cross-encoder", verbose=0)
                ranked_results = ranker.rank(query=query, docs=docs_for_reranking)
                top_ranked = ranked_results.top_k(min(3, len(docs_for_reranking)))
                
                # Get the final top documents after reranking
                final_retrieved_paragraphs = []
                final_top_sources = []
                
                for ranked_doc in top_ranked:
                    ranked_idx = docs_for_reranking.index(ranked_doc.text)
                    doc_idx = doc_indices[ranked_idx]
                    source_info = next((s for s in top_sources_data if s['idx'] == doc_idx), None)
                    if source_info:
                        source_info['reranker_score'] = ranked_doc.score
                        final_top_sources.append(source_info)
                        final_retrieved_paragraphs.append(ranked_doc.text)
                
                return final_retrieved_paragraphs, final_top_sources
                
            except Exception as e:
                logging.error(f"Error in reranking: {e}")
        
        # If reranking fails, sort by combined score
        sorted_indices = sorted(range(len(top_sources_data)), 
                               key=lambda i: top_sources_data[i]['score'], 
                               reverse=True)
        sorted_paragraphs = [retrieved_paragraphs[i] for i in sorted_indices[:3]]
        sorted_sources = [top_sources_data[i] for i in sorted_indices[:3]]
        
        return sorted_paragraphs, sorted_sources
        
    except Exception as e:
        logging.error(f"Error in document retrieval: {str(e)}")
        traceback.print_exc()
        return [], []

# synchronous wrapper for the async retrieve_relevant_documents
def retrieve_relevant_documents_sync(query, top_k=5):
    """Synchronous version of retrieve_relevant_documents for use in non-async functions"""
    import asyncio
    
    # Create a new event loop
    loop = asyncio.new_event_loop()
    
    try:
        # Run the async function in the event loop and get the result
        result = loop.run_until_complete(retrieve_relevant_documents(query, top_k))
        return result
    finally:
        # Clean up the event loop
        loop.close()


# Format image for API calls
def format_image(image):
    """Convert PIL Image to base64 for API"""
    buffered = BytesIO()
    # Convert to RGB if it has alpha channel
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG", quality=90)
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_base64

# Get context image from PDF pages
def get_context_image(top_sources):
    """Get the context image from retrieved PDF pages"""
    global page_images
    
    if not top_sources or len(top_sources) == 0:
        return None
        
    # Get the top source document
    top_source = top_sources[0]
    image_key = top_source.get('image_key')
    
    if not image_key or image_key not in page_images:
        logging.warning(f"Context image not found for key: {image_key}")
        
        # Try to find the image by reconstructing path patterns
        parts = image_key.split('_')
        if len(parts) >= 2:
            filename = '_'.join(parts[:-1])
            page_num = parts[-1]
            
            # Check different potential locations
            potential_paths = [
                os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, filename, f"page_{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
            ]
            
            for potential_path in potential_paths:
                if os.path.exists(potential_path):
                    logging.info(f"Found context image at: {potential_path}")
                    return Image.open(potential_path)
            
        return None
        
    try:
        image_path = page_images[image_key]
        if os.path.exists(image_path):
            context_image = Image.open(image_path)
            return context_image
        else:
            logging.error(f"Context image file not found: {image_path}")
    except Exception as e:
        logging.error(f"Error loading context image: {e}")
    
    return None

# Helper function to get context image path
def get_context_image_path(top_sources):
    """Get the file path for the context image"""
    global page_images
    
    if not top_sources or len(top_sources) == 0:
        return None
        
    # Get the top source document
    top_source = top_sources[0]
    image_key = top_source.get('image_key')
    
    if not image_key or image_key not in page_images:
        # Try to find the image by reconstructing path patterns
        parts = image_key.split('_')
        if len(parts) >= 2:
            filename = '_'.join(parts[:-1])
            page_num = parts[-1]
            
            # Check different potential locations
            potential_paths = [
                os.path.join(PDF_IMAGES_DIR, filename, f"{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, filename, f"page_{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{filename}_{page_num}.png")
            ]
            
            for potential_path in potential_paths:
                if os.path.exists(potential_path):
                    return potential_path
            
        return None
    
    return page_images[image_key] if image_key in page_images else None


# Get template path based on environment
def get_template_path(filename):
    """Get the correct template path based on environment"""
    # For Modal deployment
    if os.path.exists(TEMPLATES_DIR):
        return os.path.join(TEMPLATES_DIR, filename)
    
    # For local development
    local_template_dir = os.path.join(os.path.dirname(__file__), "templates")
    if os.path.exists(local_template_dir):
        return os.path.join(local_template_dir, filename)
    
    # Fallback - current directory
    return filename

# Helper functions for generating HTML components
def generate_confidence_badge(confidence):
    """Generate HTML for confidence badge with appropriate color"""
    confidence_class = 'badge-warning'
    if confidence == 'High':
        confidence_class = 'badge-success'
    elif confidence == 'Low':
        confidence_class = 'badge-error'
    
    return f'<span class="badge {confidence_class}">{confidence}</span>'

def generate_feedback_buttons(result_id):
    """Generate HTML for feedback buttons with HTMX attributes"""
    return f"""
    <div class="flex space-x-1">
        <button 
            class="btn btn-xs btn-circle btn-outline" 
            hx-post="/api/feedback"
            hx-vals='{{"id": "{result_id}", "feedback": "positive"}}'
            hx-swap="outerHTML"
            title="Positive Feedback">
            👍
        </button>
        <button 
            class="btn btn-xs btn-circle btn-outline" 
            hx-post="/api/feedback"
            hx-vals='{{"id": "{result_id}", "feedback": "negative"}}'
            hx-swap="outerHTML"
            title="Negative Feedback">
            👎
        </button>
    </div>
    """

def generate_feedback_badge(feedback):
    """Generate HTML for feedback badge based on type"""
    if not feedback:
        return ""
    
    feedback_class = "badge-success" if feedback == "positive" else "badge-error"
    return f'<span class="badge {feedback_class}">{feedback}</span>'

def create_image_thumbnail(image_path=None):
    """Create HTML for an image thumbnail, with placeholder if no image is available"""
    if image_path and os.path.exists(image_path):
        # Use actual image if available
        return f"""
        <div class="avatar">
            <div class="mask mask-squircle w-12 h-12">
                <img src="/image-thumbnail?path={image_path}" alt="Classification Image">
            </div>
        </div>
        """
    else:
        # Use placeholder if no image available
        return """
        <div class="avatar">
            <div class="mask mask-squircle w-12 h-12 bg-base-300 flex items-center justify-center">
                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-base-content/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
            </div>
        </div>
        """

# Helper function to get image path for a classification
def get_classification_image_path(result_id):
    """Attempt to find the image path for a classification result"""
    # This is a placeholder implementation - in a real application, 
    # you would store the image paths in the database or retrieve them
    # from a file system based on the result ID
    
    # For now, we'll return None to use placeholders
    return None

def get_trend_indicator(stats):
    """Calculate a trend indicator for daily classifications"""
    daily_counts = stats.get("daily_counts", [])
    
    if len(daily_counts) < 2:
        return {
            "value": 0,
            "trend": "neutral",
            "html": '<span class="text-base-content">-</span>'
        }
    
    # Get counts only
    counts = [count for _, count in daily_counts]
    
    # Calculate trend - compare first half to second half
    mid_point = len(counts) // 2
    first_half = sum(counts[:mid_point])
    second_half = sum(counts[mid_point:])
    
    if first_half > 0:
        trend_value = ((second_half - first_half) / first_half) * 100
        trend_value = round(trend_value, 1)
    else:
        trend_value = 0
    
    # Determine trend direction
    trend_direction = "positive" if trend_value > 0 else "negative" if trend_value < 0 else "neutral"
    
    # Create HTML
    if trend_direction == "positive":
        html = f"""
        <span class="inline-flex items-center text-success">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 10l7-7m0 0l7 7m-7-7v18" />
            </svg>
            {abs(trend_value)}%
        </span>
        """
    elif trend_direction == "negative":
        html = f"""
        <span class="inline-flex items-center text-error">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 14l-7 7m0 0l-7-7m7 7V3" />
            </svg>
            {abs(trend_value)}%
        </span>
        """
    else:
        html = '<span class="text-base-content">0%</span>'
    
    return {
        "value": trend_value,
        "trend": trend_direction,
        "html": html
    }


def dashboard():
    """Render the enhanced insect classification dashboard with Flowbite components and HTMX"""
    stats = get_classification_stats()
    
    # Create navigation bar
    navbar = Div(
        Div(
            A(
                Span("🐝", cls="text-xl"),
                Span("Insect Classifier", cls="ml-2 text-xl font-semibold"),
                href="/",
                cls="flex items-center"
            ),
            Div(
                A(
                    "Dashboard",
                    href="/dashboard",
                    cls="btn btn-sm btn-ghost btn-active"
                ),
                A(
                    "Classifier",
                    href="/",
                    cls="btn btn-sm btn-ghost"
                ),
                cls="flex-none"
            ),
            cls="navbar bg-base-200 rounded-lg mb-8 shadow-sm"
        ),
        cls="w-full"
    )
    
    # Line chart section
    trend_data = get_trend_indicator(stats)
    
    # Here's the fix - use NotStr instead of Raw
    Div(
        P(
            NotStr(trend_data["html"]),  # This line is fixed
            cls="text-xl font-bold",
            id="trend-indicator"
        )
    )

def generate_flowbite_table_rows(results):
    """Generate HTML for Flowbite table rows"""
    if not results:
        return """
        <tr>
            <td colspan="6" class="px-6 py-4 text-center">
                No classification results found.
            </td>
        </tr>
        """
    
    html = ""
    for id, category, confidence, feedback, created_at, context_source in results:
        # Determine confidence class for Flowbite styling
        confidence_class = 'bg-yellow-100 text-yellow-800'
        if confidence == 'High':
            confidence_class = 'bg-green-100 text-green-800'
        elif confidence == 'Low':
            confidence_class = 'bg-red-100 text-red-800'
        
        # Build the table row
        html += f"""
        <tr class="bg-white border-b dark:bg-gray-800 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600">
            <!-- Image column -->
            <td class="px-6 py-4">
                <div class="relative w-10 h-10 overflow-hidden bg-gray-100 rounded-full dark:bg-gray-600 cursor-pointer"
                     onclick="showImageModal('{id}', '{category}', '{confidence}')">
                    <svg class="absolute w-12 h-12 text-gray-400 -left-1" fill="currentColor" viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg">
                        <path fill-rule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clip-rule="evenodd"></path>
                    </svg>
                </div>
            </td>
            
            <!-- Category column -->
            <td class="px-6 py-4 font-medium text-gray-900 whitespace-nowrap dark:text-white">
                {category}
            </td>
            
            <!-- Confidence column -->
            <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {confidence_class}">
                    {confidence}
                </span>
            </td>
        """
        
        # Feedback column with HTMX support
        if feedback:
            feedback_class = 'bg-green-100 text-green-800' if feedback == 'positive' else 'bg-red-100 text-red-800'
            html += f"""
            <td class="px-6 py-4">
                <span class="px-2 py-1 rounded-full text-xs font-semibold {feedback_class}">
                    {feedback}
                </span>
            </td>
            """
        else:
            html += f"""
            <td class="px-6 py-4">
                <div class="flex space-x-1" hx-target="this" hx-swap="outerHTML">
                    <button 
                        class="text-gray-500 hover:text-green-500 focus:outline-none" 
                        hx-post="/api/feedback"
                        hx-vals='{{"id": "{id}", "feedback": "positive"}}'>
                        👍
                    </button>
                    <button 
                        class="text-gray-500 hover:text-red-500 focus:outline-none" 
                        hx-post="/api/feedback"
                        hx-vals='{{"id": "{id}", "feedback": "negative"}}'>
                        👎
                    </button>
                </div>
            </td>
            """
        
        # Date and actions columns
        html += f"""
            <td class="px-6 py-4">
                {created_at}
            </td>
            
            <td class="px-6 py-4 text-right">
                <button 
                    class="font-medium text-blue-600 dark:text-blue-500 hover:underline"
                    onclick="showImageModal('{id}', '{category}', '{confidence}')">
                    View
                </button>
            </td>
        </tr>
        """
    
    return html

# Generate classification using Claude's API for a single image
@app.function(
    image=image,
    cpu=1.0,
    timeout=300,
    volumes={DATA_DIR: bee_volume}
)
def classify_image_claude(image_data: str, options: Dict[str, bool]) -> Dict[str, Any]:
    """
    Classify insect in image using Claude's API based on provided options
    
    Args:
        image_data: Base64 encoded image
        options: Dictionary of toggle options
    
    Returns:
        Dictionary with classification results
    """
    result_id = uuid.uuid4().hex
    
    # Build additional instructions based on options
    additional_instructions = []
    format_instructions = []
    
    if options.get("detailed_description", False):
        additional_instructions.append("Provide a detailed description of the insect, focusing on shapes and colors visible in the image.")
        format_instructions.append("- Detailed Description: [shapes, colors, and distinctive features]")
        
    if options.get("plant_classification", False):
        additional_instructions.append("If there are any plants visible in the image, identify them to the best of your ability.")
        format_instructions.append("- Plant Identification: [names of visible plants, if any]")
        
    if options.get("taxonomy", False):
        additional_instructions.append("Provide taxonomic classification of the insect to the most specific level possible (Order, Family, Genus, Species).")
        format_instructions.append("- Taxonomy: [Order, Family, Genus, Species where possible]")
    
    # Get relevant context using RAG retrieval
    context_text = ""
    context_source = None
    retrieved_paragraphs = []
    top_sources = []
    
    if options.get("use_rag", True):  # Default to using RAG
        query = "insect classification"
        # Use the synchronous wrapper to call the async function
        retrieved_paragraphs, top_sources = retrieve_relevant_documents_sync(query, top_k=3)

        if retrieved_paragraphs:
            context_text = "\n\nReference Information:\n" + "\n\n".join(retrieved_paragraphs)
            if top_sources and len(top_sources) > 0:
                source = top_sources[0]
                context_source = f"{source.get('filename', 'unknown document')}, page {source.get('page', 'unknown')}"
    
    # Prepare the prompt
    categories_list = "\n".join([f"- {category}" for category in INSECT_CATEGORIES])
    additional_instructions_text = "\n".join(additional_instructions) if additional_instructions else ""
    format_instructions_text = "\n".join(format_instructions) if format_instructions else ""
    
    prompt = CLASSIFICATION_PROMPT.format(
        categories=categories_list,
        context_text=context_text,
        additional_instructions=additional_instructions_text,
        format_instructions=format_instructions_text
    )
    
    print("🔍 Sending image to Claude for classification...")
    
    try:
        # Prepare the request for Claude API
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        
        # Make the API call
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        # Extract the response content
        result = response.json()
        classification_text = result["content"][0]["text"]
        
        # Parse the classification result
        # Simple parsing based on the expected format
        lines = classification_text.strip().split("\n")
        parsed_result = {}
        
        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().replace("- ", "")
                value = value.strip()
                parsed_result[key] = value
        
        # Store essential information
        category = parsed_result.get("Main Category", "Unclassified")
        confidence = parsed_result.get("Confidence", "Low")
        description = parsed_result.get("Description", "No description provided")
        
        # Store the full result in the database
        try:
            conn = setup_database(DB_PATH)
            cursor = conn.cursor()
            
            # Include context_source in the insert
            cursor.execute(
                "INSERT INTO results (id, category, confidence, description, additional_details, context_source) VALUES (?, ?, ?, ?, ?, ?)",
                (result_id, category, confidence, description, json.dumps(parsed_result), context_source)
            )
            
            conn.commit()
            conn.close()
            
            return {
                "id": result_id,
                "category": category,
                "confidence": confidence,
                "description": description,
                "details": parsed_result,
                "context_source": context_source,
                "context_paragraphs": retrieved_paragraphs,
                "top_sources": top_sources,
                "raw_response": classification_text
            }
            
        except Exception as db_error:
            print(f"⚠️ Error saving to database: {db_error}")
            # Still return the result even if database save fails
            return {
                "id": result_id,
                "category": category,
                "confidence": confidence,
                "description": description,
                "details": parsed_result,
                "context_source": context_source,
                "context_paragraphs": retrieved_paragraphs,
                "top_sources": top_sources,
                "raw_response": classification_text,
                "db_error": str(db_error)
            }
            
    except Exception as e:
        print(f"⚠️ Error in classification: {e}")
        return {
            "error": str(e),
            "id": result_id
        }
    
# Batch classification function
@app.function(
    image=image,
    cpu=1.0,
    timeout=500,  # Longer timeout for batch processing
    volumes={DATA_DIR: bee_volume}
)
def classify_batch_claude(images_data: List[str], options: Dict[str, bool]) -> Dict[str, Any]:
    """
    Classify multiple insect images in batch using Claude's API
    
    Args:
        images_data: List of base64 encoded images (max 5)
        options: Dictionary of toggle options
    
    Returns:
        Dictionary with batch classification results
    """
    batch_id = uuid.uuid4().hex
    
    # Limit to max 5 images per batch for cost/performance
    max_images = 5
    if len(images_data) > max_images:
        images_data = images_data[:max_images]
    
    # Build additional instructions based on options
    additional_instructions = []
    format_instructions = []
    
    if options.get("detailed_description", False):
        additional_instructions.append("Provide a detailed description of the insect, focusing on shapes and colors visible in the image.")
        format_instructions.append("- Detailed Description: [shapes, colors, and distinctive features]")
        
    if options.get("plant_classification", False):
        additional_instructions.append("If there are any plants visible in the image, identify them to the best of your ability.")
        format_instructions.append("- Plant Identification: [names of visible plants, if any]")
        
    if options.get("taxonomy", False):
        additional_instructions.append("Provide taxonomic classification of the insect to the most specific level possible (Order, Family, Genus, Species).")
        format_instructions.append("- Taxonomy: [Order, Family, Genus, Species where possible]")
    
    # Get relevant context using RAG retrieval if enabled
    context_text = ""
    context_source = None
    retrieved_paragraphs = []
    top_sources = []
    
    if options.get("use_rag", True):  # Default to using RAG
        query = "insect classification"
        # Use the synchronous wrapper to call the async function
        retrieved_paragraphs, top_sources = retrieve_relevant_documents_sync(query, top_k=3)
        
        if retrieved_paragraphs:
            context_text = "\n\nReference Information:\n" + "\n\n".join(retrieved_paragraphs)
            if top_sources and len(top_sources) > 0:
                source = top_sources[0]
                context_source = f"{source.get('filename', 'unknown document')}, page {source.get('page', 'unknown')}"
    
    # Prepare the batch prompt
    categories_list = "\n".join([f"- {category}" for category in INSECT_CATEGORIES])
    additional_instructions_text = "\n".join(additional_instructions) if additional_instructions else ""
    format_instructions_text = "\n".join(format_instructions) if format_instructions else ""
    
    prompt = BATCH_PROMPT.format(
        count=len(images_data),
        categories=categories_list,
        context_text=context_text,
        additional_instructions=additional_instructions_text,
        format_instructions=format_instructions_text
    )
    
    print(f"🔍 Sending batch of {len(images_data)} images to Claude for classification...")
    
    try:
        # Prepare the request for Claude API
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Build content array with all images first
        content = []
        for img_data in images_data:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img_data
                }
            })
        
        # Add the text prompt at the end
        content.append({
            "type": "text",
            "text": prompt
        })
        
        payload = {
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1500,  # Increased for multi-image response
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }
        
        # Make the API call
        response = requests.post(CLAUDE_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        # Extract the response content
        result = response.json()
        batch_text = result["content"][0]["text"]
        
        # Parse the batch results (split by "IMAGE X:")
        image_results = []
        
        # Split by "IMAGE" keyword
        raw_sections = batch_text.split("IMAGE ")
        
        # Remove any empty initial section
        if raw_sections and not raw_sections[0].strip():
            raw_sections = raw_sections[1:]
        elif raw_sections and not raw_sections[0].strip().startswith("1:"):
            # If first section doesn't start with a number, it's probably preamble
            raw_sections = raw_sections[1:]
        
        # Process each image section
        for i, section in enumerate(raw_sections):
            if i >= len(images_data):  # Safety check
                break
                
            # Clean up the section
            if section.strip().startswith(f"{i+1}:"):
                # Remove the image number prefix
                section = section.strip()[2:].strip()
            
            # Parse this section
            lines = section.strip().split("\n")
            parsed_result = {}
            
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().replace("- ", "")
                    value = value.strip()
                    parsed_result[key] = value
            
            # Get essential fields
            result_id = f"{batch_id}_{i}"
            category = parsed_result.get("Main Category", "Unclassified")
            confidence = parsed_result.get("Confidence", "Low")
            description = parsed_result.get("Description", "No description provided")
            
            # Add to results
            image_results.append({
                "id": result_id,
                "index": i,
                "category": category,
                "confidence": confidence,
                "description": description,
                "details": parsed_result
            })
        
        # Store batch results in database
        try:
            conn = setup_database(DB_PATH)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO batch_results (batch_id, result_count, results) VALUES (?, ?, ?)",
                (batch_id, len(image_results), json.dumps(image_results))
            )
            
            conn.commit()
            conn.close()
            
            # Save results to file
            save_results_file(batch_id, {
                "batch": True,
                "results": image_results,
                "raw_response": batch_text
            })
            
        except Exception as e:
            print(f"⚠️ Error saving batch results to database: {e}")
        
        return {
            "batch_id": batch_id,
            "count": len(image_results),
            "results": image_results,
            "context_source": context_source,
            "context_paragraphs": retrieved_paragraphs,
            "top_sources": top_sources,
            "raw_response": batch_text
        }
        
    except Exception as e:
        print(f"⚠️ Error in batch classification: {e}")
        traceback.print_exc()
        return {
            "error": str(e),
            "batch_id": batch_id
        }

# Main FastHTML Server with defined routes
@app.function(
    image=image,
    volumes={DATA_DIR: bee_volume},
    cpu=1.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    """Main FastHTML Server for Bee Classifier Dashboard with RAG"""
    from rank_bm25 import BM25Okapi
    # Load RAG data at startup
    load_rag_data()
    
    # Set up the FastHTML app with required headers
    fasthtml_app, rt = fast_app(
        hdrs=(
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@3.9.2/dist/full.css"),
            Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css"),
            # Add Flowbite CSS
            Link(rel="stylesheet", href="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.0/flowbite.min.css"),
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
            # Add Flowbite JavaScript
            Script(src="https://cdnjs.cloudflare.com/ajax/libs/flowbite/2.2.0/flowbite.min.js"),
            # Add ApexCharts for better charts
            Script(src="https://cdn.jsdelivr.net/npm/apexcharts"),
            # Add custom theme styles
            Style("""
                :root {
                --color-base-100: oklch(98% 0.002 247.839);
                --color-base-200: oklch(96% 0.003 264.542);
                --color-base-300: oklch(92% 0.006 264.531);
                --color-base-content: oklch(21% 0.034 264.665);
                --color-primary: oklch(47% 0.266 120.957);  /* Green for bees */
                --color-primary-content: oklch(97% 0.014 254.604);
                --color-secondary: oklch(74% 0.234 93.635);  /* Yellow for bees */
                --color-secondary-content: oklch(13% 0.028 261.692);
                --color-accent: oklch(41% 0.234 41.252);     /* Brown accent */
                --color-accent-content: oklch(97% 0.014 254.604);
                --color-neutral: oklch(13% 0.028 261.692);
                --color-neutral-content: oklch(98% 0.002 247.839);
                --color-info: oklch(58% 0.158 241.966);
                --color-info-content: oklch(97% 0.013 236.62);
                --color-success: oklch(62% 0.194 149.214);
                --color-success-content: oklch(98% 0.018 155.826);
                --color-warning: oklch(66% 0.179 58.318);
                --color-warning-content: oklch(98% 0.022 95.277);
                --color-error: oklch(59% 0.249 0.584);
                --color-error-content: oklch(97% 0.014 343.198);
                }

                /* Original styling remains */
                
                /* Custom styling */
                .text-bee-green {
                    color: oklch(47% 0.266 120.957);
                }
                
                .bg-bee-yellow {
                    background-color: oklch(74% 0.234 93.635);
                }
                
                .custom-border {
                    border-color: var(--color-base-300);
                }

                /* Confidence level colors */
                .confidence-high {
                    color: var(--color-success);
                }
                
                .confidence-medium {
                    color: var(--color-warning);
                }
                
                .confidence-low {
                    color: var(--color-error);
                }
                
                /* Batch specific styles */
                .batch-previews {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 10px;
                    margin: 15px 0;
                }
                
                .preview-item {
                    position: relative;
                    width: 80px;
                    height: 80px;
                }
                
                .preview-img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    border-radius: 0.5rem;
                    border: 2px solid var(--color-base-300);
                }
                
                .remove-btn {
                    position: absolute;
                    top: -8px;
                    right: -8px;
                    background: var(--color-error);
                    color: white;
                    border-radius: 50%;
                    width: 20px;
                    height: 20px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    cursor: pointer;
                }
                
                /* Add Flowbite compatibility styles */
                .flowbite-card {
                    background-color: var(--color-base-100);
                    border: 1px solid var(--color-base-300);
                    border-radius: 0.5rem;
                }
                
                .flowbite-table th, .flowbite-table td {
                    padding: 0.75rem 1rem;
                }
                
                /* Custom styles for the donut chart */
                #donut-chart-container {
                    height: 320px;
                }
                
                /* Custom styles for the line chart */
                #line-chart-container {
                    height: 300px;
                }
                
                /* Table image thumbnails */
                .bee-thumbnail {
                    width: 60px;
                    height: 60px;
                    object-fit: cover;
                    border-radius: 0.25rem;
                }
                
                /* HTMX loading indicator */
                .htmx-indicator {
                    opacity: 0;
                    transition: opacity 200ms ease-in;
                }
                .htmx-request .htmx-indicator {
                    opacity: 1;
                }
                .htmx-request.htmx-indicator {
                    opacity: 1;
                }
                
                /* Pie chart colors */
                [style*="--color-1"] {
                    background-color: var(--color-primary);
                }
                [style*="--color-2"] {
                    background-color: var(--color-secondary);
                }
                [style*="--color-3"] {
                    background-color: var(--color-accent);
                }
                [style*="--color-4"] {
                    background-color: #1e88e5;
                }
                [style*="--color-5"] {
                    background-color: #43a047;
                }
                [style*="--color-6"] {
                    background-color: #ffb300;
                }
                [style*="--color-7"] {
                    background-color: #e53935;
                }
                [style*="--color-8"] {
                    background-color: #8e24aa;
                }
                [style*="--color-9"] {
                    background-color: #00acc1;
                }
                [style*="--color-10"] {
                    background-color: #f4511e;
                }
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
    #################################################
    # Dashboard Route - Enhanced with Flowbite and HTMX
    #################################################
    @rt("/dashboard")
    def dashboard():
        """Render the enhanced insect classification dashboard with Flowbite components and HTMX"""
        stats = get_classification_stats()
        
        # Create navigation bar
        navbar = Div(
            Div(
                A(
                    Span("🐝", cls="text-xl"),
                    Span("Insect Classifier", cls="ml-2 text-xl font-semibold"),
                    href="/",
                    cls="flex items-center"
                ),
                Div(
                    A(
                        "Dashboard",
                        href="/dashboard",
                        cls="btn btn-sm btn-ghost btn-active"
                    ),
                    A(
                        "Classifier",
                        href="/",
                        cls="btn btn-sm btn-ghost"
                    ),
                    cls="flex-none"
                ),
                cls="navbar bg-base-200 rounded-lg mb-8 shadow-sm"
            ),
            cls="w-full"
        )
        
        # Stats summary cards section
        summary_cards = Div(
            Div(
                Div(
                    Div(
                        H3("Total Classifications", cls="font-bold text-lg"),
                        P(str(stats["total"]), cls="text-4xl font-semibold text-primary"),
                        cls="p-6"
                    ),
                    cls="bg-base-100 rounded-lg shadow-md border custom-border"
                ),
                Div(
                    Div(
                        H3("Single Images", cls="font-bold text-lg"),
                        P(str(stats["total_single"]), cls="text-3xl font-semibold"),
                        cls="p-6"
                    ),
                    cls="bg-base-100 rounded-lg shadow-md border custom-border"
                ),
                Div(
                    Div(
                        H3("Batch Images", cls="font-bold text-lg"),
                        P(str(stats["total_batch"]), cls="text-3xl font-semibold"),
                        cls="p-6"
                    ),
                    cls="bg-base-100 rounded-lg shadow-md border custom-border"
                ),
                cls="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8"
            ),
            cls="mb-8"
        )
        
        # Calculate data for the donut chart
        total_insects = sum(count for _, count in stats["combined_category_counts"])
        pie_data = []
        start_value = 0.0
        
        # Process top 10 categories or all if less than 10
        categories_to_display = stats["combined_category_counts"][:10]
        
        for category, count in categories_to_display:
            percentage = count / total_insects if total_insects > 0 else 0
            end_value = start_value + percentage
            
            pie_data.append({
                "category": category,
                "count": count,
                "percentage": percentage * 100,  # Convert to percentage
                "start": start_value,
                "end": end_value
            })
            
            start_value = end_value
        
        # Create Donut Chart HTML with ApexCharts
        donut_chart = Div(
            H3("Insect Category Distribution", cls="font-semibold mb-4 text-center text-bee-green text-lg"),
            
            # Chart Container
            Div(
                id="donut-chart-container",
                cls="mx-auto"
            ),
            
            # HTMX-powered chart reload button
            Div(
                Button(
                    "Refresh Chart Data",
                    cls="btn btn-sm btn-outline btn-primary mt-4",
                    hx_get="/api/chart-data",
                    hx_target="#donut-chart-container",
                    hx_trigger="click",
                    hx_indicator="#chart-loading"
                ),
                Span(
                    Span(cls="loading loading-spinner loading-xs ml-2"),
                    cls="htmx-indicator",
                    id="chart-loading"
                ),
                cls="text-center mt-4"
            ),
            
            # ApexCharts initialization script
            Script(f"""
            document.addEventListener('DOMContentLoaded', function() {{
                // Extract data from server-side rendering
                const categoryData = {json.dumps([{'category': cat, 'count': count} for cat, count in categories_to_display])};
                
                // Prepare data for ApexCharts
                const labels = categoryData.map(item => item.category);
                const counts = categoryData.map(item => item.count);
                
                // Custom bee-themed colors
                const colors = [
                    '#8B5A00', // Brown
                    '#FFC107', // Yellow
                    '#A5D6A7', // Light Green
                    '#66BB6A', // Medium Green
                    '#43A047', // Dark Green
                    '#FFB74D', // Light Orange
                    '#FFA000', // Dark Amber
                    '#E65100', // Dark Orange
                    '#795548', // Medium Brown
                    '#4E342E'  // Dark Brown
                ];
                
                // Initialize ApexCharts Donut Chart
                const donutChart = new ApexCharts(document.querySelector("#donut-chart-container"), {{
                    series: counts,
                    chart: {{
                        type: 'donut',
                        height: 320,
                        fontFamily: 'inherit',
                        foreColor: 'inherit',
                        animations: {{
                            enabled: true,
                            easing: 'easeinout',
                            speed: 800
                        }}
                    }},
                    labels: labels,
                    colors: colors,
                    legend: {{
                        position: 'bottom',
                        fontSize: '14px',
                        formatter: function(seriesName, opts) {{
                            // Show category name and percentage
                            return `${{seriesName}}: ${{Math.round(opts.w.globals.series[opts.seriesIndex]/opts.w.globals.seriesTotals[0]*100)}}%`;
                        }}
                    }},
                    tooltip: {{
                        y: {{
                            formatter: function(value) {{
                                return value + " classifications";
                            }}
                        }}
                    }},
                    dataLabels: {{
                        enabled: false
                    }},
                    responsive: [{{
                        breakpoint: 480,
                        options: {{
                            chart: {{
                                height: 260
                            }},
                            legend: {{
                                position: 'bottom'
                            }}
                        }}
                    }}],
                    plotOptions: {{
                        pie: {{
                            donut: {{
                                size: '50%',
                                labels: {{
                                    show: true,
                                    name: {{
                                        show: true
                                    }},
                                    value: {{
                                        show: true,
                                        formatter: function(val) {{
                                            return val;
                                        }}
                                    }},
                                    total: {{
                                        show: true,
                                        label: 'Total',
                                        formatter: function(w) {{
                                            return w.globals.seriesTotals.reduce((a, b) => a + b, 0);
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }});
                
                // Render the chart
                donutChart.render();
                
                // Set up HTMX event listener to update chart when new data is received
                document.body.addEventListener('htmx:afterSwap', function(evt) {{
                    if (evt.detail.target.id === 'donut-chart-container') {{
                        // Parse the new data (assuming JSON response)
                        try {{
                            const newData = JSON.parse(evt.detail.xhr.response);
                            donutChart.updateSeries(newData.counts);
                            donutChart.updateOptions({{
                                labels: newData.labels
                            }});
                        }} catch(e) {{
                            console.error('Error updating chart:', e);
                        }}
                    }}
                }});
            }});
            """),
            
            cls="w-full bg-base-100 p-6 rounded-lg shadow-md border custom-border"
        )
        
        # Create Line Chart for Activity
        trend_data = get_trend_indicator(stats)
        line_chart = Div(
            Div(
                H3("Daily Classification Activity", cls="font-semibold text-bee-green text-lg"),
                
                # HTMX-powered time range selector
                Div(
                    Div(role="button", tabindex="0", cls="btn btn-sm btn-outline"),
                    Span("Last 7 days", id="current-range"),
                    Svg(
                        Path(
                            stroke="currentColor",
                            stroke_linecap="round",
                            stroke_linejoin="round",
                            stroke_width="2",
                            d="m1 1 4 4 4-4"
                        ),
                        cls="w-2.5 h-2.5 ml-1.5",
                        aria_hidden="true",
                        xmlns="http://www.w3.org/2000/svg",
                        fill="none",
                        viewBox="0 0 10 6"
                    ),
                    cls="dropdown dropdown-end"
                ),
                
                cls="flex justify-between mb-4"
            ),
            
            # Chart Container with Loading Indicator
            Div(
                Div(id="line-chart-container", cls="w-full"),
                Div(
                    Span(cls="loading loading-spinner loading-md text-primary"),
                    id="line-loading",
                    cls="htmx-indicator absolute inset-0 flex items-center justify-center bg-base-100 bg-opacity-60"
                ),
                cls="relative"
            ),
            
            # Summary Statistics
            Div(
                Div(
                    P("Total", cls="text-sm text-base-content/70"),
                    P(str(stats["total"]), cls="text-xl font-bold", id="total-classifications")
                ),
                Div(
                    P("Average / Day", cls="text-sm text-base-content/70"),
                    P(
                        str(round(stats["total"] / len(stats["daily_counts"])) if stats["daily_counts"] else 0),
                        cls="text-xl font-bold",
                        id="avg-classifications"
                    )
                ),
                Div(
                    P("Trend", cls="text-sm text-base-content/70"),
                    P(
                        NotStr(trend_data["html"]),
                        cls="text-xl font-bold",
                        id="trend-indicator"
                    )
                ),
                cls="grid grid-cols-3 gap-4 mt-4"
            ),
            
            # Line Chart Initialization Script
            Script(f"""
            document.addEventListener('DOMContentLoaded', function() {{
                // Extract data from server-side rendering
                const dailyData = {json.dumps([{'date': date, 'count': count} for date, count in stats["daily_counts"]])};
                
                // Prepare data for ApexCharts
                const dates = dailyData.map(item => item.date);
                const counts = dailyData.map(item => item.count);
                
                // Initialize the line chart
                const lineChart = new ApexCharts(document.querySelector("#line-chart-container"), {{
                    series: [{{
                        name: 'Classifications',
                        data: counts
                    }}],
                    chart: {{
                        height: 300,
                        type: 'line',
                        fontFamily: 'inherit',
                        foreColor: 'inherit',
                        toolbar: {{
                            show: false
                        }},
                        animations: {{
                            enabled: true,
                            easing: 'easeinout',
                            speed: 800
                        }}
                    }},
                    stroke: {{
                        width: 3,
                        curve: 'smooth'
                    }},
                    colors: ['oklch(47% 0.266 120.957)'], // Primary green color
                    markers: {{
                        size: 5,
                        strokeWidth: 0,
                        hover: {{
                            size: 7
                        }}
                    }},
                    xaxis: {{
                        categories: dates,
                        labels: {{
                            rotateAlways: false,
                            style: {{
                                fontSize: '12px'
                            }}
                        }}
                    }},
                    yaxis: {{
                        title: {{
                            text: 'Classifications'
                        }},
                        min: 0,
                        forceNiceScale: true
                    }},
                    tooltip: {{
                        shared: true,
                        intersect: false,
                        y: {{
                            formatter: function(value) {{
                                return value + " classifications";
                            }}
                        }}
                    }},
                    grid: {{
                        show: true,
                        borderColor: 'var(--color-base-300)',
                        strokeDashArray: 5,
                        position: 'back'
                    }},
                    fill: {{
                        type: 'gradient',
                        gradient: {{
                            shade: 'light',
                            type: "vertical",
                            shadeIntensity: 0.3,
                            inverseColors: false,
                            opacityFrom: 0.7,
                            opacityTo: 0.2,
                            stops: [0, 100]
                        }}
                    }}
                }});
                
                // Render the line chart
                lineChart.render();
            }});
            """),
            
            cls="w-full bg-base-100 p-6 rounded-lg shadow-md border custom-border"
        )
        
        # Charts Section
        charts_section = Div(
            H2("Classification Overview", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                Div(
                    donut_chart,
                    cls="w-full lg:w-1/2"
                ),
                Div(
                    line_chart,
                    cls="w-full lg:w-1/2"
                ),
                cls="flex flex-col lg:flex-row gap-6 w-full"
            ),
            cls="mb-8"
        )
        
        # Confidence & Feedback Section
        confidence_feedback_section = Div(
            Div(
                Div(
                    H3("Confidence Levels", cls="font-semibold mb-3"),
                    Table(
                        Thead(
                            Tr(
                                Th("Confidence"),
                                Th("Count"),
                            )
                        ),
                        Tbody(
                            *[
                                Tr(
                                    Td(
                                        Span(
                                            confidence,
                                            cls=f"badge {'badge-success' if confidence == 'High' else 'badge-warning' if confidence == 'Medium' else 'badge-error'}"
                                        )
                                    ),
                                    Td(str(count)),
                                )
                                for confidence, count in stats["confidence_counts"]
                            ]
                        ),
                        cls="table w-full"
                    ),
                    cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
                ),
                Div(
                    H3("User Feedback", cls="font-semibold mb-3"),
                    Table(
                        Thead(
                            Tr(
                                Th("Feedback"),
                                Th("Count"),
                            )
                        ),
                        Tbody(
                            *[
                                Tr(
                                    Td(
                                        Span(
                                            feedback,
                                            cls=f"badge {'badge-success' if feedback == 'positive' else 'badge-error'}"
                                        )
                                    ),
                                    Td(str(count)),
                                )
                                for feedback, count in stats["feedback_counts"]
                            ] if stats["feedback_counts"] else [
                                Tr(
                                    Td("No feedback yet"),
                                    Td("0")
                                )
                            ]
                        ),
                        cls="table w-full"
                    ),
                    cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
                ),
                cls="grid grid-cols-1 md:grid-cols-2 gap-6"
            ),
            cls="mb-8"
        )
        
        # DaisyUI Table for Recent Classifications
        daisyui_table = Div(
            Div(
                H3("Recent Classifications", cls="font-semibold mb-3"),
                
                # HTMX-powered refresh button
                Button(
                    Svg(
                        Path(
                            stroke_linecap="round",
                            stroke_linejoin="round",
                            stroke_width="2",
                            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                        ),
                        cls="h-4 w-4 mr-1",
                        xmlns="http://www.w3.org/2000/svg",
                        fill="none",
                        viewBox="0 0 24 24",
                        stroke="currentColor"
                    ),
                    "Refresh",
                    cls="btn btn-sm btn-outline btn-primary",
                    hx_get="/api/recent-classifications",
                    hx_target="#classifications-table-container",
                    hx_trigger="click",
                    hx_indicator="#table-loading"
                ),
                Span(
                    Span(cls="loading loading-spinner loading-xs ml-2"),
                    cls="htmx-indicator",
                    id="table-loading"
                ),
                
                cls="flex justify-between items-center mb-4"
            ),
            
            # Table Container
            Div(
                Table(
                    # Table Head
                    Thead(
                        Tr(
                            Th("Image"),
                            Th("ID"),
                            Th("Category"),
                            Th("Confidence"),
                            Th("Feedback"),
                            Th("Source"),
                            Th("Time"),
                            Th("Actions")
                        )
                    ),
                    
                    # Table Body
                    Tbody(
                        *[
                            Tr(
                                # Image Cell
                                Td(
                                    NotStr(create_image_thumbnail(get_classification_image_path(id)))
                                ),
                                
                                # ID Cell
                                Td(f"{id[:8]}..."),
                                
                                # Category Cell
                                Td(category),
                                
                                # Confidence Cell
                                Td(NotStr(generate_confidence_badge(confidence))),
                                
                                # Feedback Cell
                                Td(
                                    NotStr(generate_feedback_badge(feedback) if feedback else generate_feedback_buttons(id))
                                ),
                                
                                # Source Cell
                                Td(
                                    context_source if context_source else "None",
                                    cls="max-w-xs truncate",
                                    title=context_source if context_source else "No source"
                                ),
                                
                                # Time Cell
                                Td(created_at),
                                
                                # Actions Cell
                                Td(
                                    Div(
                                        Div(role="button", tabindex="0", cls="btn btn-xs btn-ghost m-1"),
                                        Svg(
                                            Path(
                                                stroke_linecap="round",
                                                stroke_linejoin="round",
                                                stroke_width="2",
                                                d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z"
                                            ),
                                            cls="h-4 w-4",
                                            xmlns="http://www.w3.org/2000/svg",
                                            fill="none",
                                            viewBox="0 0 24 24",
                                            stroke="currentColor"
                                        ),
                                        cls="dropdown dropdown-end"
                                    )
                                )
                            )
                            for id, category, confidence, feedback, created_at, context_source in stats["recent_classifications"]
                        ]
                    ),
                    cls="table table-zebra w-full"
                ),
                id="classifications-table-container",
                cls="overflow-x-auto"
            ),
            cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
        )
        
        # Flowbite Table for All Classifications
        flowbite_table = Div(
            Div(
                H3("All Classifications", cls="font-semibold mb-3"),
                
                # Search and Filter Controls
                Div(
                    # Search input with HTMX
                    Div(
                        Div(
                            Svg(
                                Path(
                                    stroke="currentColor",
                                    stroke_linecap="round",
                                    stroke_linejoin="round",
                                    stroke_width="2",
                                    d="m19 19-4-4m0-7A7 7 0 1 1 1 8a7 7 0 0 1 14 0Z"
                                ),
                                cls="w-4 h-4 text-gray-500 dark:text-gray-400",
                                aria_hidden="true",
                                xmlns="http://www.w3.org/2000/svg",
                                fill="none",
                                viewBox="0 0 20 20"
                            ),
                            cls="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none"
                        ),
                        Input(
                            type="text",
                            id="classification-search",
                            cls="block w-full p-2 pl-10 text-sm border border-base-300 rounded-lg bg-base-100",
                            placeholder="Search classifications...",
                            hx_post="/api/search-classifications",
                            hx_trigger="keyup changed delay:500ms",
                            hx_target="#flowbite-table-body",
                            hx_indicator="#search-indicator"
                        ),
                        Div(
                            Span(cls="loading loading-spinner loading-xs"),
                            id="search-indicator",
                            cls="htmx-indicator absolute inset-y-0 right-0 flex items-center pr-3"
                        ),
                        cls="relative w-full sm:w-64"
                    ),
                    
                    # Category filter dropdown
                    Select(
                        Option("All Categories", value=""),
                        *[
                            Option(category, value=category)
                            for category, _ in stats["category_counts"]
                        ],
                        cls="block w-full sm:w-auto p-2 text-sm border border-base-300 rounded-lg bg-base-100",
                        hx_post="/api/filter-classifications",
                        hx_trigger="change",
                        hx_target="#flowbite-table-body",
                        hx_indicator="#filter-indicator"
                    ),
                    Span(
                        cls="htmx-indicator loading loading-spinner loading-xs ml-2",
                        id="filter-indicator"
                    ),
                    
                    cls="flex flex-col sm:flex-row gap-3 w-full md:w-auto"
                ),
                
                cls="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-4"
            ),
            
            # Flowbite Table Container
            Div(
                Table(
                    # Table Head
                    Thead(
                        Tr(
                            Th(
                                Span("Image", cls="flex items-center"),
                                scope="col",
                                cls="px-6 py-3"
                            ),
                            Th(
                                Div(
                                    "Category",
                                    A(
                                        Svg(
                                            Path(
                                                d="M8.574 11.024h6.852a2.075 2.075 0 0 0 1.847-1.086 1.9 1.9 0 0 0-.11-1.986L13.736 2.9a2.122 2.122 0 0 0-3.472 0L6.837 7.952a1.9 1.9 0 0 0-.11 1.986 2.074 2.074 0 0 0 1.847 1.086Zm6.852 1.952H8.574a2.072 2.072 0 0 0-1.847 1.087 1.9 1.9 0 0 0 .11 1.985l3.426 5.05a2.123 2.123 0 0 0 3.472 0l3.427-5.05a1.9 1.9 0 0 0 .11-1.985 2.074 2.074 0 0 0-1.846-1.087Z"
                                            ),
                                            cls="w-3 h-3",
                                            aria_hidden="true",
                                            xmlns="http://www.w3.org/2000/svg",
                                            fill="currentColor",
                                            viewBox="0 0 24 24"
                                        ),
                                        href="#",
                                        cls="ml-1.5",
                                        hx_get="/api/sort-classifications?field=category&dir=asc",
                                        hx_target="#flowbite-table-body"
                                    ),
                                    cls="flex items-center"
                                ),
                                scope="col",
                                cls="px-6 py-3"
                            ),
                            Th(
                                Div(
                                    "Confidence",
                                    A(
                                        Svg(
                                            Path(
                                                d="M8.574 11.024h6.852a2.075 2.075 0 0 0 1.847-1.086 1.9 1.9 0 0 0-.11-1.986L13.736 2.9a2.122 2.122 0 0 0-3.472 0L6.837 7.952a1.9 1.9 0 0 0-.11 1.986 2.074 2.074 0 0 0 1.847 1.086Zm6.852 1.952H8.574a2.072 2.072 0 0 0-1.847 1.087 1.9 1.9 0 0 0 .11 1.985l3.426 5.05a2.123 2.123 0 0 0 3.472 0l3.427-5.05a1.9 1.9 0 0 0 .11-1.985 2.074 2.074 0 0 0-1.846-1.087Z"
                                            ),
                                            cls="w-3 h-3",
                                            aria_hidden="true",
                                            xmlns="http://www.w3.org/2000/svg",
                                            fill="currentColor",
                                            viewBox="0 0 24 24"
                                        ),
                                        href="#",
                                        cls="ml-1.5",
                                        hx_get="/api/sort-classifications?field=confidence&dir=asc",
                                        hx_target="#flowbite-table-body"
                                    ),
                                    cls="flex items-center"
                                ),
                                scope="col",
                                cls="px-6 py-3"
                            ),
                            Th(
                                Div(
                                    "Feedback",
                                    cls="flex items-center"
                                ),
                                scope="col",
                                cls="px-6 py-3"
                            ),
                            Th(
                                Div(
                                    "Date",
                                    A(
                                        Svg(
                                            Path(
                                                d="M8.574 11.024h6.852a2.075 2.075 0 0 0 1.847-1.086 1.9 1.9 0 0 0-.11-1.986L13.736 2.9a2.122 2.122 0 0 0-3.472 0L6.837 7.952a1.9 1.9 0 0 0-.11 1.986 2.074 2.074 0 0 0 1.847 1.086Zm6.852 1.952H8.574a2.072 2.072 0 0 0-1.847 1.087 1.9 1.9 0 0 0 .11 1.985l3.426 5.05a2.123 2.123 0 0 0 3.472 0l3.427-5.05a1.9 1.9 0 0 0 .11-1.985 2.074 2.074 0 0 0-1.846-1.087Z"
                                            ),
                                            cls="w-3 h-3",
                                            aria_hidden="true",
                                            xmlns="http://www.w3.org/2000/svg",
                                            fill="currentColor",
                                            viewBox="0 0 24 24"
                                        ),
                                        href="#",
                                        cls="ml-1.5",
                                        hx_get="/api/sort-classifications?field=date&dir=desc",
                                        hx_target="#flowbite-table-body"
                                    ),
                                    cls="flex items-center"
                                ),
                                scope="col",
                                cls="px-6 py-3"
                            ),
                            Th(
                                Span("Actions", cls="sr-only"),
                                scope="col",
                                cls="px-6 py-3"
                            )
                        )
                    ),
                    
                    # Table Body with HTMX support
                    Tbody(
                        NotStr(generate_flowbite_table_rows(stats["recent_classifications"])),
                        id="flowbite-table-body"
                    ),
                    
                    cls="w-full text-sm text-left text-gray-500 dark:text-gray-400 flowbite-table"
                ),
                
                # Pagination with HTMX
                    Nav(
                        Span(
                            "Showing ",
                            Span("1-10", cls="font-semibold text-gray-900 dark:text-white"),
                            " of ",
                            Span(str(stats["total_single"]), cls="font-semibold text-gray-900 dark:text-white"),
                            cls="text-sm font-normal text-gray-500 dark:text-gray-400"
                        ),
                        Ul(
                            Li(
                                A(
                                    "Previous",
                                    href="#",
                                    cls="flex items-center justify-center px-3 h-8 ml-0 leading-tight text-gray-500 bg-white border border-gray-300 rounded-l-lg hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white",
                                    hx_get="/api/classifications-page?page=prev",
                                    hx_target="#flowbite-table-body"
                                )
                            ),
                            Li(
                                A(
                                    "1",
                                    href="#",
                                    cls="flex items-center justify-center px-3 h-8 leading-tight text-gray-500 bg-white border border-gray-300 hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white",
                                    hx_get="/api/classifications-page?page=1",
                                    hx_target="#flowbite-table-body"
                                )
                            ),
                            Li(
                                A(
                                    "2",
                                    href="#",
                                    cls="flex items-center justify-center px-3 h-8 leading-tight text-gray-500 bg-white border border-gray-300 hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white",
                                    hx_get="/api/classifications-page?page=2",
                                    hx_target="#flowbite-table-body"
                                )
                            ),
                            Li(
                                A(
                                    "3",
                                    aria_current="page",
                                    href="#",
                                    cls="flex items-center justify-center px-3 h-8 text-blue-600 border border-gray-300 bg-blue-50 hover:bg-blue-100 hover:text-blue-700 dark:border-gray-700 dark:bg-gray-700 dark:text-white",
                                    hx_get="/api/classifications-page?page=3",
                                    hx_target="#flowbite-table-body"
                                )
                            ),
                            Li(
                                A(
                                    "Next",
                                    href="#",
                                    cls="flex items-center justify-center px-3 h-8 leading-tight text-gray-500 bg-white border border-gray-300 rounded-r-lg hover:bg-gray-100 hover:text-gray-700 dark:bg-gray-800 dark:border-gray-700 dark:text-gray-400 dark:hover:bg-gray-700 dark:hover:text-white",
                                    hx_get="/api/classifications-page?page=next",
                                    hx_target="#flowbite-table-body"
                                )
                            ),
                            cls="inline-flex -space-x-px text-sm h-8"
                        ),
                        cls="flex items-center justify-between p-4",
                        aria_label="Table navigation"
                    ),
                    
                    cls="relative overflow-x-auto shadow-md sm:rounded-lg"
                ),
                
                # Image Modal for Detail View
                Div(
                    Div(
                        Div(
                            H3("Classification Details", cls="text-lg leading-6 font-medium text-gray-900", id="modal-title"),
                            Div(
                                Div(
                                    # Placeholder image
                                    Svg(
                                        Path(
                                            stroke_linecap="round",
                                            stroke_linejoin="round",
                                            stroke_width="2",
                                            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
                                        ),
                                        id="modal-image-placeholder",
                                        cls="h-48 w-48 text-gray-400",
                                        fill="none",
                                        viewBox="0 0 24 24",
                                        stroke="currentColor"
                                    ),
                                    Img(
                                        id="modal-image",
                                        cls="max-h-64 hidden",
                                        src="",
                                        alt="Classification Image"
                                    ),
                                    cls="bg-gray-100 p-4 rounded-lg flex items-center justify-center"
                                ),
                                Div(
                                    P(
                                        Span("ID:", cls="font-semibold"),
                                        Span(id="modal-id")
                                    ),
                                    P(
                                        Span("Category:", cls="font-semibold"),
                                        Span(id="modal-category")
                                    ),
                                    P(
                                        Span("Confidence:", cls="font-semibold"),
                                        Span(id="modal-confidence")
                                    ),
                                    P(
                                        Span("Reference Source:", cls="font-semibold"),
                                        Span(id="modal-source"),
                                        id="modal-source-container",
                                        cls="hidden"
                                    ),
                                    P(
                                        Span("Description:", cls="font-semibold"),
                                        Span(id="modal-description"),
                                        id="modal-description-container",
                                        cls="hidden"
                                    ),
                                    cls="mt-4 text-left"
                                ),
                                cls="mt-2 px-7 py-3"
                            ),
                            Div(
                                Button(
                                    "Close",
                                    id="modal-close",
                                    cls="px-4 py-2 bg-gray-500 text-white text-base font-medium rounded-md w-full shadow-sm hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-300"
                                ),
                                cls="items-center px-4 py-3"
                            ),
                            cls="mt-3 text-center"
                        ),
                        cls="relative top-20 mx-auto p-5 border w-11/12 md:w-3/4 lg:w-1/2 shadow-lg rounded-md bg-white"
                    ),
                    id="image-modal",
                    cls="fixed hidden inset-0 bg-black bg-opacity-50 overflow-y-auto h-full w-full z-50"
                ),
                
                # Modal handling script
                Script("""
                // Modal handling functions
                function showImageModal(id, category, confidence) {
                    // Set modal content
                    document.getElementById('modal-id').textContent = id;
                    document.getElementById('modal-category').textContent = category;
                    
                    // Set confidence with appropriate color
                    const confidenceEl = document.getElementById('modal-confidence');
                    confidenceEl.textContent = confidence;
                    confidenceEl.className = confidence === 'High' ? 'text-green-600' : 
                                            confidence === 'Medium' ? 'text-yellow-600' : 'text-red-600';
                    
                    // Show modal
                    document.getElementById('image-modal').classList.remove('hidden');
                    
                    // Optional: Fetch additional details with HTMX
                    htmx.ajax('GET', '/api/classification-details/' + id, {target: '#modal-description-container'});
                }
                
                // Close modal when close button is clicked
                document.getElementById('modal-close').addEventListener('click', function() {
                    document.getElementById('image-modal').classList.add('hidden');
                });
                
                // Close modal when backdrop is clicked
                document.getElementById('image-modal').addEventListener('click', function(e) {
                    if (e.target === this) {
                        this.classList.add('hidden');
                    }
                });
                """),
                
                cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border mb-8"
            )
        
        # Tables Section with Tab Navigation
        tables_section = Div(
            H2("Classification Results", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                # Tab navigation
                Div(
                    Div(
                        Div(role="tablist", cls="tabs tabs-boxed bg-base-200 p-1 mb-6"),
                        
                        Button(
                            "Recent Classifications", 
                            cls="tab tab-active",
                            id="tab-recent",
                            onclick="showTab('recent')",
                            role="tab",
                            aria_selected="true"
                        ),
                        
                        Button(
                            "All Classifications", 
                            cls="tab",
                            id="tab-all",
                            onclick="showTab('all')",
                            role="tab",
                            aria_selected="false"
                        ),
                        
                        cls="flex justify-center"
                    ),
                    
                    # Tab content containers
                    Div(
                        # DaisyUI Table Content
                        Div(
                            daisyui_table,
                            id="tab-content-recent",
                            cls="block"
                        ),
                        
                        # Flowbite Table Content
                        Div(
                            flowbite_table,
                            id="tab-content-all",
                            cls="hidden"
                        ),
                        
                        cls="w-full"
                    ),
                    
                    # Simple tab switching script
                    Script("""
                    function showTab(tabName) {
                        // Hide all tab contents
                        document.getElementById('tab-content-recent').classList.add('hidden');
                        document.getElementById('tab-content-all').classList.add('hidden');
                        
                        // Deactivate all tabs
                        document.getElementById('tab-recent').classList.remove('tab-active');
                        document.getElementById('tab-all').classList.remove('tab-active');
                        
                        // Show selected tab content
                        document.getElementById('tab-content-' + tabName).classList.remove('hidden');
                        
                        // Activate selected tab
                        document.getElementById('tab-' + tabName).classList.add('tab-active');
                    }
                    """),
                    
                    cls="w-full"
                ),
                cls="w-full"
            ),
            cls="mb-8"
        )
        
        # Context Sources Section (for RAG stats)
        rag_section = Div(
            H2("RAG Context Usage", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                Div(
                    H3("Most Used Context Sources", cls="font-semibold mb-3"),
                    Table(
                        Thead(
                            Tr(
                                Th("Source"),
                                Th("Usage Count"),
                            )
                        ),
                        Tbody(
                            *[
                                Tr(
                                    Td(source),
                                    Td(str(count)),
                                )
                                for source, count in stats["context_counts"]
                            ] if stats["context_counts"] else [
                                Tr(
                                    Td("No context sources recorded yet"),
                                    Td("0")
                                )
                            ]
                        ),
                        cls="table w-full"
                    ),
                    cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
                ),
                cls="w-full"
            ),
            cls="mb-8"
        )
        
        return Title("Dashboard - Insect Classifier"), Main(
            Div(
                H1("Classification Dashboard", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Statistics and insights from the Insect Classifier with RAG", cls="text-center mb-8 text-base-content/70"),
                navbar,
                summary_cards,
                charts_section,
                confidence_feedback_section,
                tables_section,
                rag_section,
                cls="container mx-auto px-4 py-8 max-w-7xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )

#################################################
    # API route for chart data
    #################################################
    @rt("/api/chart-data", methods=["GET"])
    async def api_chart_data(request):
        """API endpoint to get chart data for the donut chart"""
        try:
            # Get stats
            stats = get_classification_stats()
            
            # Process the top categories
            categories_to_display = stats["combined_category_counts"][:10]
            
            # Format the data for ApexCharts
            labels = [category for category, _ in categories_to_display]
            counts = [count for _, count in categories_to_display]
            
            # Return JSON data
            return JSONResponse({
                "labels": labels,
                "counts": counts
            })
            
        except Exception as e:
            print(f"Error getting chart data: {e}")
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    #################################################
    # API route for activity chart data
    #################################################
    @rt("/api/chart-activity", methods=["GET"])
    async def api_chart_activity(request):
        """API endpoint to get activity data for the line chart"""
        try:
            # Get requested time range (default to 7 days)
            range_days = int(request.query_params.get("range", 7))
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Get daily classification counts for the requested range
            cursor.execute("""
                SELECT DATE(created_at) as date, COUNT(*) as count 
                FROM results 
                WHERE created_at >= date('now', ?) 
                GROUP BY DATE(created_at) 
                ORDER BY date ASC
            """, (f'-{range_days} days',))
            
            daily_data = cursor.fetchall()
            
            # Get total and calculate average
            total_count = sum(count for _, count in daily_data)
            avg_count = round(total_count / len(daily_data)) if daily_data else 0
            
            # Calculate trend (using simple percentage change)
            trend = 0
            if len(daily_data) >= 2:
                first_half = sum(count for _, count in daily_data[:len(daily_data)//2])
                second_half = sum(count for _, count in daily_data[len(daily_data)//2:])
                
                if first_half > 0:
                    trend = round(((second_half - first_half) / first_half) * 100, 1)
            
            # Extract dates and counts
            dates = [date for date, _ in daily_data]
            counts = [count for _, count in daily_data]
            
            conn.close()
            
            # Return JSON data
            return JSONResponse({
                "dates": dates,
                "counts": counts,
                "total": total_count,
                "average": avg_count,
                "trend": trend
            })
            
        except Exception as e:
            print(f"Error getting activity data: {e}")
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

    #################################################
    # API route for recent classifications
    #################################################
    @rt("/api/recent-classifications", methods=["GET"])
    async def api_recent_classifications(request):
        """API endpoint to get recent classification data for the table"""
        try:
            # Get limit parameter with default of 10
            limit = int(request.query_params.get("limit", 10))
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Get recent classifications
            cursor.execute("""
                SELECT id, category, confidence, feedback, created_at, context_source 
                FROM results 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (limit,))
            
            recent_classifications = cursor.fetchall()
            conn.close()
            
            # Generate HTML for the table body
            html = ""
            for id, category, confidence, feedback, created_at, context_source in recent_classifications:
                # Determine confidence class
                confidence_class = 'badge-warning'
                if confidence == 'High':
                    confidence_class = 'badge-success'
                elif confidence == 'Low':
                    confidence_class = 'badge-error'
                
                # Build the table row
                html += f"""
                <tr class="hover">
                    <td>
                        <div class="avatar cursor-pointer" onclick="document.getElementById('modal-{id[:8]}').showModal()">
                            <div class="mask mask-squircle w-12 h-12 bg-base-300 flex items-center justify-center">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6 text-base-content/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
                                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                                </svg>
                            </div>
                        </div>
                        
                        <dialog id="modal-{id[:8]}" class="modal">
                            <div class="modal-box">
                                <h3 class="font-bold text-lg mb-2">Classification: {category}</h3>
                                <!-- Modal content -->
                                <div class="modal-action">
                                    <form method="dialog">
                                        <button class="btn">Close</button>
                                    </form>
                                </div>
                            </div>
                            <form method="dialog" class="modal-backdrop">
                                <button>close</button>
                            </form>
                        </dialog>
                    </td>
                    <td class="font-mono text-xs">{id[:8]}...</td>
                    <td>{category}</td>
                    <td>
                        <span class="badge {confidence_class}">{confidence}</span>
                    </td>
                """
                
                # Feedback cell
                if feedback:
                    feedback_class = "badge-success" if feedback == "positive" else "badge-error"
                    html += f"""
                    <td>
                        <span class="badge {feedback_class}">{feedback}</span>
                    </td>
                    """
                else:
                    html += f"""
                    <td>
                        <div class="flex space-x-1">
                            <button 
                                class="btn btn-xs btn-circle btn-outline" 
                                hx-post="/api/feedback"
                                hx-vals='{{"id": "{id}", "feedback": "positive"}}'
                                hx-swap="outerHTML"
                                title="Positive Feedback">
                                👍
                            </button>
                            <button 
                                class="btn btn-xs btn-circle btn-outline" 
                                hx-post="/api/feedback"
                                hx-vals='{{"id": "{id}", "feedback": "negative"}}'
                                hx-swap="outerHTML"
                                title="Negative Feedback">
                                👎
                            </button>
                        </div>
                    </td>
                    """
                
                # Context source and time
                source_display = context_source if context_source else "None"
                html += f"""
                    <td class="max-w-xs truncate" title="{source_display}">{source_display}</td>
                    <td>{created_at}</td>
                    <td>
                        <div class="dropdown dropdown-end">
                            <label tabindex="0" class="btn btn-xs btn-ghost m-1">
                                <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 12h.01M12 12h.01M19 12h.01M6 12a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0zm7 0a1 1 0 11-2 0 1 1 0 012 0z" />
                                </svg>
                            </label>
                            <ul tabindex="0" class="dropdown-content z-[1] menu p-2 shadow bg-base-100 rounded-box w-52">
                                <li><a onclick="document.getElementById('modal-{id[:8]}').showModal()">View Details</a></li>
                                <li><a hx-get="/export-result?id={id}" hx-trigger="click" hx-swap="none">Export Result</a></li>
                            </ul>
                        </div>
                    </td>
                </tr>
                """
            
            # Return the HTML directly for HTMX to swap
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error getting recent classifications: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <tr>
                <td colspan="8" class="text-center text-error">
                    Error loading data: {str(e)}
                </td>
            </tr>
            """)

    #################################################
    # API route for feedback submission
    #################################################
    @rt("/api/feedback", methods=["POST"])
    async def api_submit_feedback(request):
        """API endpoint to submit feedback for a classification"""
        try:
            # Get form data
            form_data = await request.form()
            result_id = form_data.get("id")
            feedback = form_data.get("feedback")
            
            if not result_id or not feedback:
                return HTMLResponse("""
                <div class="text-error">Error: Missing parameters</div>
                """, status_code=400)
            
            # Validate feedback type
            if feedback not in ["positive", "negative"]:
                return HTMLResponse("""
                <div class="text-error">Error: Invalid feedback type</div>
                """, status_code=400)
            
            # Update the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Update the record
            cursor.execute(
                "UPDATE results SET feedback = ? WHERE id = ?",
                (feedback, result_id)
            )
            
            # Check if any row was affected
            if cursor.rowcount == 0:
                # Try batch results table
                cursor.execute(
                    "SELECT batch_id, results FROM batch_results WHERE batch_id = ? OR batch_id = SUBSTR(?, 1, INSTR(?, '_') - 1)",
                    (result_id, result_id, result_id)
                )
                batch_result = cursor.fetchone()
                
                if batch_result:
                    batch_id, results_json = batch_result
                    
                    # Parse the results JSON
                    results_data = json.loads(results_json)
                    
                    # Find the specific result in the batch
                    for result in results_data:
                        if result.get("id") == result_id:
                            # Update the feedback
                            result["feedback"] = feedback
                            break
                    
                    # Save the updated results back to the database
                    cursor.execute(
                        "UPDATE batch_results SET results = ? WHERE batch_id = ?",
                        (json.dumps(results_data), batch_id)
                    )
            
            conn.commit()
            conn.close()
            
            # Return updated feedback pill/badge
            feedback_class = "badge-success" if feedback == "positive" else "badge-error"
            return HTMLResponse(f"""
            <span class="badge {feedback_class}">{feedback}</span>
            """)
            
        except Exception as e:
            print(f"Error submitting feedback: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <div class="text-error">Error: {str(e)}</div>
            """, status_code=500)
        
#################################################
    # API route for searching classifications
    #################################################
    @rt("/api/search-classifications", methods=["POST"])
    async def api_search_classifications(request):
        """API endpoint to search classifications"""
        try:
            # Get form data
            form_data = await request.form()
            search_query = form_data.get("classification-search", "").strip()
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Build the query
            query = """
                SELECT id, category, confidence, feedback, created_at, context_source 
                FROM results 
                WHERE 1=1
            """
            params = []
            
            # Add search conditions if query is provided
            if search_query:
                query += """
                    AND (
                        category LIKE ? OR
                        description LIKE ? OR
                        id LIKE ? OR
                        context_source LIKE ?
                    )
                """
                search_param = f"%{search_query}%"
                params.extend([search_param, search_param, search_param, search_param])
            
            # Add order and limit
            query += " ORDER BY created_at DESC LIMIT 20"
            
            # Execute the query
            cursor.execute(query, params)
            search_results = cursor.fetchall()
            conn.close()
            
            # Generate HTML for table rows
            html = generate_flowbite_table_rows(search_results)
            
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error searching classifications: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-red-500">
                    Error searching data: {str(e)}
                </td>
            </tr>
            """)

    #################################################
    # API route for filtering classifications
    #################################################
    @rt("/api/filter-classifications", methods=["POST"])
    async def api_filter_classifications(request):
        """API endpoint to filter classifications by category"""
        try:
            # Get form data
            form_data = await request.form()
            category = form_data.get("value", "").strip()
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Build the query
            query = """
                SELECT id, category, confidence, feedback, created_at, context_source 
                FROM results 
                WHERE 1=1
            """
            params = []
            
            # Add category filter if provided
            if category:
                query += " AND category = ?"
                params.append(category)
            
            # Add order and limit
            query += " ORDER BY created_at DESC LIMIT 20"
            
            # Execute the query
            cursor.execute(query, params)
            filter_results = cursor.fetchall()
            conn.close()
            
            # Generate HTML for table rows
            html = generate_flowbite_table_rows(filter_results)
            
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error filtering classifications: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-red-500">
                    Error filtering data: {str(e)}
                </td>
            </tr>
            """)

    #################################################
    # API route for sorting classifications
    #################################################
    @rt("/api/sort-classifications", methods=["GET"])
    async def api_sort_classifications(request):
        """API endpoint to sort classifications"""
        try:
            # Get query parameters
            field = request.query_params.get("field", "date")
            direction = request.query_params.get("dir", "desc")
            
            # Map field names to database columns
            field_mapping = {
                "category": "category",
                "confidence": "confidence",
                "date": "created_at"
            }
            
            # Validate parameters
            if field not in field_mapping:
                field = "date"
            
            if direction not in ["asc", "desc"]:
                direction = "desc"
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Build and execute query
            query = f"""
                SELECT id, category, confidence, feedback, created_at, context_source 
                FROM results 
                ORDER BY {field_mapping[field]} {direction.upper()}
                LIMIT 20
            """
            
            cursor.execute(query)
            sorted_results = cursor.fetchall()
            conn.close()
            
            # Generate HTML for table rows
            html = generate_flowbite_table_rows(sorted_results)
            
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error sorting classifications: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-red-500">
                    Error sorting data: {str(e)}
                </td>
            </tr>
            """)

    #################################################
    # API route for pagination
    #################################################
    @rt("/api/classifications-page", methods=["GET"])
    async def api_classifications_page(request):
        """API endpoint to get a specific page of classification results"""
        try:
            # Get page parameter
            page = request.query_params.get("page", "1")
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Calculate offset based on page
            per_page = 10
            
            if page == "prev":
                # We don't know the current page, so just return the first page
                offset = 0
            elif page == "next":
                # Similar challenge, let's assume we're on page 1
                offset = per_page
            else:
                try:
                    page_num = int(page)
                    offset = (page_num - 1) * per_page
                except ValueError:
                    offset = 0
            
            # Get paginated results
            cursor.execute("""
                SELECT id, category, confidence, feedback, created_at, context_source 
                FROM results 
                ORDER BY created_at DESC 
                LIMIT ? OFFSET ?
            """, (per_page, offset))
            
            page_results = cursor.fetchall()
            conn.close()
            
            # Generate HTML for table rows
            html = generate_flowbite_table_rows(page_results)
            
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error getting page data: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <tr>
                <td colspan="6" class="px-6 py-4 text-center text-red-500">
                    Error loading page data: {str(e)}
                </td>
            </tr>
            """)

#################################################
    # API route for classification details
    #################################################
    @rt("/api/classification-details/{result_id}", methods=["GET"])
    async def api_classification_details(request):
        """API endpoint to get detailed information about a specific classification"""
        try:
            # Get result ID from path parameter
            result_id = request.path_params["result_id"]
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Get classification details
            cursor.execute("""
                SELECT id, category, confidence, description, additional_details, context_source 
                FROM results 
                WHERE id = ?
            """, (result_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            if not result:
                return HTMLResponse("""
                <div class="mt-4">
                    <p class="text-red-500">Classification not found</p>
                </div>
                """, status_code=404)
            
            # Extract data
            id, category, confidence, description, additional_details, context_source = result
            
            # Build response HTML
            html = f"""
            <div class="block mt-4">
                <span class="font-semibold">Description:</span>
                <p class="mt-1">{description}</p>
            </div>
            """
            
            # Add additional details if available
            if additional_details:
                try:
                    details = json.loads(additional_details)
                    html += '<div class="mt-3">'
                    for key, value in details.items():
                        if key not in ["Main Category", "Confidence", "Description"]:
                            html += f'<p><span class="font-semibold">{key}:</span> {value}</p>'
                    html += '</div>'
                except:
                    pass
            
            # Add context source if available
            if context_source:
                html += f"""
                <div class="mt-3" id="modal-source-container">
                    <span class="font-semibold">Reference Source:</span> 
                    <span id="modal-source">{context_source}</span>
                </div>
                """
            
            return HTMLResponse(html)
            
        except Exception as e:
            print(f"Error getting classification details: {e}")
            traceback.print_exc()
            return HTMLResponse(f"""
            <div class="mt-4">
                <p class="text-red-500">Error loading details: {str(e)}</p>
            </div>
            """, status_code=500)

    #################################################
    # API route for exporting results
    #################################################
    @rt("/export-result", methods=["GET"])
    async def api_export_result(request):
        """API endpoint to export a classification result as JSON"""
        try:
            # Get result ID from query parameter
            result_id = request.query_params.get("id")
            
            if not result_id:
                return JSONResponse({"error": "Missing result ID"}, status_code=400)
            
            # Connect to the database
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            cursor = conn.cursor()
            
            # Get result data
            cursor.execute("""
                SELECT id, category, confidence, description, additional_details, context_source, created_at, feedback
                FROM results 
                WHERE id = ?
            """, (result_id,))
            
            result = cursor.fetchone()
            
            if not result:
                # Check batch results
                cursor.execute("""
                    SELECT results
                    FROM batch_results
                    WHERE batch_id = ? OR batch_id = SUBSTR(?, 1, INSTR(?, '_') - 1)
                """, (result_id, result_id, result_id))
                
                batch_result = cursor.fetchone()
                
                if batch_result:
                    batch_data = json.loads(batch_result[0])
                    for item in batch_data:
                        if item.get("id") == result_id:
                            # Format the result
                            export_data = {
                                "id": item.get("id"),
                                "category": item.get("category"),
                                "confidence": item.get("confidence"),
                                "description": item.get("description"),
                                "details": item.get("details", {}),
                                "feedback": item.get("feedback"),
                                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
                            }
                            
                            # Set content disposition to trigger download
                            headers = {
                                "Content-Disposition": f"attachment; filename=classification_{result_id[:8]}.json"
                            }
                            
                            return JSONResponse(export_data, headers=headers)
                
                return JSONResponse({"error": "Result not found"}, status_code=404)
            
            # Extract data
            id, category, confidence, description, additional_details, context_source, created_at, feedback = result
            
            # Parse additional details
            details = {}
            if additional_details:
                try:
                    details = json.loads(additional_details)
                except:
                    details = {"raw": additional_details}
            
            # Format the result
            export_data = {
                "id": id,
                "category": category,
                "confidence": confidence,
                "description": description,
                "details": details,
                "context_source": context_source,
                "created_at": created_at,
                "feedback": feedback,
                "exported_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Set content disposition to trigger download
            headers = {
                "Content-Disposition": f"attachment; filename=classification_{result_id[:8]}.json"
            }
            
            return JSONResponse(export_data, headers=headers)
            
        except Exception as e:
            print(f"Error exporting result: {e}")
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)

#################################################
    # Image serving routes
    #################################################
    @rt("/image-thumbnail", methods=["GET"])
    async def serve_image_thumbnail(request):
        """Serve a thumbnail image for classifications"""
        try:
            # Get the image path from query parameters
            image_path = request.query_params.get("path", "")
            
            if not image_path or not os.path.exists(image_path):
                # Return a placeholder SVG if no image is found
                placeholder_svg = """
                <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
                    <circle cx="8.5" cy="8.5" r="1.5"></circle>
                    <polyline points="21 15 16 10 5 21"></polyline>
                </svg>
                """
                return Response(
                    content=placeholder_svg.encode("utf-8"),
                    media_type="image/svg+xml"
                )
            
            # Get the file extension to determine content type
            _, ext = os.path.splitext(image_path.lower())
            
            # Set content type based on extension
            if ext in ['.jpg', '.jpeg']:
                content_type = "image/jpeg"
            elif ext == '.png':
                content_type = "image/png"
            elif ext == '.gif':
                content_type = "image/gif"
            elif ext == '.svg':
                content_type = "image/svg+xml"
            else:
                content_type = "application/octet-stream"
            
            # Read the file
            with open(image_path, "rb") as f:
                content = f.read()
            
            # Return the image
            return Response(
                content=content,
                media_type=content_type
            )
            
        except Exception as e:
            print(f"Error serving image: {e}")
            traceback.print_exc()
            
            # Return a placeholder SVG if there's an error
            placeholder_svg = """
            <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            """
            return Response(
                content=placeholder_svg.encode("utf-8"),
                media_type="image/svg+xml"
            )

    @rt("/context-image-thumbnail", methods=["GET"])
    async def serve_context_image_thumbnail(request):
        """Serve a thumbnail of context images from the RAG system"""
        try:
            # Get source ID from query parameters
            source_id = request.query_params.get("source", "")
            
            if not source_id:
                return Response(status_code=400)
            
            # Parse source ID to get filename and page
            parts = source_id.split(", page ")
            if len(parts) != 2:
                return Response(status_code=400)
            
            filename = parts[0]
            try:
                page_num = int(parts[1])
            except ValueError:
                page_num = 1
            
            # Construct image path based on patterns used in the system
            potential_paths = []
            
            # Try to find PDF image directory
            pdf_name = os.path.splitext(filename)[0]  # Remove extension
            
            # Check different potential locations
            pdf_dir = os.path.join(PDF_IMAGES_DIR, pdf_name)
            if os.path.exists(pdf_dir) and os.path.isdir(pdf_dir):
                # PDF directory found, look for page image
                potential_paths.extend([
                    os.path.join(pdf_dir, f"{page_num-1}.png"),  # 0-indexed
                    os.path.join(pdf_dir, f"{page_num}.png"),    # 1-indexed
                    os.path.join(pdf_dir, f"page_{page_num-1}.png"),
                    os.path.join(pdf_dir, f"page_{page_num}.png")
                ])
            
            # Also check direct file paths
            potential_paths.extend([
                os.path.join(PDF_IMAGES_DIR, f"{pdf_name}_{page_num-1}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{pdf_name}_{page_num}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{pdf_name}_page{page_num-1}.png"),
                os.path.join(PDF_IMAGES_DIR, f"{pdf_name}_page{page_num}.png")
            ])
            
            # Try all potential paths
            image_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    image_path = path
                    break
            
            if not image_path:
                # Return a placeholder SVG
                placeholder_svg = """
                <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" y1="13" x2="8" y2="13"></line>
                    <line x1="16" y1="17" x2="8" y2="17"></line>
                    <line x1="10" y1="9" x2="8" y2="9"></line>
                </svg>
                """
                return Response(
                    content=placeholder_svg.encode("utf-8"),
                    media_type="image/svg+xml"
                )
            
            # Read the image file
            with open(image_path, "rb") as f:
                content = f.read()
            
            # Return the image
            return Response(
                content=content,
                media_type="image/png"
            )
            
        except Exception as e:
            print(f"Error serving context image: {e}")
            traceback.print_exc()
            
            # Return a placeholder SVG if there's an error
            placeholder_svg = """
            <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <circle cx="12" cy="12" r="10"></circle>
                <line x1="12" y1="8" x2="12" y2="12"></line>
                <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
            """
            return Response(
                content=placeholder_svg.encode("utf-8"),
                media_type="image/svg+xml"
            )
#################################################
    # API route for image classification - FIXED FOR ASYNC/AWAIT ISSUE
    #################################################
    @rt("/classify", methods=["POST"])
    async def api_classify_image(request):
        """API endpoint to classify insect image using Claude with RAG"""
        try:
            # Get image data and options from request JSON
            data = await request.json()
            image_data = data.get("image_data", "")
            options = data.get("options", {})
            
            if not image_data:
                return JSONResponse({"error": "No image data provided"}, status_code=400)
            
            result = classify_image_claude.remote(image_data, options)
            
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error classifying image: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
    
    #################################################
    # Batch Classify API Endpoint
    #################################################
    @rt("/classify-batch", methods=["POST"])
    async def api_classify_batch(request):
        """API endpoint to classify multiple insect images in batch mode with RAG"""
        try:
            # Get form data with files
            form = await request.form()
            options_json = form.get("options", "{}")
            options = json.loads(options_json)
            
            # Extract image files
            image_files = []
            for key in form.keys():
                if key.startswith("image_"):
                    image_files.append(form.get(key))
                    
            if not image_files:
                return JSONResponse({"error": "No images provided"}, status_code=400)
                
            # Limit to 5 images
            if len(image_files) > 5:
                image_files = image_files[:5]
                
            # Process each image
            base64_images = []
            for file in image_files:
                # Read file content
                content = await file.read()
                
                # Convert to base64
                base64_data = base64.b64encode(content).decode("utf-8")
                base64_images.append(base64_data)
                
            if not base64_images:
                return JSONResponse({"error": "Failed to process images"}, status_code=400)
                
            result = classify_batch_claude.remote(base64_images, options)
            
            # Return the result
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error in batch classification: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
    

    #################################################
    # Homepage Route - Unified Classifier Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the unified classifier dashboard"""
        
        # Create toggle switches for classification options
        def create_toggle(name, label, checked=False, description=None):
            toggle_input = Input(
                type="checkbox",
                name=name,
                checked="checked" if checked else None,
                cls="toggle toggle-primary mr-3"
            )
            
            label_span = Span(label)
            
            label_element = Label(
                toggle_input,
                label_span,
                cls="label cursor-pointer justify-start"
            )
            
            toggle_element = Div(
                label_element,
                cls="mb-3"
            )
            
            # If description is provided, add it to the container
            if description:
                description_p = P(description, cls="text-sm text-base-content/70 ml-10")
                # Create a new Div with both elements
                toggle_element = Div(
                    label_element,
                    description_p,
                    cls="mb-3"
                )
                        
            return toggle_element
        
        # Classification options panel with RAG option
        classification_options = Div(
            H3("Classification Options", cls="text-lg font-semibold mb-4 text-bee-green"),
            create_toggle("use_rag", "Use Context-Enhanced Classification (RAG)", True, 
                        "Enhances classification accuracy using relevant reference materials"),
            create_toggle("detailed_description", "Detailed Description (shapes, colors)"),
            create_toggle("plant_classification", "Plant Classification"),
            create_toggle("taxonomy", "Taxonomic Classification"),
            cls="mb-6 p-4 bg-base-200 rounded-lg"
        )
        
        # Unified image upload section
        upload_section = Div(
            Label("Upload Insect Images", cls="block text-xl font-medium mb-2 text-bee-green"),
            P("Upload one or more insect images (up to 5) for classification.", cls="mb-4"),
            Div(
                # Use DaisyUI file input instead of custom drag-and-drop
                Label(
                    "Select Images",
                    cls="block mb-2 text-sm font-medium"
                ),
                Input(
                    type="file",
                    name="insect_images",
                    accept="image/jpeg,image/png",
                    multiple=True,
                    cls="file-input file-input-bordered file-input-primary w-full",
                    id="image-input",
                    hx_on="change: handleFileSelection(event)"
                ),
                cls="mb-6"
            ),
            
            # Preview area - shows either single preview or batch previews
            Div(
                # Single image preview
                Img(
                    id="single-preview",
                    src="",
                    cls="max-h-64 mx-auto hidden object-contain rounded-lg border shadow-sm"
                ),
                
                # Batch previews container
                Div(
                    id="batch-previews",
                    cls="batch-previews hidden"
                ),
                
                # Count display
                Div(
                    Span("", id="image-count"),
                    cls="text-center mt-2 text-sm text-base-content/70 hidden",
                    id="count-display"
                ),
                
                cls="mb-6"
            ),
            cls="mb-8"
        )
        
        # Control panel 
        control_panel = Div(
            H2("Insect Image Classification", cls="text-xl font-bold mb-4 text-bee-green"),
            upload_section,
            classification_options,
            Button(
                "Classify Insects",
                cls="btn btn-primary w-full",
                id="classify-button",
                disabled="disabled"
            ),
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Results panel
        results_panel = Div(
            H2("Classification Results", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                Div(
                    cls="loading loading-spinner loading-lg text-primary",
                    id="loading-indicator"
                ),
                cls="flex justify-center items-center h-32 hidden",
                id="loading-indicator-parent"
            ),
            Div(
                P("Upload image(s) and click 'Classify Insects' to see results.", 
                cls="text-center text-base-content/70 italic"),
                id="results-placeholder",
                cls="text-center py-12"
            ),
            
            # Container for both single and batch results
            Div(
                # Single result container
                Div(
                    id="single-result",
                    cls="hidden"
                ),
                
                # Batch results carousel container
                Div(
                    id="batch-results",
                    cls="hidden"
                ),
                
                id="results-content",
                cls="hidden"
            ),
            
            # Actions for results
            Div(
                Button(
                    "Copy Results",
                    cls="btn btn-outline btn-accent btn-sm mr-2",
                    id="copy-button"
                ),
                Button(
                    "New Classification",
                    cls="btn btn-outline btn-primary btn-sm",
                    id="new-button"
                ),
                cls="mt-6 flex justify-end items-center gap-2 hidden",
                id="result-actions"
            ),
            cls="w-full md:w-1/2 bg-base-100 p-6 rounded-lg shadow-lg custom-border border"
        )
        
        # Navigation bar
        navbar = Div(
            Div(
                A(
                    Span("🐝", cls="text-xl"),
                    Span("Insect Classifier", cls="ml-2 text-xl font-semibold"),
                    href="/",
                    cls="flex items-center"
                ),
                Div(
                    A(
                        "Dashboard",
                        href="/dashboard",
                        cls="btn btn-sm btn-ghost"
                    ),
                    cls="flex-none"
                ),
                cls="navbar bg-base-200 rounded-lg mb-8 shadow-sm"
            ),
            cls="w-full"
        )
        
        # Add script for form handling with fixes
        form_script = Script("""
        document.addEventListener('DOMContentLoaded', function() {
            // Form elements - cache all DOM elements we'll need to reference
            const imageInput = document.getElementById('image-input');
            const singlePreview = document.getElementById('single-preview');
            const batchPreviewsContainer = document.getElementById('batch-previews');
            const countDisplay = document.getElementById('count-display');
            const imageCountElem = document.getElementById('image-count');
            const classifyButton = document.getElementById('classify-button');
            
            // Results elements - references to DOM elements for displaying results
            const loadingIndicator = document.getElementById('loading-indicator').parentElement;
            const resultsPlaceholder = document.getElementById('results-placeholder');
            const resultsContent = document.getElementById('results-content');
            const singleResult = document.getElementById('single-result');
            const batchResults = document.getElementById('batch-results');
            const resultActions = document.getElementById('result-actions');
            const copyButton = document.getElementById('copy-button');
            const newButton = document.getElementById('new-button');
            
            // Mode tracking variables
            let isBatchMode = false;
            const MAX_IMAGES = 5;
            let selectedFiles = [];
            let rawResponseText = '';
            
            // Debug elements on page load
            console.log("DOM loaded - Bee Classifier Initialized");
            console.log("Button state:", classifyButton ? (classifyButton.disabled ? "disabled" : "enabled") : "button not found");
            
            // Explicitly attach the change event listener
            // This ensures the file input triggers our handler even if the HTML attribute binding fails
            if (imageInput) {
                console.log("Setting up file input change listener");
                imageInput.addEventListener('change', function(event) {
                    console.log("File input changed - files selected:", event.target.files.length);
                    handleFileSelection(event);
                });
            } else {
                console.error("Critical Error: Image input element not found");
            }
            
            // Get options from the form controls
            function getOptions() {
                return {
                    use_rag: document.querySelector('input[name="use_rag"]').checked,
                    detailed_description: document.querySelector('input[name="detailed_description"]').checked,
                    plant_classification: document.querySelector('input[name="plant_classification"]').checked,
                    taxonomy: document.querySelector('input[name="taxonomy"]').checked
                };
            }
            
            // Handle file selection - core function that processes selected files
            window.handleFileSelection = function(event) {
                console.log("handleFileSelection called");
                const files = event.target.files;
                
                if (!files || files.length === 0) {
                    console.log("No files selected");
                    resetForm();
                    return;
                }
                
                console.log(`${files.length} files selected`);
                
                if (files.length > MAX_IMAGES) {
                    alert(`Please select a maximum of ${MAX_IMAGES} images.`);
                    resetForm();
                    return;
                }
                
                // Determine mode based on file count
                isBatchMode = files.length > 1;
                selectedFiles = Array.from(files);
                
                if (isBatchMode) {
                    // Batch mode - show multiple previews
                    console.log("Batch mode activated");
                    singlePreview.classList.add('hidden');
                    batchPreviewsContainer.classList.remove('hidden');
                    batchPreviewsContainer.innerHTML = '';
                    
                    // Create preview for each image
                    selectedFiles.forEach((file, index) => {
                        const reader = new FileReader();
                        
                        reader.onload = function(e) {
                            // Create preview container
                            const previewDiv = document.createElement('div');
                            previewDiv.className = 'preview-item';
                            previewDiv.dataset.index = index;
                            
                            // Create image preview
                            const img = document.createElement('img');
                            img.src = e.target.result;
                            img.className = 'preview-img';
                            
                            // Create remove button
                            const removeBtn = document.createElement('div');
                            removeBtn.className = 'remove-btn';
                            removeBtn.innerHTML = '×';
                            removeBtn.onclick = function() {
                                // Remove this file
                                selectedFiles.splice(index, 1);
                                
                                // Update UI
                                if (selectedFiles.length === 0) {
                                    resetForm();
                                } else if (selectedFiles.length === 1) {
                                    // Switch to single mode
                                    isBatchMode = false;
                                    showSinglePreview(selectedFiles[0]);
                                } else {
                                    // Stay in batch mode but update
                                    updateBatchPreviews();
                                }
                            };
                            
                            // Add elements
                            previewDiv.appendChild(img);
                            previewDiv.appendChild(removeBtn);
                            batchPreviewsContainer.appendChild(previewDiv);
                        };
                        
                        reader.readAsDataURL(file);
                    });
                    
                    // Update count display
                    imageCountElem.textContent = `${selectedFiles.length} images selected`;
                    countDisplay.classList.remove('hidden');
                } else {
                    // Single mode - show one preview
                    console.log("Single image mode activated");
                    showSinglePreview(selectedFiles[0]);
                }
                
                // Enable classify button - with visual feedback
                if (classifyButton) {
                    classifyButton.disabled = false;
                    classifyButton.classList.remove('opacity-50');
                    classifyButton.classList.add('hover:bg-primary-focus');
                    console.log("Classify button enabled");
                } else {
                    console.error("Critical Error: Classify button not found");
                }
            };
            
            // Show single image preview
            function showSinglePreview(file) {
                console.log("Showing single preview for file:", file.name);
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    singlePreview.src = e.target.result;
                    singlePreview.classList.remove('hidden');
                    batchPreviewsContainer.classList.add('hidden');
                    countDisplay.classList.add('hidden');
                };
                
                reader.readAsDataURL(file);
            }
            
            // Update batch previews
            function updateBatchPreviews() {
                console.log("Updating batch previews, count:", selectedFiles.length);
                batchPreviewsContainer.innerHTML = '';
                
                selectedFiles.forEach((file, index) => {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        // Create preview container
                        const previewDiv = document.createElement('div');
                        previewDiv.className = 'preview-item';
                        previewDiv.dataset.index = index;
                        
                        // Create image preview
                        const img = document.createElement('img');
                        img.src = e.target.result;
                        img.className = 'preview-img';
                        
                        // Create remove button
                        const removeBtn = document.createElement('div');
                            removeBtn.className = 'remove-btn';
                        removeBtn.innerHTML = '×';
                        removeBtn.onclick = function() {
                            // Remove this file
                            selectedFiles.splice(index, 1);
                            
                            // Update UI
                            if (selectedFiles.length === 0) {
                                resetForm();
                            } else if (selectedFiles.length === 1) {
                                // Switch to single mode
                                isBatchMode = false;
                                showSinglePreview(selectedFiles[0]);
                            } else {
                                // Stay in batch mode but update
                                updateBatchPreviews();
                            }
                        };
                        
                        // Add elements
                        previewDiv.appendChild(img);
                        previewDiv.appendChild(removeBtn);
                        batchPreviewsContainer.appendChild(previewDiv);
                    };
                    
                    reader.readAsDataURL(file);
                });
                
                // Update count
                imageCountElem.textContent = `${selectedFiles.length} images selected`;
                countDisplay.classList.remove('hidden');
            }
            
            // Reset the form to initial state
            function resetForm() {
                console.log("Resetting form");
                imageInput.value = '';
                singlePreview.src = '';
                singlePreview.classList.add('hidden');
                batchPreviewsContainer.innerHTML = '';
                batchPreviewsContainer.classList.add('hidden');
                countDisplay.classList.add('hidden');
                
                // Reset button with visual indicators
                if (classifyButton) {
                    classifyButton.disabled = true;
                    classifyButton.classList.add('opacity-50');
                    classifyButton.classList.remove('hover:bg-primary-focus');
                }
                
                selectedFiles = [];
                isBatchMode = false;
            }
            
            // Handle classify button click
            if (classifyButton) {
                classifyButton.addEventListener('click', function() {
                    console.log("Classify button clicked");
                    
                    // Show loading state
                    loadingIndicator.classList.remove('hidden');
                    resultsPlaceholder.classList.add('hidden');
                    resultsContent.classList.add('hidden');
                    resultActions.classList.add('hidden');
                    classifyButton.disabled = true;
                    classifyButton.classList.add('opacity-50');
                    
                    if (isBatchMode) {
                        // Batch mode - process multiple images
                        console.log("Processing batch of", selectedFiles.length, "images");
                        processBatchImages();
                    } else {
                        // Single mode - process one image
                        console.log("Processing single image");
                        processSingleImage();
                    }
                });
            }
            
            // Process a single image
            function processSingleImage() {
                console.log("Starting single image processing");
                
                // Get the base64 image data
                const base64Data = singlePreview.src.split(',')[1];
                
                // Send request to API
                fetch('/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image_data: base64Data,
                        options: getOptions()
                    })
                })
                .then(response => {
                    console.log("Received API response, status:", response.status);
                    return response.json();
                })
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    
                    if (data.error) {
                        console.error("Error from API:", data.error);
                        // Show error message
                        singleResult.innerHTML = `
                            <div class="alert alert-error">
                                <span>Error: ${data.error}</span>
                            </div>
                        `;
                        singleResult.classList.remove('hidden');
                        batchResults.classList.add('hidden');
                        resultsContent.classList.remove('hidden');
                        return;
                    }
                    
                    console.log("Classification successful, displaying results");
                    
                    // Display the result using the enhanced display function
                    displaySingleResult(data);
                    
                    // Show containers
                    singleResult.classList.remove('hidden');
                    batchResults.classList.add('hidden');
                    resultsContent.classList.remove('hidden');
                    resultActions.classList.remove('hidden');
                })
                .catch(error => {
                    console.error('Error classifying image:', error);
                    loadingIndicator.classList.add('hidden');
                    singleResult.innerHTML = `
                        <div class="alert alert-error">
                            <span>Error: Could not process your request. Please try again.</span>
                        </div>
                    `;
                    singleResult.classList.remove('hidden');
                    batchResults.classList.add('hidden');
                    resultsContent.classList.remove('hidden');
                    classifyButton.disabled = false;
                    classifyButton.classList.remove('opacity-50');
                });
            }
            
            // Process batch of images
            function processBatchImages() {
                console.log("Starting batch image processing");
                
                // Create form data for file upload
                const formData = new FormData();
                
                // Add all files
                selectedFiles.forEach((file, index) => {
                    formData.append(`image_${index}`, file);
                    console.log(`Added image_${index} to form data:`, file.name);
                });
                
                // Add options
                const options = getOptions();
                formData.append('options', JSON.stringify(options));
                console.log("Added options to form data:", options);
                
                // Send batch request
                fetch('/classify-batch', {
                    method: 'POST',
                    body: formData
                })
                .then(response => {
                    console.log("Received batch API response, status:", response.status);
                    return response.json();
                })
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    
                    if (data.error) {
                        console.error("Error from batch API:", data.error);
                        // Show error message
                        batchResults.innerHTML = `
                            <div class="alert alert-error">
                                <span>Error: ${data.error}</span>
                            </div>
                        `;
                        singleResult.classList.add('hidden');
                        batchResults.classList.remove('hidden');
                        resultsContent.classList.remove('hidden');
                        return;
                    }
                    
                    console.log("Batch classification successful, displaying results");
                    
                    // Save raw response for copy button
                    rawResponseText = data.raw_response;
                    
                    // Display batch results using the enhanced display function
                    displayBatchResults(data);
                    
                    // Show result sections
                    singleResult.classList.add('hidden');
                    batchResults.classList.remove('hidden');
                    resultsContent.classList.remove('hidden');
                    resultActions.classList.remove('hidden');
                })
                .catch(error => {
                    console.error('Error in batch processing:', error);
                    loadingIndicator.classList.add('hidden');
                    batchResults.innerHTML = `
                        <div class="alert alert-error">
                            <span>Error: Could not process your request. Please try again.</span>
                        </div>
                    `;
                    singleResult.classList.add('hidden');
                    batchResults.classList.remove('hidden');
                    resultsContent.classList.remove('hidden');
                    classifyButton.disabled = false;
                    classifyButton.classList.remove('opacity-50');
                });
            }
            
            // Setup copy button
            if (copyButton) {
                copyButton.addEventListener('click', function() {
                    console.log("Copy button clicked");
                    navigator.clipboard.writeText(rawResponseText);
                    copyButton.innerHTML = 'Copied!';
                    setTimeout(() => {
                        copyButton.innerHTML = 'Copy Results';
                    }, 2000);
                });
            }
            
            // Setup new button
            if (newButton) {
                newButton.addEventListener('click', function() {
                    console.log("New classification button clicked");
                    // Reset form
                    resetForm();
                    
                    // Reset results
                    resultsPlaceholder.classList.remove('hidden');
                    resultsContent.classList.add('hidden');
                    resultActions.classList.add('hidden');
                    singleResult.classList.add('hidden');
                    batchResults.classList.add('hidden');
                });
            }
            
            // Function to display single classification result with context
            function displaySingleResult(result) {
                console.log("Displaying single result:", result.category);
                const singleResult = document.getElementById('single-result');
                
                // Determine confidence class
                let confidenceClass = 'badge-warning';
                if (result.confidence === 'High') {
                    confidenceClass = 'badge-success';
                } else if (result.confidence === 'Low') {
                    confidenceClass = 'badge-error';
                }
                
                // Create result HTML
                let resultHTML = `
                    <div class="p-4 bg-base-200 rounded-lg mb-4">
                        <div class="flex justify-between items-center mb-2">
                            <h3 class="text-lg font-bold">${result.category}</h3>
                            <span class="badge ${confidenceClass}">Confidence: ${result.confidence}</span>
                        </div>
                        <p class="mb-4">${result.description}</p>
                `;
                
                // Add additional details if available
                const details = result.details;
                for (const key in details) {
                    if (key !== 'Main Category' && key !== 'Confidence' && key !== 'Description') {
                        resultHTML += `
                            <div class="mb-2">
                                <span class="font-semibold">${key}:</span>
                                <span>${details[key]}</span>
                            </div>
                        `;
                    }
                }
                
                resultHTML += `</div>`;
                
                // Add context section if available
                if (result.context_source || (result.context_paragraphs && result.context_paragraphs.length > 0)) {
                    resultHTML += `
                        <div class="collapse collapse-arrow bg-base-200 rounded-lg mb-4">
                            <input type="checkbox" />
                            <div class="collapse-title font-medium">
                                Context Information
                            </div>
                            <div class="collapse-content">
                                <div class="mb-3">
                                    <span class="font-semibold">Source:</span>
                                    <span id="context-source">${result.context_source || 'No context source available'}</span>
                                </div>
                    `;
                    
                    // Add context paragraphs if available
                    if (result.context_paragraphs && result.context_paragraphs.length > 0) {
                        resultHTML += `
                            <div class="mb-3">
                                <div class="font-semibold mb-2">Reference Text:</div>
                                <div class="text-sm bg-base-300 p-3 rounded-md">${result.context_paragraphs[0]}</div>
                            </div>
                        `;
                    }
                    
                    resultHTML += `
                            </div>
                        </div>
                    `;
                }
                
                // Add feedback controls
                resultHTML += `
                    <div class="flex items-center mt-4 mb-4">
                        <span class="text-sm mr-2">Rate this classification:</span>
                        <button class="btn btn-outline btn-sm mr-2" id="thumbs-up-button" onclick="provideFeedback('${result.id}', 'positive')">
                            👍
                        </button>
                        <button class="btn btn-outline btn-sm" id="thumbs-down-button" onclick="provideFeedback('${result.id}', 'negative')">
                            👎
                        </button>
                        <span id="feedback-message" class="text-sm ml-4"></span>
                    </div>
                `;
                
                // Add raw response in collapsible section
                resultHTML += `
                    <details class="collapse bg-base-200">
                        <summary class="collapse-title font-medium">Raw Response</summary>
                        <div class="collapse-content">
                            <pre class="text-xs whitespace-pre-wrap">${result.raw_response}</pre>
                        </div>
                    </details>
                `;
                
                // Update the container
                singleResult.innerHTML = resultHTML;
                
                // Save raw response for copy button
                rawResponseText = result.raw_response;
            }
            
            // Function to display batch results with context
            function displayBatchResults(batchResult) {
                console.log("Displaying batch results for", batchResult.results.length, "images");
                const batchResultsContainer = document.getElementById('batch-results');
                
                // Create results header
                let batchHTML = `
                    <h3 class="text-lg font-semibold mb-4 text-center">Batch Results (${batchResult.results.length} images)</h3>
                `;
                
                // Create carousel
                batchHTML += `
                    <div class="carousel">
                        <div class="carousel-inner" id="carousel-items">
                `;
                
                // Add each result as a carousel item
                batchResult.results.forEach((result, index) => {
                    // Determine confidence class
                    let confidenceClass = 'badge-warning';
                    if (result.confidence === 'High') {
                        confidenceClass = 'badge-success';
                    } else if (result.confidence === 'Low') {
                        confidenceClass = 'badge-error';
                    }
                    
                    batchHTML += `
                        <div class="carousel-item" id="slide-${index}">
                            <div class="p-4 bg-base-200 rounded-lg max-w-3xl mx-auto">
                                <div class="flex justify-between items-center mb-2">
                                    <h3 class="text-lg font-medium">Result ${index + 1} of ${batchResult.results.length}</h3>
                                    <span class="badge ${confidenceClass}">Confidence: ${result.confidence}</span>
                                </div>
                                <h4 class="text-lg font-bold mb-2">${result.category}</h4>
                                <p class="mb-4">${result.description}</p>
                    `;
                    
                    // Add additional details
                    const details = result.details;
                    for (const key in details) {
                        if (key !== 'Main Category' && key !== 'Confidence' && key !== 'Description') {
                            batchHTML += `
                                <div class="mb-2">
                                    <span class="font-semibold">${key}:</span>
                                    <span>${details[key]}</span>
                                </div>
                            `;
                        }
                    }
                    
                    // Add feedback controls
                    batchHTML += `
                                <div class="flex items-center mt-4">
                                    <span class="text-sm mr-2">Rate this classification:</span>
                                    <button class="btn btn-outline btn-sm mr-2" id="thumbs-up-button-${index}" onclick="provideFeedback('${result.id}', 'positive', 'thumbs-up-button-${index}', 'thumbs-down-button-${index}', 'feedback-message-${index}')">
                                        👍
                                    </button>
                                    <button class="btn btn-outline btn-sm" id="thumbs-down-button-${index}" onclick="provideFeedback('${result.id}', 'negative', 'thumbs-up-button-${index}', 'thumbs-down-button-${index}', 'feedback-message-${index}')">
                                        👎
                                    </button>
                                    <span id="feedback-message-${index}" class="text-sm ml-4"></span>
                                </div>
                            </div>
                        </div>
                    `;
                });
                
                // Close carousel container
                batchHTML += `
                        </div>
                `;
                
                // Add navigation if more than one result
                if (batchResult.results.length > 1) {
                    batchHTML += `
                        <button class="btn btn-circle carousel-control carousel-control-prev" id="prev-btn">❮</button>
                        <button class="btn btn-circle carousel-control carousel-control-next" id="next-btn">❯</button>
                        <div class="carousel-indicators" id="carousel-indicators">
                    `;
                    
                    // Add indicators for navigation
                    for (let i = 0; i < batchResult.results.length; i++) {
                        batchHTML += `
                            <div class="carousel-indicator ${i === 0 ? 'active' : ''}" data-index="${i}"></div>
                        `;
                    }
                    
                    batchHTML += `</div>`;
                }
                
                // Add context section if available
                if (batchResult.context_source || (batchResult.context_paragraphs && batchResult.context_paragraphs.length > 0)) {
                    batchHTML += `
                        <div class="collapse collapse-arrow bg-base-200 rounded-lg mt-4 mb-4">
                            <input type="checkbox" />
                            <div class="collapse-title font-medium">
                                Shared Context Information
                            </div>
                            <div class="collapse-content">
                                <div class="mb-3">
                                    <span class="font-semibold">Source:</span>
                                    <span>${batchResult.context_source || 'No context source available'}</span>
                                </div>
                    `;
                    
                    // Add context paragraphs if available
                    if (batchResult.context_paragraphs && batchResult.context_paragraphs.length > 0) {
                        batchHTML += `
                            <div class="mb-3">
                                <div class="font-semibold mb-2">Reference Text:</div>
                                <div class="text-sm bg-base-300 p-3 rounded-md">${batchResult.context_paragraphs[0]}</div>
                            </div>
                        `;
                    }
                    
                    batchHTML += `
                            </div>
                        </div>
                    `;
                }
                
                // Add raw response in collapsible section
                batchHTML += `
                    <details class="collapse bg-base-200 mt-4">
                        <summary class="collapse-title font-medium">Raw Response</summary>
                        <div class="collapse-content">
                            <pre class="text-xs whitespace-pre-wrap">${batchResult.raw_response}</pre>
                        </div>
                    </details>
                `;
                
                // Set HTML
                batchResultsContainer.innerHTML = batchHTML;
                
                // Save raw response for copy button
                rawResponseText = batchResult.raw_response;
                
                // Setup carousel navigation if needed
                if (batchResult.results.length > 1) {
                    console.log("Setting up carousel for", batchResult.results.length, "items");
                    const carouselItems = document.getElementById('carousel-items');
                    const prevBtn = document.getElementById('prev-btn');
                    const nextBtn = document.getElementById('next-btn');
                    const indicators = document.querySelectorAll('.carousel-indicator');
                    
                    let currentIndex = 0;
                    
                    // Show a specific slide
                    function showSlide(index) {
                        // Handle wrapping
                        if (index < 0) index = batchResult.results.length - 1;
                        if (index >= batchResult.results.length) index = 0;
                        
                        currentIndex = index;
                        console.log("Showing slide", index);
                        
                        // Update transform
                        carouselItems.style.transform = `translateX(-${index * 100}%)`;
                        
                        // Update indicators
                        indicators.forEach((dot, i) => {
                            if (i === index) {
                                dot.classList.add('active');
                            } else {
                                dot.classList.remove('active');
                            }
                        });
                    }
                    
                    // Add event listeners
                    if (prevBtn) prevBtn.addEventListener('click', () => showSlide(currentIndex - 1));
                    if (nextBtn) nextBtn.addEventListener('click', () => showSlide(currentIndex + 1));
                    
                    // Add indicator click handlers
                    indicators.forEach((indicator, i) => {
                        indicator.addEventListener('click', () => showSlide(i));
                    });
                    
                    // Initialize first slide
                    showSlide(0);
                }
            }
            
            // Global function for providing feedback
            window.provideFeedback = function(resultId, feedbackType, upButtonId, downButtonId, messageId) {
                console.log("Providing feedback:", feedbackType, "for result:", resultId);
                
                // Get button elements
                const upButton = document.getElementById(upButtonId || 'thumbs-up-button');
                const downButton = document.getElementById(downButtonId || 'thumbs-down-button');
                const messageElement = document.getElementById(messageId || 'feedback-message');
                
                if (!upButton || !downButton) {
                    console.error("Feedback buttons not found");
                    return;
                }
                
                // Update button UI state
                if (feedbackType === 'positive') {
                    upButton.classList.add('btn-success', 'btn-active');
                    downButton.classList.remove('btn-error', 'btn-active');
                } else {
                    upButton.classList.remove('btn-success', 'btn-active');
                    downButton.classList.add('btn-error', 'btn-active');
                }
                
                // Show sending message
                if (messageElement) {
                    messageElement.textContent = 'Saving feedback...';
                }
                
                // Send feedback to server
                fetch('/api/feedback', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        id: resultId,
                        feedback: feedbackType
                    })
                })
                .then(response => {
                    if (!response.ok) {
                        throw new Error('Failed to save feedback');
                    }
                    return response.json();
                })
                .then(data => {
                    console.log('Feedback saved successfully:', data);
                    if (messageElement) {
                        messageElement.textContent = 'Feedback saved!';
                        
                        // Clear message after a delay
                        setTimeout(() => {
                            messageElement.textContent = '';
                        }, 3000);
                    }
                })
                .catch(error => {
                    console.error('Error saving feedback:', error);
                    if (messageElement) {
                        messageElement.textContent = 'Error saving feedback.';
                    }
                    
                    // Reset buttons 
                    upButton.classList.remove('btn-success', 'btn-active');
                    downButton.classList.remove('btn-error', 'btn-active');
                });
            };
        });
        """)
        
        return Title("Insect Classifier"), Main(
            form_script,
            Div(
                H1("Insect Classification App", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Powered by Claude's Vision AI with RAG", cls="text-center mb-8 text-base-content/70"),
                navbar,  # Add the navbar here
                Div(
                    control_panel,
                    results_panel,
                    cls="flex flex-col md:flex-row gap-6 w-full"
                ),
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    # Return the FastHTML app
    return fasthtml_app

# When running locally
if __name__ == "__main__":
    print("Starting Insect Classification App...")
    # This section is only executed when running the script directly, not through Modal
