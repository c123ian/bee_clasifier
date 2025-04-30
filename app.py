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

# Claude API constants
CLAUDE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
        import traceback
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
        import traceback
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
        import traceback
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
            Script(src="https://unpkg.com/htmx.org@1.9.10"),
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
                
                /* Carousel styles */
                .carousel {
                    position: relative;
                    overflow: hidden;
                    width: 100%;
                }
                
                .carousel-inner {
                    display: flex;
                    transition: transform 0.5s ease;
                }
                
                .carousel-item {
                    flex: 0 0 100%;
                    width: 100%;
                }
                
                .carousel-control {
                    position: absolute;
                    top: 50%;
                    transform: translateY(-50%);
                    z-index: 10;
                }
                
                .carousel-control-prev {
                    left: 10px;
                }
                
                .carousel-control-next {
                    right: 10px;
                }
                
                .carousel-indicators {
                    display: flex;
                    justify-content: center;
                    margin-top: 15px;
                }
                
                .carousel-indicator {
                    width: 10px;
                    height: 10px;
                    border-radius: 50%;
                    background: var(--color-base-300);
                    margin: 0 5px;
                    cursor: pointer;
                }
                
                .carousel-indicator.active {
                    background: var(--color-primary);
                }
                
                /* New context section styles */
                .context-section {
                    background-color: var(--color-base-200);
                    border-radius: 0.5rem;
                    padding: 1rem;
                    margin-top: 1rem;
                }
                
                .context-image {
                    max-height: 200px;
                    object-fit: contain;
                    margin: 0 auto;
                    display: block;
                    border-radius: 0.5rem;
                    border: 1px solid var(--color-base-300);
                }
                
                .context-text {
                    background-color: var(--color-base-300);
                    padding: 0.75rem;
                    border-radius: 0.375rem;
                    font-size: 0.875rem;
                    margin-top: 0.5rem;
                }
                
                /* Button state styles */
                .btn:disabled {
                    opacity: 0.5 !important;
                    cursor: not-allowed !important;
                    pointer-events: none !important;
                }
                
                .btn:not(:disabled) {
                    cursor: pointer !important;
                    opacity: 1 !important;
                }
                
                /* Add a visible hover effect for enabled buttons */
                .btn:not(:disabled):hover {
                    filter: brightness(1.1);
                    transform: translateY(-1px);
                    transition: all 0.2s ease;
                }
                
                /* Add more obvious active state */
                .btn:not(:disabled):active {
                    transform: translateY(1px);
                }
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
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
    
    #################################################
    # Dashboard Route - Enhanced with Context Stats
    #################################################
    @rt("/dashboard")
    def dashboard():
        """Render the insect classification dashboard with RAG stats and pie chart"""
        stats = get_classification_stats()
        
        # Import the Charts.css stylesheet
        charts_css = Link(rel="stylesheet", href="https://cdn.jsdelivr.net/npm/charts.css/dist/charts.min.css")
        
        # Create navigation bar (same as homepage)
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
        
        # Categories Section with Pie Chart
        # Calculate the total for percentage calculations
        total_insects = sum(count for _, count in stats["combined_category_counts"])
        
        # Prepare data for the pie chart - we'll use the top 10 categories
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
        
        # Create table rows for pie chart
        pie_rows = []
        for data in pie_data:
            # Create a table row with styling for the pie chart segment
            # Format percentages to 1 decimal place
            pie_rows.append(
                Tr(
                    Td(
                        Span(f"{data['category']}: {data['percentage']:.1f}%", cls="data"),
                        style=f"--start: {data['start']}; --end: {data['end']};"
                    )
                )
            )
        
        # Build the pie chart using Charts.css
        pie_chart = Div(
            H3("Insect Category Distribution", cls="font-semibold mb-3 text-center"),
            Div(
                Table(
                    Caption("Category Distribution"),
                    Tbody(*pie_rows),
                    cls="charts-css pie show-labels"
                ),
                cls="mx-auto w-64 h-64"  # Set dimensions of the chart
            ),
            # Add a legend for the pie chart
            Div(
                *[
                    Div(
                        Span(cls="w-3 h-3 inline-block mr-1", 
                            style=f"background-color: var(--color-{i+1});"),
                        Span(f"{data['category']} ({data['count']})", cls="text-sm"),
                        cls="mb-1"
                    )
                    for i, data in enumerate(pie_data)
                ],
                cls="mt-4 text-center grid grid-cols-2 gap-2"
            ),
            cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
        )
        
        # Modified categories section to include pie chart
        categories_section = Div(
            H2("Classification Categories", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                Div(
                    pie_chart,  # Add the pie chart here
                    cls="w-full md:w-2/5"
                ),
                Div(
                    H3("Distribution by Category", cls="font-semibold mb-3"),
                    Table(
                        Thead(
                            Tr(
                                Th("Category"),
                                Th("Count"),
                                Th("Percentage"),
                            )
                        ),
                        Tbody(
                            *[
                                Tr(
                                    Td(category),
                                    Td(str(count)),
                                    Td(f"{count / max(stats['total'], 1) * 100:.1f}%"),
                                )
                                for category, count in stats["combined_category_counts"][:10]
                            ]
                        ),
                        cls="table table-zebra w-full"
                    ),
                    cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border w-full md:w-3/5"
                ),
                cls="flex flex-col md:flex-row gap-6 w-full"
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
        
        # Recent Classifications Section
        recent_classifications_section = Div(
            H2("Recent Classifications", cls="text-xl font-bold mb-4 text-bee-green"),
            Div(
                Table(
                    Thead(
                        Tr(
                            Th("ID"),
                            Th("Category"),
                            Th("Confidence"),
                            Th("Feedback"),
                            Th("Source"),
                            Th("Time"),
                        )
                    ),
                    Tbody(
                        *[
                            Tr(
                                Td(id[:8] + "..."),
                                Td(category),
                                Td(
                                    Span(
                                        confidence,
                                        cls=f"badge {'badge-success' if confidence == 'High' else 'badge-warning' if confidence == 'Medium' else 'badge-error'}"
                                    )
                                ),
                                Td(
                                    Span(
                                        feedback if feedback else "None",
                                        cls=f"{'badge badge-success' if feedback == 'positive' else 'badge badge-error' if feedback == 'negative' else ''}"
                                    )
                                ),
                                Td(context_source if context_source else "None"),
                                Td(created_at),
                            )
                            for id, category, confidence, feedback, created_at, context_source in stats["recent_classifications"]
                        ]
                    ),
                    cls="table table-zebra w-full"
                ),
                cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border overflow-x-auto"
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
        
        # Daily Classification Activity
        daily_activity_section = ""
        if stats["daily_counts"]:
            daily_activity_section = Div(
                H2("Daily Classification Activity", cls="text-xl font-bold mb-4 text-bee-green"),
                Div(
                    Table(
                        Thead(
                            Tr(
                                Th("Date"),
                                Th("Classifications"),
                            )
                        ),
                        Tbody(
                            *[
                                Tr(
                                    Td(date),
                                    Td(str(count)),
                                )
                                for date, count in stats["daily_counts"]
                            ]
                        ),
                        cls="table table-zebra w-full"
                    ),
                    cls="bg-base-100 p-6 rounded-lg shadow-md border custom-border"
                ),
                cls="mb-8"
            )
        
        # Add custom CSS for the pie chart colors
        pie_chart_styles = Style("""
            /* Pie chart colors */
            .charts-css.pie tbody tr:nth-child(1) {
                --color: var(--color-primary);
            }
            .charts-css.pie tbody tr:nth-child(2) {
                --color: var(--color-secondary);
            }
            .charts-css.pie tbody tr:nth-child(3) {
                --color: var(--color-accent);
            }
            .charts-css.pie tbody tr:nth-child(4) {
                --color: #1e88e5;
            }
            .charts-css.pie tbody tr:nth-child(5) {
                --color: #43a047;
            }
            .charts-css.pie tbody tr:nth-child(6) {
                --color: #ffb300;
            }
            .charts-css.pie tbody tr:nth-child(7) {
                --color: #e53935;
            }
            .charts-css.pie tbody tr:nth-child(8) {
                --color: #8e24aa;
            }
            .charts-css.pie tbody tr:nth-child(9) {
                --color: #00acc1;
            }
            .charts-css.pie tbody tr:nth-child(10) {
                --color: #f4511e;
            }
            
            /* Legend color boxes */
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
            
            /* Improve Charts.css styling for our theme */
            .charts-css.pie {
                --chart-bg: transparent;
                height: 250px;
                max-width: 250px;
                margin: 0 auto;
            }
            
            .charts-css.pie .data {
                font-size: 10px;
                color: transparent;
            }
            
            .charts-css caption {
                margin-bottom: 1rem;
                font-weight: bold;
            }
        """)
        
        return Title("Dashboard - Insect Classifier"), Main(
            charts_css,  # Add the Charts.css stylesheet
            pie_chart_styles,  # Add custom styles for the pie chart
            Div(
                H1("Classification Dashboard", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Statistics and insights from the Insect Classifier with RAG", cls="text-center mb-8 text-base-content/70"),
                navbar,
                summary_cards,
                categories_section,  # Updated to include pie chart
                confidence_feedback_section,
                recent_classifications_section,
                rag_section,
                daily_activity_section,
                cls="container mx-auto px-4 py-8 max-w-7xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
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
    # Batch Classify API Endpoint - FIXED
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
    # API route for saving feedback
    #################################################
    @rt("/api/feedback", methods=["POST"])
    async def api_save_feedback(request):
        """API endpoint to save user feedback on classification results"""
        try:
            # Get feedback data from request JSON
            data = await request.json()
            result_id = data.get("id", "")
            feedback = data.get("feedback", "")
            
            if not result_id or not feedback:
                return JSONResponse({"error": "Missing required parameters"}, status_code=400)
            
            # Validate feedback type
            if feedback not in ["positive", "negative"]:
                return JSONResponse({"error": "Invalid feedback type"}, status_code=400)
            
            # Save feedback to database
            try:
                conn = sqlite3.connect(DB_PATH, timeout=30.0)
                cursor = conn.cursor()
                
                # Update the feedback column for the specific result
                cursor.execute(
                    "UPDATE results SET feedback = ? WHERE id = ?",
                    (feedback, result_id)
                )
                
                # Check if any row was affected
                if cursor.rowcount == 0:
                    # Try to find the result in batch_results table
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
                
                return JSONResponse({"success": True, "id": result_id, "feedback": feedback})
                
            except Exception as e:
                print(f"⚠️ Error saving feedback to database: {e}")
                return JSONResponse({"error": str(e)}, status_code=500)
                    
        except Exception as e:
            print(f"Error processing feedback: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    #################################################
    # Context Image Endpoint
    #################################################
    @rt("/context-image", methods=["GET"])
    async def get_context_image(request):
        """Serve a context image from the PDF documents"""
        try:
            # Get the image path from query parameters
            image_path = request.query_params.get("path", "")
            
            if not image_path or not os.path.exists(image_path):
                return JSONResponse({"error": "Context image not found"}, status_code=404)
            
            # Get the image content type
            content_type = "image/png"  # Default to PNG
            if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
                content_type = "image/jpeg"
            
            # Read the image file
            with open(image_path, "rb") as f:
                content = f.read()
            
            # Return the image
            return Response(content=content, media_type=content_type)
            
        except Exception as e:
            print(f"Error serving context image: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    return fasthtml_app

# When running locally
if __name__ == "__main__":
    print("Starting Insect Classification App...")
    # This section is only executed when running the script directly, not through Modal
