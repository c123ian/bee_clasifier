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
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

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

# Create custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "libsm6", "libxrender1", "libxext6")
    .pip_install(
        "requests",
        "python-fasthtml==0.12.0",
        "torch",
        "numpy",
        "pandas",
        "Pillow",
        "matplotlib",
        "rerankers",
        "rank-bm25",
        "nltk",
        "faiss-cpu",
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

# Format image for API calls
def format_image(image):
    """Convert PIL Image to base64 for API"""
    buffered = io.BytesIO()
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

# Generate classification using Claude's API for a single image
@app.function(
    image=image,
    cpu=1.0,
    timeout=300,
    volumes={DATA_DIR: bee_volume}
)
async def classify_image_claude(image_data: str, options: Dict[str, bool]) -> Dict[str, Any]:
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
    query = "insect classification"
    if options.get("use_rag", True):  # Default to using RAG
        retrieved_paragraphs, top_sources = await retrieve_relevant_documents(query)
        context_text = ""
        
        if retrieved_paragraphs:
            context_text = "\n\nReference Information:\n" + "\n\n".join(retrieved_paragraphs)
            if top_sources and len(top_sources) > 0:
                source = top_sources[0]
                context_source = f"{source.get('filename', 'unknown document')}, page {source.get('page', 'unknown')}"
            else:
                context_source = None
        else:
            context_source = None
    else:
        retrieved_paragraphs = []
        top_sources = []
        context_text = ""
        context_source = None
    
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
                "raw_response": classification_text,
                "db_error": str(db_error)
            }
            
    except Exception as e:
        print(f"⚠️ Error in classification: {e}")
        return {
            "error": str(e),
            "id": result_id
        }

@app.function(
    image=image,
    volumes={DATA_DIR: bee_volume},
    cpu=1.0,
    timeout=3600
)
@modal.asgi_app()
def serve():
    from rank_bm25 import BM25Okapi
    """Main FastHTML Server for Bee Classifier Dashboard with RAG"""
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
                cls="flex justify-center items-center h-32 hidden"
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
        
        # Add script for unified form handling
        form_script = Script("""
        document.addEventListener('DOMContentLoaded', function() {
            // Form elements
            const imageInput = document.getElementById('image-input');
            const singlePreview = document.getElementById('single-preview');
            const batchPreviewsContainer = document.getElementById('batch-previews');
            const countDisplay = document.getElementById('count-display');
            const imageCountElem = document.getElementById('image-count');
            const classifyButton = document.getElementById('classify-button');
            
            // Results elements
            const loadingIndicator = document.getElementById('loading-indicator').parentElement;
            const resultsPlaceholder = document.getElementById('results-placeholder');
            const resultsContent = document.getElementById('results-content');
            const singleResult = document.getElementById('single-result');
            const batchResults = document.getElementById('batch-results');
            const resultActions = document.getElementById('result-actions');
            const copyButton = document.getElementById('copy-button');
            const newButton = document.getElementById('new-button');
            
            // Mode tracking
            let isBatchMode = false;
            const MAX_IMAGES = 5;
            let selectedFiles = [];
            let rawResponseText = '';
            
            // Get options from the form
            function getOptions() {
                return {
                    use_rag: document.querySelector('input[name="use_rag"]').checked,
                    detailed_description: document.querySelector('input[name="detailed_description"]').checked,
                    plant_classification: document.querySelector('input[name="plant_classification"]').checked,
                    taxonomy: document.querySelector('input[name="taxonomy"]').checked
                };
            }
            
            // Handle file selection
            window.handleFileSelection = function(event) {
                const files = event.target.files;
                
                if (!files || files.length === 0) {
                    resetForm();
                    return;
                }
                
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
                    showSinglePreview(selectedFiles[0]);
                }
                
                // Enable classify button
                classifyButton.disabled = false;
            };
            
            // Show single image preview
            function showSinglePreview(file) {
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
            
            // Reset the form
            function resetForm() {
                imageInput.value = '';
                singlePreview.src = '';
                singlePreview.classList.add('hidden');
                batchPreviewsContainer.innerHTML = '';
                batchPreviewsContainer.classList.add('hidden');
                countDisplay.classList.add('hidden');
                classifyButton.disabled = true;
                selectedFiles = [];
                isBatchMode = false;
            }
            
            // Handle classify button click
            classifyButton.addEventListener('click', function() {
                // Show loading state
                loadingIndicator.classList.remove('hidden');
                resultsPlaceholder.classList.add('hidden');
                resultsContent.classList.add('hidden');
                resultActions.classList.add('hidden');
                classifyButton.disabled = true;
                
                if (isBatchMode) {
                    // Batch mode - process multiple images
                    processBatchImages();
                } else {
                    // Single mode - process one image
                    processSingleImage();
                }
            });
            
            // Process a single image
            function processSingleImage() {
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
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    
                    if (data.error) {
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
                });
            }
            
            // Process batch of images
            function processBatchImages() {
                // Create form data for file upload
                const formData = new FormData();
                
                // Add all files
                selectedFiles.forEach((file, index) => {
                    formData.append(`image_${index}`, file);
                });
                
                // Add options
                formData.append('options', JSON.stringify(getOptions()));
                
                // Send batch request
                fetch('/classify-batch', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.classList.add('hidden');
                    
                    if (data.error) {
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
                });
            }
            
            // Setup copy button
            copyButton.addEventListener('click', function() {
                navigator.clipboard.writeText(rawResponseText);
                copyButton.innerHTML = 'Copied!';
                setTimeout(() => {
                    copyButton.innerHTML = 'Copy Results';
                }, 2000);
            });
            
            // Setup new button
            newButton.addEventListener('click', function() {
                // Reset form
                resetForm();
                
                // Reset results
                resultsPlaceholder.classList.remove('hidden');
                resultsContent.classList.add('hidden');
                resultActions.classList.add('hidden');
                singleResult.classList.add('hidden');
                batchResults.classList.add('hidden');
            });
            
            // Function to display single classification result with context
            function displaySingleResult(result) {
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
                    prevBtn.addEventListener('click', () => showSlide(currentIndex - 1));
                    nextBtn.addEventListener('click', () => showSlide(currentIndex + 1));
                    
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
                // Get button elements
                const upButton = document.getElementById(upButtonId || 'thumbs-up-button');
                const downButton = document.getElementById(downButtonId || 'thumbs-down-button');
                const messageElement = document.getElementById(messageId || 'feedback-message');
                
                // Update button UI state
                if (feedbackType === 'positive') {
                    upButton.classList.add('btn-success', 'btn-active');
                    downButton.classList.remove('btn-error', 'btn-active');
                } else {
                    upButton.classList.remove('btn-success', 'btn-active');
                    downButton.classList.add('btn-error', 'btn-active');
                }
                
                // Show sending message
                messageElement.textContent = 'Saving feedback...';
                
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
                    messageElement.textContent = 'Feedback saved!';
                    
                    // Clear message after a delay
                    setTimeout(() => {
                        messageElement.textContent = '';
                    }, 3000);
                })
                .catch(error => {
                    console.error('Error saving feedback:', error);
                    messageElement.textContent = 'Error saving feedback.';
                    
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
        """Render the insect classification dashboard with RAG stats"""
        
        # Get statistics data
        stats = get_classification_stats()
        
        # Format category counts for React component
        category_data = []
        for category, count in stats["combined_category_counts"]:
            category_data.append({"name": category, "value": count})
        
        # Format daily counts for the line chart
        daily_data = []
        for date, count in stats["daily_counts"]:
            daily_data.append({"date": date, "count": count})
        daily_data.reverse()  # Show oldest to newest for the line chart
        
        # Format confidence data for the donut chart
        confidence_data = []
        for confidence, count in stats["confidence_counts"]:
            confidence_data.append({"name": confidence, "value": count})
        
        # Format feedback data for a pie chart
        feedback_data = []
        positive_count = 0
        negative_count = 0
        for feedback, count in stats["feedback_counts"]:
            if feedback == "positive":
                positive_count = count
                feedback_data.append({"name": "👍 Positive", "value": count})
            elif feedback == "negative":
                negative_count = count
                feedback_data.append({"name": "👎 Negative", "value": count})
        
        # Helper function for confidence classes
        def get_confidence_class(confidence):
            if confidence == 'High':
                return 'confidence-high'
            elif confidence == 'Low':
                return 'confidence-low'
            return 'confidence-medium'
        
        # Create the dashboard HTML
        dashboard_stats = Div(
            H2("Classification Statistics", cls="text-2xl font-bold mb-6 text-bee-green"),
            
            Div(
                Div(
                    Div(
                        H3("Total Classifications", cls="text-lg font-medium text-base-content/70"),
                        Div(
                            Span(f"{stats['total']:,}", cls="text-4xl font-bold text-primary"),
                            cls="mt-2"
                        ),
                        cls="p-4 bg-base-200 rounded-lg"
                    ),
                    cls="w-full md:w-1/3"
                ),
                
                Div(
                    Div(
                        H3("Single Images", cls="text-lg font-medium text-base-content/70"),
                        Div(
                            Span(f"{stats['total_single']:,}", cls="text-4xl font-bold text-secondary"),
                            cls="mt-2"
                        ),
                        cls="p-4 bg-base-200 rounded-lg"
                    ),
                    cls="w-full md:w-1/3"
                ),
                
                Div(
                    Div(
                        H3("Batch Images", cls="text-lg font-medium text-base-content/70"),
                        Div(
                            Span(f"{stats['total_batch']:,}", cls="text-4xl font-bold text-accent"),
                            cls="mt-2"
                        ),
                        cls="p-4 bg-base-200 rounded-lg"
                    ),
                    cls="w-full md:w-1/3"
                ),
                
                cls="flex flex-col md:flex-row gap-4 mb-6"
            ),
            
            Div(
                Div(
                    H3("Insect Diversity", cls="text-xl font-semibold mb-4 text-bee-green"),
                    Div(
                        id="insect-diversity-chart",
                        cls="h-72"
                    ),
                    cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border mb-6"
                ),
                cls="w-full"
            ),
            
            Div(
                Div(
                    H3("Confidence Levels", cls="text-xl font-semibold mb-4 text-bee-green"),
                    Div(
                        id="confidence-chart",
                        cls="h-72"
                    ),
                    cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border"
                ),
                
                Div(
                    H3("Classifications Over Time", cls="text-xl font-semibold mb-4 text-bee-green"),
                    Div(
                        id="time-series-chart",
                        cls="h-72"
                    ),
                    cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border"
                ),
                
                cls="flex flex-col md:flex-row gap-6 mb-6"
            ),
            
            # New section for feedback statistics
            Div(
                H3("User Feedback", cls="text-xl font-semibold mb-4 text-bee-green"),
                Div(
                    # Feedback summary
                    Div(
                        Div(
                            Span("👍", cls="text-4xl"),
                            Div(
                                Span(f"{positive_count:,}", cls="text-3xl font-bold text-success"),
                                P("Positive Ratings", cls="text-sm opacity-75"),
                                cls="ml-4"
                            ),
                            cls="flex items-center mb-4"
                        ),
                        Div(
                            Span("👎", cls="text-4xl"),
                            Div(
                                Span(f"{negative_count:,}", cls="text-3xl font-bold text-error"),
                                P("Negative Ratings", cls="text-sm opacity-75"),
                                cls="ml-4"
                            ),
                            cls="flex items-center"
                        ),
                        cls="px-8 py-6 bg-base-200 rounded-lg"
                    ),
                    # Donut chart for feedback
                    Div(
                        Div(
                            id="feedback-chart",
                            cls="h-64"
                        ),
                        cls="flex-1 bg-base-200 rounded-lg px-4 py-6"
                    ),
                    cls="flex flex-col md:flex-row gap-6"
                ),
                cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border mb-6"
            ),
            
            # New section for context source statistics
            (Div(
                H3("Context Sources", cls="text-xl font-semibold mb-4 text-bee-green"),
                Table(
                    Thead(
                        Tr(
                            Th("Source Document", cls="px-4 py-2"),
                            Th("Usage Count", cls="px-4 py-2"),
                            Th("Percentage", cls="px-4 py-2")
                        ),
                        cls="bg-base-200"
                    ),
                    Tbody(
                        *[
                            Tr(
                                Td(source, cls="px-4 py-2"),
                                Td(str(count), cls="px-4 py-2 text-center"),
                                Td(
                                    f"{(count / stats['total'] * 100):.1f}%", 
                                    cls="px-4 py-2 text-center"
                                ),
                                cls="border-b border-base-200 hover:bg-base-200"
                            )
                            for source, count in stats["context_counts"]
                        ] if stats["context_counts"] else [
                            Tr(
                                Td("No context sources found", cls="px-4 py-2 text-center font-italic", colspan="3")
                            )
                        ],
                        cls=""
                    ),
                    cls="table w-full"
                ),
                cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border mb-6"
            ) if stats["context_counts"] else ""),
            
            Div(
                H3("Recent Classifications", cls="text-xl font-semibold mb-4 text-bee-green"),
                Table(
                    Thead(
                        Tr(
                            Th("ID", cls="px-4 py-2"),
                            Th("Category", cls="px-4 py-2"),
                            Th("Confidence", cls="px-4 py-2"),
                            Th("Context Source", cls="px-4 py-2"),
                            Th("Feedback", cls="px-4 py-2"),
                            Th("Date", cls="px-4 py-2")
                        ),
                        cls="bg-base-200"
                    ),
                    Tbody(
                        *[
                            Tr(
                                Td(Id[:8] + "...", cls="px-4 py-2 text-sm"),
                                Td(Category, cls="px-4 py-2"),
                                Td(
                                    Span(
                                        Confidence, 
                                        cls=f"badge {get_confidence_class(Confidence)}"
                                    ),
                                    cls="px-4 py-2"
                                ),
                                Td(ContextSource or "None", cls="px-4 py-2 text-sm"),
                                Td(
                                    Span("👍" if Feedback == "positive" else "👎" if Feedback == "negative" else "—"),
                                    cls="px-4 py-2 text-center"
                                ),
                                Td(Date.split()[0], cls="px-4 py-2 text-sm"),
                                cls="border-b border-base-200 hover:bg-base-200"
                            )
                            for Id, Category, Confidence, Feedback, Date, ContextSource in stats["recent_classifications"]
                        ] if stats["recent_classifications"] else [
                            Tr(
                                Td("No recent classifications found", cls="px-4 py-2 text-center font-italic", colspan="6")
                            )
                        ],
                        cls=""
                    ),
                    cls="table w-full"
                ),
                cls="p-4 bg-base-100 rounded-lg shadow-lg custom-border border mb-6"
            ),
            
            Div(
                A(
                    "Back to Classifier",
                    href="/",
                    cls="btn btn-primary"
                ),
                cls="mt-6"
            ),
            
            cls="container mx-auto px-4 max-w-6xl"
        )
        
        # Add visualization React components
        visualization_script = Script(f"""
        // Helper function for confidence classes
        function getConfidenceClass(confidence) {{
            if (confidence === 'High') return 'confidence-high';
            if (confidence === 'Low') return 'confidence-low';
            return 'confidence-medium';
        }}
        
        // Category data for the pie chart
        const categoryData = {json.dumps(category_data)};
        
        // Daily data for line chart
        const dailyData = {json.dumps(daily_data)};
        
        // Confidence data for donut chart
        const confidenceData = {json.dumps(confidence_data)};
        
        // Feedback data for pie chart
        const feedbackData = {json.dumps(feedback_data)};
        
        // Color scheme for charts
        const COLORS = [
            '#4caf50', '#8bc34a', '#cddc39', 
            '#ffc107', '#ff9800', '#ff5722',
            '#f44336', '#e91e63', '#9c27b0', 
            '#673ab7'
        ];
        
        // Confidence level colors
        const CONFIDENCE_COLORS = {{
            'High': '#4caf50',
            'Medium': '#ffc107',
            'Low': '#f44336'
        }};
        
        // Feedback colors
        const FEEDBACK_COLORS = {{
            '👍 Positive': '#4caf50',
            '👎 Negative': '#f44336'
        }};
        
        // Render the insect diversity pie chart
        function renderInsectDiversityChart() {{
            const container = document.getElementById('insect-diversity-chart');
            if (!container || categoryData.length === 0) return;
            
            // Create SVG
            const width = container.clientWidth;
            const height = container.clientHeight;
            const radius = Math.min(width, height) / 2 * 0.8;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${{width / 2}},${{height / 2}})`);
            
            // Compute the total
            const total = categoryData.reduce((sum, entry) => sum + entry.value, 0);
            
            // Create pie layout
            const pie = d3.pie()
                .sort(null)
                .value(d => d.value);
            
            const data_ready = pie(categoryData);
            
            // Create arcs
            const arc = d3.arc()
                .innerRadius(0)
                .outerRadius(radius);
            
            const outerArc = d3.arc()
                .innerRadius(radius * 0.9)
                .outerRadius(radius * 0.9);
            
            // Add the arcs
            svg.selectAll('allSlices')
                .data(data_ready)
                .enter()
                .append('path')
                .attr('d', arc)
                .attr('fill', (d, i) => COLORS[i % COLORS.length])
                .attr('stroke', 'white')
                .style('stroke-width', '2px')
                .style('opacity', 0.7);
            
            // Add labels
            svg.selectAll('allLabels')
                .data(data_ready)
                .enter()
                .append('text')
                .text(d => {{
                    // Only show label if it's a significant slice (>3% of total)
                    const percent = (d.data.value / total * 100).toFixed(1);
                    if (percent < 3) return '';
                    return `${{d.data.name}} (${{percent}}%)`;
                }})
                .attr('transform', d => {{
                    const pos = outerArc.centroid(d);
                    const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
                    pos[0] = radius * 0.99 * (midangle < Math.PI ? 1 : -1);
                    return `translate(${{pos[0]}}, ${{pos[1]}})`;
                }})
                .style('text-anchor', d => {{
                    const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
                    return (midangle < Math.PI ? 'start' : 'end');
                }})
                .style('font-size', '12px');
            
            // Add lines
            svg.selectAll('allPolylines')
                .data(data_ready)
                .enter()
                .append('polyline')
                .attr('points', d => {{
                    const posA = arc.centroid(d);
                    const posB = outerArc.centroid(d);
                    const posC = outerArc.centroid(d);
                    const midangle = d.startAngle + (d.endAngle - d.startAngle) / 2;
                    posC[0] = radius * 0.95 * (midangle < Math.PI ? 1 : -1);
                    
                    // Only show lines for slices with labels (>3% of total)
                    const percent = (d.data.value / total * 100).toFixed(1);
                    if (percent < 3) return '';
                    
                    return [posA, posB, posC];
                }})
                .style('fill', 'none')
                .style('stroke', 'gray')
                .style('stroke-width', 1);
        }}
        
        // Render the time series chart
        function renderTimeSeriesChart() {{
            const container = document.getElementById('time-series-chart');
            if (!container || dailyData.length === 0) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            const margin = {{top: 20, right: 30, bottom: 40, left: 50}};
            const innerWidth = width - margin.left - margin.right;
            const innerHeight = height - margin.top - margin.bottom;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${{margin.left}},${{margin.top}})`);
            
            // Create scales
            const x = d3.scaleBand()
                .domain(dailyData.map(d => d.date))
                .range([0, innerWidth])
                .padding(0.1);
            
            const y = d3.scaleLinear()
                .domain([0, d3.max(dailyData, d => d.count) * 1.1])
                .range([innerHeight, 0]);
            
            // Add the x-axis
            svg.append('g')
                .attr('transform', `translate(0,${{innerHeight}}`)
                .call(d3.axisBottom(x))
                .selectAll("text")
                .attr("y", 10)
                .attr("x", -5)
                .attr("dy", ".35em")
                .attr("transform", "rotate(-45)")
                .style("text-anchor", "end");
            
            // Add the y-axis
            svg.append('g')
                .call(d3.axisLeft(y));
            
            // Add the line
            svg.append('path')
                .datum(dailyData)
                .attr('fill', 'none')
                .attr('stroke', '#4caf50')
                .attr('stroke-width', 2)
                .attr('d', d3.line()
                    .x(d => x(d.date) + x.bandwidth() / 2)
                    .y(d => y(d.count))
                );
            
            // Add circles
            svg.selectAll('circle')
                .data(dailyData)
                .enter()
                .append('circle')
                .attr('cx', d => x(d.date) + x.bandwidth() / 2)
                .attr('cy', d => y(d.count))
                .attr('r', 5)
                .attr('fill', '#4caf50')
                .attr('stroke', 'white')
                .attr('stroke-width', 2);
            
            // Add labels
            svg.selectAll('text.value')
                .data(dailyData)
                .enter()
                .append('text')
                .attr('x', d => x(d.date) + x.bandwidth() / 2)
                .attr('y', d => y(d.count) - 10)
                .attr('text-anchor', 'middle')
                .text(d => d.count)
                .style('font-size', '10px');
        }}
        
        // Render the confidence donut chart
        function renderConfidenceChart() {{
            const container = document.getElementById('confidence-chart');
            if (!container || confidenceData.length === 0) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            const radius = Math.min(width, height) / 2 * 0.8;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${{width / 2}},${{height / 2}})`);
            
            // Compute the total
            const total = confidenceData.reduce((sum, entry) => sum + entry.value, 0);
            
            // Create pie layout
            const pie = d3.pie()
                .sort(null)
                .value(d => d.value);
            
            const data_ready = pie(confidenceData);
            
            // Create arcs
            const arc = d3.arc()
                .innerRadius(radius * 0.5)  // Donut hole
                .outerRadius(radius);
            
            // Add the arcs
            svg.selectAll('allSlices')
                .data(data_ready)
                .enter()
                .append('path')
                .attr('d', arc)
                .attr('fill', d => CONFIDENCE_COLORS[d.data.name] || '#999')
                .attr('stroke', 'white')
                .style('stroke-width', '2px')
                .style('opacity', 0.8);
            
            // Add center text
            svg.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '0em')
                .style('font-size', '16px')
                .style('font-weight', 'bold')
                .text('Confidence');
            
            svg.append('text')
                .attr('text-anchor', 'middle')
                .attr('dy', '1.5em')
                .style('font-size', '14px')
                .text(`Total: ${{total}}`);
            
            // Add labels
            svg.selectAll('allLabels')
                .data(data_ready)
                .enter()
                .append('text')
                .text(d => `${{d.data.name}}: ${{d.data.value}} (${{(d.data.value / total * 100).toFixed(1)}}%)`)
                .attr('transform', d => {{
                    const pos = arc.centroid(d);
                    pos[0] = pos[0] * 1.6;
                    pos[1] = pos[1] * 1.6;
                    return `translate(${{pos[0]}}, ${{pos[1]}})`;
                }})
                .style('text-anchor', 'middle')
                .style('font-size', '12px');
        }}
        
        // Render the feedback pie chart
        function renderFeedbackChart() {{
            const container = document.getElementById('feedback-chart');
            if (!container || feedbackData.length === 0) return;
            
            const width = container.clientWidth;
            const height = container.clientHeight;
            const radius = Math.min(width, height) / 2 * 0.8;
            
            const svg = d3.select(container)
                .append('svg')
                .attr('width', width)
                .attr('height', height)
                .append('g')
                .attr('transform', `translate(${{width / 2}},${{height / 2}})`);
            
            // Compute the total
            const total = feedbackData.reduce((sum, entry) => sum + entry.value, 0);
            
            // No feedback data case
            if (total === 0) {{
                svg.append('text')
                    .attr('text-anchor', 'middle')
                    .style('font-size', '16px')
                    .text('No feedback data available yet');
                return;
            }}
            
            // Create pie layout
            const pie = d3.pie()
                .sort(null)
                .value(d => d.value);
            
            const data_ready = pie(feedbackData);
            
            // Create arcs
            const arc = d3.arc()
                .innerRadius(0)  // Full pie, not donut
                .outerRadius(radius);
            
            // Add the arcs
            svg.selectAll('allSlices')
                .data(data_ready)
                .enter()
                .append('path')
                .attr('d', arc)
                .attr('fill', d => FEEDBACK_COLORS[d.data.name] || '#999')
                .attr('stroke', 'white')
                .style('stroke-width', '2px')
                .style('opacity', 0.8);
            
            // Add labels
            svg.selectAll('allLabels')
                .data(data_ready)
                .enter()
                .append('text')
                .text(d => `${{d.data.name}}: ${{(d.data.value / total * 100).toFixed(1)}}%`)
                .attr('transform', d => {{
                    const pos = arc.centroid(d);
                    return `translate(${{pos[0]}}, ${{pos[1]}})`;
                }})
                .style('text-anchor', 'middle')
                .style('font-size', '14px')
                .style('font-weight', 'bold')
                .style('fill', 'white');
        }}
        
        // Initialize all charts once DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {{
            renderInsectDiversityChart();
            renderTimeSeriesChart();
            renderConfidenceChart();
            renderFeedbackChart();
        }});
        """)
        
        return Title("Insect Classification Dashboard"), Main(
            visualization_script,
            Div(
                H1("Insect Classification Dashboard", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Statistics and visualizations of classification results", cls="text-center mb-8 text-base-content/70"),
                dashboard_stats,
                cls="py-8"
            ),
            Script(src="https://d3js.org/d3.v7.min.js"),  # Add D3.js for visualizations
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    
    #################################################
    # API route for image classification
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
            
            # Call the classification function
            result = await classify_image_claude.remote(image_data, options)
            
            # Add context image URL if available
            if result.get("top_sources") and len(result["top_sources"]) > 0:
                context_image_path = get_context_image_path(result["top_sources"])
                if context_image_path:
                    result["context_image_url"] = f"/context-image?path={context_image_path}"
            
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error classifying image: {e}")
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
                
            # Send batch to Claude
            result = await classify_batch_claude.remote(base64_images, options)
            
            # Add context image URL if available
            if result.get("top_sources") and len(result["top_sources"]) > 0:
                context_image_path = get_context_image_path(result["top_sources"])
                if context_image_path:
                    result["context_image_url"] = f"/context-image?path={context_image_path}"
            
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

if __name__ == "__main__":
    print("Starting Insect Classification App...")
# Main FastHTML Server with defined routes
