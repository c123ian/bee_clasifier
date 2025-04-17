import modal
import os
import sqlite3
import uuid
import time
import json
import base64
import requests
from typing import Optional, Dict, Any, List

from fasthtml.common import *
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

# Define app
app = modal.App("bee_classifier")

# Constants and directories
DATA_DIR = "/data"
RESULTS_FOLDER = "/data/classification_results"
DB_PATH = "/data/bee_classifier.db"
STATUS_DIR = "/data/status"

# Claude API constants
CLAUDE_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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
    .apt_install("git")
    .pip_install(
        "requests",
        "python-fasthtml==0.12.0"
    )
)

# Look up data volume for storing results
try:
    bee_volume = modal.Volume.lookup("bee_volume", create_if_missing=True)
except modal.exception.NotFoundError:
    bee_volume = modal.Volume.persisted("bee_volume")

# Base prompt template for Claude
CLASSIFICATION_PROMPT = """
You are an expert entomologist specializing in insect identification. Your task is to analyze the 
provided image and classify the insect(s) visible. 

Please categorize the insect into one of these categories:
{categories}

{additional_instructions}

Format your response as follows:
- Main Category: [the most likely category from the list]
- Confidence: [High, Medium, or Low]
- Description: [brief description of what you see]
{format_instructions}

IMPORTANT: Just provide the formatted response above with no additional explanation or apology.
"""

# Batch prompt template for Claude
BATCH_PROMPT = """
You are an expert entomologist specializing in insect identification. Your task is to analyze {count} 
images of insects and classify each one.

For EACH image, categorize the insect into one of these categories:
{categories}

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

# Setup database for classification results
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
    
    # Prepare the prompt
    categories_list = "\n".join([f"- {category}" for category in INSECT_CATEGORIES])
    additional_instructions_text = "\n".join(additional_instructions) if additional_instructions else ""
    format_instructions_text = "\n".join(format_instructions) if format_instructions else ""
    
    prompt = CLASSIFICATION_PROMPT.format(
        categories=categories_list,
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
            
            cursor.execute(
                "INSERT INTO results (id, category, confidence, description, additional_details) VALUES (?, ?, ?, ?, ?)",
                (result_id, category, confidence, description, json.dumps(parsed_result))
            )
            
            conn.commit()
            conn.close()
            
            # Save results to file
            save_results_file(result_id, parsed_result)
            
        except Exception as e:
            print(f"⚠️ Error saving to database: {e}")
            raise e
        
        return {
            "id": result_id,
            "category": category,
            "confidence": confidence,
            "description": description,
            "details": parsed_result,
            "raw_response": classification_text
        }
        
    except Exception as e:
        print(f"⚠️ Error classifying image: {e}")
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
    
    # Prepare the batch prompt
    categories_list = "\n".join([f"- {category}" for category in INSECT_CATEGORIES])
    additional_instructions_text = "\n".join(additional_instructions) if additional_instructions else ""
    format_instructions_text = "\n".join(format_instructions) if format_instructions else ""
    
    prompt = BATCH_PROMPT.format(
        count=len(images_data),
        categories=categories_list,
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
    """Main FastHTML Server for Bee Classifier Dashboard"""
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
        def create_toggle(name, label, checked=False):
            return Div(
                Label(
                    Input(
                        type="checkbox",
                        name=name,
                        checked="checked" if checked else None,
                        cls="toggle toggle-primary mr-3"
                    ),
                    Span(label),
                    cls="label cursor-pointer justify-start"
                ),
                cls="mb-3"
            )
        
        # Classification options panel
        classification_options = Div(
            H3("Classification Options", cls="text-lg font-semibold mb-4 text-bee-green"),
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
                Label(
                    Div(
                        Span("Click or drag images here", cls="text-lg text-center"),
                        P("Select single or multiple files (JPEG, PNG)", cls="text-sm text-center mt-2"),
                        cls="flex flex-col items-center justify-center h-full"
                    ),
                    Input(
                        type="file",
                        name="insect_images",
                        accept="image/jpeg,image/png",
                        multiple=True,
                        cls="hidden",
                        id="image-input"
                    ),
                    cls="w-full h-40 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer hover:bg-base-200 transition-colors"
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
                    detailed_description: document.querySelector('input[name="detailed_description"]').checked,
                    plant_classification: document.querySelector('input[name="plant_classification"]').checked,
                    taxonomy: document.querySelector('input[name="taxonomy"]').checked
                };
            }
            
            // Handle image upload
            imageInput.addEventListener('change', function(event) {
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
            });
            
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
                    
                    // Save raw response for copy button
                    rawResponseText = data.raw_response;
                    
                    // Determine confidence class
                    let confidenceClass = 'confidence-medium';
                    if (data.confidence === 'High') {
                        confidenceClass = 'confidence-high';
                    } else if (data.confidence === 'Low') {
                        confidenceClass = 'confidence-low';
                    }
                    
                    // Create result HTML
                    let resultHTML = `
                        <div class="p-4 bg-base-200 rounded-lg mb-4">
                            <div class="flex justify-between items-center mb-2">
                                <h3 class="text-lg font-bold">${data.category}</h3>
                                <span class="badge ${confidenceClass}">Confidence: ${data.confidence}</span>
                            </div>
                            <p class="mb-4">${data.description}</p>
                    `;
                    
                    // Add additional details if available
                    const details = data.details;
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
                    
                    // Add raw response in collapsible section
                    resultHTML += `
                        <details class="collapse bg-base-200">
                            <summary class="collapse-title font-medium">Raw Response</summary>
                            <div class="collapse-content">
                                <pre class="text-xs whitespace-pre-wrap">${data.raw_response}</pre>
                            </div>
                        </details>
                    `;
                    
                    // Update results content
                    singleResult.innerHTML = resultHTML;
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
                    
                    // Create carousel for batch results
                    createBatchResultsCarousel(data);
                    
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
            
            // Create carousel for batch results
            function createBatchResultsCarousel(data) {
                const results = data.results || [];
                
                if (results.length === 0) {
                    batchResults.innerHTML = `
                        <div class="alert alert-warning">
                            <span>No classification results were returned.</span>
                        </div>
                    `;
                    return;
                }
                
                // Start building carousel HTML
                let carouselHTML = `
                    <h3 class="text-lg font-semibold mb-4 text-center">Batch Results (${results.length} images)</h3>
                    <div class="carousel">
                        <div class="carousel-inner" id="carousel-items">
                `;
                
                // Add each result as a carousel item
                results.forEach((result, index) => {
                    // Determine confidence class
                    let confidenceClass = 'confidence-medium';
                    if (result.confidence === 'High') {
                        confidenceClass = 'confidence-high';
                    } else if (result.confidence === 'Low') {
                        confidenceClass = 'confidence-low';
                    }
                    
                    carouselHTML += `
                        <div class="carousel-item" id="slide-${index}">
                            <div class="p-4 bg-base-200 rounded-lg max-w-3xl mx-auto">
                                <div class="flex justify-between items-center mb-2">
                                    <h3 class="text-lg font-medium">Result ${index + 1} of ${results.length}</h3>
                                    <span class="badge ${confidenceClass}">Confidence: ${result.confidence}</span>
                                </div>
                                <h4 class="text-lg font-bold mb-2">${result.category}</h4>
                                <p class="mb-4">${result.description}</p>
                    `;
                    
                    // Add additional details
                    const details = result.details;
                    for (const key in details) {
                        if (key !== 'Main Category' && key !== 'Confidence' && key !== 'Description') {
                            carouselHTML += `
                                <div class="mb-2">
                                    <span class="font-semibold">${key}:</span>
                                    <span>${details[key]}</span>
                                </div>
                            `;
                        }
                    }
                    
                    carouselHTML += `
                            </div>
                        </div>
                    `;
                });
                
                // Close carousel container
                carouselHTML += `
                        </div>
                `;
                
                // Add navigation if more than one result
                if (results.length > 1) {
                    carouselHTML += `
                        <button class="btn btn-circle carousel-control carousel-control-prev" id="prev-btn">❮</button>
                        <button class="btn btn-circle carousel-control carousel-control-next" id="next-btn">❯</button>
                        <div class="carousel-indicators" id="carousel-indicators">
                    `;
                    
                    // Add indicators for navigation
                    for (let i = 0; i < results.length; i++) {
                        carouselHTML += `
                            <div class="carousel-indicator ${i === 0 ? 'active' : ''}" data-index="${i}"></div>
                        `;
                    }
                    
                    carouselHTML += `</div>`;
                }
                
                // Add raw response in collapsible section
                carouselHTML += `
                    <details class="collapse bg-base-200 mt-4">
                        <summary class="collapse-title font-medium">Raw Response</summary>
                        <div class="collapse-content">
                            <pre class="text-xs whitespace-pre-wrap">${data.raw_response}</pre>
                        </div>
                    </details>
                `;
                
                // Set HTML
                batchResults.innerHTML = carouselHTML;
                
                // Setup carousel navigation if needed
                if (results.length > 1) {
                    const carouselItems = document.getElementById('carousel-items');
                    const prevBtn = document.getElementById('prev-btn');
                    const nextBtn = document.getElementById('next-btn');
                    const indicators = document.querySelectorAll('.carousel-indicator');
                    
                    let currentIndex = 0;
                    
                    // Show a specific slide
                    function showSlide(index) {
                        // Handle wrapping
                        if (index < 0) index = results.length - 1;
                        if (index >= results.length) index = 0;
                        
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
                }
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
            
            // Set up drag and drop
            const dropzone = document.querySelector('label[for="image-input"]');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                });
            });
            
            // Highlight on drag
            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() {
                    dropzone.classList.add('bg-base-200');
                });
            });
            
            // Remove highlight on drag leave/drop
            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, function() {
                    dropzone.classList.remove('bg-base-200');
                });
            });
            
            // Handle file drop
            dropzone.addEventListener('drop', function(e) {
                const files = e.dataTransfer.files;
                
                if (files.length > 0) {
                    // Check max count
                    if (files.length > MAX_IMAGES) {
                        alert(`Please select a maximum of ${MAX_IMAGES} images.`);
                        return;
                    }
                    
                    // Update file input
                    const dataTransfer = new DataTransfer();
                    for (let i = 0; i < files.length; i++) {
                        dataTransfer.items.add(files[i]);
                    }
                    imageInput.files = dataTransfer.files;
                    
                    // Trigger change event
                    const event = new Event('change');
                    imageInput.dispatchEvent(event);
                }
            });
        });
        """)
        
        return Title("Insect Classifier"), Main(
            form_script,
            Div(
                H1("Insect Classification App", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Powered by Claude's Vision AI", cls="text-center mb-8 text-base-content/70"),
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
    # Single Image Classify API Endpoint
    #################################################
    @rt("/classify", methods=["POST"])
    async def api_classify_image(request):
        """API endpoint to classify insect image using Claude"""
        try:
            # Get image data and options from request JSON
            data = await request.json()
            image_data = data.get("image_data", "")
            options = data.get("options", {})
            
            if not image_data:
                return JSONResponse({"error": "No image data provided"}, status_code=400)
            
            # Call the classification function
            result = classify_image_claude.remote(image_data, options)
            
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error classifying image: {e}")
            return JSONResponse({"error": str(e)}, status_code=500)
    
    #################################################
    # Batch Classify API Endpoint
    #################################################
    @rt("/classify-batch", methods=["POST"])
    async def api_classify_batch(request):
        """API endpoint to classify multiple insect images in batch mode"""
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
            result = classify_batch_claude.remote(base64_images, options)
            
            return JSONResponse(result)
                
        except Exception as e:
            print(f"Error in batch classification: {e}")
            import traceback
            traceback.print_exc()
            return JSONResponse({"error": str(e)}, status_code=500)
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Insect Classification App...")
