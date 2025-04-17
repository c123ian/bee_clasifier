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
CLAUDE_API_KEY = "sk-ant-api03-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
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

# Function to save results to file
def save_results_file(result_id, image_data, result_content):
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
    conn.commit()
    return conn

# Generate classification using Claude's API
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
            save_results_file(result_id, image_data, parsed_result)
            
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
            # Add custom theme styles from https://daisyui.com/theme-generator/
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
                --radius-selector: 0rem;
                --radius-field: 0.5rem;
                --radius-box: 2rem;
                --size-selector: 0.25rem;
                --size-field: 0.25rem;
                --border: 1px;
                }

                /* Custom styling for better contrast */
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
            """),
        )
    )
    
    # Ensure database exists
    setup_database(DB_PATH)
    
    #################################################
    # Homepage Route - Bee Classifier Dashboard
    #################################################
    @rt("/")
    def homepage():
        """Render the bee classifier dashboard"""
        
        # Image upload section
        image_upload = Div(
            Label("Upload Insect Image", cls="block text-xl font-medium mb-2 text-bee-green"),
            P("Upload an image of an insect to classify it using Claude's vision capabilities.", cls="mb-4"),
            Div(
                Label(
                    Div(
                        Span("Click or drag image here", cls="text-lg text-center"),
                        P("Supported formats: JPEG, PNG", cls="text-sm text-center mt-2"),
                        cls="flex flex-col items-center justify-center h-full"
                    ),
                    Input(
                        type="file",
                        name="insect_image",
                        accept="image/jpeg,image/png",
                        cls="hidden",
                        id="image-input"
                    ),
                    cls="w-full h-40 border-2 border-dashed rounded-lg flex items-center justify-center cursor-pointer hover:bg-base-200 transition-colors"
                ),
                cls="mb-6"
            ),
            Div(
                Img(
                    id="image-preview",
                    src="",
                    cls="max-h-64 mx-auto hidden object-contain rounded-lg border shadow-sm"
                ),
                cls="mb-6"
            ),
            cls="mb-8"
        )
        
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
        
        # Classification options
        classification_options = Div(
            H3("Classification Options", cls="text-lg font-semibold mb-4 text-bee-green"),
            create_toggle("detailed_description", "Detailed Description (shapes, colors)"),
            create_toggle("plant_classification", "Plant Classification"),
            create_toggle("taxonomy", "Taxonomic Classification"),
            cls="mb-6 p-4 bg-base-200 rounded-lg"
        )
        
        # Controls panel
        controls_panel = Div(
            H2("Insect Image Classification", cls="text-xl font-bold mb-4 text-bee-green"),
            image_upload,
            classification_options,
            Button(
                "Classify Insect",
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
                P("Upload an image and click 'Classify Insect' to see results.", 
                  cls="text-center text-base-content/70 italic"),
                id="results-placeholder",
                cls="text-center py-12"
            ),
            Div(
                id="results-content",
                cls="hidden"
            ),
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
        
        # Add script for form handling
        form_script = Script("""
        document.addEventListener('DOMContentLoaded', function() {
            const imageInput = document.getElementById('image-input');
            const imagePreview = document.getElementById('image-preview');
            const classifyButton = document.getElementById('classify-button');
            const loadingIndicator = document.getElementById('loading-indicator');
            const resultPlaceholder = document.getElementById('results-placeholder');
            const resultsContent = document.getElementById('results-content');
            const resultActions = document.getElementById('result-actions');
            const copyButton = document.getElementById('copy-button');
            const newButton = document.getElementById('new-button');
            
            // Handle image upload
            imageInput.addEventListener('change', function(event) {
                const file = event.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    
                    reader.onload = function(e) {
                        // Show image preview
                        imagePreview.src = e.target.result;
                        imagePreview.classList.remove('hidden');
                        
                        // Enable classify button
                        classifyButton.disabled = false;
                    };
                    
                    reader.readAsDataURL(file);
                }
            });
            
            // Handle classify button click
            classifyButton.addEventListener('click', function() {
                // Get the base64 image data (remove metadata)
                const base64Data = imagePreview.src.split(',')[1];
                
                // Get toggle states
                const options = {
                    detailed_description: document.querySelector('input[name="detailed_description"]').checked,
                    plant_classification: document.querySelector('input[name="plant_classification"]').checked,
                    taxonomy: document.querySelector('input[name="taxonomy"]').checked
                };
                
                // Show loading indicator
                loadingIndicator.parentElement.classList.remove('hidden');
                resultPlaceholder.classList.add('hidden');
                resultsContent.classList.add('hidden');
                resultActions.classList.add('hidden');
                classifyButton.disabled = true;
                
                // Send request to API
                fetch('/classify', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        image_data: base64Data,
                        options: options
                    })
                })
                .then(response => response.json())
                .then(data => {
                    // Hide loading indicator
                    loadingIndicator.parentElement.classList.add('hidden');
                    
                    if (data.error) {
                        // Show error message
                        resultsContent.innerHTML = `
                            <div class="alert alert-error">
                                <span>Error: ${data.error}</span>
                            </div>
                        `;
                        resultsContent.classList.remove('hidden');
                        return;
                    }
                    
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
                    resultsContent.innerHTML = resultHTML;
                    resultsContent.classList.remove('hidden');
                    resultActions.classList.remove('hidden');
                    
                    // Setup copy button
                    copyButton.onclick = function() {
                        navigator.clipboard.writeText(data.raw_response);
                        copyButton.innerText = "Copied!";
                        setTimeout(() => copyButton.innerText = "Copy Results", 2000);
                    };
                    
                    // Setup new button
                    newButton.onclick = function() {
                        // Reset the form
                        imageInput.value = '';
                        imagePreview.src = '';
                        imagePreview.classList.add('hidden');
                        classifyButton.disabled = true;
                        resultPlaceholder.classList.remove('hidden');
                        resultsContent.classList.add('hidden');
                        resultActions.classList.add('hidden');
                    };
                })
                .catch(error => {
                    console.error('Error classifying image:', error);
                    loadingIndicator.parentElement.classList.add('hidden');
                    resultsContent.innerHTML = `
                        <div class="alert alert-error">
                            <span>Error: Could not process your request. Please try again.</span>
                        </div>
                    `;
                    resultsContent.classList.remove('hidden');
                    classifyButton.disabled = false;
                });
            });
            
            // Make the label act as a dropzone
            const dropzone = document.querySelector('label[for="image-input"]');
            
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropzone.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropzone.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropzone.classList.add('bg-base-200');
            }
            
            function unhighlight() {
                dropzone.classList.remove('bg-base-200');
            }
            
            dropzone.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    imageInput.files = files;
                    const event = new Event('change');
                    imageInput.dispatchEvent(event);
                }
            }
        });
        """)
        
        return Title("Insect Classifier"), Main(
            form_script,
            Div(
                H1("Insect Classification App", cls="text-3xl font-bold text-center mb-2 text-bee-green"),
                P("Powered by Claude's Vision AI", cls="text-center mb-8 text-base-content/70"),
                Div(
                    controls_panel,
                    results_panel,
                    cls="flex flex-col md:flex-row gap-6 w-full"
                ),
                cls="container mx-auto px-4 py-8 max-w-6xl"
            ),
            cls="min-h-screen bg-base-100",
            data_theme="light"
        )
    
    #################################################
    # Classify API Endpoint
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
    
    # Return the FastHTML app
    return fasthtml_app

if __name__ == "__main__":
    print("Starting Insect Classification App...")
