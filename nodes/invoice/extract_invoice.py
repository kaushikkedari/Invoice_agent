#  extract_invoice_text.py

import os
import json
import re
import mimetypes
import pandas as pd
from typing import Dict, Any, Optional, List, Union, TypedDict, Literal
from pathlib import Path
import logging
from dotenv import load_dotenv
import base64 # Added for image encoding

# LangChain Imports
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
# from langchain_google_genai import ChatGoogleGenerativeAI # REMOVED
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage # Added for vision prompt

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

# Load environment variables (e.g., for API key)
load_dotenv()

# Image Processing Imports
try:
    from PIL import Image
    # import google.generativeai as genai # REMOVED
    # from google.api_core.exceptions import InvalidArgument # REMOVED
    IMAGE_PROCESSING_AVAILABLE = True
except ImportError:
    IMAGE_PROCESSING_AVAILABLE = False
    print("WARNING: Optional dependencies for image processing not found. Install with: pip install pillow")
    
# Import the AppState definition
# Assuming your AppState is defined in mypackage.state
# Adjust the import path if your structure is different
try:
    from mypackage.state import AppState
    APP_STATE_AVAILABLE = True
except ImportError:
    APP_STATE_AVAILABLE = False
    print("Warning: mypackage.state.AppState not available. Node function will use local type definitions.")
    # Define a minimal mock of the required state structure if not available
    class AppState(TypedDict):
        input_type: Literal["image", "pdf"]
        raw_input: Union[str, bytes]
        detected_file_type: Optional[Literal["pdf", "image"]]
        extracted_invoice_data: Optional[Dict[str, Any]]
        error: Optional[str]

# --- Configuration & Setup ---
# Using print statements for logging as per the example style
print("Invoice Text Extraction Module Loading...")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") # Use GOOGLE_API_KEY common for Google services
if not GOOGLE_API_KEY:
    print("CRITICAL WARNING: GOOGLE_API_KEY environment variable not set.")
    # Consider raising an error or exiting if the key is essential
    # raise ValueError("GOOGLE_API_KEY must be set in the environment variables.")

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    # No immediate error raise, functions below will check ai_component.llm
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    ai_component = None # Set to None if init fails

# --- Helper Functions ---
# These functions contain the core logic for file loading and LLM interaction

def _load_file_content(file_path: str) -> Dict[str, Any]:
    """
    Loads the content from the invoice file (PDF or image) and performs
    extraction to get text/visual content.

    Args:
        file_path: Path to the invoice file.

    Returns:
        Dict containing file_type, raw_text (for PDFs), and any error messages.
    """
    result = {
        "file_type": None,
        "raw_text": None,
        "error": None
    }

    # Check if file exists
    if not os.path.exists(file_path):
        result["error"] = f"File not found: {file_path}"
        return result

    # Detect file type using mimetypes
    mime_type, _ = mimetypes.guess_type(file_path)
    print(f"Detected MIME type for {Path(file_path).name}: {mime_type}")

    # Process based on file type
    if mime_type == 'application/pdf':
        result["file_type"] = "pdf"
        try:
            print(f"Processing PDF: {file_path}")
            loader = PyMuPDF4LLMLoader(file_path)
            documents: List[Document] = loader.load()
            if not documents:
                result["error"] = "PyMuPDF4LLMLoader extracted no content from PDF."
                return result
                
            # Concatenate content from all pages/documents
            result["raw_text"] = "\n\n".join([doc.page_content for doc in documents])
            print(f"Successfully extracted {len(result['raw_text'])} characters from PDF.")
        except Exception as e:
            result["error"] = f"Error processing PDF {file_path}: {str(e)}"
            return result
            
    elif mime_type and mime_type.startswith('image/'):
        result["file_type"] = "image"
        # We'll process the image directly with the vision model later
        # Just verify we can open it
        try:
            with Image.open(file_path) as img:
                # Just checking if we can open the image
                width, height = img.size
                print(f"Successfully verified image: {file_path} ({width}x{height})")
                # We don't set raw_text here as we'll use Vision model directly
        except Exception as e:
            result["error"] = f"Error verifying image file {file_path}: {str(e)}"
            return result
    else:
        result["error"] = f"Unsupported file type: {mime_type}. Please provide a PDF or image file."
        
    return result


def _extract_invoice_data_with_llm(file_path: str, file_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses the AIComponent (configured LLM) to extract structured data.
    Handles both PDF (text) and Image (vision) input.
    """
    result = {"extracted_data": None, "error": None}
    
    # Check if AIComponent and its LLM are available
    if not ai_component or not ai_component.llm:
        result["error"] = "LLM (via AIComponent) not initialized. Check config and .env."
        print(result["error"])
        return result

    file_type = file_info.get("file_type")
    
    # Define the JSON schema we want to extract
    # This maps to your database schema fields for better downstream integration
    json_schema = {
        "invoice_number": "string or NULL", 
        "invoice_date": "YYYY-MM-DD or string or NULL", 
        "Invoice_ate": "YYYY-MM-DD or string or NULL", 
        "vendor_name": "string or NULL",
        "vendor_city": "string or NULL", 
        "vendor_country": "string or NULL", 
        "purchase_order_number": "string or NULL", 
        "payment_terms": "string or NULL", 
        "subtotal_amount": "number or NULL", 
        "tax_amount": "number or NULL", 
        "shipping_cost": "number or NULL", 
        "total_amount": "number or NULL", 
        "line_items": [
            {
                "description": "string or NULL", 
                "hsn": "string or NULL",
                "quantity": "number or NULL", 
                "unit_of_measure": "string or NULL", 
                "unit_price": "number or NULL", 
                "line_total": "number or NULL" 
            }
        ]
    }

    # Convert schema to string for prompt insertion
    schema_string = json.dumps(json_schema, indent=2)

    try:
        # For PDF files, use text-based LLM with the extracted text
        if file_type == "pdf":
            raw_text = file_info.get("raw_text")
            if not raw_text or not raw_text.strip():
                result["error"] = "No text content extracted from PDF."
                return result
                
            print("Using AIComponent LLM to extract invoice data from PDF text...")
            
            # Create the prompt (Keep template as is)
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AI assistant specializing in extracting key information from vendor invoice documents.
                Extract invoice data from the provided text and format it as JSON according to the provided schema.
                Follow these guidelines:
                - Extract ONLY what's explicitly present in the text. DO NOT make up or guess any values.
                - If a field is not found, set it to null.
                - For amounts, extract as numbers (remove currency symbols, thousand separators, etc.)
                - For dates, convert to YYYY-MM-DD format when possible.
                - Extract all line items found in the invoice.
                - Ensure the JSON is valid and properly formatted.
                - For dates, convert to YYYY-MM-DD format when possible.
                - For amounts, extract as numbers (remove currency symbols, thousand separators, etc.)
                 
                
                """),
                ("human", """
                Extract information from this invoice text and return it as JSON:
                
                Invoice Text:
                ```
                {text}
                ```
                
                JSON Schema:
                {schema}
                
                Return ONLY the JSON object containing the extracted information.
                """)
            ])
            
            # Set up the JSON output parser
            json_parser = JsonOutputParser()
            
            # Create and run the extraction chain using AIComponent
            extraction_chain = prompt_template | ai_component.llm | json_parser
            
            try:
                extracted_json = extraction_chain.invoke({
                    "text": raw_text,
                    "schema": schema_string
                })
                result["extracted_data"] = extracted_json
                print("Successfully extracted structured data from PDF invoice.")
                
            except Exception as e:
                result["error"] = f"LLM processing error (PDF Text): {str(e)}"
                print(result["error"])
                return result
                
        # For image files, use AIComponent LLM with multimodal input
        elif file_type == "image":
            if not IMAGE_PROCESSING_AVAILABLE:
                result["error"] = "Image processing libraries (Pillow) not available. Cannot process image."
                return result
                
            print("Using AIComponent LLM (Azure 4o-mini via deployment) to extract invoice data from image...")
            
            try:
                # --- Prepare Multimodal Input for AzureChatOpenAI --- 
                # 1. Encode Image to Base64
                mime_type = file_info.get("mime_type", mimetypes.guess_type(file_path)[0])
                with open(file_path, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                image_url = f"data:{mime_type};base64,{base64_image}"
                
                # Prompt for the vision model
                prompt = f"""
                You are an expert at analyzing invoices images.
                Extract information from this invoice image and provide it as a JSON object.
                The invoice image is a screenshot of 
                
                Extract the following details from the invoice:
                - invoice_number: The unique identifier for this invoice
                - invoice_date: The date when the invoice was issued
                - due_date: When payment is due (if present)
                - vendor_name: The company sending the invoice
                - vendor_tax_id: Tax identification number of the vendor
                - vendor_city: City where the vendor is located
                - vendor_country: Country where the vendor is located
                - purchase_order_number: Any referenced PO number
                - order_date: Date when the order was placed
                - payment_terms: Payment terms mentioned
                - subtotal_amount: Subtotal before tax
                - tax_amount: Tax amount
                - shipping_cost: Cost for shipping/delivery
                - total_amount: The total amount due
                - currency: The currency used (USD, EUR, etc.)
                - line_items: Array of items with description, hsn (if available), quantity, unit_of_measure, unit_price, and line_total
                
                Return ONLY valid JSON with these fields. Use null for any missing fields.
                The JSON schema should match:
                {schema_string}
                """
                
                # Construct the message list for invoke
                messages = [
                    HumanMessage(
                        content=[
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url, "detail": "auto"} # detail: auto lets model choose
                            }
                        ]
                    )
                ]
                # --- End Multimodal Input Prep ---

                # 3. Invoke the AIComponent LLM
                # Note: We expect the LLM response content to be the JSON string.
                # We might need a parser here if the model wraps it.
                response = ai_component.invoke(messages)
                
                # Extract the text response content
                if hasattr(response, 'content'):
                    llm_output_str = response.content
                else:
                    llm_output_str = str(response)
                
                # 4. Parse the JSON response
                try:
                    # Look for JSON content between triple backticks if present
                    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', llm_output_str)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        # If no code block found, assume the whole output is JSON
                        json_str = llm_output_str.strip()
                    
                    # Parse the JSON
                    extracted_json = json.loads(json_str)
                    result["extracted_data"] = extracted_json
                    print("Successfully extracted structured data from image invoice via AIComponent.")
                    
                except json.JSONDecodeError as e:
                    result["error"] = f"Failed to parse Vision model response as JSON via AIComponent: {e}\nRaw Response: {llm_output_str[:500]}..."
                    print(result["error"])
                    return result
                    
            except Exception as e:
                result["error"] = f"Vision model processing error via AIComponent: {str(e)}"
                print(result["error"])
                return result
        else:
            result["error"] = f"Unsupported file type passed to LLM extraction: {file_type}"
            
    except Exception as e:
        result["error"] = f"Unexpected error during LLM extraction: {str(e)}"
        print(result["error"])
        
    return result


# --- Main Extraction Function (Standalone Use) ---

def extract_invoice_data(file_path: str) -> Dict[str, Any]:
    """
    Main function to extract structured data from an invoice file.
    This is the primary function that should be called from other modules
    or for standalone use.
    
    Args:
        file_path: Path to the invoice file (PDF or image)
        
    Returns:
        Dictionary containing:
        - extracted_data: The structured invoice data as a dictionary (or None on failure)
        - error: Any error message (or None on success)
        - file_type: The detected file type ('pdf' or 'image')
        - raw_text: The extracted text content (for PDFs only, useful for debugging)
    """
    print(f"Starting invoice data extraction for: {file_path}")
    
    # Initialize the result structure
    result = {
        "extracted_data": None,
        "error": None,
        "file_type": None,
        "raw_text": None
    }
    
    # Validate input
    if not file_path or not isinstance(file_path, str):
        result["error"] = "Invalid file path provided."
        return result
        
    try:
        # Step 1: Load and process the file content
        file_info = _load_file_content(file_path)
        
        # Add file info to results
        result["file_type"] = file_info.get("file_type")
        result["raw_text"] = file_info.get("raw_text")
        
        # Check for errors in file loading
        if file_info.get("error"):
            result["error"] = file_info["error"]
            print(f"Error loading file: {result['error']}")
            return result
            
        # Step 2: Extract structured data using appropriate LLM
        extraction_result = _extract_invoice_data_with_llm(file_path, file_info)
        
        # Add extraction results
        result["extracted_data"] = extraction_result.get("extracted_data")
        
        # Check for errors in extraction
        if extraction_result.get("error"):
            result["error"] = extraction_result["error"]
            print(f"Error extracting data: {result['error']}")
        else:
            print("Invoice data extraction completed successfully.")
            
    except Exception as e:
        result["error"] = f"Unexpected error during extraction: {str(e)}"
        print(f"Exception during extraction: {result['error']}")
        
    return result


# --- LangGraph Node Function ---

def extract_invoice_node(state: AppState) -> Dict[str, Any]:
    """
    LangGraph node function that extracts structured data from an invoice file.
    This is designed to be used as part of a larger LangGraph workflow.
    
    Args:
        state: The current state of the workflow (AppState)
        
    Returns:
        Dictionary with updates to be applied to the workflow state
    """
    print("\n--- Executing Invoice Extraction Node ---")
    
    # Initialize the updates dictionary
    updates = {
        "detected_file_type": None,
        "extracted_invoice_data": None,
        "error": None
    }
    
    # Extract file path from state
    file_path = None
    input_type = state.get("input_type") # Keep track of the classified type
    raw_input = state.get("raw_input")
    
    # Get the file path directly from raw_input if it's a string (path)
    if isinstance(raw_input, str):
        file_path = raw_input
    elif isinstance(raw_input, bytes):
        # If it's bytes, we can't process it as a file path here
        updates["error"] = "Binary data provided instead of file path."
        return updates
    else:
        updates["error"] = f"Invalid raw_input type for invoice node: {type(raw_input)}"
        return updates

    # Store the verified path
    updates["input_file_path"] = file_path
    
    # Validate the file path
    if not file_path or not isinstance(file_path, str):
        updates["error"] = "No valid file path provided in state."
        return updates
    
    if not os.path.exists(file_path):
        updates["error"] = f"File not found at {file_path}."
        return updates
    
    try:
        # Call the standalone function to do the actual work
        result = extract_invoice_data(file_path)
        
        # Map result fields to state updates
        updates["detected_file_type"] = result.get("file_type")
        updates["extracted_invoice_data"] = result.get("extracted_data")
        
        # For PDF files, also store the raw text
        if result.get("file_type") == "pdf" and result.get("raw_text"):
            updates["invoice_text_extraction"] = result.get("raw_text")
        
        # Check for errors
        if result.get("error"):
            updates["error"] = result.get("error")
            print(f"Error in extraction: {updates['error']}")
        else:
            print("Invoice data extracted successfully.")
            # Print a sample of the extracted data
            if updates["extracted_invoice_data"]:
                invoice_data = updates["extracted_invoice_data"]
                print(f"Invoice #{invoice_data.get('invoice_number', 'N/A')} from {invoice_data.get('vendor_name', 'N/A')}")
                print(f"Total: {invoice_data.get('total_amount', 'N/A')} {invoice_data.get('currency', '')}")
                print(f"Line items: {len(invoice_data.get('line_items', []))}")
    except Exception as e:
        updates["error"] = f"Unexpected error in extract_invoice_node: {str(e)}"
        print(f"Exception: {updates['error']}")
    
    print("--- Completed Invoice Extraction Node ---")
    return updates


# --- Example Usage ---

if __name__ == "__main__":
    print("\n=== Invoice Data Extraction Example ===\n")
    
    # Check for Google API key
    if not os.getenv("GEMINI_API_KEY"):
        print("WARNING: GOOGLE_API_KEY environment variable is not set.")
        print("Please set this environment variable before running.")
        print("Example: export GOOGLE_API_KEY='your_api_key_here'")
        print("Or in Windows: set GOOGLE_API_KEY=your_api_key_here")
        exit(1)
    
    # Path to your invoice file (PDF or image)
    # Replace with the path to your actual invoice file
    invoice_file_path = r"Invoices\Invalid\17_invoice_clpo.pdf"  # Set this to your file path"
    
    # Prompt for file path if not set
    if not invoice_file_path:
        print("Please enter the path to your invoice file (PDF or image):")
        invoice_file_path = input("> ").strip()
    
    # Verify the file exists
    if not os.path.exists(invoice_file_path):
        print(f"Error: File not found at {invoice_file_path}")
        exit(1)
    
    print(f"Processing invoice file: {invoice_file_path}")
    
    # Run the extraction
    result = extract_invoice_data(invoice_file_path)
    
    print("\n=== Processing Results ===\n")
    
    # Check for errors
    if result.get("error"):
        print(f"ERROR: {result['error']}")
    else:
        print("✅ Processing completed successfully!")
        
    # Display file information
    print(f"\nFile: {Path(invoice_file_path).name}")
    print(f"Type: {result.get('file_type')}")
    
    # Display extracted data
    extracted_data = result.get("extracted_data")
    if extracted_data:
        print("\n--- Extracted Invoice Data ---\n")
        print(json.dumps(extracted_data, indent=2))
        
        # Summary of key information
        # print("\n--- Invoice Summary ---")
        # print(f"Vendor: {extracted_data.get('vendor_name', 'N/A')}")
        # print(f"Invoice #: {extracted_data.get('invoice_number', 'N/A')}")
        # print(f"Date: {extracted_data.get('invoice_date', 'N/A')}")
        # print(f"PO #: {extracted_data.get('purchase_order_number', 'N/A')}")
        # print(f"Total: {extracted_data.get('total_amount', 'N/A')} {extracted_data.get('currency', '')}")
        
        # Line items summary
        line_items = extracted_data.get("line_items", [])
        if line_items:
            print(f"\nLine Items: {len(line_items)}")
            for i, item in enumerate(line_items, 1):
                desc = item.get("description", "N/A")
                # Truncate long descriptions
                if desc and len(desc) > 30:
                    desc = desc[:27] + "..."
                qty = item.get("quantity", "N/A")
                total = item.get("line_total", "N/A")
                print(f"  {i}. {desc} - Qty: {qty}, Total: {total}")
    else:
        print("\nNo data was extracted.")
        
        # Show raw text for debugging if available
        if result.get("raw_text"):
            print("\n--- Raw Text Preview (first 500 chars) ---")
            print(result["raw_text"][:500] + "..." if len(result["raw_text"]) > 500 else result["raw_text"])
            
    # Now test the node function if AppState is available
    if APP_STATE_AVAILABLE:
        print("\n\n=== Testing Node Function ===\n")
        # Create a mock state
        mock_state = {
            "input_type": "text",
            "raw_input": invoice_file_path
        }
        
        # Run the node function
        node_updates = extract_invoice_node(mock_state)
        
        print("\n--- Node Function Results ---")
        if node_updates.get("error"):
            print(f"Error: {node_updates['error']}")
        else:
            print("Node execution successful!")
            print(f"Detected file type: {node_updates.get('detected_file_type')}")
            if node_updates.get("extracted_invoice_data"):
                print("Invoice data successfully extracted.")
                # Don't repeat the full output since we already showed it above
    else:
        print("\nSkipping node function test as AppState is not available.")
