from typing import Dict, Literal
# from ..mypackage.state import AppState # Changed to absolute import
from mypackage.state import AppState

#Classification logic

def classify_input_logic(raw_input: str) -> str:
    """Classify the input into a specific category:
    'pdf' for PDF files, 'image' for image files (jpg, png, jpeg),
    'text' otherwise."""
    if not isinstance(raw_input, str): # Basic type check
         return "error" # Or some other appropriate handling
    raw = raw_input.lower()
    if raw.endswith(".pdf"):
        return "pdf" # Classify PDFs separately
    elif raw.endswith((".jpg", ".jpeg", ".png")):
        return "image"
    return "text"
# Node to perform the classification
def classification_node(state: AppState) -> Dict:
    """
    LangGraph node that classifies the 'raw_input' and updates the state.
    """
    print("---CLASSIFYING INPUT---")
    raw_input = state.get("raw_input")
    if raw_input is None:
         print("Error: raw_input not found in state.")
         # Decide how to handle this - maybe set type to 'error' or raise exception
         return {"input_type": "error"} 

    input_type = classify_input_logic(raw_input)
    print(f"Input classified as: {input_type}")
    return {"input_type": input_type}
