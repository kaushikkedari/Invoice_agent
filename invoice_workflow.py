import os
from typing import Dict, Any, Annotated, Union, List
from dotenv import load_dotenv

# LangGraph import
from langgraph.graph import StateGraph

# Import nodes from each branch
from nodes.invoice.extract_invoice import extract_invoice_node
from nodes.invoice.query_po_data import load_po_data_node
from nodes.invoice.execute_invoice import execute_invoice_node
from nodes.invoice.validate import validate_invoice_node
from nodes.input_router import classification_node
from nodes.invoice.correct_po_query import correct_po_query_node

# Query branch nodes
from nodes.query.anlyze_query import analyze_query_node
from nodes.query.generate_query import generate_query_node
from nodes.query.execute_query import execute_query_node
from nodes.query.visualize_data import visualize_data_node

# Import app state
from mypackage.state import AppState

# Load environment variables
load_dotenv()

# --- Define Max Correction Attempts ---
MAX_CORRECTION_ATTEMPTS = 3

# --- Define Router Logic ---
def route_by_input_type(state: AppState) -> str:
    """
    Router function to determine which branch to execute based on input type.
    """
    input_type = state.get("input_type")
    if not input_type:
        print("WARNING: Input type not set, defaulting to 'text'")
        return "text"
    
    print(f"Routing based on input type: {input_type}")
    if input_type.lower() in ["image", "pdf"]:
        return "invoice"
    else:
        return "text"

# --- Define Error Checkers ---
def check_for_errors(state: AppState) -> str:
    """Check if state contains errors and return appropriate next node."""
    if state.get("error"):
        error_msg = state.get("error")
        print(f"Error detected: {error_msg}")
        return "error_handler"
    return "continue"

# --- NEW: Decision Function for Invoice Query Execution ---
def decide_after_invoice_execution(state: AppState) -> str:
    """
    Decides the next step after attempting to execute the PO query.
    Routes to correction if a DB error occurred and attempts remain.
    Routes to validation on success.
    Routes to error handler otherwise.
    """
    # Check specifically for the db_error_message set by execute_invoice
    db_error = state.get("db_error_message")
    if db_error:
        print(f"DB Error detected during invoice query execution: {db_error}")
        # Check correction attempts
        attempts = state.get("po_query_correction_attempts", 0)
        if attempts < MAX_CORRECTION_ATTEMPTS:
            print(f"Attempting correction (Attempt {attempts + 1}/{MAX_CORRECTION_ATTEMPTS})")
            return "correct_query" # Route to correction node
        else:
            print(f"Maximum correction attempts ({MAX_CORRECTION_ATTEMPTS}) reached. Routing to error handler.")
            # Set a final error message indicating correction failed
            state["error"] = f"Failed to execute PO query after {MAX_CORRECTION_ATTEMPTS} correction attempts. Last DB error: {db_error}"
            return "error_handler"
    elif state.get("error"): # Check for other general errors
        print(f"General error detected after invoice query execution: {state.get('error')}")
        return "error_handler"
    else:
        # Query executed successfully
        print("Invoice query executed successfully. Proceeding to validation.")
        return "validate_invoice"

# --- Setup LangGraph Workflows ---
def create_invoice_workflow():
    """
    Create the workflow for invoice processing branch.
    """
    # Create the graph with AppState
    graph = StateGraph(AppState)
    
    # Add nodes to the graph
    graph.add_node("extract_invoice", extract_invoice_node)
    graph.add_node("query_po_data", load_po_data_node)
    graph.add_node("execute_invoice", execute_invoice_node)
    graph.add_node("validate_invoice", validate_invoice_node)
    graph.add_node("correct_po_query", correct_po_query_node)
    
    # --- NEW: Add simple error handler node ---
    def invoice_error_handler_node(state: AppState):
        print("--- Invoice Workflow Error Handler ---")
        print(f"Workflow stopped due to error: {state.get('error')}")
        # Ensure error is propagated if not already set
        return {"error": state.get('error') or "Unknown error in invoice workflow"}
    graph.add_node("error_handler", invoice_error_handler_node)
    
    # Connect nodes with conditional routing for error handling
    graph.add_conditional_edges(
        "extract_invoice",
        check_for_errors, # Generic check is okay here
        {
            "continue": "query_po_data",
            "error_handler": "error_handler" # Route to error handler
        }
    )
    
    # Check for errors after PO data query generation
    graph.add_conditional_edges(
        "query_po_data",
        check_for_errors, # Generic check okay here
        {
            "continue": "execute_invoice",
            "error_handler": "error_handler"
        }
    )
    
    # --- UPDATED: Conditional edges after executing query ---
    graph.add_conditional_edges(
        "execute_invoice",
        decide_after_invoice_execution, # Use specific decision function
        {
            "validate_invoice": "validate_invoice",
            "correct_query": "correct_po_query",
            "error_handler": "error_handler"
        }
    )
    
    # --- NEW: Edge from correction node back to execution ---
    graph.add_edge("correct_po_query", "execute_invoice")
    
    # Check for errors after validation (optional, could just end)
    graph.add_conditional_edges(
        "validate_invoice",
        check_for_errors,
        {
            "continue": END,
            "error_handler": "error_handler"
        }
    )

    # --- NEW: Connect error handler to END ---
    graph.add_edge("error_handler", END)
    
    # Set the entry point
    graph.set_entry_point("extract_invoice")
    
    # Return the StateGraph object (not compiled yet)
    return graph

def create_query_workflow():
    """
    Create the workflow for text query processing branch.
    """
    # Create the graph with AppState
    graph = StateGraph(AppState)
    
    # Add nodes to the graph
    graph.add_node("analyze_query", analyze_query_node)
    graph.add_node("generate_query", generate_query_node)
    graph.add_node("execute_query", execute_query_node)
    graph.add_node("visualize_data", visualize_data_node)
    
    # Simple error handler node (can be expanded)
    def error_handler_node(state: AppState):
        print("--- Query Workflow Error Handler ---")
        print(f"Workflow stopped due to error: {state.get('error')}")
        return {"error": state.get('error')}
    graph.add_node("error_handler", error_handler_node)

    # Connect nodes with conditional routing for error handling
    # Check for errors after query analysis
    graph.add_conditional_edges(
        "analyze_query",
        check_for_errors,
        {
            "continue": "generate_query",
            "error_handler": "error_handler"
        }
    )
    
    # Check for errors after query generation
    graph.add_conditional_edges(
        "generate_query",
        check_for_errors,
        {
            "continue": "execute_query",
            "error_handler": "error_handler"
        }
    )

    # Check for errors after query execution
    graph.add_conditional_edges(
        "execute_query",
        check_for_errors,
        {
            "continue": "visualize_data",
            "error_handler": "error_handler"
        }
    )
    
    # Check for errors after visualization
    graph.add_conditional_edges(
        "visualize_data",
        check_for_errors, 
        {
            "continue": END,
            "error_handler": "error_handler"
        }
    )

    # Connect error handler to END
    graph.add_edge("error_handler", END)
    
    # Set the entry point
    graph.set_entry_point("analyze_query")
    
    # Return the StateGraph object (not compiled yet)
    return graph

def create_main_workflow():
    """
    Create the main workflow that routes between query and invoice branches.
    """
    # Create the main graph with AppState
    main_graph = StateGraph(AppState)
    
    # Add the classification node
    main_graph.add_node("classify_input", classification_node)
    
    # Create and compile the subgraphs
    invoice_subgraph_compiled = create_invoice_workflow().compile()
    query_subgraph_compiled = create_query_workflow().compile()
    
    # Add the *compiled* subgraphs as nodes to the main graph
    main_graph.add_node("invoice_branch", invoice_subgraph_compiled)
    main_graph.add_node("query_branch", query_subgraph_compiled)
    
    # Connect the router node to the appropriate subgraph node
    main_graph.add_conditional_edges(
        "classify_input",
        route_by_input_type,
        {
            "invoice": "invoice_branch",
            "text": "query_branch",
            "error": END
        }
    )
    
    # Connect the subgraph nodes to the main graph's END
    main_graph.add_edge("invoice_branch", END)
    main_graph.add_edge("query_branch", END)
    
    # Set the main graph's entry point
    main_graph.set_entry_point("classify_input")
    
    # Compile the main graph
    compiled_main_graph = main_graph.compile()
    
    return compiled_main_graph

# Special state value for graph end
END = "__end__"

# Create the main workflow
main_workflow = create_main_workflow()

def process_input(raw_input: Union[str, bytes], input_type: str = None) -> Dict[str, Any]:
    """
    Process user input through the appropriate workflow branch.
    
    Args:
        raw_input: The user's raw input (text query or file path)
        input_type: Optional override for input type
        
    Returns:
        The final state from the workflow
    """
    initial_state = {
        "raw_input": raw_input,
        "input_type": input_type
    }
    
    # Create a checkpoint store if needed for persistence
    # checkpointer = JsonCheckpoint(os.path.join("checkpoints", "invoice_workflow"))
    
    # Execute the workflow
    result = main_workflow.invoke(initial_state)
    return result

# For testing the workflow directly
if __name__ == "__main__":
    import sys
    
    # Check if we have an argument
    if len(sys.argv) < 2:
        print("Usage: python invoice_workflow.py <query_text_or_file_path>")
        sys.exit(1)
    
    input_value = sys.argv[1]
    
    # Detect input type
    if os.path.exists(input_value) and input_value.lower().endswith((".jpg", ".jpeg", ".png", ".pdf")):
        print(f"Processing file: {input_value}")
        result = process_input(input_value, "image")
    else:
        print(f"Processing query: {input_value}")
        result = process_input(input_value, "text")
    
    # Print the result
    print("\n--- Workflow Result ---")
    for key, value in result.items():
        if key != "raw_input":  # Skip printing raw input to avoid clutter
            value_preview = str(value)
            if len(value_preview) > 200:
                value_preview = value_preview[:200] + "..."
            print(f"{key}: {value_preview}") 