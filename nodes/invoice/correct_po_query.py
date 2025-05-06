import os
import json
from typing import Dict, Any, Optional

# LangChain Imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# State Import - Adjust path as necessary if your structure differs
# Assuming AppState is in mypackage.state
try:
    from mypackage.state import AppState
except ImportError:
    print("Warning: mypackage.state.AppState not available. Using local type definitions.")
    # Define a minimal mock if AppState is not available
    from typing import TypedDict, List
    class AppState(TypedDict):
        invoice_code: Optional[str]
        failed_sql_query: Optional[str]
        db_error_message: Optional[str]
        po_query_correction_attempts: Optional[int]
        extracted_invoice_data: Optional[Dict[str, Any]]
        error: Optional[str]
        # Add other potentially required fields for context if needed

# Helper Imports (assuming similar structure to other nodes)
# Adapt the path '..' based on your actual project structure
try:
    # Need schema loading capability, similar to query_po_data.py
    from ..query.database_schema_fallback import DATABASE_SCHEMA as FALLBACK_SCHEMA
    VECTORSTORE_PATH = "schema_vectorstore" # Path to your FAISS index directory
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    VECTORSTORE_AVAILABLE = True
    print("Loaded vector store dependencies for schema loading.")
except ImportError:
    VECTORSTORE_AVAILABLE = False
    FALLBACK_SCHEMA = "SCHEMA_UNAVAILABLE" # Define a clear fallback
    print("Warning: Vector store or fallback schema dependencies not found. Correction quality may be reduced.")

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

# Load environment variables
load_dotenv()

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    if ai_component.llm is None:
         # Raise error here because correction relies on the LLM
         raise ValueError("Failed to initialize LLM via AIComponent for correction. Check config and .env.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize AIComponent for correction - {e}")
    raise SystemExit(f"AIComponent initialization failed: {e}")


# --- Helper Function: Get Schema ---
# Simplified version - uses vector store if possible, else fallback.
# Consider consolidating schema loading logic into a shared utility if used often.
def _get_database_schema() -> str:
    """ Attempts to fetch the database schema for correction context. """
    schema = None
    if VECTORSTORE_AVAILABLE:
        api_key = os.getenv("GOOGLE_API_KEY")
        vectorstore_full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', VECTORSTORE_PATH))
        if api_key and os.path.exists(vectorstore_full_path):
            try:
                print(f"Attempting to load FAISS vector store from: {vectorstore_full_path} for correction")
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                vector_store = FAISS.load_local(
                    vectorstore_full_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                schema_query = "database schema description"
                results = vector_store.similarity_search(schema_query, k=1)
                if results and results[0].page_content:
                    schema = results[0].page_content
                    print("Successfully retrieved schema from VectorDB for correction.")
                else:
                    print("Info: Schema query returned no results from VectorDB for correction.")
            except Exception as e:
                print(f"Warning: Failed to load/query VectorDB for correction: {e}. Falling back...")
        elif not api_key:
             print("Info: GEMINI_API_KEY not set for vectorDB schema loading.")
        elif not os.path.exists(vectorstore_full_path):
             print(f"Info: Vector store path '{vectorstore_full_path}' not found for correction.")

    if schema is None:
        print("Using fallback schema for correction.")
        schema = FALLBACK_SCHEMA # Use the imported or default fallback
        if schema == "SCHEMA_UNAVAILABLE":
            print("CRITICAL WARNING: No database schema available for correction.")

    return schema


# --- Correction Prompt Template ---
CORRECTION_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", """You are an expert PostgreSQL query debugging and correction assistant.
Your task is to analyze a failed SQL query intended to fetch Purchase Order (PO) details for invoice validation, understand the database error message, and generate a corrected query based strictly on the provided database schema.

**Goal Context:**
The original goal was to fetch PO header, line item, and vendor details matching a specific Purchase Order Number (`purchase_order_number`) found in an invoice (`{po_number_from_invoice}`). The data is needed to validate the invoice against the PO.

**Database Schema Definition (Retrieved - ABSOLUTE Source of Truth):**
```sql
{schema}
```
**CRITICAL:** This schema is the *only* valid source for table and column names, casing, and structure. Use it exclusively. Pay attention to column names like `hsn` (in `items`) if relevant.

**Failed SQL Query:**
```sql
{failed_query}
```

**Database Error Message:**
```
{db_error}
```

## CRITICAL RULES FOR QUERY CORRECTION: ##
1.  **Analyze Error:** Understand the `Database Error Message` to pinpoint the cause of failure (e.g., syntax error, non-existent column/table, incorrect join).
2.  **Analyze Failed Query:** Review the `Failed SQL Query` in light of the error message and the schema.
3.  **SCHEMA NAMES ONLY (NON-NEGOTIABLE):** The corrected query MUST use the exact lowercase table and column names as they appear **verbatim** in the **Database Schema Definition**.
    *   **DO NOT** invent, guess, approximate, or hallucinate names. NO EXCEPTIONS.
    *   **DO NOT** change casing.
    *   **DO NOT** modify underscores.
    *   Correct any names in the failed query that do not strictly match the schema.
4.  **QUALIFY ALL COLUMNS:** ALWAYS qualify column names with their exact table name from the schema (e.g., `purchase_orders.vendorid`). Correct if the failed query missed this.
5.  **Correct Joins:** Ensure JOIN clauses link the correct tables (`purchase_orders`, `purchase_order_line_items`, `vendors`, and potentially `items` if HSN/item details were needed) using the exact column names specified in the schema for relationships (`vendorid`, `po_id`, `itemid`).
6.  **Correct SELECT:** Ensure all selected columns exist in the schema and are qualified. The query should select enough detail for invoice validation (PO header info, vendor info, line item details including description, quantity, price, total, potentially HSN via join).
7.  **Correct WHERE:** Ensure the `WHERE` clause correctly filters `purchase_orders.ponumber` against the target `purchase_order_number` (`{po_number_from_invoice}`) using the exact schema names.
8.  **Syntax & Format:**
    *   Generate ONLY a single, complete, corrected PostgreSQL SELECT statement.
    *   Do NOT include any surrounding text, explanations, comments, or markdown formatting (like ```sql```).
    *   End the query with a semicolon `;`.
9.  **FINAL CHECK (MANDATORY):** Before outputting, meticulously verify: Does every single table name and column name used in the generated corrected query exist EXACTLY (including casing and underscores) in the **Database Schema Definition** provided above? If not, correct it before outputting.

Generate the corrected PostgreSQL query based *only* on the schema, the failed query, the error message, and the goal context.
"""),
    # Note: 'human' message is kept minimal as system prompt contains all context.
    ("human", "Correct the failed SQL query based on the provided details.")
])


# --- LangGraph Node Function ---

def correct_po_query_node(state: AppState) -> Dict[str, Any]:
    """
    Attempts to correct a failed SQL query generated for fetching PO data.
    """
    print("\n---NODE: Correct PO Query---")
    updates: Dict[str, Any] = {
        "error": None # Clear previous node errors if correction is attempted
    }

    failed_query = state.get("failed_sql_query")
    db_error = state.get("db_error_message")
    extracted_invoice_data = state.get("extracted_invoice_data")
    current_attempts = state.get("po_query_correction_attempts", 0)

    # --- Input Validation ---
    if not failed_query:
        updates["error"] = "Correction Error: Missing 'failed_sql_query' in state."
        print(f"Error: {updates['error']}")
        # Cannot proceed without the query that failed
        return updates # Or potentially route to a different error state

    if not db_error:
        updates["error"] = "Correction Error: Missing 'db_error_message' in state."
        print(f"Error: {updates['error']}")
        # Cannot effectively correct without the error
        return updates

    if not extracted_invoice_data or not extracted_invoice_data.get("purchase_order_number"):
         updates["error"] = "Correction Error: Missing invoice data or PO number for context."
         print(f"Error: {updates['error']}")
         # Correction might be less effective without context
         # Decide if you want to proceed or fail here
         # return updates # Option to fail early

    po_number_context = extracted_invoice_data.get("purchase_order_number", "UNKNOWN_PO") if extracted_invoice_data else "UNKNOWN_PO"


    # --- Get Schema ---
    db_schema = _get_database_schema()
    if db_schema == "SCHEMA_UNAVAILABLE":
        updates["error"] = "Correction Error: Cannot correct query without database schema."
        print(f"Error: {updates['error']}")
        # Cannot proceed without schema
        return updates


    # --- Attempt Correction ---
    try:
        print(f"Attempting correction for PO: {po_number_context} (Attempt {current_attempts + 1})")
        print(f"Failed Query:\n{failed_query}")
        print(f"DB Error:\n{db_error}")

        # --- Use AIComponent for the LLM call ---
        correction_chain = CORRECTION_PROMPT_TEMPLATE | ai_component.llm | StrOutputParser()

        corrected_sql = correction_chain.invoke({
            "schema": db_schema,
            "failed_query": failed_query,
            "db_error": db_error,
            "po_number_from_invoice": po_number_context
        })

        # --- Clean and Validate Corrected SQL ---
        corrected_sql = corrected_sql.strip()
        # Remove potential markdown
        if corrected_sql.startswith("```sql"):
            corrected_sql = corrected_sql[len("```sql"):].strip()
        elif corrected_sql.startswith("```"):
            corrected_sql = corrected_sql[len("```"):].strip()
        if corrected_sql.endswith("```"):
            corrected_sql = corrected_sql[:-len("```")].strip()

        if not corrected_sql.upper().startswith("SELECT"):
            raise ValueError(f"Correction attempt did not produce a valid SELECT query. Output: {corrected_sql}")

        if not corrected_sql.endswith(';'):
             corrected_sql += ';'

        print(f"Generated Corrected SQL Query:\n{corrected_sql}")

        # Update state with the corrected query and increment attempt counter
        updates["invoice_code"] = corrected_sql # Overwrite the failed query
        updates["po_query_correction_attempts"] = current_attempts + 1
        updates["failed_sql_query"] = None # Clear the failed query marker
        updates["db_error_message"] = None # Clear the error message

    except Exception as e:
        error_msg = f"Correction Error: Failed during LLM call or processing - {e}"
        print(f"Error: {error_msg}")
        updates["error"] = error_msg
        # Keep the attempt counter incremented even if correction fails
        updates["po_query_correction_attempts"] = current_attempts + 1
        # Decide if you want to clear invoice_code or leave the last failed one
        # updates["invoice_code"] = None # Option to clear

    print("---NODE: Correct PO Query Completed---")
    return updates


# --- Standalone Execution Example ---
if __name__ == '__main__':
    import pprint

    print("\n--- Running Standalone Test for correct_po_query_node ---")

    # --- Mock Input State ---
    mock_invoice_data = {
      "invoice_number": "INV-TEST-123",
      "purchase_order_number": "PO-TEST-999",
      # other fields...
    }

    # Example: Correcting a common error (wrong column name)
    mock_state_wrong_column: AppState = {
        "extracted_invoice_data": mock_invoice_data,
        "failed_sql_query": "SELECT po.po_id, po.ponumber, po.total_amnt FROM purchase_orders po WHERE po.ponumber = 'PO-TEST-999';", # Incorrect 'total_amnt'
        "db_error_message": 'ERROR: column po.total_amnt does not exist\nLINE 1: SELECT po.po_id, po.ponumber, po.total_amnt FROM purchase_o...\n                                     ^',
        "po_query_correction_attempts": 0,
        # Ensure other fields needed by AppState mock are present if strict
        "invoice_code": None, # This would be set by the correction
        "error": None
    }

     # Example: Correcting a join error
    mock_state_join_error: AppState = {
        "extracted_invoice_data": mock_invoice_data,
        "failed_sql_query": """
            SELECT
                po.po_id,
                po.ponumber,
                po.totalamount,
                pol.quantityordered,
                v.vendorname
            FROM purchase_orders po
            INNER JOIN purchase_order_line_items pol ON po.poid = pol.poid -- Incorrect join column case/name
            INNER JOIN vendors v ON po.vendorid = v.vendorid
            WHERE po.ponumber = 'PO-TEST-999';
            """,
        "db_error_message": 'ERROR: column po.poid does not exist',
        "po_query_correction_attempts": 1, # Simulate second attempt
        "invoice_code": None,
        "error": None
    }


    # --- Test Execution ---
    if not os.getenv("GEMINI_API_KEY"):
         print("\n*** WARNING: GEMINI_API_KEY not set. LLM call will likely fail. ***")
         # sys.exit(1) # Optional: exit if key is mandatory

    print("\nTesting with wrong column error...")
    result_wrong_column = correct_po_query_node(mock_state_wrong_column)
    print("\n--- Result (Wrong Column) ---")
    pprint.pprint(result_wrong_column)
    if result_wrong_column.get("invoice_code"):
        print("\nCorrected Query:")
        print(result_wrong_column["invoice_code"])

    print("\nTesting with join error...")
    result_join_error = correct_po_query_node(mock_state_join_error)
    print("\n--- Result (Join Error) ---")
    pprint.pprint(result_join_error)
    if result_join_error.get("invoice_code"):
        print("\nCorrected Query:")
        print(result_join_error["invoice_code"])


    print("\nStandalone test finished.") 