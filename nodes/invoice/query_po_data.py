import os
import json
import pandas as pd
from typing import Dict, Any, Optional

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from ..query.database_schema_fallback import DATABASE_SCHEMA
# Vector Store Imports (Added)
try:
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    VECTORSTORE_AVAILABLE = True
except ImportError:
    VECTORSTORE_AVAILABLE = False
    print("Warning: FAISS/Google Embeddings libraries not found. Install with: pip install faiss-cpu langchain-community langchain-google-genai")

# Import AppState for type hinting
try:
    from mypackage.state import AppState
except ImportError:
    print("Warning: mypackage.state.AppState not available. Using local type definitions.")
    # Define a minimal mock if AppState is not available
    from typing import TypedDict, Literal, Union
    class AppState(TypedDict):
        extracted_invoice_data: Optional[Dict[str, Any]]
        invoice_code: Optional[str]
        error: Optional[str]

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

# --- Configuration ---
VECTORSTORE_PATH = "schema_vectorstore" # Path to your FAISS index directory

# --- Default Fallback Schema --- (Used if VectorDB and file loading fail)
# Attempt to load from actual fallback file first
try:
    from ..query.database_schema_fallback import DATABASE_SCHEMA as FALLBACK_SCHEMA
    print("Loaded fallback schema from file.")
except (ImportError, ModuleNotFoundError):
    print("Warning: Could not import fallback schema from file. Using hardcoded default.")
    FALLBACK_SCHEMA = "purchase_orders(po_id, ponumber, vendorid, orderdate, totalamount, currency), purchase_order_line_items(polineid, po_id, itemid, quantityordered, unitprice, linetotal), vendors(vendorid, vendorname)"

DEFAULT_SCHEMA = FALLBACK_SCHEMA # Assign to the constant name used later

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    # No immediate error raise, functions below will check ai_component.llm
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    ai_component = None # Set to None if init fails

# --- Helper Functions ---

def _get_database_schema() -> str:
    """
    Attempts to fetch the database schema, prioritizing VectorDB (FAISS),
    then falling back to the imported file schema, then a default string.
    
    Returns:
        A string containing the database schema.
    """
    schema = None
    
    # --- 1. Attempt to fetch from FAISS VectorDB --- 
    if VECTORSTORE_AVAILABLE:
        api_key = os.getenv("GOOGLE_API_KEY")
        if api_key and os.path.exists(VECTORSTORE_PATH):
            try:
                print(f"Attempting to load FAISS vector store from: {VECTORSTORE_PATH}")
                # Adjust embedding model if needed
                embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
                
                vector_store = FAISS.load_local(
                    VECTORSTORE_PATH, 
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print("FAISS vector store loaded successfully.")
                
                schema_query = "database schema description"
                print(f"Querying vector store for: '{schema_query}'")
                results = vector_store.similarity_search(schema_query, k=1)
                
                if results and results[0].page_content:
                    schema = results[0].page_content
                    print("Successfully retrieved schema from VectorDB.")
                else:
                    print("Info: Schema query returned no results or empty content from VectorDB.")
                    
            except Exception as e:
                print(f"Warning: Failed to load or query FAISS VectorDB at '{VECTORSTORE_PATH}': {e}. Falling back...")
        elif not api_key:
            print("Info: GOOGLE_API_KEY not set, cannot load embeddings for VectorDB.")
        elif not os.path.exists(VECTORSTORE_PATH):
             print(f"Info: Vector store path '{VECTORSTORE_PATH}' not found.")
    else:
        print("Info: FAISS library not available, skipping VectorDB lookup.")
        
    # --- 2. Use Fallback Schema from file (if VectorDB failed) ---
    if schema is None:
        print("Using fallback schema from file or default.")
        schema = DEFAULT_SCHEMA # This now refers to the schema loaded from file or the hardcoded one
            
    return schema

def _generate_po_sql(invoice_data: Dict[str, Any], db_schema: str) -> Optional[str]:
    """
    Uses the AIComponent LLM to generate a *single* SQL query to fetch PO data.
    """
    po_number = invoice_data.get("purchase_order_number")
    if not po_number:
        print("Error: Cannot generate SQL without purchase_order_number in invoice_data.")
        return None
        
    print(f"Generating SQL query for PO Number: {po_number} using AIComponent...")
    
    # Check if AI Component is available
    if not ai_component or not ai_component.llm:
        print("Error: LLM (via AIComponent) not initialized. Cannot generate SQL.")
        return None
        
    try:
        # Convert invoice data to JSON string for the prompt
        invoice_data_str = json.dumps(invoice_data, indent=2)
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are an expert PostgreSQL query generator. Your task is to write a single SQL SELECT query to fetch all database records required for validating an invoice against a purchase order (PO).
             You will be given the database schema and the PO number from the invoice.

## Database Schema Definition (SOURCE OF TRUTH - CHARACTER-FOR-CHARACTER EXACT MATCHING REQUIRED) ##
```sql
{schema}
```

## Target Purchase Order Number ##
`{po_number}`

## !!! CRITICAL WARNING ABOUT COLUMN NAMES !!! ##
THE #1 CAUSE OF QUERY FAILURES IS ADDING UNDERSCORES TO COLUMN NAMES THAT DON'T HAVE THEM.

Examples of CRITICAL ERRORS you MUST AVOID:
✓ CORRECT: `polineid` (exactly as in schema)  
✗ WRONG: `po_line_id` (adding underscores)

✓ CORRECT: `totalamount` (exactly as in schema)  
✗ WRONG: `total_amount` (adding underscores)

✓ CORRECT: `ponumber` (exactly as in schema)  
✗ WRONG: `po_number` (adding underscores)

You MUST COPY-PASTE column names from the schema EXACTLY - character-for-character - with NO modifications whatsoever.
ANY deviation from exact schema naming, even a single underscore difference, will cause COMPLETE QUERY FAILURE.

## Guidelines for Query Generation ##

1.  **Required Data:** The query MUST retrieve all necessary details for invoice validation. This requires joining `purchase_orders`, `vendors`, `purchase_order_line_items`, AND the `items` table.

2.  **EXACT CHARACTER-FOR-CHARACTER SCHEMA ADHERENCE (ZERO TOLERANCE):**
    *   You MUST COPY-PASTE column and table names EXACTLY as they appear in the schema WITHOUT ANY CHANGES.
    *   CRITICAL WARNING: The #1 cause of query failures is adding underscores (_) to column names that don't have them. NEVER add, remove, or modify underscores.
    *   IMMEDIATELY SELF-CHECK: Verify each column name character-by-character against the schema, with particular attention to underscores.
    *   Column names like "polineid", "totalamount", "ponumber", etc. MUST remain EXACTLY as shown in the schema - NEVER transform to "po_line_id", "total_amount", "po_number", etc.
    *   If unsure about a column name, ALWAYS default to the EXACT schema version - do not guess or "fix" perceived naming inconsistencies.
    *   WARNING: Any deviation from exact schema naming, even a single character difference, will cause COMPLETE QUERY FAILURE.

3.  **SELECT Clause (Specific Columns Required):**
    *   You MUST select specific columns from the joined tables. Do NOT use `*`.
    *   **From `purchase_orders`:** Include `po_id`, `ponumber`, `orderdate`, `expecteddeliverydate`, `subtotalamount`, `taxamount`, `shippingcost`, `totalamount`, `currency`, `status`, `vendorid` (for joining). **Use exact schema names.**
    *   **From `vendors` (JOINED):** Include `vendorname` and any available address/contact columns specified in the schema. **Use exact schema names.**
    *   **From `purchase_order_line_items` (JOINED):** Include `polineid`, `itemid` (for joining), `itemdescription`, `quantityordered`, `unitofmeasure`, `unitprice`, `linetotal`, `linestatus`. **Use exact schema names.**
    *   **From `items` (JOINED):** You **MUST** include the `hsn` column (e.g., `items.hsn`). **Use the exact schema name for `hsn`.**
    *   **Qualify ALL columns** with their table name or alias.

4.  **JOINS (Mandatory Tables):**
    *   You **MUST** construct `INNER JOIN` clauses to link:
        *   `purchase_orders` with `vendors` (using `vendorid`)
        *   `purchase_orders` with `purchase_order_line_items` (using `po_id`)
        *   `purchase_order_line_items` with `items` (using `itemid`)
    *   The `ON` conditions MUST use the exact column names for keys as specified in the **Database Schema Definition**.

5.  **FILTERING (WHERE Clause):**
    *   Use a `WHERE` clause to filter `purchase_orders` based on the `ponumber` column.
    *   The filter MUST compare `purchase_orders.ponumber` (using the exact schema name) to the target PO number: `purchase_orders.ponumber = '{po_number}'`.

6.  **MANDATORY SELF-CORRECTION PROCESS:**
   *   After drafting your query, perform a character-by-character verification against the provided schema.
   *   For EACH table and column name, cross-reference with the exact schema text, paying special attention to underscores.
   *   If you detect ANY deviation, immediately correct it before finalizing your response.
   *   Perform this verification AGAIN after corrections.

7.  **Output Format:**
    *   Generate ONLY the single, complete PostgreSQL SELECT statement.
    *   Do NOT include any surrounding text or explanations.
    *   End the query with a semicolon `;`.

Generate the PostgreSQL query to fetch specific columns from purchase_orders, vendors, purchase_order_line_items, AND items, including items.hsn, for the target PO Number.
"""),
            ("human", f"""Generate a PostgreSQL query to retrieve PO validation data for PO Number: '{po_number}'.

YOUR ABSOLUTE TOP PRIORITY IS COLUMN NAME ACCURACY:
- COPY column names EXACTLY from schema with ZERO modifications
- NEVER add underscores to column names (e.g., use 'polineid' NOT 'po_line_id')
- Perform COLUMN-BY-COLUMN verification against the schema

Required tables:
- purchase_orders
- vendors
- purchase_order_line_items
- items (must include items.hsn)

Character-for-character schema adherence is MANDATORY.
""")
        ])
        
        # --- Use AIComponent for the LLM call ---
        sql_chain = prompt_template | ai_component.llm | StrOutputParser()
        
        generated_sql = sql_chain.invoke({
            "schema": db_schema,
            "invoice_data": invoice_data_str,
            "po_number": po_number
        })
        
        # --- ADDED CLEANING STEP --- 
        # Clean potential markdown code block formatting
        generated_sql = generated_sql.strip()
        if generated_sql.startswith("```sql"):
            # Remove ```sql prefix and leading/trailing whitespace/newlines
            generated_sql = generated_sql[len("```sql"):].strip()
        elif generated_sql.startswith("```"): # Handle case without 'sql' language tag
             generated_sql = generated_sql[len("```"):].strip()
             
        if generated_sql.endswith("```"):
            # Remove ``` suffix and leading/trailing whitespace/newlines
            generated_sql = generated_sql[:-len("```")].strip()
        # --- END CLEANING STEP --- 

        # Basic validation/cleanup (now runs on cleaned SQL)
        generated_sql = generated_sql.strip()
        if not generated_sql.upper().startswith("SELECT"):
            print(f"Error: Generated output doesn't look like a SELECT query after cleaning: {generated_sql}")
            return None
            
        # Add trailing semicolon if missing
        if not generated_sql.endswith(';'):
             generated_sql += ';'
             
        print("Generated SQL Query (cleaned):", generated_sql)
        return generated_sql
            
    except Exception as e:
        print(f"Error during SQL generation with AIComponent: {e}")
        return None

# --- LangGraph Node Function ---

def load_po_data_node(state: AppState) -> Dict[str, Any]:
    """
    LangGraph node: Gets DB schema (VectorDB -> File -> Default),
    then uses an LLM to generate a *single* SQL query string to fetch 
    PO/Vendor/Line Item data based on the extracted invoice PO number.
    The generated query is stored in the state for a later execution node.
    """
    print("\n---NODE: Load PO Data (Generate SQL Only)---")
    updates: Dict[str, Any] = {
        "invoice_code": None,
        "error": None
    }
    
    invoice_data = state.get("extracted_invoice_data")
    
    if not invoice_data:
        updates["error"] = "Cannot generate PO query: Extracted invoice data is missing."
        print(f"Error: {updates['error']}")
        return updates
        
    po_number = invoice_data.get("purchase_order_number")
    
    if not po_number:
        print("Info: No purchase_order_number in invoice data. No PO query generated.")
        # No query to generate, but not necessarily an error for the workflow
        updates["invoice_code"] = None 
        return updates

    # Step 1: Get Database Schema 
    db_schema = _get_database_schema()
    if not db_schema:
        updates["error"] = "Critical error: Failed to obtain any database schema."
        print(f"Error: {updates['error']}")
        return updates

    # Step 2: Generate the single SQL Query using LLM
    generated_sql = _generate_po_sql(invoice_data, db_schema)
    
    if not generated_sql:
        updates["error"] = f"Failed to generate SQL query for PO number '{po_number}'."
        print(f"Error: {updates['error']}")
        # Leave invoice_code as None
    else:
        updates["invoice_code"] = generated_sql
        print("Successfully generated SQL query.")
        
    print("--- Completed Load PO Data Node (Generate SQL Only) ---")
    return updates

# --- Standalone Execution Example ---

if __name__ == '__main__':
    import pprint 

    print("\n--- Running Standalone Test for load_po_data_node (Generate SQL Only) ---")

    # --- Mock Input State ---
    # Use the actual extracted data you provided
    extracted_data_from_invoice = {
  "invoice_number": "INV-2024-ERR-005",
  "invoice_date": "2024-08-20",
  "Invoice_ate": None,
  "vendor_name": "National Industrial Supply",
  "vendor_city": "Chicago",
  "vendor_country": None,
  "purchase_order_number": "PO-2024-09006",
  "payment_terms": "Net 30",
  "subtotal_amount": 1245.0,
  "tax_amount": 99.6,
  "shipping_cost": 80.0,
  "total_amount": 1424.6,
  "line_items": [
    {
      "description": "Business Laptop 14-inch",
      "sku": None,
      "quantity": 1,
      "unit_of_measure": None,
      "unit_price": 825.0,
      "line_total": 825.0
    },
    {
      "description": "USB 3.0 Docking Station",
      "sku": None,
      "quantity": 1,
      "unit_of_measure": None,
      "unit_price": 135.0,
      "line_total": 135.0
    },
    {
      "description": "Windows Pro License",
      "sku": None,
      "quantity": 1,
      "unit_of_measure": None,
      "unit_price": 195.0,
      "line_total": 195.0
    },
    {
      "description": "Office 365 Business Standard License",
      "sku": None,
      "quantity": 1,
      "unit_of_measure": None,
      "unit_price": 90.0,
      "line_total": 90.0
    }
  ]
}

    mock_state_with_po: AppState = {
        "extracted_invoice_data": extracted_data_from_invoice,
        "invoice_code": None,
        "error": None,
         # Add other required AppState fields if the mock definition is strict
         "input_type": "text", # Or "pdf"/"image" depending on original input type
         "raw_input": "invoice_file_path_placeholder.pdf" # Placeholder path
        # Add any other fields from AppState initialized to None or default
    }

    
    # --- Test Execution ---
    print("\nTesting with PO Number...")
    if not os.getenv("GEMINI_API_KEY"):
         print("\n*** WARNING: GEMINI_API_KEY not set. LLM call will likely fail. ***")
         
    result_with_po = load_po_data_node(mock_state_with_po)
    print("\n--- Result (With PO) ---")
    pprint.pprint(result_with_po)
    if result_with_po.get("invoice_code"):
        print("\nGenerated Query:")
        print(result_with_po["invoice_code"])
    elif not result_with_po.get("error"):
         print("\nNo query generated (as expected or due to LLM issue without error). Check logs.")
    else:
        print("\nNode finished with an error.")

    print("\nStandalone test finished.") 