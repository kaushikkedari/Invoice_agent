import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from mypackage.state import AppState  # Import for package usage
import sys
import pandas as pd
import re
import sqlparse

# Calculate the project root directory to import from vectordb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from vectordb.load_vectordb import load_schema_to_vectordb, get_relevant_schema

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

load_dotenv()

# --- Initialize Vector Database ---
try:
    # Try to load an existing vector store
    import os
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    
    if os.path.exists(os.path.join(project_root, "schema_vectorstore")):
        print("Loading existing vector database...")
        # Create embeddings model first
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY") # Ensure this uses GOOGLE_API_KEY
        )
        # Then load the vector store with the embeddings
        vectorstore = FAISS.load_local(
            os.path.join(project_root, "schema_vectorstore"),
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print("Creating new vector database...")
        vectorstore = load_schema_to_vectordb(os.path.join(project_root, "vectordb/database_schema.json"))
        # Save for future use
        vectorstore.save_local(os.path.join(project_root, "schema_vectorstore"))
except Exception as e:
    print(f"Error initializing vector database: {e}")
    # Fallback to hardcoded schema if vector DB fails
    from .database_schema_fallback import DATABASE_SCHEMA
    print("Using fallback database schema")

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    if ai_component.llm is None:
         raise ValueError("Failed to initialize LLM via AIComponent for error correction. Check config and .env.")
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    raise

# Error correction prompt
error_correction_prompt = """You are an expert PostgreSQL query debugger tasked with fixing a failed SQL query. Your primary focus is to match column names EXACTLY as they appear in the schema.

** CRITICAL RULE: COLUMN NAMES MUST BE COPIED CHARACTER-FOR-CHARACTER FROM THE SCHEMA**

**Database Schema (THE ONLY SOURCE OF TRUTH):**
```sql
{db_schema}
```

**Original User Query:**
`{user_query}`

**Failed SQL Query:**
```sql
{query_code}
```

**Error Message:**
```
{error_message}
```

**Previous Correction Attempts:**
{previous_attempts}

**WARNING: COLUMN NAME UNDERSCORE ERRORS **
The #1 most common error is adding or removing underscores in column names:

|  CORRECT (as in schema) |  WRONG (with wrong underscores) |
|--------------------------|----------------------------------|
| `contactperson`          | `contact_person`                 |
| `polineid`               | `po_line_id` or `pol_line_id`    |
| `ponumber`               | `po_number`                      |
| `total_sale_amount`      | `totalamount` or `total_amount`  |

**SPECIAL ATTENTION:** The `sales_transaction` table has `total_sale_amount` (with underscores) while other tables like `purchase_orders` have `totalamount` (without underscores). Do not mix these up!

**MANDATORY COLUMN NAME VERIFICATION STEPS:**
1. Identify each column in the query (e.g., vendors.contact_person)
2. Find the EXACT column name in the schema (e.g., vendors.contactperson)
3. Copy the column name CHARACTER-FOR-CHARACTER from the schema
4. DO NOT add or remove underscores or make any other modifications
5. Verify again by comparing each character between your corrected query and the schema

**REQUIRED CORRECTION PROCESS:**
1. First, analyze the exact error message to identify which column(s) caused the error
2. Look up the correct column name in the schema using exact character-by-character matching
3. Replace ANY incorrect column names with the EXACT names from the schema
4. Always qualify column names with their table names/aliases
5. Make minimal changes needed to fix the specific error
6. VERIFY each column name again by comparing character-by-character with the schema

**FINAL OUTPUT:**
RESPOND ONLY with the corrected SQL query - no explanations or markdown.

Generate the corrected PostgreSQL query now."""

# --- NEW: Add column name validation functions ---
def parse_schema_for_validation():
    """Parse DATABASE_SCHEMA into a structured dict for fast validation."""
    schema_dict = {}
    
    # Get schema from fallback (most reliable source for validation)
    from .database_schema_fallback import DATABASE_SCHEMA
    schema_text = DATABASE_SCHEMA
    
    # Simple regex-based parsing for table and column names
    table_pattern = r'([a-zA-Z_]+)\s*[:][^:]+?(?=\n\w+\s*:|$)'
    column_pattern = r'(\w+)\s*\('
    
    # Find all tables
    tables = re.findall(table_pattern, schema_text, re.DOTALL)
    
    for table in tables:
        table = table.strip()
        # Get the section for this table
        table_section_pattern = f"{table}\\s*:[^:]+?(?=\\n\\w+\\s*:|$)"
        table_section = re.search(table_section_pattern, schema_text, re.DOTALL)
        
        if table_section:
            section_text = table_section.group(0)
            # Extract column names
            columns = []
            for line in section_text.split('\n'):
                line = line.strip()
                if '(' in line and ')' in line:
                    # Extract column names
                    col_match = re.search(r'^\s*([a-zA-Z0-9_]+)', line)
                    if col_match:
                        col_name = col_match.group(1).strip()
                        if col_name and col_name != table:
                            columns.append(col_name)
            
            schema_dict[table] = columns
    
    print(f"Parsed schema for validation: {len(schema_dict)} tables")
    return schema_dict

# Initialize validation schema cache
SCHEMA_VALIDATION_DICT = None

def get_validation_schema():
    """Get or initialize the schema validation dictionary."""
    global SCHEMA_VALIDATION_DICT
    if SCHEMA_VALIDATION_DICT is None:
        SCHEMA_VALIDATION_DICT = parse_schema_for_validation()
    return SCHEMA_VALIDATION_DICT

def validate_sql_column_names(sql_query):
    """
    Validate that all column names in SQL query exist in the schema.
    Returns list of errors if any columns don't match the schema.
    """
    if not sql_query:
        return ["Empty SQL query"]
    
    errors = []
    schema_dict = get_validation_schema()
    
    try:
        # Parse the SQL query
        parsed = sqlparse.parse(sql_query)
        if not parsed:
            return ["Failed to parse SQL query"]
        
        # Extract column references
        for statement in parsed:
            for token in statement.flatten():
                # Look for qualified column names (table.column)
                if isinstance(token, sqlparse.sql.Identifier) and '.' in token.value:
                    parts = token.value.split('.')
                    if len(parts) == 2:
                        table_name, column_name = parts
                        
                        # Check if table exists
                        if table_name not in schema_dict:
                            errors.append(f"Table '{table_name}' not found in schema")
                            continue
                        
                        # Check if column exists in table
                        if column_name not in schema_dict[table_name]:
                            # Special case check for totalamount vs total_sale_amount confusion
                            if table_name == 'sales_transaction' and column_name == 'totalamount':
                                errors.append(f"Column '{column_name}' not found in table '{table_name}'. Did you mean 'total_sale_amount'?")
                            else:
                                errors.append(f"Column '{column_name}' not found in table '{table_name}'")
                                # Suggest similar column names if available
                                similar_columns = [col for col in schema_dict[table_name] 
                                                  if column_name.replace('_', '') == col.replace('_', '')
                                                  or column_name.lower() in col.lower()]
                                if similar_columns:
                                    errors.append(f"  Did you mean one of these? {', '.join(similar_columns)}")
        
        return errors
    except Exception as e:
        return [f"Error during SQL validation: {e}"]

# --- END NEW ---

# --- LangGraph Node ---
def execute_query_node(state: AppState) -> Dict[str, Any]:
    """
    Executes the generated SQL query against the PostgreSQL database
    using LangChain's SQLDatabase utility. Includes a feedback loop
    that attempts to fix errors by regenerating the query up to 3 times.
    
    Returns query results both as text and as a pandas DataFrame.
    """
    print("---EXECUTING QUERY---")
    query_code = state.get("query_code")
    raw_input = state.get("raw_input", "")  # Original user query
    error_message = None
    query_result = None
    result_df = None
    
    # Track correction attempts
    max_attempts = 3
    attempt_count = 0
    correction_history = []

    # --- Input Validation ---
    if not query_code or not isinstance(query_code, str):
        print("Error: Invalid or missing 'query_code' in state.")
        return {"error": "Invalid or missing query code."}

    if not all([DB_NAME, DB_USER, DB_PASSWORD]):
        error_msg = "Error: Database credentials (DB_NAME, DB_USER, DB_PASSWORD) are not set in environment variables."
        print(error_msg)
        return {"error": error_msg}

    # --- NEW: Pre-execute SQL validation ---
    validation_errors = validate_sql_column_names(query_code)
    if validation_errors:
        print(f"SQL validation failed: {validation_errors}")
        error_message = f"SQL column name validation failed: {validation_errors}"
        # Still attempt to execute, but record this for potential correction
        correction_history.append({
            "attempt": attempt_count,
            "query": query_code, 
            "error": f"Pre-execution validation: {validation_errors}",
            "correction": None
        })
    # --- END NEW ---
    
    # Create PostgreSQL connection string
    connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    
    # Try to initialize the database connection once
    try:
        print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
        db = SQLDatabase.from_uri(connection_string)
        print("Connection successful.")
    except Exception as e:
        # If connection fails, return immediately
        error_msg = f"Error connecting to database: {e}"
        print(error_msg)
        return {"error": error_msg}
    
    # --- Query Execution and Error Correction Loop ---
    current_query = query_code
    
    while attempt_count < max_attempts:
        try:
            print(f"Execution attempt {attempt_count + 1}/{max_attempts}")
            print(f"Executing SQL:\n{current_query}")
            
            # Execute the query and get text result
            result = db.run(current_query)
            
            # ADDED: Execute the query again to get DataFrame result
            # Use SQLAlchemy engine directly
            import sqlalchemy
            from sqlalchemy import text
            engine = sqlalchemy.create_engine(connection_string)
            with engine.connect() as connection:
                # Use pandas to execute the query and get a DataFrame
                result_df = pd.read_sql_query(text(current_query), connection)
            
            # Success! Set result and break the loop
            query_result = result
            print(f"Query executed successfully.")
            print(f"Result preview: {result[:500]}..." if len(result) > 500 else f"Result: {result}")
            print(f"DataFrame shape: {result_df.shape}")
            break
            
        except Exception as e:
            # Query failed
            error_message = str(e)
            print(f"Error executing query (attempt {attempt_count + 1}): {error_message}")
            
            # If this was our last attempt, break and return the error
            if attempt_count >= max_attempts - 1:
                print(f"Maximum correction attempts ({max_attempts}) reached without success.")
                break
                
            # Try to correct the query using AIComponent
            corrected_query = _generate_corrected_query(
                current_query, 
                error_message, 
                raw_input, 
                correction_history,
                ai_component # Pass the component instance
            )
            
            # Add to correction history
            correction_history.append({
                "attempt": attempt_count + 1,
                "query": current_query,
                "error": error_message,
                "correction": corrected_query
            })
            
            # Update current query for next attempt
            current_query = corrected_query
            attempt_count += 1
            
            print(f"Generated corrected query (attempt {attempt_count}):\n{current_query}")
    
    # --- Return State Update ---
    result_dict = {}
    
    if query_result is not None:
        # Success case
        result_dict["query_result"] = query_result
        # ADDED: Include the DataFrame in the state
        if result_df is not None:
            result_dict["result_dataframe"] = result_df
        if correction_history:
            result_dict["correction_history"] = correction_history
    else:
        # Error case
        result_dict["error"] = error_message
        result_dict["correction_history"] = correction_history
    
    return result_dict

# Modified to accept AIComponent instance
def _generate_corrected_query(query_code: str, error_message: str, user_query: str, 
                              correction_history: List[Dict], ai_component: AIComponent) -> str:
    """
    Generates a corrected query based on the error message using the provided AIComponent.
    """
    # Check if LLM is available in the component
    if not ai_component or not ai_component.llm:
         print("Error: AIComponent not available or LLM not initialized for query correction.")
         return query_code # Return original query if correction cannot be attempted
         
    try:
        # Get relevant schema from vector database for the error correction
        try:
            # Use the vector database to get relevant schema
            relevant_schema = get_relevant_schema(vectorstore, user_query)
            print("Using schema from vector database for error correction")
        except Exception as e:
            print(f"Error retrieving from vector database: {e}")
            # Fallback to hardcoded schema if retrieval fails
            from .database_schema_fallback import DATABASE_SCHEMA
            relevant_schema = DATABASE_SCHEMA
            print("Using fallback schema for error correction")
            
        # Format previous attempts for the prompt
        previous_attempts_str = ""
        if correction_history:
            previous_attempts_str = "Previous correction attempts:\n"
            for i, attempt in enumerate(correction_history):
                previous_attempts_str += f"Attempt {i+1}:\n```sql\n{attempt['correction']}\n```\n"
                previous_attempts_str += f"Error: {attempt['error']}\n\n"
        else:
            previous_attempts_str = "No previous correction attempts."
        
        # --- Use AIComponent for the LLM call ---
        # Create the correction chain
        correction_chain = ChatPromptTemplate.from_template(error_correction_prompt) | ai_component.llm | StrOutputParser()
        
        # Invoke the chain
        corrected_query = correction_chain.invoke({
            "db_schema": relevant_schema,
            "user_query": user_query,
            "query_code": query_code,
            "error_message": error_message,
            "previous_attempts": previous_attempts_str
        })
        
        # Clean up the response
        corrected_query = corrected_query.strip()
        if corrected_query.startswith("```sql"):
            corrected_query = corrected_query[6:].strip()
        if corrected_query.endswith("```"):
            corrected_query = corrected_query[:-3].strip()
        
        return corrected_query
    
    except Exception as e:
        print(f"Error during query correction: {e}")
        # If correction fails, return the original query to avoid making things worse
        return query_code


# Example of how you might test this node independently (optional)
if __name__ == '__main__':
    import sys
    import pprint

    # Make sure to set the GEMINI_API_KEY environment variable before running
    if not os.getenv("GEMINI_API_KEY"):
        print("Cannot run test: GEMINI_API_KEY environment variable not set.")
    # Ensure DB environment variables are set (.env should be loaded)
    elif not all([DB_NAME, DB_USER, DB_PASSWORD]):
        print("Cannot run test: Database credentials (DB_NAME, DB_USER, DB_PASSWORD) are not set.")
    else:
        # Test with a query that contains an error (using a non-existent table)
        print("\n--- Test Case: Query with Error ---")
        test_query = """SELECT
    vendor.vendorname,
    SUM(purchase_orders.totalamount)
FROM
    purchase_orders
INNER JOIN
    vendor ON purchase_orders.vendorid = vendor.vendorid
GROUP BY
    vendor.vendorname
ORDER BY
    SUM(purchase_orders.totalamount) DESC
LIMIT 5;"""
        test_user_query = "Show the top 5 vendors based on the total TotalAmount of their purchase orders"
              
        test_state = AppState(
            query_code=test_query,
            raw_input=test_user_query
        )
        result = execute_query_node(test_state)
        print("\nResult:")
        pprint.pprint(result)