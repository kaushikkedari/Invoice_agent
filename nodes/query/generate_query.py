from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
from typing import Dict
from mypackage.state import AppState # Assuming state.py is two levels up
import sys
import os

# Calculate the project root directory to import from vectordb
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from vectordb.load_vectordb import load_schema_to_vectordb, get_relevant_schema

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

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
            google_api_key=os.getenv("GOOGLE_API_KEY")
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

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    if ai_component.llm is None:
         raise ValueError("Failed to initialize LLM via AIComponent. Check config and .env.")
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    # Depending on requirements, you might want to raise the error
    # or handle it gracefully (e.g., by setting ai_component to None and checking later)
    raise 

# --- Prompt Template --- REVISED with enhanced column name accuracy ---
code_gen_prompt_template = ChatPromptTemplate.from_template(
   """You are an expert SQL developer, tasked with converting a structured query specification into a precise, executable PostgreSQL query.

**YOUR CRITICAL RESPONSIBILITY: CHARACTER-BY-CHARACTER COLUMN and TABLE NAME ACCURACY**
**DATABASE SCHEMA (THE ONE AND ONLY SOURCE OF TRUTH):**
```sql
{DATABASE_SCHEMA}
```

**PARSED QUERY REQUIREMENT:** 
{parsed_query_json}

**ORIGINAL USER QUERY:** 
{raw_input}

**YOUR TASK:**
Generate a valid PostgreSQL query that fulfills the requirements specified in the parsed query. You MUST adhere to the following rules:

1. **EXACT CHARACTER-FOR-CHARACTER SCHEMA ADHERENCE (ZERO TOLERANCE):**
   - Column and table names must EXACTLY match what appears in the provided schema, maintaining exact lettercase and underscore placement.
   - DO NOT add underscores where they don't exist in schema names.
   - DO NOT remove underscores where they exist in schema names.
   - Column names like "polineid", "totalamount", "ponumber" MUST remain exactly as shown in the schema - NEVER transform to "po_line_id", "total_amount", "po_number".
   - Conversely, if schema shows a column as "total_sale_amount" with underscores, NEVER change to "totalsaleamount".

**COMMON CONFUSING COLUMN NAMES (CRITICAL DISTINCTION):**
- sales_transaction table has "total_sale_amount" (WITH underscores)
- purchase_orders table has "totalamount" (WITHOUT underscores)
- NEVER use "totalamount" with sales_transaction table
- NEVER use "total_sale_amount" with purchase_orders table

**VERIFICATION STEPS (MANDATORY):**
- For each column in your generated SQL, verify character-by-character with schema
- Double-check all table and column names by referencing the exact schema text
- Pay special attention to underscores - they must match schema exactly
- Verify that sales_transaction uses total_sale_amount (not totalamount)
- Verify that purchase_orders uses totalamount (not total_amount)

2. **SQL FEATURES:**
   - Always qualify column names with their table or table alias: "table.column" not just "column".
   - Use appropriate JOIN syntax based on the join conditions specified.
   - Apply all requested filtering, aggregation, and sorting conditions.
   - Add aliases for computed columns if they would help readability.
   - Implement PostgreSQL-compliant SQL syntax.

3. **OUTPUT REQUIREMENTS:**
   * Provide ONLY the SQL query with no explanation or descriptions.
   * Do not include any markdown formatting elements (like ```sql).
   * Return a complete, runnable SQL query that meets all requirements.

FINAL VALIDATION:
- CHARACTER-BY-CHARACTER check each column and table name against schema
- Verify all column qualifications are correct
- Confirm all referenced tables exist in schema
- Verify the query accurately implements the parsed requirements

**GENERATE THE SQL QUERY NOW:**
"""
)

# --- LangGraph Node ---
def generate_query_node(state: AppState) -> Dict:
    """
    Generates PostgreSQL query code based on the structured query analysis
    and the original user query for context.
    """
    print("---GENERATING QUERY CODE---")
    parsed_query = state.get("parsed_query")
    raw_input = state.get("raw_input") # Get the original query
    error = None
    query_code = None

    if not parsed_query or not isinstance(parsed_query, dict):
        print("Error: Invalid or missing 'parsed_query' in state.")
        return {"error": "Invalid or missing parsed query details."}
    if not raw_input or not isinstance(raw_input, str):
        print("Warning: Missing 'raw_input' in state, context might be limited.")
        raw_input = "[Original query not available]" # Provide a default if missing

    try:
        # Get relevant schema from vector database
        try:
            # Use the vector database to get relevant schema
            relevant_schema = get_relevant_schema(vectorstore, raw_input)
            print("Using schema from vector database")
        except Exception as e:
            print(f"Error retrieving from vector database: {e}")
            # Fallback to hardcoded schema if retrieval fails
            from .database_schema_fallback import DATABASE_SCHEMA
            relevant_schema = DATABASE_SCHEMA
            print("Using fallback schema")

        # Convert parsed_query dict to JSON string for the prompt
        # Ensure None values are handled correctly
        parsed_query_json = json.dumps(parsed_query, indent=2, default=str)

        # --- Use AIComponent for the LLM call ---
        # analysis_chain = code_gen_prompt_template | llm | StrOutputParser() # OLD
        analysis_chain = code_gen_prompt_template | ai_component.llm | StrOutputParser() # NEW - Use the LLM instance from AIComponent

        # Invoke the chain
        query_code = analysis_chain.invoke({
            "DATABASE_SCHEMA": relevant_schema, 
            "raw_input": raw_input,
            "parsed_query_json": json.dumps(parsed_query, indent=2)
        })

        # Check if the response is just the query or includes ```sql ... ```
        query_code = query_code.strip()
        if query_code.startswith("```sql"):
            query_code = query_code[len("```sql"):].strip()
        if query_code.endswith("```"):
            query_code = query_code[:-len("```")].strip()

        # Relaxed validation check
        if not query_code or "SELECT" not in query_code.upper():
             print(f"Warning: Generated code might not be a valid SELECT query: {query_code}")
             # Keep as warning unless it's completely empty

    except Exception as e:
        print(f"Error during query code generation LLM call: {e}")
        error = f"An error occurred during query code generation: {e}"

    if error:
        return {"error": error}
    else:
        # Return the generated code to update the state
        return {"query_code": query_code}

# Example of how you might test this node independently (optional)
if __name__ == '__main__':
    # Make sure to set the GEMINI_API_KEY environment variable before running
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
    else:
        # Example parsed_query (replace with actual output from analyze_query_node)
        test_parsed_query_1 = {'output_format': 'table',
 'parsed_query': {'aggregation': 'sum of totalamount per vendorname',
                  'columns': ['vendors.vendorname',
                              'SUM(purchase_orders.totalamount)'],
                  'filters': None,
                  'join_conditions': 'join purchase_orders and vendors on '
                                     'purchase_orders.vendorid = '
                                     'vendors.vendorid',
                  'sorting': 'descending by SUM(purchase_orders.totalamount) '
                             'LIMIT 5',
                  'tables': ['purchase_orders', 'vendors']}}
        
        test_raw_input_1 = "Show the top 5 vendors based on the total TotalAmount of their purchase orders"

        # Create AppState with both raw_input and parsed_query
        test_state_1 = AppState(
            raw_input=test_raw_input_1,
            parsed_query=test_parsed_query_1['parsed_query'] # Pass only the parsed_query dict
        )

        result_1 = generate_query_node(test_state_1)
        print("\n--- Node Result 1 ---")
        import pprint
        pprint.pprint(result_1)
#
#         test_parsed_query_2 = {
#             "tables": ["Vendors"],
#             "columns": ["VendorName", "VendorRating"],
#             "filters": "VendorRating > 4.5",
#             "aggregation": None,
#             "join_conditions": None,
#             "sorting": "VendorName ascending"
#         }
#         test_state_2 = AppState(parsed_query=test_parsed_query_2)
#         result_2 = generate_query_node(test_state_2)
#         print("\n--- Node Result 2 ---")
#         pprint.pprint(result_2)
