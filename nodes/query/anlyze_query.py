from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
from typing import Dict
from dotenv import load_dotenv
from mypackage.state import AppState  # Correct relative import for package usage
import sys
import os

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


# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    if ai_component.llm is None:
         raise ValueError("Failed to initialize LLM via AIComponent. Check config and .env.")
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    # Depending on requirements, you might want to raise the error
    # or handle it gracefully
    raise

# --- Prompt Template --- EXTENSIVELY OPTIMIZED FOR COLUMN NAME ACCURACY ---
prompt_template = PromptTemplate.from_template(
   """You are an expert query analyst specializing in understanding natural language requests for a PostgreSQL database. 
      Your primary goal is to translate a user's query into a structured JSON object that precisely outlines the required data retrieval and processing steps, strictly adhering **ONLY** to the provided database schema.
      
**YOUR CRITICAL RESPONSIBILITY: CHARACTER-BY-CHARACTER COLUMN and TABLE NAME ACCURACY**
**DATABASE SCHEMA (THE ONE AND ONLY SOURCE OF TRUTH):**
```sql
{{DATABASE_SCHEMA}}
```

**COLUMN NAME ACCURACY IS THE #1 REQUIREMENT**
The schema above contains the ONLY acceptable table and column names. NEVER deviate from these exact names.

**QUERY TO ANALYZE:** "{query}"

B.  **EXACT CHARACTER-FOR-CHARACTER SCHEMA ADHERENCE (ZERO TOLERANCE):**
    *   You MUST COPY-PASTE column and table names EXACTLY as they appear in the schema WITHOUT ANY CHANGES.
    *   CRITICAL WARNING: The #1 cause of query failures is adding underscores (_) to column names that don't have them. NEVER add, remove, or modify underscores.
    *   IMMEDIATELY SELF-CHECK: Verify each column name character-by-character against the schema, with particular attention to underscores.
    *   Column names like "polineid", "totalamount", "ponumber", etc. MUST remain EXACTLY as shown in the schema - NEVER transform to "po_line_id", "total_amount", "po_number", etc.
    *   If unsure about a column name, ALWAYS default to the EXACT schema version - do not guess or "fix" perceived naming inconsistencies.
    *   WARNING: Any deviation from exact schema naming, even a single character difference, will cause COMPLETE QUERY FAILURE.
    *   If a requested entity or field is not present in the schema above, indicate this impossibility implicitly by omitting it or using `null` in the JSON, rather than inventing a name.

C.  **PostgreSQL Functions:**
* When the query implies data transformations or operations that require database functions (like date formatting, extraction, calculations, aggregations), describe these using common PostgreSQL function names or concepts.
* For date formatting (e.g., getting 'YYYY-MM'), use the concept of TO_CHAR(column, 'Format').
* For extracting date parts (e.g., year, month), use the concept of EXTRACT(PART FROM column).
* For aggregations, use standard names like COUNT(), SUM(), AVG(), MAX(), MIN().
* Do NOT use function names specific to other database systems (like strftime, GETDATE, DATEDIFF, etc.).
*   Use only PostgreSQL-compatible concepts.

**COLUMN NAME VERIFICATION CHECKLIST:**
- Each column name MUST be copied EXACTLY from the schema
- NEVER add underscores to "improve readability" 
- NEVER split compound words into separate words
- NEVER guess column names - refer ONLY to the schema
- VERIFY each column name CHARACTER BY CHARACTER against schema

**IMPORTANT COLUMN NAME DISTINCTION:**
The following column names look similar but are different across tables:
- sales_transaction table has "total_sale_amount" (WITH underscores)
- purchase_orders table has "totalamount" (WITHOUT underscores)
- NEVER use "totalamount" with sales_transaction
- NEVER use "total_sale_amount" with purchase_orders

**MANDATORY COLUMN NAME VERIFICATION PROCESS:**
1. For EACH column you include in your response:
   a. Find the exact column in the schema
   b. Copy the name CHARACTER-BY-CHARACTER from schema
   c. Double-check for absence/presence of underscores
   d. Never add underscores where none exist in schema
2. After generating your full response, RE-VERIFY all table and column names again

# MANDATORY SELF-CORRECTION PROCESS:
1. After generating the schema fields, perform a character-by-character verification against the provided schema.
2. For EACH table and column name, cross-reference with the exact schema text, paying special attention to underscores.
3. If you detect ANY deviation, immediately correct it before finalizing your response.

1. **parsed_query**: A JSON object containing:
   - **tables**: Array of exact table names from schema
   - **columns**: Array of columns with table qualification (e.g., "table.column")
   - **filters**: String representing WHERE/HAVING conditions or null
   - **aggregation**: String describing grouping operations or null
   - **join_conditions**: String describing table relationships or null
   - **sorting**: String describing ORDER BY preferences or null

2. **output_format**: String - either "text", "table", or "plot"
   - Use "plot" for multi-row data that shows trends or distributions
   - Use "table" for data best viewed in rows/columns
   - Use "text" for simple facts or single values

3. **visualization_type**: Optional string (e.g., "pie") for plot customization

**ESSENTIAL COLUMN NAME RULES:**
- ALL column names MUST EXACTLY match schema names
- NEVER add, remove, or modify underscores
- ALWAYS qualify columns with their table names
- VERIFY every column name character-by-character

EXAMPLE CORRECT OUTPUT:
```json
{{
  "parsed_query": {{
    "tables": ["purchase_orders", "vendors"],
    "columns": ["vendors.vendorname", "purchase_orders.totalamount"],
    "filters": "purchase_orders.status = 'Completed'",
    "aggregation": "sum of purchase_orders.totalamount grouped by vendors.vendorname",
    "join_conditions": "join purchase_orders and vendors on purchase_orders.vendorid = vendors.vendorid",
    "sorting": "descending by sum of purchase_orders.totalamount"
  }},
  "output_format": "table"
}}
```

**FINAL VERIFICATION STEP (MANDATORY):**
Before submitting your response, compare EACH column name in your JSON to the schema one final time. Check specifically for incorrect addition of underscores.

OUTPUT ONLY THE VALID JSON OBJECT - NO EXPLANATION OR ADDITIONAL TEXT.
"""
)

# --- LangGraph Node ---
def analyze_query_node(state: AppState) -> Dict:
    """
    Analyzes the raw input query using an LLM to extract structured information
    and determine the desired output format.
    """
    print("---ANALYZING QUERY---")
    raw_input = state.get("raw_input")
    error = None
    parsed_output = {}

    # --- NEW: Check for explicit chart type preference in query ---
    chart_preference = None
    if raw_input and isinstance(raw_input, str):
        raw_input_lower = raw_input.lower()
        
        # Log the raw input for debugging
        print(f"DEBUG: Raw input query: {raw_input}")
        
        # Check for pie chart request
        if any(phrase in raw_input_lower for phrase in ["pie chart", "pie graph", "as a pie", "in a pie", "using pie"]):
            chart_preference = "pie"
            print(f"DEBUG: Detected pie chart request in query: '{raw_input}'")
        # Check for bar chart request
        elif any(phrase in raw_input_lower for phrase in ["bar chart", "bar graph", "as a bar", "in a bar", "using bar"]):
            chart_preference = "bar"
            print(f"DEBUG: Detected bar chart request in query: '{raw_input}'")
        # Check for line chart request
        elif any(phrase in raw_input_lower for phrase in ["line chart", "line graph", "as a line", "in a line", "using line", "trend chart", "time series"]):
            chart_preference = "line"
            print(f"DEBUG: Detected line chart request in query: '{raw_input}'")
    # --- END NEW ---

    if not raw_input or not isinstance(raw_input, str):
        print("Error: Invalid or missing 'raw_input' in state for query analysis.")
        return {"error": "Invalid or missing query input."}

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

        # Create the analysis chain
        analysis_chain = prompt_template | ai_component.llm | StrOutputParser()

        # Invoke the chain
        llm_response = analysis_chain.invoke({
            "DATABASE_SCHEMA": relevant_schema, # Use the schema retrieved from vector DB
            "query": raw_input
        })
        print(f"LLM Response:\n{llm_response}") # Log the raw response for debugging

        # Attempt to parse the JSON response
        try:
            # --- MORE ROBUST CLEANING with DEBUGGING --- 
            print(f"DEBUG: Raw LLM Response:\n---\n{llm_response}\n---") # DEBUG
            llm_response_cleaned = llm_response.strip()
            json_str_to_parse = llm_response_cleaned # Default to cleaned response
            print(f"DEBUG: After strip():\n---\n{json_str_to_parse}\n---") # DEBUG

            if llm_response_cleaned.startswith("```json"):
                print("DEBUG: Detected ```json prefix") # DEBUG
                # Remove prefix ```json and strip whitespace/newlines
                json_str_to_parse = llm_response_cleaned[len("```json"):].strip() 
                print(f"DEBUG: After removing prefix:\n---\n{json_str_to_parse}\n---") # DEBUG
                # If the result now ends with ```, remove it and strip again
                if json_str_to_parse.endswith("```"):
                    print("DEBUG: Detected ``` suffix after prefix removal") # DEBUG
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()
                    print(f"DEBUG: After removing suffix:\n---\n{json_str_to_parse}\n---") # DEBUG
            elif llm_response_cleaned.startswith("```"):
                print("DEBUG: Detected ``` prefix (no language tag)") # DEBUG
                # Handle case where ``` has no language tag
                json_str_to_parse = llm_response_cleaned[len("```"):].strip()
                print(f"DEBUG: After removing prefix (no tag):\n---\n{json_str_to_parse}\n---") # DEBUG
                if json_str_to_parse.endswith("```"):
                    print("DEBUG: Detected ``` suffix after prefix removal (no tag)") # DEBUG
                    json_str_to_parse = json_str_to_parse[:-len("```")].strip()
                    print(f"DEBUG: After removing suffix (no tag):\n---\n{json_str_to_parse}\n---") # DEBUG
            # --- END ROBUST CLEANING ---
            
            print(f"DEBUG: Final string passed to json.loads():\n---\n{json_str_to_parse}\n---") # DEBUG
            # Parse the potentially cleaned string
            parsed_output = json.loads(json_str_to_parse) 
            
            # Basic validation of expected structure
            if not isinstance(parsed_output.get("parsed_query"), dict) or \
               not isinstance(parsed_output.get("output_format"), str):
                raise ValueError("Parsed JSON does not have the expected structure (missing 'parsed_query' dict or 'output_format' string).")
            print(f"Parsed Query: {parsed_output.get('parsed_query')}")
            print(f"Output Format: {parsed_output.get('output_format')}")
            
            # --- NEW: Extract visualization_type if present ---
            viz_type = parsed_output.get("visualization_type")
            if viz_type:
                print(f"DEBUG: LLM suggested visualization_type: {viz_type}")
                # If LLM suggests a pie chart, set chart_preference
                if viz_type.lower() == "pie":
                    chart_preference = "pie"
                    print(f"DEBUG: Setting chart_preference to 'pie' based on LLM visualization_type")
                elif viz_type.lower() in ["bar", "line"]:
                    chart_preference = viz_type.lower()
                    print(f"DEBUG: Setting chart_preference to '{viz_type.lower()}' based on LLM visualization_type")
            # --- END NEW ---

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse LLM response as JSON. Response: {llm_response}. Error: {e}")
            error = f"Failed to parse analysis result: {e}"
        except ValueError as e:
             print(f"Error: Parsed JSON structure incorrect. Response: {llm_response}. Error: {e}")
             error = f"Analysis result has incorrect structure: {e}"


    except Exception as e:
        print(f"Error during query analysis LLM call: {e}")
        error = f"An error occurred during query analysis: {e}"

    if error:
        return {"error": error}
    else:
        # Return the successfully parsed data to update the state
        result = {
            "parsed_query": parsed_output.get("parsed_query"),
            "output_format": parsed_output.get("output_format")
        }
        
        # --- NEW: Add chart preference to output if detected ---
        if chart_preference:
            result["chart_preference"] = chart_preference
            # Explicitly log that we're adding this preference
            print(f"DEBUG: Adding chart_preference='{chart_preference}' to query analysis output")
        # --- END NEW ---
        
        return result

# Example of how you might test this node independently (optional)
if __name__ == '__main__':
    import sys
    import os
    import pprint

    # Make sure to set the GEMINI_API_KEY environment variable before running
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable not set.")
    else:
        # Example 3
        test_state_3 = AppState(raw_input=""") Show the top 5 vendors based on the total TotalAmount of their purchase orders""", input_type='text')
        result_3 = analyze_query_node(test_state_3)
        print("\n--- Node Result 3 ---")
        pprint.pprint(result_3)

