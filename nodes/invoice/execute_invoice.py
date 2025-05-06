import os
import sys
import json
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Try importing AppState (keeping this part as it was)
try:
    from mypackage.state import AppState
    APP_STATE_AVAILABLE = True
except ImportError:
    APP_STATE_AVAILABLE = False
    print("Warning: AppState not available. Standalone mode only.")

# Database connection libraries check
try:
    import psycopg2
    DB_LIBS_AVAILABLE = True
except ImportError:
    DB_LIBS_AVAILABLE = False
    print("Warning: Database libraries not found. Install with: pip install psycopg2-binary sqlalchemy pandas")

load_dotenv(r"D:\April_task\OCR_PoC_working_base code\.env")

# --- Database Configuration ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def execute_sql_query(query: str) -> Tuple[bool, Optional[pd.DataFrame], Optional[str], int]:
    """
    Executes a SQL query and returns the results as a DataFrame.

    Args:
        query: SQL query string to execute

    Returns:
        Tuple containing:
        - success: Boolean indicating if query was successful
        - df: Pandas DataFrame with results if successful, None otherwise
        - error: Error message if not successful, None otherwise
        - row_count: Number of rows returned
    """
    success = False
    df = None
    error = None
    row_count = 0

    if not DB_LIBS_AVAILABLE:
        error = "Database libraries not installed"
        return success, df, error, row_count

    if not query or not query.strip():
        error = "No SQL query provided"
        return success, df, error, row_count

    # Check if environment variables are set
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
        error = "Database configuration incomplete. Check environment variables."
        return success, df, error, row_count

    connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    engine = None  # Initialize engine outside try block

    try:
        print(f"Connecting to database '{DB_NAME}' on {DB_HOST}:{DB_PORT}...")
        engine = create_engine(connection_string)
        print("Database engine created.")

        print(f"Executing query: {query[:150]}{'...' if len(query) > 150 else ''}")

        # Execute the query and get results as DataFrame
        with engine.connect() as connection:
            df = pd.read_sql_query(query, connection)

        print("Query executed successfully.")
        success = True
        row_count = len(df)
        print(f"Retrieved {row_count} records.")

    except SQLAlchemyError as e:
        error = f"Database error: {str(e)}"
        print(f"Error: {error}")

    except Exception as e:
        error = f"Unexpected error: {str(e)}"
        print(f"Error: {error}")

    finally:
        # Ensure engine is disposed
        if engine:
            engine.dispose()

    return success, df, error, row_count

def get_sql_result_as_dataframe(query: str) -> Tuple[pd.DataFrame, str]:
    """
    A simplified wrapper that returns the DataFrame and any error message.
    
    Args:
        query: SQL query string to execute
        
    Returns:
        Tuple containing:
        - df: Pandas DataFrame with results (empty DataFrame if error)
        - error: Error message if any, empty string otherwise
    """
    success, df, error, _ = execute_sql_query(query)
    if not success:
        return pd.DataFrame(), error or "Unknown error occurred"
    return df, ""

def execute_invoice_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Executes the generated SQL query against the PostgreSQL database
    using the query stored in state['query_code'] or state['invoice_code'].
    Returns the result data as a list of dictionaries in the state.
    """
    print("\n---NODE: Execute Invoice Query---")

    updates = {
        "po_data": None,  # This will now be a list of dictionaries
        "error": None
    }

    # Check for query in either invoice_code or query_code (support both)
    query = state.get("invoice_code") or state.get("query_code")

    if not query:
        updates["error"] = "No SQL query found in state (check invoice_code or query_code)"
        print(f"Error: {updates['error']}")
        return updates

    # Execute the query using the DataFrame function
    success, df, error, row_count = execute_sql_query(query)

    if not success:
        updates["error"] = error or "Unknown error after query execution."
        print(f"Query execution failed: {updates['error']}")
        return updates

    # Convert DataFrame to list of dictionaries for state consistency
    po_data_list = []
    if df is not None and not df.empty:
        try:
            # Convert DataFrame to list of dictionaries
            po_data_list = df.to_dict(orient='records')
            print(f"DataFrame successfully converted to list of {len(po_data_list)} dictionaries.")
        except Exception as e:
            conversion_error = f"Failed to convert DataFrame to dictionary list: {e}"
            print(f"Error: {conversion_error}")
            # Decide how to handle conversion error: error out or proceed with empty list?
            # Let's error out for now to make the issue visible.
            updates["error"] = conversion_error
            return updates

    # Set po_data to the list of dictionaries
    updates["po_data"] = po_data_list

    print(f"Successfully retrieved {row_count} records")

    # Print previews based on the DataFrame (df) before conversion
    if df is not None and not df.empty:
        print("\n=== Data Preview (DataFrame - First 5 Records) ===\n")
        # Use to_string() for better console formatting of DataFrame head
        print(df.head().to_string()) 

        print("\n=== DataFrame Info ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Optionally print a sample of the dictionary format
        if po_data_list:
             print("\n=== Dictionary Format (First Record Sample) ===")
             print(json.dumps(po_data_list[0], indent=2, default=str))


    print("---NODE: Execute Invoice Query Completed---")
    return updates


# --- Example Usage ---
if __name__ == "__main__":
    print("\n=== SQL Query Execution Example (DataFrame Output) ===\n")

    # Check for database configuration
    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
        print("ERROR: Database configuration incomplete. Please set these environment variables:")
        print("  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        exit(1)

    # Example query - customize this or take from command line
    sample_query = """
    SELECT
  po.*,
  pol.*,
  v.*
FROM
  purchase_orders po
JOIN
  purchase_order_line_items pol ON po.po_id = pol.po_id
JOIN
  vendors v ON po.vendorid = v.vendorid
WHERE
  po.ponumber = 'PO-2024-09001';
    """

    print("Executing sample query...")
    
    # Call the DataFrame output version
    success, df, error, row_count = execute_sql_query(sample_query)

    print("\n=== Query Results (DataFrame Output) ===\n")
    
    if success:
        print(f"Successfully retrieved {row_count} records")
        
        if not df.empty:
            # Display DataFrame information
            print("\nDataFrame Info:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            # Display first few rows
            print("\nDataFrame Preview (First 5 rows):")
            print(df.head().to_string())
            
            # Optional: Display specific column data
            print("\nAccessing specific columns:")
            if 'ponumber' in df.columns:
                print(f"PO Numbers: {df['ponumber'].unique()}")
            if 'totalamount' in df.columns:
                print(f"Total Amount (sum): {df['totalamount'].sum()}")
    else:
        print(f"Query failed: {error}")

    # Test node function if AppState is available
    if APP_STATE_AVAILABLE:
        print("\n\n=== Testing Node Function (now returning list[dict]) ===\n")

        # Create mock state with a query
        mock_state = {
            "query_code": sample_query,
            # Add other required state fields if needed
        }

        # Run the node function
        node_updates = execute_invoice_node(mock_state)

        print("\n--- Node Function Results ---")
        if node_updates.get("error"):
            print(f"Error: {node_updates['error']}")
        else:
            print("Node execution successful!")
            po_data = node_updates.get("po_data") # This is now list[dict]
            if po_data is not None:
                print(f"Number of records (dictionaries): {len(po_data)}")
                if len(po_data) > 0:
                     print(f"Type of po_data: {type(po_data)}")
                     print(f"Type of first element: {type(po_data[0])}")
                     print("\nComplete first record (Dictionary Format):")
                     print(json.dumps(po_data[0], indent=2, default=str))
                     print(json.dumps(po_data[1], indent=2, default=str))
                     
            else:
                 print("po_data in state is None or empty.")

    else:
        print("\nSkipping node function test as AppState is not available.")

# Helper functions for working with the DataFrame results

def get_dataframe_as_dict_list(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert a DataFrame to a list of dictionaries (one per row)
    Useful for compatibility with code expecting the old JSON format
    """
    if df is None or df.empty:
        return []
    return df.to_dict(orient='records')

def get_dataframe_as_json(df: pd.DataFrame, indent: int = 2) -> str:
    """
    Convert a DataFrame to a JSON string
    Useful for compatibility with code expecting the old JSON format
    """
    if df is None or df.empty:
        return json.dumps([], indent=indent)
    return df.to_json(orient='records', date_format='iso', indent=indent)