import streamlit as st
import pandas as pd
import io
import os
import tempfile
import plotly.graph_objects as go
import json
from PIL import Image
from typing import Dict, Any, List, Optional, Union
import traceback # Added for detailed error logging
import datetime # Added for filenames

# Import our workflow
from invoice_workflow import process_input

# Set page configuration
st.set_page_config(
    page_title="Invoice Cortex Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---

def display_query_results(result: Dict[str, Any]):
    """Display the results from the query workflow branch."""
    st.subheader("Query Results")
    
    # Create tabs for the different output representations
    summary_tab, graph_tab, table_tab, sql_tab = st.tabs(["Summary", "Graph", "Table", "SQL Query"])
    
    # Tab 1: Summary
    with summary_tab:
        if result.get("summary"):
            st.markdown("**Summary:**")
            st.write(result["summary"])
        else:
            st.info("No summary available for this query.")
    
    # Tab 2: Graph (Visualization)
    with graph_tab:
        if result.get("plotly_chart_json"):
            try:
                chart_json = result["plotly_chart_json"]
                # Check if chart_json is already a dict or needs loading
                if isinstance(chart_json, str):
                    chart_data = json.loads(chart_json)
                elif isinstance(chart_json, dict):
                    chart_data = chart_json # Already a dict
                else:
                    raise TypeError("plotly_chart_json is not a string or dictionary")
                
                fig = go.Figure(data=chart_data['data'], layout=chart_data['layout'])
                st.plotly_chart(fig, use_container_width=True)
            except json.JSONDecodeError as e:
                st.error(f"Error decoding Plotly chart JSON: {e}")
                st.text(result["plotly_chart_json"]) # Show raw json on error
            except Exception as e:
                st.error(f"Error displaying Plotly chart: {e}")
        else:
            st.info("No visualization available for this query.")
    
    # Tab 3: Table
    with table_tab:
        df_to_download = None # Initialize df for download button
        # Check for and display table_data (from visualize_data_node)
        if result.get("table_data"):
            try:
                table_data = result["table_data"]
                if isinstance(table_data, list) and table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df)
                    df_to_download = df # Assign for download
                elif isinstance(table_data, list) and not table_data:
                    st.info("The query returned an empty table_data list.")
                else:
                    st.warning(f"Table data is not in the expected list format. Type: {type(table_data)}")
                    # Try to display it anyway if possible
                    st.write("Attempting to display data:")
                    st.write(table_data)
            except Exception as e:
                st.error(f"Error displaying table_data: {e}")
        # Fallback: Check for result_dataframe if table_data wasn't found
        elif result.get("result_dataframe") is not None:
            try:
                df = result["result_dataframe"]
                if isinstance(df, pd.DataFrame) and not df.empty: # Ensure it's a non-empty DataFrame
                    st.dataframe(df)
                    df_to_download = df # Assign for download
                elif isinstance(df, pd.DataFrame) and df.empty:
                    st.info("The result DataFrame is empty.")
                else:
                     st.warning(f"result_dataframe is not a DataFrame. Type: {type(df)}")
                     st.write(df) # Show the raw data
            except Exception as e:
                st.error(f"Error displaying result_dataframe: {e}")
        # Display raw query result if it's just text and no dataframe/table was shown
        elif result.get("query_result") and isinstance(result["query_result"], str):
            st.text(result["query_result"])
        else:
            st.info("No table data available for this query.")
            
        # --- KEPT: Download Button for Table Data ---
        if df_to_download is not None:
            try:
                csv_data = df_to_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Table Data (CSV) 💾",
                    data=csv_data,
                    file_name=f"query_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                 st.warning(f"Could not prepare table data for download: {e}")
            
    # Tab 4: SQL Query
    with sql_tab:
        if result.get("query_code"):
            st.code(result["query_code"], language="sql")
        else:
            st.info("No SQL query generated for this request.")

    # Display any errors
    if result.get("error"):
        st.error(f"Workflow Error: {result['error']}")

# --- MODIFIED ---
def display_single_invoice_result(result: Dict[str, Any], original_filename: str):
    """Displays the results for a single processed invoice, focusing on status and discrepancies."""
    
    validation_details = result.get("validation_result")
    status = result.get("validation_status") or (validation_details.get("status") if validation_details else "unknown")
    summary = validation_details.get("summary", "") if validation_details else ""
    discrepancies = validation_details.get("discrepancies", []) if validation_details else []

    # --- Display Status First ---
    if status == "valid":
        st.success("✅ Status: VALID")
        # --- REMOVED Summary and Generic Message ---
        # if summary:
        #     st.caption(f"Summary: {summary}")
        # st.info("Key fields (Vendor, PO Number, Totals, Line Items) were compared and matched.")
        
        # --- ADDED: Display list of key validated fields ---
        st.markdown("**Validated Fields:**")
        key_validated_fields = [
            "Vendor Name",
            "PO Number",
            "Invoice Date",
            "Total Amount",
            "Line Item Quantities",
            "Line Item Prices",
            "HSN Codes (where applicable)"
        ]
        for field in key_validated_fields:
            st.markdown(f"- ✅ {field}")
        
    elif status == "invalid":
        reason = validation_details.get("reason", "") if validation_details else ""
        # Construct a concise reason if not explicitly provided
        if not reason and discrepancies:
            reason = ", ".join(list(set([d.get('field', 'Unknown Field') for d in discrepancies])))
        elif not reason:
             reason = "Validation Failed (Unknown Reason)"
             
        st.error(f"❌ Status: INVALID - {reason}")
        if summary:
            st.write(f"**Summary:** {summary}")
        
        # Show specific discrepancies if available
        if discrepancies:
            st.subheader("Discrepancies Found:")
            for item in discrepancies:
                field = item.get('field', 'Unknown Field')
                inv_val = item.get('invoice_value', 'N/A')
                po_val = item.get('po_value', 'N/A')
                notes = item.get('notes', 'No details')
                
                col1, col2, col3 = st.columns([2,3,3]) # Adjust column ratios as needed
                with col1:
                    # Add red cross next to the field name
                    st.markdown(f"**Field:** {field} ") 
                with col2:
                    st.markdown(f"**Invoice:** `{str(inv_val)} ❌`")
                with col3:
                     st.markdown(f"**PO:** `{str(po_val)} `")
                if notes:
                    st.caption(f"Note: {notes}")
                st.markdown("---") # Separator
        else:
            st.info("No specific discrepancy details were provided in the validation result, although the status is invalid.")

    elif status == "needs_review": # Handle the review status
         st.warning("⚠️ Status: NEEDS REVIEW")
         if summary:
             st.write(f"**Summary:** {summary}")
         if validation_details and validation_details.get("details"):
              st.caption(f"Details: {validation_details['details']}")
         # Optionally show raw response if available in 'details' or other field (Removed as per request)
         # if validation_details.get("raw_response"):
         #      with st.expander("Raw LLM Response Fragment (for Debugging)"):
         #           st.code(validation_details["raw_response"])

    elif result.get("error"):
        # Handle cases where validation didn't run due to prior error
        st.error(f"🛑 Error: {result['error']}")
        st.info("Processing stopped before validation could be completed.")
        # REMOVED partial data display and download button

    else: # Fallback for other unknown statuses or missing validation results
        st.info(f"Status: {str(status).upper()}")
        st.write("Validation did not produce a standard result.")
        # REMOVED JSON display

    # --- REMOVED PO Data Display ---
    # --- REMOVED Extracted Data Display and Download Button ---
    # --- REMOVED Validation Result Download Button ---

# --- Main App ---
def main():
    st.title("Invoice Cortex Agent")
    
    # Create tabs for the different functionalities
    query_tab, invoice_tab = st.tabs(["Data Query", "Invoice Validation"])
    
    # Query Processing Tab
    with query_tab:
        st.header("Data Query")
        st.write("Enter a natural language query about purchase orders, vendors, or sales data.")
        
        # Text input for query
        query_text = st.text_area("Enter your query:", 
                             "Show me the top 5 vendors by total purchase order amounts.", 
                             height=100)
        
        # Process button
        if st.button("Process Query", key="process_query"):
            if query_text:
                with st.spinner("Processing query..."):
                    try:
                        # Call the workflow
                        # Ensure the user query is passed correctly in initial state
                        initial_state = {
                            "raw_input": query_text,
                            "input_type": "text",
                            "user_query": query_text # Pass the query explicitly if needed by nodes
                        }
                        result = process_input(raw_input=query_text, input_type="text") 
                        # Display results
                        display_query_results(result)
                    except Exception as e:
                         st.error(f"An error occurred during workflow execution: {e}")
                         st.text(traceback.format_exc()) # Show traceback for debugging
            else:
                st.warning("Please enter a query.")
    
    # Invoice Validation Tab
    with invoice_tab:
        st.header("Invoice Validation")
        st.write("Upload one or more invoice images or PDFs to extract and validate against purchase order data.")
        
        # --- MODIFIED: File uploader for multiple files ---
        uploaded_files = st.file_uploader("Upload Invoices (PDF, JPG, PNG)", 
                                       type=["pdf", "jpg", "jpeg", "png"],
                                       accept_multiple_files=True) # Allow multiple files
        
        # Initialize session state to store results
        if 'invoice_results' not in st.session_state:
            st.session_state.invoice_results = {}

        if uploaded_files:
             st.write(f"Ready to process {len(uploaded_files)} file(s).")
             # Process button
             if st.button("Process Invoices", key="process_invoices"):
                 
                 # Clear previous results before processing new batch
                 st.session_state.invoice_results = {} 
                 
                 progress_bar = st.progress(0, text="Initializing...")
                 total_files = len(uploaded_files)
                 results_dict = {} # Store results here {filename: result_dict}
                 
                 with st.spinner(f"Processing {total_files} invoices sequentially..."):
                     for i, uploaded_file in enumerate(uploaded_files):
                         file_path = None # Ensure file_path is reset
                         current_file_name = uploaded_file.name
                         progress_text = f"Processing file {i+1}/{total_files}: {current_file_name}"
                         progress_bar.progress((i) / total_files, text=progress_text) # Update progress before processing
                         
                         try:
                             # Save the uploaded file to a temporary location
                             with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(current_file_name)[1]) as tmp_file:
                                 tmp_file.write(uploaded_file.getvalue())
                                 file_path = tmp_file.name
                             
                             # Determine input type (image or pdf) for the workflow
                             file_extension = os.path.splitext(current_file_name)[1].lower()
                             input_type = "image" if file_extension in ['.jpg', '.jpeg', '.png'] else "pdf"

                             # Call the workflow with the file path
                             # The workflow is designed for single input, so we call it per file
                             result = process_input(file_path, input_type)
                             
                             results_dict[current_file_name] = result
                         
                         except Exception as e:
                             st.error(f"Error processing {current_file_name}: {e}")
                             st.text(traceback.format_exc()) # Show traceback
                             # Store error information
                             results_dict[current_file_name] = {"error": f"Failed during processing: {e}"}
                         finally:
                             # Clean up the temporary file
                             if file_path and os.path.exists(file_path):
                                 try:
                                     os.unlink(file_path)
                                 except Exception as unlink_err:
                                      st.warning(f"Could not delete temp file {file_path}: {unlink_err}")
                             # Update progress bar after processing this file is done
                             progress_bar.progress((i + 1) / total_files, text=progress_text) 
                             
                 # Store results in session state after processing all files
                 st.session_state.invoice_results = results_dict
                 progress_bar.progress(1.0, text=f"Completed processing {total_files} invoice(s).") # Final progress update
                 # Consider hiding progress bar after a short delay or leaving it at 100%
                 # progress_bar.empty() # Remove progress bar after completion
                 st.success(f"Finished processing {total_files} invoice(s).")


        # --- MODIFIED: Display results from session state ---
        if st.session_state.invoice_results:
            st.subheader("Processing Results")
            for filename, result_data in st.session_state.invoice_results.items():
                with st.expander(f"Results for: {filename}", expanded=True): # Expand by default
                     # Pass the original filename for use in download buttons (though none are used now)
                     display_single_invoice_result(result_data, filename) 

if __name__ == "__main__":
    main() 