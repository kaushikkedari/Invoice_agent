import streamlit as st
import pandas as pd
import io
import os
import tempfile
import plotly.graph_objects as go
import json
from PIL import Image
from typing import Dict, Any, List, Optional, Union

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
        # Check for and display table_data (from visualize_data_node)
        if result.get("table_data"):
            try:
                table_data = result["table_data"]
                if isinstance(table_data, list) and table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df)
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
                if not df.empty:
                    st.dataframe(df)
                else:
                    st.info("The dataframe is empty.")
            except Exception as e:
                st.error(f"Error displaying result_dataframe: {e}")
        # Display raw query result if it's just text and no dataframe/table was shown
        elif result.get("query_result") and isinstance(result["query_result"], str):
            st.text(result["query_result"])
        else:
            st.info("No table data available for this query.")
    
    # Tab 4: SQL Query
    with sql_tab:
        if result.get("query_code"):
            st.code(result["query_code"], language="sql")
        else:
            st.info("No SQL query generated for this request.")

    # Display any errors
    if result.get("error"):
        st.error(f"Workflow Error: {result['error']}")

def display_invoice_results(result: Dict[str, Any]):
    """Display the results from the invoice workflow branch."""
    
    # Display PO data
    if result.get("po_data"):
        st.subheader("Purchase Order Data")
        try:
            # Convert list of dictionaries to DataFrame for better display
            po_df = pd.DataFrame(result["po_data"])
            st.dataframe(po_df)
        except Exception as e:
            st.error(f"Error displaying PO data: {e}")
            st.json(result["po_data"])
    
    # Display validation results
    if result.get("validation_result"):
        st.subheader("Validation Result")
        
        validation_details = result["validation_result"] # Contains status, summary, discrepancies etc.
        status = result.get("validation_status") or validation_details.get("status", "unknown") # Get status reliably
        
        # Get discrepancies if available
        discrepancies = validation_details.get("discrepancies", [])
        
        # Prepare matched and mismatched field lists
        matched_fields = []
        mismatched_fields = []
        invoice_data = result.get("extracted_invoice_data", {})
        
        if invoice_data:
            if discrepancies:
                discrepancy_fields_lower = [d.get('field', '').lower() for d in discrepancies]
                
                # Populate mismatched fields from discrepancies
                for item in discrepancies:
                    mismatched_fields.append({
                        'field': item.get('field', 'Unknown Field'),
                        'invoice_value': item.get('invoice_value', 'N/A'),
                        'po_value': item.get('po_value', 'N/A'),
                        'reason': item.get('notes', 'Mismatch')
                    })
                
                # Populate matched fields (fields in invoice but not in discrepancies)
                if result.get("po_data"):
                    po_data = result.get("po_data", [{}])[0] # Assuming single PO match for simplicity
                    for key, inv_value in invoice_data.items():
                        if key != "line_items" and key.lower() not in discrepancy_fields_lower:
                            po_value = "N/A"
                            if key in po_data:
                                po_value = po_data[key]
                            elif key.replace("_", "") in po_data:
                                po_value = po_data[key.replace("_", "")]
                            matched_fields.append({
                                'field': key,
                                'invoice_value': inv_value,
                                'po_value': po_value,
                                'reason': 'Match'
                            })
            # If no discrepancies explicitly listed, populate based on status
            elif status == "invalid":
                 for key, value in invoice_data.items():
                     if key != "line_items":
                         mismatched_fields.append({
                            'field': key,
                            'invoice_value': value,
                            'po_value': 'Not found in PO',
                            'reason': validation_details.get("reason", "PO data not available")
                        })
            elif status == "valid":
                 if result.get("po_data"):
                     po_data = result.get("po_data", [{}])[0]
                     for key, inv_value in invoice_data.items():
                        if key != "line_items":
                            po_value = "N/A"
                            if key in po_data:
                                po_value = po_data[key]
                            elif key.replace("_", "") in po_data:
                                po_value = po_data[key.replace("_", "")]
                            matched_fields.append({
                                'field': key,
                                'invoice_value': inv_value,
                                'po_value': po_value,
                                'reason': 'Match'
                            })
                 else:
                      for key, value in invoice_data.items():
                         if key != "line_items":
                            matched_fields.append({
                                'field': key,
                                'invoice_value': value,
                                'po_value': 'N/A',
                                'reason': 'Assumed Match (no PO data)'
                            })

        # --- Display Logic --- 
        
        if status == "valid":
            st.success("Status: VALID")
            if validation_details.get("summary"):
                st.caption(f"Summary: {validation_details['summary']}")
            
            # --- VALID INVOICE: Show ONLY List of Matched Field Names --- 
            if matched_fields:
                st.subheader("Validated Fields")
                # Iterate and display each matched field name with a checkmark
                for item in matched_fields:
                    st.markdown(f"✅ {item['field']}")
            else:
                st.info("No matched fields found to display.")
                
        elif status == "invalid":
            reason = validation_details.get("reason")
            summary = validation_details.get("summary", "No summary provided.")
            
            # Construct a concise reason
            if reason:
                 display_reason = reason
            elif discrepancies:
                 display_reason = ", ".join([d.get('field', 'Unknown Field') for d in discrepancies[:2]]) + ("..." if len(discrepancies) > 2 else "")
            else:
                 display_reason = "Validation Failed"
                 
            st.error(f"Status: INVALID - {display_reason}")
            st.write(f"**Summary:** {summary}")
            
            # --- INVALID INVOICE: Show ONLY List of Mismatched Field Details --- 
            if mismatched_fields:
                st.subheader("Discrepancies")
                # Iterate and display each mismatched field comparison
                for item in mismatched_fields:
                    st.markdown(f"**{item['field']} Invoice:** {item['invoice_value']}")
                    st.markdown(f"**{item['field']} PO:** {item['po_value']}")
                    st.markdown(f"**Reason:** {item['reason']}")
                    st.markdown("---") # Add a separator
            else:
                st.info("No specific discrepancies identified, but the overall status is invalid.")
        
        else: # Fallback for unknown status
            st.info(f"Status: {str(status).upper()}")
            # Display the raw JSON as a fallback
            st.json(validation_details) 
            
    # Display errors if any
    if result.get("error"):
        st.error(f"Error: {result['error']}")

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
                         import traceback
                         st.text(traceback.format_exc()) # Show traceback for debugging
            else:
                st.warning("Please enter a query.")
    
    # Invoice Validation Tab
    with invoice_tab:
        st.header("Invoice Validation")
        st.write("Upload an invoice image or PDF to extract and validate against purchase order data.")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload Invoice (PDF, JPG, PNG)", 
                                       type=["pdf", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded file
            if uploaded_file.type.startswith('image/'):
                st.image(Image.open(uploaded_file), caption="Uploaded Invoice", use_container_width=True)
            elif uploaded_file.type == 'application/pdf':
                st.write("PDF uploaded successfully")
            
            # Process button
            if st.button("Process Invoice", key="process_invoice"):
                with st.spinner("Processing invoice..."):
                    # Save the uploaded file to a temporary location
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        file_path = tmp_file.name
                    
                    try:
                        # Call the workflow with the file path
                        result = process_input(file_path, "image")
                        
                        # Display results
                        display_invoice_results(result)
                    except Exception as e:
                        st.error(f"Error processing invoice: {e}")
                    finally:
                        # Clean up the temporary file
                        if os.path.exists(file_path):
                            os.unlink(file_path)

if __name__ == "__main__":
    main() 