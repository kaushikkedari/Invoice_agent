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

# Set page configuration with custom theme
st.set_page_config(
    page_title="Invoice Cortex Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF6B35;
        color: white;
    }
    .stButton>button {
        background-color: #FF6B35;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #E85D2C;
        color: black;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---

def display_query_results(result: Dict[str, Any]):
    """Display the results from the query workflow branch."""
    st.markdown("### 📊 Query Results")
    
    # Create tabs for the different output representations
    summary_tab, graph_tab, table_tab, sql_tab = st.tabs(["📝 Summary", "📈 Graph", "📋 Table", "💻 SQL Query"])
    
    # Tab 1: Summary
    with summary_tab:
        if result.get("summary"):
            st.markdown("#### Summary")
            st.markdown(f'<div class="info-box">{result["summary"]}</div>', unsafe_allow_html=True)
        else:
            st.info("ℹ️ No summary available for this query.")
    
    # Tab 2: Graph (Visualization)
    with graph_tab:
        if result.get("plotly_chart_json"):
            try:
                chart_json = result["plotly_chart_json"]
                if isinstance(chart_json, str):
                    chart_data = json.loads(chart_json)
                elif isinstance(chart_json, dict):
                    chart_data = chart_json
                else:
                    raise TypeError("plotly_chart_json is not a string or dictionary")
                
                fig = go.Figure(data=chart_data['data'], layout=chart_data['layout'])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error displaying Plotly chart: {e}")
        else:
            st.info("ℹ️ No visualization available for this query.")
    
    # Tab 3: Table
    with table_tab:
        df_to_download = None
        if result.get("table_data"):
            try:
                table_data = result["table_data"]
                if isinstance(table_data, list) and table_data:
                    df = pd.DataFrame(table_data)
                    st.dataframe(df, use_container_width=True)
                    df_to_download = df
                elif isinstance(table_data, list) and not table_data:
                    st.info("ℹ️ The query returned an empty table_data list.")
                else:
                    st.warning(f"⚠️ Table data is not in the expected list format. Type: {type(table_data)}")
                    st.write("Attempting to display data:")
                    st.write(table_data)
            except Exception as e:
                st.error(f"❌ Error displaying table_data: {e}")
        elif result.get("result_dataframe") is not None:
            try:
                df = result["result_dataframe"]
                if isinstance(df, pd.DataFrame) and not df.empty:
                    st.dataframe(df, use_container_width=True)
                    df_to_download = df
                elif isinstance(df, pd.DataFrame) and df.empty:
                    st.info("ℹ️ The result DataFrame is empty.")
                else:
                    st.warning(f"⚠️ result_dataframe is not a DataFrame. Type: {type(df)}")
                    st.write(df)
            except Exception as e:
                st.error(f"❌ Error displaying result_dataframe: {e}")
        elif result.get("query_result") and isinstance(result["query_result"], str):
            st.text(result["query_result"])
        else:
            st.info("ℹ️ No table data available for this query.")
            
        if df_to_download is not None:
            try:
                csv_data = df_to_download.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Table Data (CSV)",
                    data=csv_data,
                    file_name=f"query_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(f"⚠️ Could not prepare table data for download: {e}")
            
    # Tab 4: SQL Query
    with sql_tab:
        if result.get("query_code"):
            st.code(result["query_code"], language="sql")
        else:
            st.info("ℹ️ No SQL query generated for this request.")

    if result.get("error"):
        st.error(f"❌ Workflow Error: {result['error']}")

# --- MODIFIED ---
def display_single_invoice_result(result: Dict[str, Any], original_filename: str):
    """Displays the results for a single processed invoice, focusing on status and discrepancies."""
    
    validation_details = result.get("validation_result")
    status = result.get("validation_status") or (validation_details.get("status") if validation_details else "unknown")
    summary = validation_details.get("summary", "") if validation_details else ""
    discrepancies = validation_details.get("discrepancies", []) if validation_details else []

    # Get the extracted invoice data and PO data for comparison
    invoice_data = result.get("extracted_invoice_data", {})
    # Corrected PO data handling:
    # For headers, we can take from the first PO line (assuming consistency for a given PO)
    po_data_for_headers = result.get("po_data", [{}])[0] if result.get("po_data") else {}
    # For line items, we need the full list of PO lines
    po_line_items_list = result.get("po_data", [])

    # --- Display Status First ---
    if status == "valid":
        st.markdown('<div class="success-box">✅ Status: VALID</div>', unsafe_allow_html=True)
        
        # --- ADDED: Display list of key validated fields ---
        st.markdown("#### Validated Fields")
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
             
        st.markdown(f'<div class="error-box">❌ Status: INVALID - {reason}</div>', unsafe_allow_html=True)
        if summary:
            st.write(f"**Summary:** {summary}")

        # Initialize verified_fields and verified_line_items at the start of this block
        verified_fields = []
        verified_line_items = []
        
        # --- Display Mismatched Fields (Discrepancies) ---
        if discrepancies:
            st.markdown("#### 🔍 Mismatched Fields")
            for item in discrepancies:
                field = item.get('field', 'Unknown Field')
                inv_val = item.get('invoice_value', 'N/A')
                po_val = item.get('po_value', 'N/A')
                notes = item.get('notes', 'No details')
                
                # Use columns for better layout
                col_field, col_inv, col_po = st.columns([2,3,3])
                with col_field:
                    st.markdown(f"**Field:** {field}")
                with col_inv:
                    st.markdown(f"**Invoice:** `{str(inv_val)}` ❌")
                with col_po:
                     st.markdown(f"**PO:** `{str(po_val)}` ✅")
                if notes:
                    st.caption(f"Note: {notes}")
                st.markdown("---") # Separator

        # --- NEW: Auto-detect matching fields ---
        if invoice_data and po_data_for_headers: # Use po_data_for_headers here
            # Check header fields
            field_mappings = {
                "purchase_order_number": "ponumber",
                "vendor_name": "vendorname", # Added Vendor Name
                "total_amount": "totalamount",
                "subtotal_amount": "subtotalamount",
                "tax_amount": "taxamount",
                "shipping_cost": "shippingcost",
                "currency": "currency"
            }
            
            for inv_field, po_field in field_mappings.items():
                inv_value = invoice_data.get(inv_field)
                po_value = po_data_for_headers.get(po_field) # Use po_data_for_headers
                if inv_value is not None and po_value is not None:
                    # Convert to same type for comparison
                    try:
                        inv_value = float(inv_value) if isinstance(inv_value, (int, float)) else str(inv_value).strip()
                        po_value = float(po_value) if isinstance(po_value, (int, float)) else str(po_value).strip()
                        if inv_value == po_value:
                            verified_fields.append({
                                "field": inv_field.replace("_", " ").title(),
                                "invoice_value": inv_value,
                                "po_value": po_value,
                                "notes": "Matches PO data"
                            })
                    except (ValueError, TypeError):
                        continue

            # Check line items
            invoice_line_items = invoice_data.get("line_items", [])
            # po_line_items = [item for item in po_data.get("line_items", []) if item.get("itemdescription")] # Old logic using incorrect po_data structure for lines
            
            for inv_item in invoice_line_items:
                # Try to find a matching PO line item
                for po_item in po_line_items_list: # Iterate through the full list of PO lines
                    inv_desc = inv_item.get("description", "").strip().lower()
                    po_desc = po_item.get("itemdescription", "").strip().lower()

                    if inv_desc and po_desc and inv_desc == po_desc:
                        current_matching_fields = []
                        # Check HSN
                        inv_hsn = inv_item.get("hsn")
                        po_hsn = po_item.get("hsn")
                        if inv_hsn is not None and po_hsn is not None and str(inv_hsn).strip() == str(po_hsn).strip():
                            current_matching_fields.append("HSN Code")
                        
                        # Check quantity (even if overall discrepancy, list if it matches for this pair)
                        inv_qty = inv_item.get("quantity")
                        po_qty = po_item.get("quantityordered")
                        try:
                            if inv_qty is not None and po_qty is not None and float(inv_qty) == float(po_qty):
                                current_matching_fields.append("Quantity")
                        except (ValueError, TypeError):
                            pass

                        # Check unit price
                        inv_up = inv_item.get("unit_price")
                        po_up = po_item.get("unitprice")
                        try:
                            if inv_up is not None and po_up is not None and float(inv_up) == float(po_up):
                                current_matching_fields.append("Unit Price")
                        except (ValueError, TypeError):
                            pass
                            
                        # Check line total
                        inv_lt = inv_item.get("line_total")
                        po_lt = po_item.get("linetotal")
                        try:
                            if inv_lt is not None and po_lt is not None and float(inv_lt) == float(po_lt):
                                current_matching_fields.append("Line Total")
                        except (ValueError, TypeError):
                            pass
                            
                        if current_matching_fields:
                            verified_line_items.append({
                                "invoice_line_description": inv_item.get("description"),
                                "matched_po_line_description": po_item.get("itemdescription"),
                                "fields_matched": current_matching_fields,
                                "notes": f"These specific fields match PO: {', '.join(current_matching_fields)}"
                            })
                        break # Found a match for inv_item, move to next inv_item

        # --- Display Verified Header Fields ---
        if verified_fields:
            st.markdown("#### ✅ Additionally Validated Fields")
            for field in verified_fields:
                st.markdown(f"- **{field['field']}** ✅")
            st.markdown("---")

        # --- Display Verified Line Item Details ---
        if verified_line_items:
            st.markdown("#### 📋 Validated Fields Line Item Details")
            for item in verified_line_items:
                st.markdown(f"**For Invoice Line Item:** `{item['invoice_line_description']}`")
                if item.get("fields_matched"):
                    st.markdown("Matching PO Details Found For:")
                    for matched_field in item["fields_matched"]:
                        st.markdown(f"- {matched_field} ✅")
                if item.get("notes"):
                    st.caption(f"Note: {item['notes']}")
                st.markdown("---")
        
    elif status == "needs_review":
         st.markdown('<div class="warning-box">⚠️ Status: NEEDS REVIEW</div>', unsafe_allow_html=True)
         if summary:
             st.write(f"**Summary:** {summary}")
         if validation_details and validation_details.get("details"):
              st.caption(f"Details: {validation_details['details']}")

    elif result.get("error"):
        st.markdown(f'<div class="error-box">🛑 Error: {result["error"]}</div>', unsafe_allow_html=True)
        st.info("Processing stopped before validation could be completed.")

    else:
        st.info(f"Status: {str(status).upper()}")
        st.write("Validation did not produce a standard result.")

# --- Main App ---
def main():
    st.markdown("# 📊 Invoice Cortex Agent")
    
    # Create tabs for the different functionalities
    query_tab, invoice_tab = st.tabs(["🔍 Data Query", "📄 Invoice Validation"])
    
    # Query Processing Tab
    with query_tab:
        st.markdown("### 🔍 Data Query")
        st.markdown("Enter a natural language query about purchase orders, vendors, or sales data.")
        
        # Text input for query
        query_text = st.text_area("Enter your query:", 
                             "Show me the top 5 vendors by total purchase order amounts.", 
                             height=100)
        
        # Process button
        if st.button("🚀 Process Query", key="process_query"):
            if query_text:
                with st.spinner("⏳ Processing query..."):
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
                         st.error(f"❌ An error occurred during workflow execution: {e}")
                         st.text(traceback.format_exc()) # Show traceback for debugging
            else:
                st.warning("⚠️ Please enter a query.")
    
    # Invoice Validation Tab
    with invoice_tab:
        st.markdown("### 📄 Invoice Validation")
        st.markdown("Upload one or more invoice images or PDFs to extract and validate against purchase order data.")
        
        # --- MODIFIED: File uploader for multiple files ---
        uploaded_files = st.file_uploader("📤 Upload Invoices (PDF, JPG, PNG)", 
                                       type=["pdf", "jpg", "jpeg", "png"],
                                       accept_multiple_files=True) # Allow multiple files
        
        # Initialize session state to store results
        if 'invoice_results' not in st.session_state:
            st.session_state.invoice_results = {}

        if uploaded_files:
             st.markdown(f"📦 Ready to process {len(uploaded_files)} file(s).")
             # Process button
             if st.button("🚀 Process Invoices", key="process_invoices"):
                 
                 # Clear previous results before processing new batch
                 st.session_state.invoice_results = {} 
                 
                 progress_bar = st.progress(0, text="⏳ Initializing...")
                 total_files = len(uploaded_files)
                 results_dict = {} # Store results here {filename: result_dict}
                 
                 with st.spinner(f"⏳ Processing {total_files} invoices sequentially..."):
                     for i, uploaded_file in enumerate(uploaded_files):
                         file_path = None # Ensure file_path is reset
                         current_file_name = uploaded_file.name
                         progress_text = f"�� Processing file {i+1}/{total_files}: {current_file_name}"
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
                             st.error(f"❌ Error processing {current_file_name}: {e}")
                             st.text(traceback.format_exc()) # Show traceback
                             # Store error information
                             results_dict[current_file_name] = {"error": f"Failed during processing: {e}"}
                         finally:
                             # Clean up the temporary file
                             if file_path and os.path.exists(file_path):
                                 try:
                                     os.unlink(file_path)
                                 except Exception as unlink_err:
                                      st.warning(f"⚠️ Could not delete temp file {file_path}: {unlink_err}")
                             # Update progress bar after processing this file is done
                             progress_bar.progress((i + 1) / total_files, text=progress_text) 
                             
                 # Store results in session state after processing all files
                 st.session_state.invoice_results = results_dict
                 progress_bar.progress(1.0, text=f"✅ Completed processing {total_files} invoice(s).") # Final progress update
                 # Consider hiding progress bar after a short delay or leaving it at 100%
                 # progress_bar.empty() # Remove progress bar after completion
                 st.success(f"🎉 Finished processing {total_files} invoice(s).")


        # --- MODIFIED: Display results from session state ---
        if st.session_state.invoice_results:
            st.markdown("### 📊 Processing Results")
            for filename, result_data in st.session_state.invoice_results.items():
                with st.expander(f"📄 Results for: {filename}", expanded=True): # Expand by default
                     # Pass the original filename for use in download buttons (though none are used now)
                     display_single_invoice_result(result_data, filename) 

if __name__ == "__main__":
    main() 