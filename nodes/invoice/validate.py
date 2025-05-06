import os
import sys
import json
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# LangChain & LLM Imports
# from langchain_google_genai import ChatGoogleGenerativeAI # REMOVED
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

# Try importing AppState
try:
    from mypackage.state import AppState
    APP_STATE_AVAILABLE = True
except ImportError:
    APP_STATE_AVAILABLE = False
    print("Warning: AppState not available. Standalone mode only.")
    # Define a minimal mock state for standalone execution
    class AppState(dict): pass

load_dotenv()

# --- REMOVE Old LLM Configuration ---
# try:
#     GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
#     if not GOOGLE_API_KEY:
#         raise ValueError("GOOGLE_API_KEY environment variable not set.")
#     llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=GOOGLE_API_KEY, temperature=0.1)
# except Exception as e:
#     print(f"CRITICAL ERROR: Failed to initialize LLM - {e}")
#     raise SystemExit(f"LLM initialization failed: {e}")

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    if ai_component.llm is None:
         # Raise error here because validation relies on the LLM
         raise ValueError("Failed to initialize LLM via AIComponent for validation. Check config and .env.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize AIComponent for validation - {e}")
    raise SystemExit(f"AIComponent initialization failed: {e}")


# --- Validation Prompt ---
VALIDATION_PROMPT_TEMPLATE = """You are an expert Accounts Payable assistant AI. Your task is to validate an extracted invoice against the corresponding Purchase Order (PO) data retrieved from the database.

**Invoice Data:**
```json
{invoice_data_json}
```

**Purchase Order (PO) Data:**
(Note: This contains details for each line item found for the PO number)
```json
{po_data_json}
```

**Validation Criteria & Rules for Reporting:**
1.  **PO Number Match:** Confirm the `purchase_order_number` in the invoice matches the `ponumber` in the PO data.
2.  **Vendor Match:** Compare the `vendor_name` (and potentially address details if available) between the invoice and PO data (vendors table fields like `vendorname`).
3.  **Line Item Details:** For each line item in the invoice:
    *   Attempt to match it with a corresponding line item in the PO data based on description and/or `hsn` (`itemdescription`, `items.hsn` which should be joined into the PO data).
    *   **HSN Code:** If an `hsn` value is present in the invoice line item, it **MUST** exactly match the `hsn` value for the corresponding matched PO item. If not, this is a **major discrepancy** for that line item.
    *   **Quantity:** Compare `quantity` (invoice) with `quantityordered` (PO). If they differ, this is a discrepancy for that line item.
    *   **Unit Price:** Compare `unit_price`. Minor rounding (< $0.01) is acceptable; significant differences are discrepancies for that line item.
    *   **Line Total:** Compare `line_total`. Minor rounding (< $0.01) is acceptable; significant differences are discrepancies for that line item.
    *   Flag any invoice line items that cannot be matched to a PO line item.
    *   Flag any PO line items not present on the invoice (unless PO status indicates partial fulfillment; assume full invoice expected for now).
4.  **Overall Totals:** Compare `subtotal_amount`, `tax_amount`, and `total_amount`. Minor rounding (< $0.01) is acceptable; significant differences are discrepancies.
5.  **Invoice Date:** Compare `invoice_date` with `orderdate` from PO. If dates are semantically different (not just format), it's a discrepancy.
6.  **Invoice Number:** Note `invoice_number` from invoice (for record-keeping).
7.  **Expected Delivery Date:** Compare `expected_delivery_date` (invoice) with `deliverydate` or `expecteddeliverydate` (PO).
8.  **PO Status Check:** If PO `status` is 'Closed' or 'Fully Received', an invoice for it is generally a discrepancy.

**Output Format & Instructions:**
Respond ONLY with a valid JSON object. Ensure all string values in the JSON are properly escaped.

*   `status`: String - "valid" or "invalid".
    *   "valid": All critical details (PO Number, Vendor, Totals, HSN Codes, Line Item Quantities/Prices) match within acceptable tolerances.
    *   "invalid": Any significant discrepancy found.
*   `summary`: String - A brief (1-2 sentence) summary. If invalid, **clearly state the primary reason(s) based ONLY on the actual listed discrepancies** (e.g., "Invalid due to HSN mismatch on one line item and total amount difference.").
*   `discrepancies`: List[Dict] - **ONLY list actual differences here.**
    *   Each dictionary MUST represent a **genuine mismatch** between the invoice and PO data.
    *   `field`: String - Specific field with the discrepancy (e.g., "Total Amount", "Line Item Quantity for 'Product X'", "Vendor Name", "HSN Code for 'Product Y'"). Be precise, especially for line items (e.g., reference the item by its description).
    *   `invoice_value`: The actual value from the invoice that is mismatched.
    *   `po_value`: The actual value from the PO that it was compared against.
    *   `notes`: String - **Explanation of WHY this is a discrepancy.** Example: "Invoice quantity (10) is less than PO quantity (12)." **CRITICAL: If `invoice_value` and `po_value` in this entry are identical, DO NOT list it as a discrepancy; your reasoning is flawed. Find the true discrepancy or state that the values matched if they did.**
*   `verified_header_fields`: List[Dict] (OPTIONAL - include if `status` is 'invalid' but some header fields *do* match).
    *   List header-level fields (PO Number, Vendor Name, Invoice Date, Totals) that **exactly matched** their PO counterparts.
    *   Each dictionary: `field`, `invoice_value`, `po_value`, `notes` (e.g., "Matches PO data.").
*   `verified_line_item_details`: List[Dict] (OPTIONAL - include if `status` is 'invalid' but some line items have *partially* matching details).
    *   For an invoice line item matched to a PO line item: if some sub-fields (e.g., quantity, unit price) match, but another (e.g., HSN) is a discrepancy for *that same line item*, list the matching parts here.
    *   Each dictionary: `invoice_line_description`, `matched_po_line_description`, `fields_matched` (list, e.g., ["Quantity", "Unit Price"]), `notes` (e.g., "Quantity and price match PO, but HSN code (see discrepancies) was incorrect.").
*   `confidence`: Float (0.0 to 1.0) - Your confidence in the `status`.

**Example of a Valid Response:**
```json
{{
  "status": "valid",
  "summary": "Invoice details align perfectly with the purchase order.",
  "discrepancies": [],
  "confidence": 0.99
}}
```

**Example of an Invalid Response (Updated):**
```json
{{
  "status": "invalid",
  "summary": "Invalid due to HSN code mismatch for 'Widget Type B' and Vendor Name incorrect. PO number and totals match. 'Widget Type A' quantity/price also match.",
  "discrepancies": [
    {{
      "field": "Vendor Name",
      "invoice_value": "Wrong Vendor Inc.",
      "po_value": "Supplier Corp",
      "notes": "Vendor name on invoice (Wrong Vendor Inc.) does not match PO (Supplier Corp)."
    }},
    {{
      "field": "HSN Code for 'Widget Type B'",
      "invoice_value": "HSN-INV-002",
      "po_value": "HSN-PO-001",
      "notes": "HSN code on invoice (HSN-INV-002) for 'Widget Type B' does not match PO HSN (HSN-PO-001)."
    }}
    // IF quantity for Widget Type B was ALSO a mismatch, it would be a SEPARATE entry:
    // {{
    //   "field": "Line Item Quantity for 'Widget Type B'",
    //   "invoice_value": 5,
    //   "po_value": 10,
    //   "notes": "Invoice quantity (5) for 'Widget Type B' does not match PO quantity (10)."
    // }}
  ],
  "verified_header_fields": [
    {{
      "field": "Purchase Order Number",
      "invoice_value": "PO12345",
      "po_value": "PO12345",
      "notes": "Matches PO data."
    }},
    {{
      "field": "Total Amount",
      "invoice_value": 150.75,
      "po_value": 150.75,
      "notes": "Matches PO data."
    }}
  ],
  "verified_line_item_details": [
    {{
      "invoice_line_description": "Widget Type A", // Assuming Widget Type A had no discrepancies
      "matched_po_line_description": "Widget Type A",
      "fields_matched": ["Quantity", "Unit Price", "Line Total", "HSN Code"],
      "notes": "All details for this line item match PO."
    }}
    // Widget Type B would NOT be in verified_line_item_details if its HSN was a discrepancy, 
    // because the goal of this section is to highlight items that have some matching parts DESPITE an issue on the SAME line.
    // If Widget Type B quantity ALSO mismatched, it would be in discrepancies.
    // If Widget Type B quantity MATCHED but HSN mismatched, then it *could* appear here with fields_matched: ["Quantity"], and notes clarifying HSN is in discrepancies.
  ],
  "confidence": 0.85
}}
```

Analyze the provided data and generate the validation JSON object. **Double-check that any field listed in `discrepancies` truly represents a difference between invoice and PO values, and that your `notes` accurately reflect this difference.**
"""

# --- LangGraph Node ---

def validate_invoice_node(state: AppState) -> Dict[str, Any]:
    """
    Compares extracted invoice data with PO data using an LLM to validate.
    Handles the case where the PO number was not found in the database.
    Outputs only 'valid' or 'invalid' status.
    """
    print("---NODE: Validate Invoice---")

    updates: Dict[str, Any] = {
        "validation_status": None,
        "validation_result": None,
        "error": None
    }

    # Get data from state
    extracted_invoice_data = state.get("extracted_invoice_data")
    po_data = state.get("po_data") # Expected to be list[dict] or None/empty list

    # --- Input Checks ---
    if not extracted_invoice_data:
        updates["error"] = "Validation Error: Missing extracted invoice data in state."
        updates["validation_status"] = "invalid" # Default to invalid if core data missing
        updates["validation_result"] = {"reason": "Missing Invoice Data", "details": updates["error"]}
        print(f"Error: {updates['error']}")
        return updates

    invoice_po_number = extracted_invoice_data.get("purchase_order_number")
    if not invoice_po_number:
        # If the *invoice* itself is missing a PO number, it's invalid for PO matching.
        updates["validation_status"] = "invalid"
        updates["validation_result"] = {
            "reason": "Missing PO Number in Invoice",
            "summary": "The extracted invoice does not contain a Purchase Order number. Cannot validate against PO data.",
            "discrepancies": [{
                "field": "Purchase Order Number",
                "invoice_value": None,
                "po_value": None,
                "notes": "PO number missing from invoice."
            }]
        }
        print("Error: Invoice missing PO number. Cannot perform PO validation.")
        return updates

    # --- Handle Invalid PO Number Case ---
    # Check if po_data is None or an empty list
    if not po_data:
        print(f"PO Data for Invoice PO '{invoice_po_number}' not found in database.")
        updates["validation_status"] = "invalid"
        updates["validation_result"] = {
            "reason": "Invalid Purchase Order Number",
            "summary": f"Purchase Order Number '{invoice_po_number}' provided in the invoice was not found in the database records.",
            "discrepancies": [{
                "field": "Purchase Order Number",
                "invoice_value": invoice_po_number,
                "po_value": None,
                "notes": "PO number from invoice not found in database."
            }]
        }
        print("---NODE: Validate Invoice Completed (PO Not Found - Invalid)---")
        return updates

    # --- Proceed with LLM Validation --- (Uses ai_component)
    try:
        # Prepare data for the prompt
        # Ensure data is JSON serializable (handle dates, decimals if necessary)
        invoice_data_json = json.dumps(extracted_invoice_data, indent=2, default=str)
        po_data_json = json.dumps(po_data, indent=2, default=str)

        # Setup prompt and parser
        prompt = ChatPromptTemplate.from_template(VALIDATION_PROMPT_TEMPLATE)
        json_parser = JsonOutputParser()
        # --- Use AIComponent for the LLM call ---
        # chain = prompt | llm | json_parser # OLD
        chain = prompt | ai_component.llm | json_parser # NEW

        print("Sending data to LLM (via AIComponent) for validation...")
        llm_result = chain.invoke({
            "invoice_data_json": invoice_data_json,
            "po_data_json": po_data_json
        })

        print("LLM validation result received.")
        # Basic check on the structure
        if isinstance(llm_result, dict) and "status" in llm_result and "summary" in llm_result:
            # Ensure status is only 'valid' or 'invalid'
            llm_status = llm_result.get("status")
            if llm_status not in ["valid", "invalid"]:
                print(f"Warning: LLM returned unexpected status '{llm_status}'. Defaulting to 'invalid'.")
                llm_status = "invalid"
                llm_result["status"] = "invalid" # Correct the result dict as well
                llm_result["summary"] = f"Original LLM status was '{llm_status}'. Reviewed and marked as invalid. {llm_result.get('summary', '')}"

            updates["validation_status"] = llm_status
            updates["validation_result"] = llm_result
            print(f"Validation Status: {updates['validation_status']}")
            print(f"Validation Summary: {llm_result.get('summary')}")
        else:
            raise OutputParserException("LLM response missing required keys ('status', 'summary').")

    except OutputParserException as e:
        error_msg = f"Validation LLM Error: Failed to parse LLM response - {e}"
        print(f"Error: {error_msg}")
        updates["error"] = error_msg
        # Fallback status to invalid
        # Fallback status
        updates["validation_status"] = "needs_review"
        updates["validation_result"] = {
            "reason": "LLM Parsing Error",
            "details": error_msg,
            "raw_response": str(e) # Include raw error if possible
        }
    except Exception as e:
        error_msg = f"Validation LLM Error: An unexpected error occurred - {e}"
        print(f"Error: {error_msg}")
        updates["error"] = error_msg
        # Fallback status
        updates["validation_status"] = "needs_review"
        updates["validation_result"] = {
            "reason": "Unexpected LLM Error",
            "details": error_msg
        }

    print("---NODE: Validate Invoice Completed---")
    return updates

# --- Example Usage ---
if __name__ == "__main__":
    print("=== Invoice Validation Node Example ===")

    # --- Actual Data from Previous Steps ---

    # Data extracted from extract_invoice.py (lines 407-442)
    actual_extracted_invoice_data = {
  "invoice_number": "INV-2024-ERR-001",
  "invoice_date": "2024-03-24",
  "Invoice_ate": None,
  "vendor_name": "Regional Office Supplies",
  "vendor_city": "Mumbai",
  "vendor_country": None,
  "purchase_order_number": "PO-2024-09001",
  "payment_terms": "Net 30",
  "subtotal_amount": 15612.0,
  "tax_amount": 2341.8,
  "shipping_cost": 500.0,
  "total_amount": 18453.8,
  "line_items": [
    {
      "description": "A4 Printer Paper (Ream, 500 Sheets)",
      "sku": None,
      "quantity": 12,
      "unit_of_measure": None,
      "unit_price": 672.0,
      "line_total": 8064.0
    },
    {
      "description": "Desk Stapler - Standard",
      "sku": None,
      "quantity": 6,
      "unit_of_measure": None,
      "unit_price": 1258.0,
      "line_total": 7548.0
    }
  ]
}

      

    # Data retrieved from execute_invoice.py (lines 274-382)
    actual_po_data = [
        {
  "po_id": 9001,
  "ponumber": "PO-2024-09001",
  "vendorid": 1002,
  "orderdate": "2024-03-10",
  "expecteddeliverydate": "2024-03-25",
  "shippingaddressline1": "1 Corporate Drive",
  "shippingaddressline2": "Tech Park One",
  "shippingcity": "Pimpri-Chinchwad",
  "shippingstate": "Maharashtra",
  "shippingpostalcode": "411019",
  "shippingcountry": "India",
  "paymentterms": "Net 30",
  "subtotalamount": 13010.0,
  "taxamount": 2341.8,
  "shippingcost": 500.0,
  "totalamount": 15851.8,
  "currency": "INR",
  "status": "Fully Received",
  "buyercontactperson": "Amit Singh",
  "notes": None,
  "createdat": "2023-02-15 09:30:00",
  "updatedat": "2025-04-01 14:00:00",
  "polineid": 12001,
  "linenumber": 1,
  "itemid": 502,
  "itemdescription": "A4 Printer Paper (Ream, 500 Sheets)",
  "quantityordered": 10,
  "unitofmeasure": "Ream",
  "unitprice": 672.0,
  "linetotal": 6720.0,
  "lineexpecteddeliverydate": None,
  "quantityreceived": 10,
  "actualdeliverydate": "2024-03-24",
  "linestatus": "Fully Received",
  "vendorname": "Regional Office Supplies",
  "contactperson": "Ankit Patel",
  "email": "sales@regionaloffice.in",
  "phonenumber": "+91 22 2800 5678",
  "addressline1": "15 Andheri Ind Est",
  "addressline2": None,
  "city": "Mumbai",
  "state": "Maharashtra",
  "postalcode": "400059",
  "country": "India",
  "region": "APAC",
  "taxid": "27AAAPR5678E2Z9",
  "defaultpaymentterms": "Net 30",
  "vendorrating": 4.0
},
{
  "po_id": 9001,
  "ponumber": "PO-2024-09001",
  "vendorid": 1002,
  "orderdate": "2024-03-10",
  "expecteddeliverydate": "2024-03-25",
  "shippingaddressline1": "1 Corporate Drive",
  "shippingaddressline2": "Tech Park One",
  "shippingcity": "Pimpri-Chinchwad",
  "shippingstate": "Maharashtra",
  "shippingpostalcode": "411019",
  "shippingcountry": "India",
  "paymentterms": "Net 30",
  "subtotalamount": 13010.0,
  "taxamount": 2341.8,
  "shippingcost": 500.0,
  "totalamount": 15851.8,
  "currency": "INR",
  "status": "Fully Received",
  "buyercontactperson": "Amit Singh",
  "notes": None,
  "createdat": "2023-02-15 09:30:00",
  "updatedat": "2025-04-01 14:00:00",
  "polineid": 12002,
  "linenumber": 2,
  "itemid": 513,
  "itemdescription": "Desk Stapler - Standard",
  "quantityordered": 5,
  "unitofmeasure": "Each",
  "unitprice": 1258.0,
  "linetotal": 6290.0,
  "lineexpecteddeliverydate": None,
  "quantityreceived": 5,
  "actualdeliverydate": "2024-03-24",
  "linestatus": "Fully Received",
  "vendorname": "Regional Office Supplies",
  "contactperson": "Ankit Patel",
  "email": "sales@regionaloffice.in",
  "phonenumber": "+91 22 2800 5678",
  "addressline1": "15 Andheri Ind Est",
  "addressline2": None,
  "city": "Mumbai",
  "state": "Maharashtra",
  "postalcode": "400059",
  "country": "India",
  "region": "APAC",
  "taxid": "27AAAPR5678E2Z9",
  "defaultpaymentterms": "Net 30",
  "vendorrating": 4.0
}
]

    # --- Run Test with Actual Data ---
    print("--- Test Case with Actual Data ---")
    # Use the actual data loaded above
    state_actual = AppState(extracted_invoice_data=actual_extracted_invoice_data, po_data=actual_po_data)
    result_actual = validate_invoice_node(state_actual)
    print("Result (Actual Data):")
    print(json.dumps(result_actual, indent=2, default=str))

    # Keep the commented out sections for other test cases if needed later
    # print("--- Test Case 2: Invalid PO Number ---")
    # ... (code remains commented out)

    # print("--- Test Case 3: Mismatched Data ---")
    # ... (code remains commented out)

    print("=== Validation Node Example Completed ===")
