# Invoice Validation & Query Processing System

A complete LangGraph-based system for processing and validating invoices against purchase order data, as well as executing natural language queries against a database.

## Features

### Query Processing Branch
- Natural language to SQL query conversion
- Intelligent schema-based query generation
- Query execution against PostgreSQL database
- Results visualization in tables and charts

### Invoice Processing Branch
- PDF/Image invoice extraction
- Automated validation against Purchase Order data
- Line item matching
- Detailed validation reporting

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL database with the appropriate schema
- Google Gemini API key

### Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-directory]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with the following variables:
```
GEMINI_API_KEY=your_gemini_api_key
DB_HOST=your_db_host
DB_PORT=your_db_port
DB_NAME=your_db_name
DB_USER=your_db_username
DB_PASSWORD=your_db_password
```

## Usage

### Running the Streamlit UI

Run the Streamlit application:
```bash
streamlit run app_ui.py
```

This will open a web interface with two tabs:
1. **SQL Query Processing**: Enter natural language queries about purchase orders, vendors, or sales.
2. **Invoice Validation**: Upload invoices (PDF/images) to validate against purchase order data.

### Workflow Architecture

The system uses LangGraph to create a workflow with two main branches:

#### Query Branch
- `analyze_query_node`: Parses user query to structured format
- `generate_query_node`: Generates SQL based on analysis
- `execute_query_node`: Executes SQL against database

#### Invoice Branch
- `classification_node`: Determines input type
- `extract_invoice_node`: Extracts data from PDF/images
- `load_po_data_node`: Generates SQL to fetch related PO data
- `execute_invoice_node`: Fetches PO data from database
- `validate_invoice_node`: Validates invoice against PO data

## Project Structure

```
├── app_ui.py                  # Streamlit UI
├── invoice_workflow.py        # LangGraph workflow definition
├── requirements.txt           # Dependencies
├── .env                       # Environment variables (create this)
├── nodes/                     # LangGraph node implementations
│   ├── invoice/               # Invoice processing nodes
│   │   ├── extract_invoice.py # Invoice data extraction
│   │   ├── query_po_data.py   # SQL generation for PO data
│   │   ├── execute_invoice.py # Database query execution
│   │   └── validate.py        # Invoice validation logic
│   └── query/                 # Query processing nodes
│       ├── anlyze_query.py    # Query analysis
│       ├── generate_query.py  # SQL generation
│       └── execute_query.py   # Query execution
├── input_router.py            # Input classification node
├── mypackage/                 # Common utilities
│   └── state.py               # State definition
└── vectordb/                  # Vector database for schema
    ├── load_vectordb.py       # Vector DB loading utilities
    └── database_schema.json   # Database schema description
```

## Running from Command Line

You can also run the workflow directly:

```bash
python invoice_workflow.py "Show me the top 5 vendors by total purchase order amounts"
# or
python invoice_workflow.py path/to/invoice.pdf
``` 