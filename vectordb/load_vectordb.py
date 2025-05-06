# load the JSON file into python dictionary
import json
import os
from typing import Dict, List, Any
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

def load_schema_to_vectordb(schema_path: str = 'vectordb\database_schema.json') -> FAISS:
    """
    Loads database schema from JSON and converts it into a FAISS vector database
    for semantic retrieval during query processing.
    
    Args:
        schema_path: Path to the database schema JSON file
    
    Returns:
        FAISS vector store containing schema information
    """
    # Load the JSON file into python dictionary
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    
    # Initialize embeddings model
    embed_model = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    
    # Extract schema information into text chunks for embedding
    schema_texts = []
    schema_metadatas = []
    
    # Process tables and their columns
    for table in schema['database_schema']['tables']:
        table_name = table['table_name']
        table_desc = table['description']
        
        # Create a document for the table itself
        table_text = f"Table: {table_name}\nDescription: {table_desc}\n"
        schema_texts.append(table_text)
        schema_metadatas.append({"type": "table", "name": table_name})
        
        # Create documents for each column
        for column in table['columns']:
            col_name = column['column_name']
            col_type = column['data_type']
            constraints = column.get('constraints', [])
            col_desc = column.get('description', '')
            
            constraints_str = ", ".join(constraints) if constraints else "None"
            col_text = f"Table: {table_name}, Column: {col_name}\nData Type: {col_type}\nConstraints: {constraints_str}\nDescription: {col_desc}\n"
            
            schema_texts.append(col_text)
            schema_metadatas.append({
                "type": "column", 
                "table": table_name, 
                "name": col_name, 
                "data_type": col_type
            })
    
    # Process relationships
    for rel in schema['database_schema']['relationships']:
        from_table = rel['from_table']
        from_col = rel['from_column']
        to_table = rel['to_table']
        to_col = rel['to_column']
        rel_desc = rel['description']
        
        rel_text = f"Relationship: {from_table}.{from_col} -> {to_table}.{to_col}\nDescription: {rel_desc}\n"
        
        schema_texts.append(rel_text)
        schema_metadatas.append({
            "type": "relationship",
            "from_table": from_table,
            "from_column": from_col,
            "to_table": to_table,
            "to_column": to_col
        })
    
    # Create FAISS vector store
    vectorstore = FAISS.from_texts(
        texts=schema_texts,
        embedding=embed_model,
        metadatas=schema_metadatas
    )
    
    print(f"Successfully created FAISS vector store with {len(schema_texts)} schema elements")
    return vectorstore

def get_relevant_schema(vectorstore: FAISS, query: str, k: int = 10) -> str:
    """
    Retrieves schema information relevant to the query from the vector store.
    
    Args:
        vectorstore: FAISS vector store containing schema information
        query: User's natural language query
        k: Number of relevant documents to retrieve
        
    Returns:
        Formatted string containing relevant schema information
    """
    # Search the vector store for relevant schema information
    results = vectorstore.similarity_search(query, k=k)
    
    # Format the results into a structured schema description
    formatted_schema = "DATABASE SCHEMA:\n\n"
    
    # Track what we've added to avoid duplicates
    added_tables = set()
    added_relationships = []
    
    for doc in results:
        content = doc.page_content
        metadata = doc.metadata
        
        if metadata["type"] == "table" and metadata["name"] not in added_tables:
            formatted_schema += f"{content}\n"
            added_tables.add(metadata["name"])
        
        elif metadata["type"] == "column" and metadata["table"] in added_tables:
            formatted_schema += f"{content}\n"
        
        elif metadata["type"] == "relationship":
            rel_key = f"{metadata['from_table']}.{metadata['from_column']} -> {metadata['to_table']}.{metadata['to_column']}"
            if rel_key not in added_relationships:
                formatted_schema += f"{content}\n"
                added_relationships.append(rel_key)
    
    return formatted_schema

# Example usage
if __name__ == "__main__":
    # Create vector database from schema
    vectorstore = load_schema_to_vectordb()
    
    # Test retrieval with sample query
    test_query = "What were the total sales for each vendor last month?"
    relevant_schema = get_relevant_schema(vectorstore, test_query)
    print("\nRelevant Schema for Query:")
    print(relevant_schema)
    
    # Save the vector store for later use
    vectorstore.save_local("schema_vectorstore")