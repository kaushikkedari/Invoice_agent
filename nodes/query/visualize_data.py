import os
import io
import base64
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # Import seaborn for better plots
from typing import Dict, Any, Optional, List # Ensure List is imported
import traceback # For detailed error logging
import plotly.express as px
import plotly.graph_objects as go
import json
from decimal import Decimal

# --- Assume these are imported/defined elsewhere ---
# from .utils import analyze_dataframe_for_plot, generate_chosen_visualization, format_as_text_table # Example import
# from .llm_setup import llm, summarize_prompt # Example import
from mypackage.state import AppState
from langchain_core.output_parsers import StrOutputParser # For summary chain

# --- NEW: Import AIComponent ---
from mypackage.llm_provider import AIComponent

# --- Placeholder Definitions (Replace with your actual functions) ---

# Make sure these functions are defined or imported correctly based on your project structure
# For demonstration, including simplified versions here. USE YOUR FULL VERSIONS.

def analyze_dataframe_for_plot(df: pd.DataFrame) -> Dict[str, Any]:
    """Placeholder: Replace with your full analysis function."""
    print("INFO: Using placeholder analyze_dataframe_for_plot")
    if df is None or df.empty: return {'type': 'none', 'reason': 'No data'}
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not numeric_cols: return {'type': 'none', 'reason': 'No numeric cols'}
    if len(df.columns) >= 2:
        return {'type': 'bar', 'x_col': df.columns[0], 'y_col': numeric_cols[0], 'reason': 'Placeholder: Found category & numeric'}
    else:
        return {'type': 'histogram', 'col': numeric_cols[0], 'reason': 'Placeholder: Single numeric col'}

def generate_chosen_visualization(df: pd.DataFrame, plot_info: Dict[str, Any]) -> Optional[str]:
    """Placeholder: Replace with your full visualization function."""
    print(f"INFO: Using placeholder generate_chosen_visualization for type {plot_info.get('type')}")
    plot_type = plot_info.get('type', 'none')
    if plot_type == 'none' or df.empty: return None
    try:
        plt.figure(figsize=(8, 5)) # Smaller default size maybe
        if plot_type == 'bar':
            sns.barplot(x=plot_info['x_col'], y=plot_info['y_col'], data=df.head(15))
            plt.xticks(rotation=45, ha='right')
        elif plot_type == 'histogram':
            sns.histplot(df[plot_info['col']], kde=True)
        else: # Default fallback plot
             df[plot_info.get('y_col', df.select_dtypes(include=np.number).columns[0])].plot(kind='line')
             plt.xticks(rotation=45, ha='right')

        plt.title(f"Placeholder Plot ({plot_type})")
        plt.tight_layout()
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        plot_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return f"data:image/png;base64,{plot_base64}"
    except Exception as e:
        print(f"Placeholder plot error: {e}")
        plt.close()
        return None

def format_as_text_table(data: List[Dict]) -> str:
     """Placeholder: Replace with your table formatting function."""
     print("INFO: Using placeholder format_as_text_table")
     if not data: return "No data."
     try:
          return pd.DataFrame(data).to_string(index=False, max_rows=10) # Limit rows for display
     except:
          return str(data[:10]) # Basic fallback

# --- NEW: Initialize AIComponent ---
try:
    ai_component = AIComponent() # Uses settings from config.py/.env
    # No immediate error raise, functions below will check ai_component.llm
except Exception as e:
    print(f"Error initializing AIComponent: {e}")
    ai_component = None # Set to None if init fails

# --- Plotly Chart Generation Functions ---

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Optional[str]:
    """Generates a Plotly bar chart JSON string."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Error: Columns '{x_col}' or '{y_col}' not found in DataFrame for bar chart.")
        return None
    try:
        # Ensure y-column is numeric for bar chart aggregation if needed, or suitable type
        try:
            df[y_col] = pd.to_numeric(df[y_col], errors='coerce')
            df_cleaned = df.dropna(subset=[y_col])
            if df_cleaned.empty:
                 print(f"Warning: Y-column '{y_col}' for bar chart has no numeric data after coercion.")
                 # Optionally try plotting counts if y_col was intended as numeric but failed
                 # fig = px.bar(df, x=x_col) # Counts occurrences of x_col
            else:
                 fig = px.bar(df_cleaned, x=x_col, y=y_col, title=title)

        except ValueError:
             # If y_col is non-numeric maybe user wants counts?
             print(f"Warning: Y-column '{y_col}' is not numeric. Attempting count plot.")
             fig = px.bar(df, x=x_col, title=f"Counts of {x_col}") # Plot counts if y is non-numeric

        return fig.to_json()
    except Exception as e:
        print(f"Error creating bar chart: {e}")
        return None

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Optional[str]:
    """Generates a Plotly line chart JSON string."""
    if x_col not in df.columns or y_col not in df.columns:
        print(f"Error: Columns '{x_col}' or '{y_col}' not found in DataFrame for line chart.")
        return None
    try:
        df_copy = df.copy() # Work on copy to avoid state side-effects
        # Ensure y-column is numeric
        df_copy[y_col] = pd.to_numeric(df_copy[y_col], errors='coerce')
        df_copy = df_copy.dropna(subset=[y_col])
        if df_copy.empty:
             print(f"Warning: Y-column '{y_col}' for line chart has no numeric data after coercion. Cannot plot line.")
             return None

        # Attempt to convert x_col to datetime and sort
        try:
            df_copy[x_col] = pd.to_datetime(df_copy[x_col])
            df_copy = df_copy.sort_values(by=x_col)
            print(f"INFO: Successfully converted '{x_col}' to datetime and sorted for line chart.")
        except Exception:
             print(f"Warning: Column '{x_col}' could not be reliably converted to datetime. Attempting to plot sequence as is after sorting numerically/lexicographically if possible.")
             try:
                 # Try sorting even if not datetime (might be numeric sequence or string)
                 df_copy = df_copy.sort_values(by=x_col)
             except Exception as sort_err:
                 print(f"Warning: Could not sort by '{x_col}': {sort_err}. Plotting in original order.")

        fig = px.line(df_copy, x=x_col, y=y_col, title=title, markers=True)
        return fig.to_json()
    except Exception as e:
        print(f"Error creating line chart: {e}")
        return None

def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str) -> Optional[str]:
    """Generates a Plotly pie chart JSON string."""
    if names_col not in df.columns or values_col not in df.columns:
        print(f"Error: Columns '{names_col}' or '{values_col}' not found in DataFrame for pie chart.")
        return None
    try:
        print(f"DEBUG: Creating pie chart with names_col={names_col}, values_col={values_col}")
        # Ensure values column is numeric
        df_copy = df.copy()
        df_copy[values_col] = pd.to_numeric(df_copy[values_col], errors='coerce')
        df_cleaned = df_copy.dropna(subset=[values_col])
        
        if df_cleaned.empty:
            print(f"Warning: Values column '{values_col}' contains no valid numeric data after coercion.")
            return None
            
        # Add detailed debugging info
        print(f"DEBUG: Pie chart data - {len(df_cleaned)} rows after cleaning")
        print(f"DEBUG: Unique values in names column: {df_cleaned[names_col].nunique()}")
        print(f"DEBUG: Sum of values column: {df_cleaned[values_col].sum()}")
        
        # Limit slices for readability
        if df_cleaned[names_col].nunique() > 15:
             print(f"Warning: Too many unique values in '{names_col}' ({df_cleaned[names_col].nunique()}). Grouping smaller slices into 'Other'.")
             # Keep top 14 + other
             top_n = df_cleaned.nlargest(14, values_col)
             other_sum = df_cleaned[~df_cleaned.index.isin(top_n.index)][values_col].sum()
             if other_sum > 0:
                  other_row = pd.DataFrame([{names_col: 'Other', values_col: other_sum}])
                  df_final = pd.concat([top_n, other_row], ignore_index=True)
             else:
                  df_final = top_n
        else:
             df_final = df_cleaned

        # Add specific Plotly pie chart configurations for better appearance
        fig = px.pie(
            df_final, 
            names=names_col, 
            values=values_col, 
            title=title,
            hole=0.3,  # Creates a donut chart effect for better readability
            color_discrete_sequence=px.colors.qualitative.Set3  # Use a color scheme with good contrast
        )
        
        # Enhance layout for better readability
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            insidetextorientation='radial'
        )
        
        # Add better legend placement
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )
        
        chart_json = fig.to_json()
        print(f"DEBUG: Successfully created pie chart JSON (length: {len(chart_json)})")
        return chart_json
    except Exception as e:
        print(f"Error creating pie chart: {e}")
        traceback.print_exc()  # Add detailed traceback for better debugging
        return None

# --- LLM Interaction Functions (Use AIComponent) ---

def get_llm_visualization_suggestion(current_df: pd.DataFrame, user_preference: str = None) -> Dict[str, Any]:
    """
    Uses LLM via AIComponent to suggest visualization type and columns.
    
    Args:
        current_df: The DataFrame to visualize
        user_preference: Optional user preference for chart type ('bar', 'line', 'pie')
    """
    if not ai_component or not ai_component.llm:
        print("ERROR: LLM (via AIComponent) not initialized, cannot get visualization suggestion.")
        return {"chart_type": "none", "x_col": None, "y_col": None, "reason": "LLM not available."}

    # Log user preference if provided
    if user_preference:
        print(f"User has requested visualization type: {user_preference}")
    
    # Prepare prompt for LLM
    prompt = f"""Analyze the following DataFrame structure and first 5 rows to determine the best chart type (bar, line, pie, or none) for visualization.
DataFrame Columns: {current_df.columns.tolist()}
DataFrame dtypes: \n{current_df.dtypes.to_string()}
First 5 rows:\n{current_df.head().to_string()}

Consider the data types and number of columns/rows.
- Use 'bar' for comparing numeric values across distinct categories. Requires one categorical-like column (text, object, category) and one numeric column.
- Use 'line' for trends over time or sequence. Requires one sequential/datetime-like column and one numeric column. Best if the sequential column can be ordered.
- Use 'pie' for proportions of a whole or distribution across categories. Requires one categorical column with relatively few unique values (ideally ≤10) and one numeric column that represents a meaningful proportion or part of a whole. Pie charts work best when showing how a total value is divided across categories (e.g., sales by product category, expenses by department).
- Use 'none' if the data is unsuitable (e.g., single value, too complex, text-only, no clear numeric column, or only one column).

{f"USER PREFERENCE: The user has specifically requested a '{user_preference}' chart if feasible with this data." if user_preference else ""}

IMPORTANT: When the user has requested a specific chart type, prioritize that chart type if the data structure allows for it. For pie charts specifically, be more lenient - if the data has a categorical column (even with up to 15-20 unique values) and a numeric column that can represent parts of a whole, a pie chart can be used.

Respond ONLY with a JSON object containing:
{{
  "chart_type": "<bar|line|pie|none>",
  "x_col": "<column_name_for_x_axis_or_names>",
  "y_col": "<column_name_for_y_axis_or_values>",
  "reason": "<brief_explanation_for_choice>"
}}
Map the conceptual X/Y or Names/Values to 'x_col' and 'y_col' in the JSON. Ensure column names exactly match the DataFrame Columns list. If chart_type is 'none', set x_col and y_col to null.

Example for bar: {{"chart_type": "bar", "x_col": "VendorName", "y_col": "TotalAmount", "reason": "Comparing total amounts across vendors."}}
Example for line: {{"chart_type": "line", "x_col": "OrderDate", "y_col": "Revenue", "reason": "Showing revenue trends over time."}}
Example for pie: {{"chart_type": "pie", "x_col": "ProductCategory", "y_col": "Sales", "reason": "Showing distribution of sales across product categories."}}
Example for none: {{"chart_type": "none", "x_col": null, "y_col": null, "reason": "Data contains only textual descriptions."}}

DataFrame Analysis Request: Determine the best chart type."""

    try:
        response = ai_component.invoke(prompt)
        # Assuming response is a LangChain AIMessage or similar, access content
        if hasattr(response, 'content'):
            response_content = response.content
        else:
            response_content = str(response) # Fallback if it's just a string

        # --- ADDED CLEANING STEP --- 
        # Clean potential markdown code block formatting
        response_content = response_content.strip()
        if response_content.startswith("```json"):
            # Remove ```json prefix and leading/trailing whitespace/newlines
            response_content = response_content[len("```json"):].strip()
        elif response_content.startswith("```"): # Handle case without 'json' language tag
             response_content = response_content[len("```"):].strip()
             
        if response_content.endswith("```"):
            # Remove ``` suffix and leading/trailing whitespace/newlines
            response_content = response_content[:-len("```")].strip()
        # --- END CLEANING STEP --- 

        suggestion = json.loads(response_content)

        # Validate suggestion structure more robustly
        if not isinstance(suggestion, dict) or not all(k in suggestion for k in ["chart_type", "x_col", "y_col", "reason"]):
             raise ValueError(f"LLM response missing required keys or not a dict. Response: {response_content}")

        # Validate column names if chart type is not 'none'
        chart_type = suggestion.get("chart_type")
        x_col = suggestion.get("x_col")
        y_col = suggestion.get("y_col")

        # --- NEW: Apply user preference if feasible ---
        if user_preference and user_preference in ['bar', 'line', 'pie'] and chart_type != 'none':
            # Check if we can honor the user preference with the current data
            if user_preference == 'pie':
                # For pie charts, check if we have categorical and numeric columns
                if x_col in current_df.columns and y_col in current_df.columns:
                    unique_values = current_df[x_col].nunique()
                    # Only use pie if categorical column doesn't have too many unique values
                    if unique_values <= 20:  # Relaxed constraint for user preference
                        print(f"Honoring user preference for pie chart. Using columns: {x_col} (names) and {y_col} (values)")
                        chart_type = 'pie'
                        suggestion["chart_type"] = 'pie'
                        suggestion["reason"] = f"User requested pie chart showing distribution of {y_col} across {x_col} categories."
            elif user_preference in ['bar', 'line']:
                # For bar/line, use the columns LLM selected but change chart type
                if x_col in current_df.columns and y_col in current_df.columns:
                    print(f"Honoring user preference for {user_preference} chart with columns: {x_col} and {y_col}")
                    chart_type = user_preference
                    suggestion["chart_type"] = user_preference
                    suggestion["reason"] = f"User requested {user_preference} chart showing {y_col} vs {x_col}."
        # --- END NEW ---

        if chart_type != 'none':
            # Allow null columns if explicitly intended by LLM for 'none' type
            if x_col is not None and x_col not in current_df.columns:
                 raise ValueError(f"LLM suggested invalid x_col '{x_col}' for chart type '{chart_type}'. Valid columns: {current_df.columns.tolist()}. Response: {response_content}")
            if y_col is not None and y_col not in current_df.columns:
                 raise ValueError(f"LLM suggested invalid y_col '{y_col}' for chart type '{chart_type}'. Valid columns: {current_df.columns.tolist()}. Response: {response_content}")
            # Allow same column if user explicitly asks for histogram/countplot maybe?
            # For now, enforce different columns for bar/line/pie.
            if chart_type in ["bar", "line", "pie"] and x_col == y_col and x_col is not None:
                 raise ValueError(f"LLM suggested the same column for x/names and y/values: '{x_col}'. Response: {response_content}")
            # Check if columns are actually provided for non-'none' types
            if chart_type != 'none' and (x_col is None or y_col is None):
                 raise ValueError(f"LLM suggested chart type '{chart_type}' but did not provide valid column names (x:'{x_col}', y:'{y_col}'). Response: {response_content}")


        return suggestion
    except json.JSONDecodeError as e:
        print(f"Error decoding LLM JSON response: {e}\nResponse: {response_content}")
        return {"chart_type": "none", "x_col": None, "y_col": None, "reason": f"Error decoding LLM suggestion JSON: {e}. Response: {response_content[:200]}..."}
    except ValueError as e: # Catch validation errors
         print(f"Error validating LLM suggestion: {e}")
         return {"chart_type": "none", "x_col": None, "y_col": None, "reason": f"LLM suggestion validation failed: {e}"}
    except Exception as e:
        print(f"Error interacting with LLM for visualization suggestion: {e}")
        # Consider logging traceback here: import traceback; traceback.print_exc()
        return {"chart_type": "none", "x_col": None, "y_col": None, "reason": f"Error interacting with LLM for suggestion: {e}"}


def get_llm_summary(current_df: Optional[pd.DataFrame], query: Optional[str], current_result_text: Optional[str]) -> str:
    """Generates a detailed summary of the data using LLM via AIComponent."""
    if not ai_component or not ai_component.llm:
        print("ERROR: LLM (via AIComponent) not initialized, cannot generate summary.")
        return "Summary generation failed: LLM not available."

    # Prepare prompt parts
    # --- ENHANCED Prompt with better data inclusion ---
    prompt_parts = ["Generate a detailed natural language summary of the key findings in the following data, which is the result of a user query."]
    if query:
        prompt_parts.append(f"User Query: {query}")

    data_preview = ""
    if current_df is not None and not current_df.empty:
        total_rows = len(current_df)
        prompt_parts.append(f"Total Results: {total_rows} rows with columns: {current_df.columns.tolist()}")
        
        # Add descriptive statistics for numeric columns
        numeric_cols = current_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            prompt_parts.append("\nNumeric Column Statistics:")
            stats_df = current_df[numeric_cols].describe().round(2)
            prompt_parts.append(stats_df.to_string())
        
        # Add value counts for categorical columns (with reasonable number of unique values)
        categorical_overview = []
        for col in current_df.columns:
            if col not in numeric_cols:  # Skip numeric columns already covered in statistics
                unique_count = current_df[col].nunique()
                if 1 <= unique_count <= 10:  # Only show distribution for columns with reasonable numbers of categories
                    categorical_overview.append(f"\nDistribution of {col}:")
                    value_counts = current_df[col].value_counts().head(10)  # Limit to top 10 values
                    categorical_overview.append(value_counts.to_string())
        
        if categorical_overview:
            prompt_parts.append("\nCategory Distributions:")
            prompt_parts.extend(categorical_overview)
        
        # Determine how many rows to include in the preview based on total size
        if total_rows <= 10:
            # If small result set, include all rows
            preview_rows = total_rows
            preview_text = f"\nComplete Dataset ({total_rows} rows):\n{current_df.to_string()}"
        elif total_rows <= 50:
            # For medium result sets, include first 15 rows
            preview_rows = 15
            preview_text = f"\nFirst {preview_rows} rows of {total_rows}:\n{current_df.head(preview_rows).to_string()}"
        else:
            # For large result sets, include samples from beginning, middle and end
            preview_text = f"\nFirst 5 rows of {total_rows}:\n{current_df.head(5).to_string()}"
            middle_idx = total_rows // 2
            middle_start = max(0, middle_idx - 2)
            middle_end = min(total_rows, middle_idx + 3)
            preview_text += f"\n\nMiddle rows ({middle_start}-{middle_end-1}) of {total_rows}:\n{current_df.iloc[middle_start:middle_end].to_string()}"
            preview_text += f"\n\nLast 5 rows of {total_rows}:\n{current_df.tail(5).to_string()}"
        
        prompt_parts.append(preview_text)
        
    elif current_result_text:
        preview_text = current_result_text
        # Only truncate if very long
        if len(preview_text) > 2000:  # Increased from previous 300 char limit
            preview_text = preview_text[:2000] + f"... (truncated, total length: {len(current_result_text)} chars)"
        prompt_parts.append(f"Result Data (Text):\n{preview_text}")
    else:
        prompt_parts.append("The query returned no data or the data format was not recognized.")

    # Provide detailed instructions for the summary
    prompt_parts.append("""
Based on the query and the data provided, please generate a detailed summary that includes:

1. The main findings or answer to the user's query
2. Key statistics or patterns (totals, averages, etc.)
3. Notable observations (trends, outliers, interesting data points)
4. Important context from the data that might be relevant 

Focus on specifics rather than general descriptions. Include precise numbers, dates, and values from the data where relevant.
""")
    # --- END ENHANCED Prompt ---
    
    prompt = "\n".join(prompt_parts)
    
    # Check if the prompt is too large and truncate if necessary
    if len(prompt) > 10000:  # Reasonable limit for most LLM contexts
        print(f"WARNING: Summary prompt is very large ({len(prompt)} chars). Truncating.")
        prompt = prompt[:10000] + "\n\n[Prompt truncated due to length]"

    try:
        response = ai_component.invoke(prompt)
        if hasattr(response, 'content'):
             summary = response.content
        else:
             summary = str(response)
        return summary.strip()
    except Exception as e:
        print(f"Error generating summary with LLM: {e}")
        return f"Could not generate summary due to an error: {e}"


# --- LangGraph Node --- (Checks ai_component.llm)

def visualize_data_node(state: AppState) -> Dict[str, Any]:
    """
    Analyzes the result dataframe, generates a Plotly visualization JSON **only if the
    requested output_format is 'plot'**, and generates a text summary using an LLM.
    """
    print("--- Running Visualize Data Node ---")
    current_df = state.get("result_dataframe")
    query = state.get("user_query")
    query_result = state.get("query_result") # Get raw query result for context
    
    # --- ENHANCED: Check for chart type preference (multiple sources) ---
    # Try to get chart preference from state through multiple possible keys
    chart_preference = state.get("chart_preference")  # Explicit preference passed in state
    if not chart_preference:
        chart_preference = state.get("chart_type")  # Alternative key
    if not chart_preference:
        chart_preference = state.get("preferred_chart")  # Another alternative key
    if not chart_preference:
        chart_preference = state.get("viz_type")  # One more alternative key
        
    # Check if chart preference is in any visualization config objects
    if not chart_preference:
        viz_config = state.get("visualization_config")
        if isinstance(viz_config, dict) and "type" in viz_config:
            chart_preference = viz_config.get("type")
            print(f"DEBUG: Found chart preference '{chart_preference}' in visualization_config")
    
    # Debug prints to track data
    print(f"DEBUG: State keys available: {list(state.keys())}")
    print(f"DEBUG: Original query: {query}")
    print(f"DEBUG: Initial chart_preference from state: {chart_preference}")
    
    # If no explicit preference set, try to infer from the query text
    if not chart_preference and query:
        query_lower = query.lower()
        # More verbose logging of the query parsing
        print(f"DEBUG: Looking for chart type in query: '{query_lower}'")
        
        # Enhanced pie chart detection with many more variations
        pie_phrases = [
            "pie chart", "pie graph", "as a pie", "in a pie", "using pie", 
            "show pie", "create pie", "make pie", "want pie", "need pie",
            "display as pie", "display in pie", "show as pie", "show in pie",
            "distribution", "breakdown", "proportion", "percentage", 
            "split by", "division of", "allocation of", "composition of"
        ]
        
        if any(phrase in query_lower for phrase in pie_phrases):
            chart_preference = "pie"
            print(f"DEBUG: Inferred chart preference 'pie' from query text: '{query}'")
        elif "bar chart" in query_lower or "bar graph" in query_lower or "as a bar" in query_lower or "using bar" in query_lower:
            chart_preference = "bar"
            print(f"DEBUG: Inferred chart preference 'bar' from query text: '{query}'")
        elif "line chart" in query_lower or "line graph" in query_lower or "as a line" in query_lower or "trend" in query_lower or "over time" in query_lower:
            chart_preference = "line"
            print(f"DEBUG: Inferred chart preference 'line' from query text: '{query}'")
            
    # Add stronger priority for pie charts related to specific data contexts
    if chart_preference is None and query and current_df is not None and not current_df.empty:
        query_lower = query.lower()
        # Check for common pie chart use cases in query
        if any(term in query_lower for term in ["vendor", "category", "type", "distribution"]):
            # Check if we have a potential categorical column and numeric column for a pie chart
            categorical_cols = current_df.select_dtypes(include=['object', 'category']).columns
            numeric_cols = current_df.select_dtypes(include=[np.number]).columns
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                # Potential candidate for pie chart if categorical column doesn't have too many values
                sample_col = categorical_cols[0]
                if current_df[sample_col].nunique() <= 15:  # Good candidate for pie
                    chart_preference = "pie"
                    print(f"DEBUG: Auto-selecting pie chart based on data structure and query context")
    
    print(f"DEBUG: Final chart_preference: {chart_preference}")
    # --- END NEW ---
    
    # Get the requested output format
    requested_output_format = state.get("output_format", "table") # Default to table if missing
    print(f"Requested output format: {requested_output_format}") # Log the format

    chart_json = None
    table_data = None # Initialize table data output
    summary = "Summary could not be generated." # Default summary
    error_message = None # Track errors encountered in this node

    # Check if LLM is available early via the component
    if not ai_component or not ai_component.llm:
        print("Warning: LLM (via AIComponent) not available. Skipping summary generation and visualization suggestion.")
        summary = "Summary and Visualization Suggestion skipped: LLM not available."
    else:
        # Determine result text for summary if no dataframe
        current_result_text = None
        if current_df is None or current_df.empty:
            if isinstance(query_result, str):
                current_result_text = query_result
            elif query_result is not None: # Catch lists, dicts, etc.
                try:
                    current_result_text = json.dumps(query_result)
                except Exception:
                     current_result_text = str(query_result)

        # 1. Generate Summary using LLM (Always happens if LLM is available)
        try:
            summary = get_llm_summary(current_df, query, current_result_text)
        except Exception as e:
            print(f"Error during summary generation: {e}")
            summary = f"Error generating summary: {e}"
            if not error_message: error_message = f"Summary generation failed: {e}"

    # --- NEW: Format DataFrame if output is 'table' ---
    if requested_output_format == 'table' and current_df is not None and not current_df.empty:
        try:
            # Convert DataFrame to list of dictionaries - common format for JSON/UI
            # Handle potential date/datetime/decimal types that aren't directly JSON serializable
            current_df_serializable = current_df.copy()
            for col in current_df_serializable.select_dtypes(include=[np.datetime64, 'datetime', 'datetime64[ns]', 'datetimetz']).columns:
                 current_df_serializable[col] = current_df_serializable[col].astype(str)
            for col in current_df_serializable.select_dtypes(include=[np.timedelta64, 'timedelta', 'timedelta64[ns]']).columns:
                 current_df_serializable[col] = current_df_serializable[col].astype(str)
            # Convert decimals specifically
            for col in current_df_serializable.select_dtypes(include=['object']).columns:
                # --- Corrected check for NA values using pd.isna() ---
                if current_df_serializable[col].isna().any():
                    # Handle potential mix of Decimal and None/NA if necessary
                    # This example converts numeric types (including Decimal) to strings, NAs stay None
                    current_df_serializable[col] = current_df_serializable[col].apply(
                        lambda x: str(float(x)) if isinstance(x, (Decimal, float, int)) and pd.notna(x) else None
                    )
                # --- Check if ALL non-NA values are Decimal ---
                elif current_df_serializable[col].dropna().apply(lambda x: isinstance(x, Decimal)).all():
                    # If all non-NA are Decimals, convert to float string (NA handling included in apply above)
                    # This case might be covered by the above, but kept for potential specific logic
                    current_df_serializable[col] = current_df_serializable[col].apply(
                         lambda x: str(float(x)) if isinstance(x, Decimal) and pd.notna(x) else None
                    )
                # Add handling for other complex object types if necessary
            
            table_data = current_df_serializable.to_dict('records')
            print(f"Formatted DataFrame to list of {len(table_data)} dictionaries for table output.")
        except Exception as e:
            print(f"Error formatting DataFrame to dict for table output: {e}")
            summary += f" (Error formatting table data: {e})"
            if not error_message: error_message = f"Failed to format table data: {e}"
    # ----------------------------------------------------

    # 2. Attempt Visualization only if DataFrame is suitable AND requested_output_format is 'plot'
    # --- MODIFIED CONDITION --- 
    if requested_output_format == 'plot' and current_df is not None and not current_df.empty:
        print("Output format is 'plot', attempting visualization...")
        if len(current_df) == 1 and len(current_df.columns) == 1:
            print("Data is a single value. Skipping visualization.")
            summary += " (Visualization skipped: data is a single value)"
        elif len(current_df.columns) < 1 :
            print("Data has no columns. Skipping visualization.")
            summary += " (Visualization skipped: data has no columns)"
        else:
            # Proceed with visualization attempt only if LLM is available
            if ai_component and ai_component.llm: # Check again here
                try:
                    print("Getting LLM visualization suggestion...")
                    # --- UPDATED: Pass chart_preference to LLM visualization function ---
                    suggestion = get_llm_visualization_suggestion(current_df, user_preference=chart_preference)
                    chart_type = suggestion.get("chart_type", "none")
                    x_col = suggestion.get("x_col")
                    y_col = suggestion.get("y_col")
                    reason = suggestion.get("reason", "N/A")
                    print(f"LLM Suggestion: Type={chart_type}, X/Names={x_col}, Y/Values={y_col}, Reason={reason}")

                    if chart_type != 'none' and x_col and y_col:
                        title = f"Visualization for Query" # Keep title simple
                        
                        # Enhanced logging
                        print(f"DEBUG: Creating {chart_type} chart with x/names={x_col}, y/values={y_col}")
                        
                        # Give explicit priority to pie chart when requested
                        if chart_preference == "pie" and chart_type != "pie":
                            print(f"DEBUG: Override - User requested pie chart but LLM suggested {chart_type}")
                            # Check if data suitable for pie chart
                            try:
                                # Check if categorical column has reasonable number of values
                                unique_count = current_df[x_col].nunique()
                                if unique_count <= 20:  # Reasonable for a pie
                                    print(f"DEBUG: Switching to pie chart as per user preference")
                                    chart_type = "pie"
                            except Exception as e:
                                print(f"ERROR checking column uniqueness: {e}")
                                # Continue with LLM suggestion
                        
                        # Create the chart based on final chart type
                        if chart_type == "bar":
                            chart_json = create_bar_chart(current_df, x_col, y_col, title)
                        elif chart_type == "line":
                            chart_json = create_line_chart(current_df, x_col, y_col, title)
                        elif chart_type == "pie":
                            chart_json = create_pie_chart(current_df, names_col=x_col, values_col=y_col, title=title)
                        
                        if chart_json is None:
                            fail_reason = f"Failed to create {chart_type} chart with columns '{x_col}' and '{y_col}'. Check logs."
                            print(fail_reason)
                            summary += f" (Visualization generation failed: {fail_reason})"
                            if not error_message: error_message = fail_reason
                        else:
                            print(f"Successfully generated {chart_type} chart JSON.")

                    else:
                        print(f"Visualization skipped based on LLM suggestion or invalid columns: {reason}")
                        summary += f" (Visualization skipped: {reason})"

                except Exception as e:
                    print(f"Error during visualization attempt: {e}")
                    error_msg = f"Failed during visualization: {e}"
                    summary += f" (Visualization failed due to an error: {e})"
                    if not error_message: error_message = error_msg
            else:
                 summary += " (Visualization skipped: LLM not available for suggestion)"
    elif requested_output_format != 'plot':
         print(f"Visualization skipped: Requested output format is '{requested_output_format}', not 'plot'.")
         summary += f" (Visualization skipped: format '{requested_output_format}')"
    else:
        print("No DataFrame found or DataFrame is empty. Skipping visualization.")
        if "Error generating summary" not in summary and "LLM not available" not in summary:
            summary += " (Visualization skipped: No suitable data frame)"


    # 3. Update state
    update_state = {
        "summary": summary,
        "plotly_chart_json": chart_json,
        "table_data": table_data # Add formatted table data to state
    }
    if error_message and not state.get("error"):
         update_state["error"] = error_message

    return update_state


# --- Example Usage --- (Keep as is, but relies on env vars for the configured provider)
# if __name__ == "__main__":
#     import pprint

#     # Test Case 1: Valid DataFrame Input
#     print("\n--- Test Case 1: Valid DataFrame ---")
#     test_df = pd.DataFrame({
#         'Category': ['A', 'B', 'C', 'D', 'E', 'F'],
#         'Value': [50, 85, 32, 68, 75, 41],
#         'Date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06'])
#     })
#     test_state1 = AppState(
#         result_dataframe=test_df,
#         query_code="SELECT Category, Value, Date FROM sales_data",
#         query_result="Category,Value,Date\nA,50,2023-01-01\nB,85,2023-01-02..." # Example raw
#     )
#     result1 = visualize_data_node(test_state1)
#     pprint.pprint(result1)
#     # Check if plot data exists (optional)
#     if result1.get("visualization_output", {}).get("plot_image"):
#         print("Plot generated (length):", len(result1["visualization_output"]["plot_image"]))
#     else:
#         print("Plot not generated.")


#     # Test Case 2: Empty DataFrame Input
#     print("\n--- Test Case 2: Empty DataFrame ---")
#     test_state2 = AppState(
#         result_dataframe=pd.DataFrame({'colA': [], 'colB': []}),
#         query_code="SELECT colA, colB FROM nodata_table WHERE condition='false'",
#         query_result="" # Empty raw result
#     )
#     result2 = visualize_data_node(test_state2)
#     pprint.pprint(result2)


#     # Test Case 3: DataFrame with No Numeric Data
#     print("\n--- Test Case 3: No Numeric Data ---")
#     test_df_nonum = pd.DataFrame({
#         'Name': ['Alice', 'Bob', 'Charlie'],
#         'City': ['New York', 'London', 'Paris']
#     })
#     test_state3 = AppState(
#         result_dataframe=test_df_nonum,
#         query_code="SELECT Name, City FROM users",
#         query_result="Name,City\nAlice,New York\nBob,London..."
#     )
#     result3 = visualize_data_node(test_state3)
#     pprint.pprint(result3)