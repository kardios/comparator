import streamlit as st
import fitz # PyMuPDF for robust PDF text extraction
from openai import OpenAI # OpenAI library for LLM interaction
from st_copy_to_clipboard import st_copy_to_clipboard # For convenient copying of results
import os # Not directly used for secrets, but often useful for environment variables

# --- Configuration for API Keys ---
# API keys are securely loaded from Streamlit Secrets.
# For local development:
#   1. Create a folder named '.streamlit' in your project's root directory.
#   2. Inside '.streamlit', create a file named 'secrets.toml'.
#   3. Add your OpenAI API key to 'secrets.toml' like this:
#      OPENAI_API_KEY = "sk-your_openai_api_key_here"
#   4. Add '.streamlit/secrets.toml' to your .gitignore file to prevent accidental commits.
# For Streamlit Community Cloud deployment:
#   Configure your 'OPENAI_API_KEY' directly in the app's settings dashboard under 'Secrets'.

# --- PDF Text Extraction Function ---
def extract_text_from_pdf(pdf_file):
    """
    Extracts text content from an uploaded PDF file using PyMuPDF.
    Handles potential errors during extraction.
    """
    text = ""
    try:
        # PyMuPDF's fitz.open can read directly from a file-like object's byte stream.
        # This is efficient for files uploaded via st.file_uploader.
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        for page in doc:
            text += page.get_text() # Extract plain text from each page.
        doc.close() # Always close the document to release resources.
    except Exception as e:
        # Display an error message to the user if PDF text extraction fails.
        st.error(f"Error extracting text with PyMuPDF: {e}")
        text = "" # Ensure an empty string is returned on error.
    return text

# --- Prompt Template for Document Comparison ---
# This template provides instructions to the LLM on how to perform the comparison
# and what aspects to focus on. It ensures consistency across different LLM calls.
COMPARISON_PROMPT_TEMPLATE = """Compare the following two documents and highlight their key similarities and differences.
Document 1:
{doc1_text}

Document 2:
{doc2_text}

Please provide a concise and clear comparison, focusing on:
1.  **Main Topics/Themes:** What are the central subjects or ideas discussed in each document?
2.  **Key Arguments/Points:** What are the most important arguments or specific points made in each document?
3.  **Similarities:** What aspects, facts, or conclusions are common to both documents?
4.  **Differences/Discrepancies:** What are the major differences, conflicting information, or unique points in each document?
5.  **Overall Tone and Purpose:** How would you describe the tone (e.g., formal, informal, objective, persuasive) and the primary purpose of each document?

Structure your comparison clearly with headings for each point.
"""

# --- OpenAI LLM Integration Function for GPT-4.1 ---
def get_gpt41_comparison(doc1_text, doc2_text, model_name="gpt-4.1"):
    """
    Calls the OpenAI API for document comparison using the 'gpt-4.1' model,
    implementing the `client.responses.create(model=..., input=string)` syntax.
    """
    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    if not openai_api_key:
        return "ERROR: OpenAI API key not found in Streamlit Secrets. Please configure it (`OPENAI_API_KEY`)."

    client = OpenAI(api_key=openai_api_key)

    # LLMs have a maximum input token limit (context window).
    # This value truncates the input documents if they exceed a certain size,
    # ensuring they fit within the LLM's context window. Adjust as needed.
    max_tokens_per_doc = 60000 # Example: a high limit, actual model limits vary

    # Format the prompt with the extracted document texts.
    # Text is truncated here if it's too long, to avoid exceeding context limits.
    input_text_for_llm = COMPARISON_PROMPT_TEMPLATE.format(
        doc1_text=doc1_text[:max_tokens_per_doc],
        doc2_text=doc2_text[:max_tokens_per_doc]
    )

    try:
        # --- OpenAI API call implementation as per your exact instruction ---
        response = client.responses.create(
            model=model_name, # The model identifier, e.g., "gpt-4.1"
            input=input_text_for_llm # The input string for the LLM
        )
        return response.output_text # Extract the generated text from the response
    except Exception as e:
        # Catch any exceptions during the API call (e.g., network issues, invalid API key, rate limits).
        return f"ERROR: An error occurred with GPT-4.1 ({model_name}) comparison: {e}"

# --- OpenAI LLM Integration Function for O3 ---
# Renamed from get_o3_comparison_o4_mini for clarity as per user's request for "o3"
def get_o3_comparison(doc1_text, doc2_text, model_name="o3"):
    """
    Calls the OpenAI API for document comparison using the 'o3' model,
    implementing the `client.responses.create(model=..., reasoning=..., input=[...])` syntax.
    """
    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    if not openai_api_key:
        return "ERROR: OpenAI API key not found in Streamlit Secrets. Please configure it (`OPENAI_API_KEY`)."

    client = OpenAI(api_key=openai_api_key)

    max_tokens_per_doc = 60000

    prompt_content = COMPARISON_PROMPT_TEMPLATE.format(
        doc1_text=doc1_text[:max_tokens_per_doc],
        doc2_text=doc2_text[:max_tokens_per_doc]
    )

    try:
        # --- OpenAI API call implementation as per your exact instruction for O3 ---
        response = client.responses.create(
            model=model_name, # The model identifier, e.g., "o3"
            reasoning={"effort": "medium"}, # Specific reasoning parameter
            input=[ # Input is a list containing a message object
                {
                    "role": "user",
                    "content": prompt_content
                }
            ]
        )
        return response.output_text # Extract the generated text from the response
    except Exception as e:
        # Catch any exceptions during the API call.
        return f"ERROR: An error occurred with O3 ({model_name}) comparison: {e}"

# --- Streamlit Application User Interface ---
st.set_page_config(layout="wide", page_title="LLM Document Comparator")
st.title("📄 LLM Document Comparison Tool (OpenAI Models)")
st.markdown("Upload two PDF documents and compare them using different OpenAI LLMs based on your specified API call syntaxes.")

# Sidebar for LLM settings and API key information
st.sidebar.header("LLM Settings & API Keys")

# Define the specific OpenAI models available for selection in the UI.
# The keys are display names, and values are the actual model identifiers for the API calls.
# Changed to directly use "gpt-4.1" and "o3" as per your request.
openai_models_for_selection = {
    "gpt-4.1": "gpt-4.1", # Model name for GPT-4.1 as provided
    "o3": "o3" # Model name for O3 as provided
}

# Allow the user to select one or more OpenAI models to run for comparison.
selected_llm_display_names = st.sidebar.multiselect(
    "Select OpenAI Models to run for comparison:",
    list(openai_models_for_selection.keys()),
    default=["gpt-4.1", "o3"] # Default selections updated to new names
)

st.sidebar.markdown("""
**API Key Configuration:**
* **OpenAI API Key:** `OPENAI_API_KEY`

This key **must be configured in your Streamlit secrets**.
* For **local development**, save it in `.streamlit/secrets.toml`.
* For **Streamlit Community Cloud**, set it in your app's settings dashboard.
""")

# --- Main Area for Document Upload ---
st.header("Upload Documents")

# Use Streamlit columns to arrange two file uploaders side-by-side for a cleaner layout.
col1, col2 = st.columns(2)

with col1:
    st.subheader("Document 1")
    # File uploader widget for the first PDF document.
    uploaded_file1 = st.file_uploader("Upload PDF Document 1", type="pdf", key="doc1")

with col2:
    st.subheader("Document 2")
    # File uploader widget for the second PDF document.
    uploaded_file2 = st.file_uploader("Upload PDF Document 2", type="pdf", key="doc2")

# Button to trigger the comparison process.
compare_button = st.button("Run Comparison(s)")

# --- Document Comparison Logic ---
if compare_button:
    # Perform input validation before proceeding with expensive operations.
    if not uploaded_file1 or not uploaded_file2:
        st.warning("Please upload both PDF documents to compare.")
    elif not selected_llm_display_names:
        st.warning("Please select at least one LLM to run for comparison.")
    else:
        # Check if the required OpenAI API key is configured.
        openai_key_configured = st.secrets.get("OPENAI_API_KEY")
        if not openai_key_configured:
            st.error("OpenAI API key (`OPENAI_API_KEY`) not found in Streamlit Secrets. Please configure it to proceed.")
        else:
            # Step 1: Extract text from PDFs with a spinner to indicate activity.
            with st.spinner("Extracting text from PDFs..."):
                doc1_text = extract_text_from_pdf(uploaded_file1)
                doc2_text = extract_text_from_pdf(uploaded_file2)

            # Check if text extraction was successful.
            if not doc1_text or not doc2_text:
                st.error("Could not extract text from one or both PDFs. Please ensure they are not scanned images or malformed.")
            else:
                st.success("Text extracted successfully! Initiating LLM comparison(s)...")

                # Step 2: Prepare columns for displaying comparison results side-by-side.
                num_cols = len(selected_llm_display_names)
                # Create columns; if no LLMs are selected, it will be an empty list (though warning prevents this).
                result_cols = st.columns(num_cols) if num_cols > 0 else []

                # Step 3: Iterate through selected LLMs and run comparison for each.
                for i, display_name in enumerate(selected_llm_display_names):
                    # Get the actual model identifier from the display name.
                    model_id = openai_models_for_selection[display_name]
                    with result_cols[i]: # Place the output in the appropriate column.
                        st.subheader(f"Comparison by {display_name}")
                        with st.spinner(f"Running {display_name} comparison..."):
                            # Call the appropriate OpenAI comparison function based on the selected LLM.
                            if display_name == "gpt-4.1":
                                comparison_result = get_gpt41_comparison(doc1_text, doc2_text, model_name=model_id)
                            elif display_name == "o3":
                                comparison_result = get_o3_comparison(doc1_text, doc2_text, model_name=model_id)
                            else:
                                comparison_result = f"ERROR: Unrecognized LLM selected: {display_name}"

                            st.markdown(comparison_result) # Display the LLM's comparison output using Markdown.

                            # Add a "Copy to Clipboard" button below each result for user convenience.
                            # The 'key' must be unique for each button instance.
                            st_copy_to_clipboard(
                                comparison_result,
                                key=f"copy_{model_id.replace('.', '_').replace('-', '_')}"
                            )

                st.success("All selected comparisons complete! Please review the outputs above.")
