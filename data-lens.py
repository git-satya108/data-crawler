import os
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure API key is properly set
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("Missing OpenAI API Key. Please check your environment variables.")
    st.stop()

# Initialize OpenAI model
llm = OpenAI(api_token=api_key)

# Module 1: Load and combine CSV/Excel files
@st.cache_data
def load_combined_data(files):
    combined_df = pd.DataFrame()
    for file in files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df


# Module 2: Handle natural language queries using OpenAI
def handle_nlp_query(smart_df, query):
    try:
        result = smart_df.chat(query)
        return result
    except KeyError as e:
        st.error(f"KeyError: {e} - The API response structure may have changed.")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


# Main Streamlit App Interface
def main():
    st.title("Gen AI-Powered Data Analysis with NLP")

    # Step 1: Upload Files
    uploaded_files = st.file_uploader(
        "Upload your Excel or CSV files",
        accept_multiple_files=True,
        type=["csv", "xlsx"]
    )

    # Step 2: Combine Files into a Single DataFrame
    if uploaded_files:
        combined_data = load_combined_data(uploaded_files)
        st.write("### Combined Data Preview")
        st.dataframe(combined_data.head())

        # Step 3: Initialize SmartDataframe with OpenAI LLM (DISABLE VECTORSTORE)
        smart_df = SmartDataframe(combined_data, config={"llm": llm, "use_vectorstore": False})

        # Step 4: Enter Natural Language Query
        query = st.text_input("Chat with your Data")

        if st.button("Run Query in Natural Language"):
            if query:
                result = handle_nlp_query(smart_df, query)
                if result:
                    st.write("### Query Result")
                    st.dataframe(result) if isinstance(result, pd.DataFrame) else st.write(result)
                else:
                    st.error("No results found. Try modifying your query.")


# Run the Streamlit App
if __name__ == "__main__":
    main()
