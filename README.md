# Chat PDF using AWS Bedrock

This repository contains a Python application that allows users to chat with PDF documents using Amazon Bedrock. It leverages the Amazon Titan Embeddings Model for text embeddings and integrates multiple language models (LLMs) like Claude and Llama2 for generating responses. The application uses Streamlit for the web interface.

## Features

- Ingest PDF documents and split them into manageable chunks.
- Create vector embeddings using Amazon Titan Embeddings Model.
- Store and retrieve vectors using FAISS.
- Query PDF documents using language models like Claude and Llama2.
- Interactive web interface built with Streamlit.

## Installation

1. Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2. Install the required dependencies using the provided `Pipfile`:
    ```bash
    pipenv install
    ```

3. Set up AWS credentials to use Amazon Bedrock services.

## Usage

1. **Data Ingestion:**

    Place your PDF files in the `data` directory. The application will automatically load and process these files.

2. **Run the Streamlit App:**

    ```bash
    streamlit run app.py
    ```

3. **Update or Create Vector Store:**

    Use the sidebar option "Vectors Update" to ingest data and create vector embeddings. This process involves loading PDF documents, splitting them into chunks, and generating vector embeddings stored locally.

4. **Ask Questions:**

    - Type your question in the text input box.
    - Choose between "Claude Output" or "Llama2 Output" to get answers from the respective language model.
  
<img width="1117" alt="image" src="https://github.com/ambreen002/ChatWithPDF-Langchain-Bedrock/assets/36915142/0188359c-a558-4807-b1c7-5438b6579a12">

## Code Overview

### Imports

- **General:**
  - `json`, `os`, `sys`: Standard Python libraries.
  - `boto3`: AWS SDK for Python.
  - `streamlit as st`: Streamlit for creating web applications.

- **Amazon Bedrock:**
  - `BedrockEmbeddings`, `Bedrock`: Embedding and LLM classes from LangChain Community.

- **Data Ingestion:**
  - `numpy as np`: NumPy for numerical operations.
  - `RecursiveCharacterTextSplitter`, `PyPDFDirectoryLoader`: For text splitting and PDF loading from LangChain Community.

- **Vector Store:**
  - `FAISS`: Vector store for similarity search.

- **LLM Models:**
  - `PromptTemplate`, `RetrievalQA`: Prompt templates and QA chains from LangChain.

### Functions

- **`data_ingestion`**: Loads and processes PDF documents, splitting them into chunks.
- **`get_vector_store`**: Creates and stores vector embeddings using FAISS.
- **`get_claude_llm`**: Initializes and returns the Claude LLM.
- **`get_llama2_llm`**: Initializes and returns the Llama2 LLM.
- **`get_response_llm`**: Generates a response from the specified LLM using a query and the vector store.

### Main Application

The main function sets up the Streamlit page, handles user input, and displays responses from the chosen LLM.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Amazon Bedrock](https://aws.amazon.com/bedrock/)
- [LangChain Community](https://www.langchain.com/)
