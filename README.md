# braintrust-local-eval

---

LLM as a Judge and the Braintrust Factuality Scorer  
   
## Executive Summary  
   
Evaluating the output of Large Language Models (LLMs) is a complex challenge. Traditional metrics are often insufficient for capturing nuances like factual accuracy, coherence, and relevance. The **"LLM as a Judge"** concept has emerged as a powerful and scalable solution, using a highly capable LLM to evaluate the outputs of another AI system.  
   
Braintrust leverages this concept directly in its **autoevals** library, with the **Factuality scorer** being a prime example. This scorer uses a "judge" model to semantically compare a model's generated answer against a predefined, ground-truth expected answer. This report details this mechanism, explaining how the input, output (the model's answer), and expected values are used in this evaluation process.  
   
## 1. The "LLM as a Judge" Concept  
   
"LLM as a Judge" is an evaluation technique where a powerful, state-of-the-art LLM (the **"judge"**) is prompted to assess the quality of text generated by another model (the **"system under test"**).  
   
### Why It's Necessary  
   
- **Beyond Simple Metrics**: Metrics like ROUGE or BLEU measure lexical overlap but fail to capture semantic meaning. An answer can be factually wrong but use many of the same words as the correct answer.  
   
- **Scalability Over Human Evaluation**: While human evaluation is the gold standard, it is slow, expensive, and difficult to scale, making it impractical for the rapid, iterative development cycles of modern AI applications.  
   
- **Nuanced Criteria**: LLM judges can be instructed to evaluate outputs based on complex, subjective criteria like helpfulness, tone, safety, and factual correctness—tasks that are difficult to codify with traditional algorithms.  
   
### How It Works  
   
The core mechanism involves three components:  
   
1. **The System's Output**: The text generated by the model you are testing.  
   
2. **The Evaluation Criteria**: A rubric or set of instructions provided to the judge LLM. This can include a ground-truth answer for comparison.  
   
3. **The Judge's Prompt**: A carefully engineered prompt that presents the system's output to the judge and asks it to return a score or analysis based on the criteria.  
   
## 2. Braintrust's Implementation: The Factuality Scorer  
   
Braintrust's **Factuality scorer** is a direct and practical application of the "LLM as a Judge" concept. It is designed to measure whether the model's generated output is factually consistent with a provided ground truth.  
   
The scorer relies on three key pieces of data for each evaluation test case:  
   
| Parameter  | Role in Evaluation                                                                                                                                                | In Your Script                                                                                                                                              |  
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|  
| **input**  | **Context**<br>The user's original prompt (e.g., "how do I request a domain to be whitelisted?"). This provides the judge with the necessary context to understand the output and expected values. | `input`                                                                                                                                                   |  
| **output** | **The Submission**<br>The actual, generated response from your RAG pipeline (`run_rag_task`). This is the piece of text being judged.                                             | `output`                                                                                                                                                  |  
| **expected** | **The Ground Truth**<br>A hand-crafted, ideal answer that is considered factually correct (e.g., "To request a domain to be whitelisted, please ensure..."). This is the reference standard the judge uses for comparison. | `expected`                                                                                                                                              |  
   
## 3. How the Factuality Judgment Works Step-by-Step  
   
When you run an Eval with the Factuality scorer, Braintrust orchestrates the following process behind the scenes:  
   
### Step 1: The Judge is Summoned  
   
Braintrust uses a powerful, state-of-the-art LLM (like GPT-4) as its **"judge."** This model is separate from the one you are testing.  
   
### Step 2: The Judge Receives its Instructions  
   
A sophisticated prompt is sent to the judge model. This prompt is engineered to guide the model's reasoning process. It includes:  
   
- **The Task**:    
  "You are comparing a submitted answer to a ground-truth answer. Your goal is to determine if the submitted answer is factually consistent with the ground-truth."  
   
- **The "Ground Truth"**:    
  The content of the `expected` variable.  
   
- **The "Submission"**:    
  The content of the `output` variable (the model's generated answer).  
   
### Step 3: Chain-of-Thought (CoT) Reasoning  
   
The judge doesn't just give a score. It is prompted to **"think"** step-by-step. This is a crucial detail that makes the evaluation transparent and reliable. The judge model will typically:  
   
1. **Identify** the key facts or claims made in the expected (ground-truth) answer.  
   
2. **Check** if those same facts are present and correctly stated in the output (the submission).  
   
3. **Identify** any information in the output that contradicts the expected answer.  
   
4. **Identify** any information in the output that is not mentioned in the expected answer (a potential hallucination).  
   
5. **Formulate** a final rationale for its score.  
   
### Step 4: The Verdict (The Score)  
   
Based on its reasoning, the judge selects a conclusion that maps to a numerical score from **0.0** to **1.0**. The logic generally follows this rubric:  
   
- **Score 1.0**:    
  The output is a perfect or near-perfect factual match with the expected answer. It may be rephrased but contains no conflicting information.  
   
- **Score 0.5 - 0.9 (Partial Credit)**:    
  The output is mostly correct but may be missing some key facts present in the expected answer. It does not contain contradictions.  
   
- **Score 0.0**:    
  The output contains one or more statements that directly contradict the expected answer. This indicates a factual error or hallucination.  
   
## Example from Your Script  
   
Let's trace the process for your first test case:  
   
- **Input**:    
  "how do I request a domain to be whitelisted?"  
   
- **Expected**:    
  "To request a domain to be whitelisted, please ensure you have your manager's approval. Once you have that, you can proceed with the request. Note that only Enterprise Accounts may be whitelisted, and the domain send limit uses a 24-hour rolling period..."  
   
- **Output** (from your RAG model):    
  Let's assume your model returns:  
  
  ```json  
  {  
    "response": "To get a domain whitelisted, you must get your manager to approve it. Only enterprise accounts can be whitelisted.",  
    "used_provided_context": true  
  }  
  ```  
   
The Braintrust Factuality judge would:  
   
1. **Analyze expected**:  
  
   - Identifies key facts:  
  
     - (a) Manager approval required  
  
     - (b) For Enterprise Accounts only  
  
     - (c) 24-hour rolling period mentioned  
   
2. **Analyze output**:  
  
   - Sees that facts (a) and (b) are correctly represented.  
   
3. **Compare**:  
  
   - Notes that fact (c) is missing from the output.  
   
4. **Reason**:  
  
   - The judge's internal monologue (Chain-of-Thought) might be:  
  
     > "The submission correctly states the need for manager approval and the restriction to enterprise accounts. However, it omits the detail about the 24-hour rolling period mentioned in the ground truth. It does not contradict the ground truth, but it is incomplete."  
   
5. **Score**:  
  
   - Because the output is factually consistent but incomplete, it would likely receive a high but not perfect score, such as **0.7** or **0.8**.  
   
## Conclusion  
   
Braintrust's **Factuality scorer** is a powerful application of the "LLM as a Judge" paradigm. It moves beyond simple text matching to perform a semantic, reason-based evaluation. By providing a clean expected ground-truth value, you empower a sophisticated LLM judge to perform a nuanced, scalable, and transparent assessment of your model's factual accuracy, dramatically improving the speed and quality of your evaluation workflow.

---


# RAG Pipeline Evaluation with Braintrust, Azure OpenAI, and MongoDB

This project provides a complete, runnable Python script for evaluating a Retrieval-Augmented Generation (RAG) pipeline. It uses Azure OpenAI for generating embeddings and chat responses, MongoDB Atlas Vector Search for information retrieval, and Braintrust for sophisticated, fact-based evaluation.

The script includes two evaluation methods:

1.  **Braintrust Evaluation (`run_eval`)**: Logs the experiment to your Braintrust dashboard and uses the `Factuality` scorer to semantically compare the generated output against a ground truth.
2.  **Local Validation (`validate_locally`)**: Prints a simple "expected vs. got" comparison directly to your console for rapid, local testing.

## Features

  - **End-to-End RAG**: Implements the full RAG workflow: embedding, vector search, and augmented generation.
  - **Azure OpenAI Integration**: Utilizes the `openai` library to connect to Azure OpenAI for both `embeddings` and `chat.completions`.
  - **MongoDB Vector Search**: Connects to a MongoDB database using `pymongo` to perform efficient similarity searches with the `$vectorSearch` aggregation pipeline.
  - **Dual Evaluation Framework**:
      - Leverages the `braintrust` SDK for robust, production-grade evaluation and experiment tracking.
      - Includes a simple local validation function for quick sanity checks without external dependencies.
  - **Secure Configuration**: Relies on environment variables for managing sensitive API keys and connection strings.
  - **Customizable Prompts**: Features a detailed system prompt that instructs the model on its task, context usage, and specific output formatting (Slack `mrkdwn` in a JSON object).

## Prerequisites

Before you begin, ensure you have the following:

1.  **Python 3.8+**.
2.  **Braintrust Account**: A free Braintrust account and an [API Key](https://www.google.com/search?q=https://www.braintrustdata.com/app/settings/api-keys).
3.  **Azure Subscription**:
      - Access to Azure OpenAI Service.
      - An Azure OpenAI Endpoint and API Key.
      - Two model deployments in Azure AI Studio:
          - A chat model (e.g., `gpt-35-turbo`, `o3-mini`).
          - An embedding model (e.g., `text-embedding-ada-002`).
4.  **MongoDB Atlas Cluster**:
      - A MongoDB cluster with a database and collection.
      - Your data loaded into the collection. Each document must contain the vector embedding.
      - A MongoDB Atlas Vector Search Index configured on your collection.

### MongoDB Vector Index Example

Your vector search index, defined in MongoDB Atlas, should target the field containing your embeddings. Based on the script, the index should be configured like this:

  - **Index Name**: `embeddings_1_search_index` (or your chosen name)
  - **Target Field**: `embeddings`

## Setup and Configuration

### 1\. Clone or Download the Script

Save the provided Python script as `braintrust-local-eval.py`.

### 2\. Install Dependencies

It's recommended to use a virtual environment. Create a `requirements.txt` file with the following content:

```txt
braintrust
autoevals
pymongo
openai
python-dotenv
```

Then, install the packages:

```bash
pip install -r requirements.txt
```

### 3\. Configure Environment Variables

The script is configured using environment variables. Create a file named `.env` in the same directory and add your credentials. **Never commit this file to version control.**

```ini
# .env file

# 1. Braintrust API Key
BRAINTRUST_API_KEY="sk-..."

# 2. Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT="https://<your-resource-name>.openai.azure.com/"
AZURE_OPENAI_API_KEY="your_azure_api_key"

# 3. MongoDB Configuration
MONGO_URI="mongodb+srv://<user>:<password>@<cluster-url>.mongodb.net/?retryWrites=true&w=majority"
```

The script will automatically load these variables if `python-dotenv` is installed.

### 4\. Update Script Constants

Review the configuration section in the Python script and ensure the following constants match your specific setup in Azure and MongoDB.

```python
# Azure deployment names must match those in your Azure AI Studio.
AZURE_CHAT_DEPLOYMENT_NAME = "o3-mini"
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002"

# MongoDB database, collection, and index names.
DB_NAME = "knowledge_base"
COLLECTION_NAME = "embeddings_1"
VECTOR_INDEX_NAME = "embeddings_1_search_index"
```

## Usage

To run both the local validation and the Braintrust evaluation, simply execute the script from your terminal:

```bash
python braintrust-local-eval.py
```

### Expected Output

You will see output in your console from two stages:

1.  **Braintrust Evaluation (`run_eval`)**:
    The script will first log its connection to MongoDB and then start the evaluation. Upon completion, you will see a message prompting you to view the results online.

    ```
    Successfully connected to MongoDB.
    Starting Braintrust evaluation for RAG task...
    Generating embedding for query: 'how do i request a domain to be whitelisted?'
    Performing vector search in MongoDB...
    Retrieved 5 documents from MongoDB.
    ...
    Evaluation complete. Check your Braintrust dashboard for the 'Azure-OpenAI-RAG-Factuality-Check' project.
    ```

    You can then log in to your Braintrust account to analyze the scores, outputs, and metadata for each test case.

2.  **Local Validation (`validate_locally`)**:
    Next, the script will run the local test, printing a detailed, color-coded comparison for each item in your test dataset. This is useful for immediate feedback.

    ```
    --- Starting Local Validation ---

    ========== Test Case 1 ==========

    --- Comparison ---
    INPUT:    how do i request a domain to be whitelisted?
    EXPECTED: To request a domain to be whitelisted, please ensure you have your manager's approval...
    GOT:      {"response":"To request a domain to be whitelisted, you need approval from your manager...","used_provided_context":true}
    =================================
    RESULT:   Contains expected text.

    ========== Test Case 2 ==========
    ...
    ```

## Customization

### Modifying the Test Data

To add or change the evaluation dataset, simply edit the `TEST_DATASET` list in the script. Each item requires an `input` (the question) and an `expected` value (the ideal answer for scoring).

```python
TEST_DATASET = [
    {
        "input": "What is my team's budget for Q3?",
        "expected": "Your team's budget for Q3 is $50,000, allocated for software and travel.",
    },
    # Add more test cases here
]
```

### Adjusting the RAG Prompt

The core logic for the language model is defined in the `messages` array within the `run_rag_task` function. You can modify the `system` prompt to change the model's instructions, persona, or output format.

### Tweaking Vector Search

In the `perform_vector_search` function, you can adjust the `$vectorSearch` parameters to optimize retrieval:

  - `numCandidates`: The number of nearest neighbors to consider during the search. Increasing this can improve accuracy at the cost of performance.
  - `limit`: The final number of documents to return.

```python
import os
import sys
from pymongo import MongoClient
from braintrust import Eval
from autoevals import Factuality
from openai import AzureOpenAI, APIError, AuthenticationError

# --- Configuration ---
# It's best practice to set secrets as environment variables.
# For example, in your terminal: `export BRAINTRUST_API_KEY="your_key_here"`

# 1. Braintrust API Key
BRAINTRUST_API_KEY = os.environ.get("BRAINTRUST_API_KEY", "sk-")
os.environ["BRAINTRUST_API_KEY"] = BRAINTRUST_API_KEY

# 2. Azure OpenAI Configuration
AZURE_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT", "https://.openai.azure.com/")
AZURE_API_KEY = os.environ.get("AZURE_OPENAI_API_KEY", "")
AZURE_API_VERSION = "2024-12-01-preview"

# Deployment names must match those in your Azure AI Studio.
#AZURE_CHAT_DEPLOYMENT_NAME = "gpt-35-turbo"
AZURE_CHAT_DEPLOYMENT_NAME = "o3-mini"
# Add a deployment name for the model used to create embeddings for vector search.
AZURE_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada-002" # Replace with your embedding model deployment

# 3. MongoDB Configuration (Inspired by the provided Flask App)
# In your terminal: `export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true&w=majority"`
MONGO_URI = "" #os.environ.get("MONGO_URI") # This is required for the script to run
DB_NAME = "" # e.g., "knowledge_base"
COLLECTION_NAME = "embeddings_1" # e.g., "embeddings"
VECTOR_INDEX_NAME = "embeddings_1_search_index" # e.g., "vector_index"
# --- Client Initialization ---
# Initialize Azure OpenAI Client
azure_client = None
try:
    if not AZURE_ENDPOINT or not AZURE_API_KEY:
        raise ValueError("Azure endpoint or API key is not configured. Please set the environment variables.")

    azure_client = AzureOpenAI(
        api_version=AZURE_API_VERSION,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
    )
except Exception as e:
    print(f"Fatal Error: Could not initialize AzureOpenAI client. Check your configuration. Details: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize MongoDB Client
mongo_client = None
if MONGO_URI:
    try:
        mongo_client = MongoClient(MONGO_URI)
        db = mongo_client[DB_NAME]
        collection = db[COLLECTION_NAME]
        print("Successfully connected to MongoDB.")
    except Exception as e:
        print(f"Fatal Error: Could not connect to MongoDB. Check MONGO_URI. Details: {e}", file=sys.stderr)
        sys.exit(1)
else:
    print("Warning: MONGO_URI is not set. The RAG task will not be able to query the database.", file=sys.stderr)


# --- Helper Functions for RAG ---

def get_embedding(text: str, model: str = AZURE_EMBEDDING_DEPLOYMENT_NAME) -> list[float]:
    """Generates a vector embedding for a given text using Azure OpenAI."""
    if not azure_client:
        raise ValueError("Azure client is not initialized.")
    try:
        response = azure_client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}", file=sys.stderr)
        return []

def perform_vector_search(vector: list[float]) -> list[dict]:
    """
    Performs a $vectorSearch query in MongoDB to find relevant documents.
    This function is similar to the `knowledge_repo.vector_search` in the provided Flask app.
    """
    if not mongo_client:
        print("Cannot perform vector search, MongoDB client not initialized.", file=sys.stderr)
        return []

    # This is the aggregation pipeline that uses the vector index
    pipeline = [
        {
            "$vectorSearch": {
                "index": VECTOR_INDEX_NAME,
                "path": "embeddings",  # The field in your documents that contains the vector
                "queryVector": vector,
                "numCandidates": 150, # Number of candidates to consider
                "limit": 5          # Number of results to return
            }
        },
        {
            "$project": {
                "_id": 0,
                "score": {"$meta": "vectorSearchScore"},
                "title": 1,
                "text": 1,
                "source": 1,
            }
        }
    ]
    try:
        results = list(collection.aggregate(pipeline))
        return results
    except Exception as e:
        print(f"Error during vector search in MongoDB: {e}", file=sys.stderr)
        return []


# --- Task Definition (RAG Workflow) ---

def run_rag_task(input_prompt: str) -> str:
    """
    Executes the full Retrieval-Augmented Generation (RAG) task:
    1. Generates an embedding for the input.
    2. Retrieves context from MongoDB via vector search.
    3. Generates a final response using the retrieved context.
    """
    if not azure_client:
        return "Error: Azure OpenAI client is not initialized."
    if not mongo_client:
        return "Error: MongoDB client is not initialized."

    # 1. Get query vector
    print(f"Generating embedding for query: '{input_prompt}'")
    query_vector = get_embedding(input_prompt)
    if not query_vector:
        return "Error: Failed to generate embedding for the query."

    # 2. Query MongoDB for context
    print("Performing vector search in MongoDB...")
    context_docs = perform_vector_search(query_vector)
    if not context_docs:
        print("No context found from vector search.", file=sys.stderr)
        # Fallback: answer without context
        context_str = "No specific context was found."
    else:
        print(f"Retrieved {len(context_docs)} documents from MongoDB.")
        # Format the context for the prompt
        context_str = "\n".join([f"- {doc['text']}" for doc in context_docs])

    # 3. Generate response from the chat model with the retrieved context
    
    try:
        response = azure_client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT_NAME,
            messages=[
                    {
                        "role": "system",
                        "content": f"""
                            Answer the query to the best of your ability with the provided context ONLY and what you know.
                            Do not make up any part of your response.
                            ALWAYS RESPOND IN SLACK MARKDOWN FORMAT (mrkdwn), which is different than classic markdown.More actions
                            ## Slack Markdown vs. Standard Markdown: Key Differences

                                Slack's `mrkdwn` is a custom version of Markdown tailored for messaging. Here's a concise comparison:

                                Links:
                                Slack: \<https://www.google.com/search?q=url|displayText\>
                                Standard: [displayText](https://www.google.com/search?q=url)

                                Emphasis:
                                Slack: *bold*, *italics*, \~strikethrough\~
                                Standard: **bold** or **bold**, *italics* or *italics*, \~\~strikethrough\~\~ (GFM)

                                Headings (\# H1):
                                Slack: Not in messages; limited in Posts.
                                Standard: Fully supported.

                                Images & Tables:
                                Slack: No direct Markdown syntax (uploads for images).
                                Standard:  for images; table syntax (GFM).

                                HTML:
                                Slack: Not supported.
                                Standard: Often allows inline HTML.

                                Slack-Specific:
                                Features like mentions (@user, \#channel) and date formatting. Standard Markdown lacks these.

                                In essence, Slack `mrkdwn` is simpler and includes platform-specific features, while standard Markdown is more robust for general content creation.
                            Provide the answer in JSON, with the answer for the user in the field `response` and whether the provided context was used in `used_provided_context`.
                            [context]{context_str}[/context]
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                            [response_criteria]
                            - Provide a concise answer to the question to the best of your ability based on the context and what you know.
                            - Provide the answer in slack markdown, which is different than classic markdown. Keep in mind the block max size is 4000 characters.
                                ## Slack Markdown vs. Standard Markdown: Key Differences

                                Slack's `mrkdwn` is a custom version of Markdown tailored for messaging. Here's a concise comparison:

                                Links:
                                Slack: \<https://www.google.com/search?q=url|displayText\>
                                Standard: [displayText](https://www.google.com/search?q=url)

                                Emphasis:
                                Slack: *bold*, *italics*, \~strikethrough\~
                                Standard: **bold** or **bold**, *italics* or *italics*, \~\~strikethrough\~\~ (GFM)

                                Headings (\# H1):
                                Slack: Not in messages; limited in Posts.
                                Standard: Fully supported.

                                Images & Tables:
                                Slack: No direct Markdown syntax (uploads for images).
                                Standard:  for images; table syntax (GFM).

                                HTML:
                                Slack: Not supported.
                                Standard: Often allows inline HTML.

                                Slack-Specific:
                                Features like mentions (@user, \#channel) and date formatting. Standard Markdown lacks these.

                                In essence, Slack `mrkdwn` is simpler and includes platform-specific features, while standard Markdown is more robust for general content creation.
                            [/response_criteria]
                            
                            [question]
                            {input_prompt}
                            [/question]
                        """
                    }
                ],
                stream=False,
                response_format={"type": "json_object"}
            )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An unexpected error occurred during chat completion: {e}", file=sys.stderr)
        return ""


# --- Braintrust Evaluation ---
# Define a shared dataset for both local validation and Braintrust evaluation.
TEST_DATASET = [
    {
        "input": "how do i request a domain to be whitelisted?",
        "expected": "To request a domain to be whitelisted, please ensure you have your manager's approval. Once you have that, you can proceed with the request. Note that only Enterprise Accounts may be whitelisted, and the domain send limit uses a 24-hour rolling period. If you have previously whitelisted a domain, it does not guarantee it won't be blacklisted in the future.",
    },
    {
        "input": "What is the capital of France?",
        "expected": "The capital of France is Paris.",
    },
]

def run_eval():
    """Sets up and runs the Braintrust evaluation for the RAG task."""
    if not MONGO_URI:
        print("Fatal Error: MONGO_URI is not set. Cannot run evaluation.", file=sys.stderr)
        sys.exit(1)

    print("Starting Braintrust evaluation for RAG task...")
    eval_name = "Azure-OpenAI-RAG-Factuality-Check"
    Eval(
      eval_name,
      data=lambda: TEST_DATASET, # Use the shared dataset for evaluation
      task=run_rag_task, # The task now performs the entire RAG workflow
      scores=[Factuality],
    )
    print(f"Evaluation complete. Check your Braintrust dashboard for the '{eval_name}' project.")
def validate_locally():
    """
    Runs the RAG task on a local dataset and prints the 'expected' vs 'got'
    output directly to the console for immediate validation.
    """
    print("--- Starting Local Validation ---")

    # 1. Iterate through each test case in the dataset
    for i, item in enumerate(TEST_DATASET):
        print(f"\n{'='*10} Test Case {i+1} {'='*10}")
        input_prompt = item["input"]
        expected_output = item["expected"]

        # 2. Get the actual ("got") output from your RAG task
        got_output = run_rag_task(input_prompt)

        # 3. Print the results for comparison
        print("\n--- Comparison ---")
        print(f"INPUT:    {input_prompt}")
        print(f"EXPECTED: {expected_output}")
        print(f"GOT:      {got_output}")
        print(f"{'='*33}")

        # 4. Optional: A simple programmatic check.
        # Note: For LLM outputs, a direct string comparison (==) is often too strict
        # because valid answers can be phrased differently. The `Factuality` scorer
        # you use in Braintrust performs a more sophisticated, semantic comparison.
        if expected_output.lower() in str(got_output).lower():
             print("RESULT:   Contains expected text.")
        else:
             print("RESULT:   Does NOT contain expected text.")
if __name__ == "__main__":
    run_eval()
    # Option 1: Run the local validation to see "expected vs. got" in your console.
    validate_locally()
```
