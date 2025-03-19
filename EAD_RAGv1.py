import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from openai import OpenAI
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize fallback documents for RAG retrieval
docs = [
    Document(page_content="Exploratory Data Analysis involves summarizing data characteristics."),
    Document(page_content="Handling missing values is crucial for improving model performance."),
]

# Initialize vectorstore if docs are available
vectorstore = FAISS.from_documents(docs, embeddings) if docs else None

# Function to load dataset from file or URL
def load_data(file, url):
    try:
        if file:
            df = pd.read_csv(file.name)
        elif url.strip():
            response = requests.get(url.strip())
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))
        else:
            raise ValueError("Please upload a file or enter a valid URL.")

        # Ensure df is a DataFrame
        if isinstance(df, pd.Series):
            df = df.to_frame()

    except Exception as e:
        return None, f"Error loading data: {str(e)}"
    return df, None

# Retrieve relevant context from RAG
def retrieve_context(query, k=3):
    if not docs:
        return "No additional context available."
    retrieved_docs = vectorstore.similarity_search(query, k=k)
    return "\n---\n".join([doc.page_content for doc in retrieved_docs])

# Generate AI-based summary with RAG
def generate_summary(df, rag_depth):
    if df is None:
        return "Error: No dataset loaded."
    
    # Generate dataset overview
    data_overview = "\n".join([
        f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns.",
        "Column Types:\n" + df.dtypes.to_string(),
        "\nMissing Values:\n" + df.isnull().sum().to_string(),
        "\nDescriptive Stats:\n" + df.describe().to_string()
    ])

    retrieved_context = retrieve_context("Exploratory Data Analysis best practices", k=rag_depth)

    prompt = f"""
    You are a data analysis expert. Given the dataset details below and additional context from relevant documents:

    Dataset details:
    {data_overview}

    Relevant Context:
    {retrieved_context}

    Provide:
    1. Concise summary of the data.
    2. Key insights (patterns, trends, anomalies).
    3. Detailed preprocessing recommendations (data types, missing values, cleaning), referencing relevant context where applicable.

    Structure your response in clear markdown format.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=1500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Generate visualizations for numerical columns
def visualize(df):
    if df is None:
        return None, "Error: No dataset available for visualization."

    numerical_cols = df.select_dtypes(include='number').columns
    if len(numerical_cols) == 0:
        return None, "No numerical columns available for visualization."

    num_cols = min(len(numerical_cols), 4)  # Limit to 4 columns

    fig, axs = plt.subplots(nrows=num_cols, ncols=2, figsize=(12, 4 * num_cols))

    # Ensure axs is always 2D for consistent indexing
    if num_cols == 1:
        axs = [axs]

    for i, col in enumerate(numerical_cols[:num_cols]):
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue', ax=axs[i][0])
        axs[i][0].set_title(f'Distribution of {col}', fontsize=14)

        sns.boxplot(x=df[col].dropna(), color='orange', ax=axs[i][1])
        axs[i][1].set_title(f'Boxplot of {col}', fontsize=14)

    plt.tight_layout()
    viz_path = "visualizations.png"
    plt.savefig(viz_path)
    plt.close()

    return viz_path, None

# Main function combining everything
def eda_analysis(file, url, rag_depth):
    df, load_error = load_data(file, url)
    if load_error:
        return load_error, None, None
    
    summary = generate_summary(df, rag_depth)
    viz_image, viz_error = visualize(df)
    
    if viz_error:
        return summary, None, df
    return summary, viz_image, df

# Gradio UI with enhanced layout and styling
with gr.Blocks(theme=gr.themes.Soft(primary_hue="emerald"), title="AI-Enhanced EDA") as iface:
    # Header Section
    gr.Markdown("""
    <div style='text-align:center; padding:20px; background: linear-gradient(135deg, #2c3e50, #3498db); border-radius:10px;'>
        <h1 style='color: white;'>🔍 AI-Powered Exploratory Data Analysis</h1>
        <p style='color: #ecf0f1;'>Upload your dataset or provide a URL for instant analysis</p>
    </div>
    """)

    # Main Content Container
    with gr.Tabs():
        with gr.TabItem("📊 Data Analysis"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    file_input = gr.File(label="📁 Upload CSV", file_types=[".csv"])
                    url_input = gr.Textbox(label="🌍 CSV URL", placeholder="Paste dataset URL here...")
                    rag_slider = gr.Slider(minimum=1, maximum=5, value=3, label="Context Retrieval Depth (RAG)")
                    analyze_btn = gr.Button("🚀 Start Analysis", variant="primary")
                    clear_btn = gr.Button("🔄 Clear", variant="secondary")

                with gr.Column(scale=2, min_width=800):
                    summary_md = gr.Markdown("# Analysis Results\n*Upload data to begin*")
                    viz_img = gr.Image(label="Automated Visualizations", interactive=True, show_download_button=True)
                    data_preview = gr.Dataframe(value=pd.DataFrame(), row_count=10, interactive=True)

        with gr.TabItem("❓ How It Works"):
            gr.Markdown("""
            ## Workflow Overview
            1. **Data Input**: Upload CSV or provide URL.
            2. **Automated Analysis**:
               - Data profiling & quality checks.
               - Context-aware insights via RAG.
               - Smart visualization generation.
            3. **Interactive Exploration**:
               - Expandable results sections.
               - Tabbed visualization interface.
            """)

    analyze_btn.click(eda_analysis, inputs=[file_input, url_input, rag_slider], outputs=[summary_md, viz_img, data_preview])
    clear_btn.click(lambda: [None, None, None, None, None], outputs=[file_input, url_input, summary_md, viz_img, data_preview])

if __name__ == "__main__":
    iface.launch()
