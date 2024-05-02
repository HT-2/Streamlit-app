from haystack.document_stores.in_memory import InMemoryDocumentStore
document_store = InMemoryDocumentStore()
from datasets import load_dataset
from haystack import Document
dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()
from haystack.components.embedders import SentenceTransformersTextEmbedder
text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
retriever = InMemoryEmbeddingRetriever(document_store)
from haystack.components.builders import PromptBuilder

template = """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""

prompt_builder = PromptBuilder(template=template)
import os
from google.colab import userdata
os.environ["GOOGLE_API_KEY"] = "AIzaSyAqqiRDjFQsmVZ2hIkR7UMWShx6QVxq_Kc"
os.environ["HF_API_TOKEN"] = "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"
from haystack_integrations.components.generators.google_ai import GoogleAIGeminiGenerator
generator = GoogleAIGeminiGenerator(model="gemini-pro")
from haystack import Pipeline

from haystack import Pipeline
# Load the pipeline
basic_rag_pipeline = Pipeline()
# Add components to your pipeline
basic_rag_pipeline.add_component("text_embedder", text_embedder)
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
basic_rag_pipeline.connect("retriever", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder", "llm")

import streamlit as st
# Create a Streamlit app
def main():
    user_input = st.text_input("Enter a question:")
    if st.button("Submit"):
        try:
            response = basic_rag_pipeline.run({"text_embedder": {"text": user_input}, "prompt_builder": {"question": user_input}})
            st.write("Answer:", response["llm"]["replies"][0])
        except Exception as e:
            st.write(f"An error occurred: {e}")

if __name__ == '__main__':
    main()
