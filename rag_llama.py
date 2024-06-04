from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from huggingface_hub import login

login(token="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")

# Set trust_remote_code=True
Settings.trust_remote_code = True

documents = SimpleDirectoryReader("data").load_data()
print(documents)

# bge-base embedding model
Settings.embed_model = HuggingFaceEmbedding(model_name="microsoft/Phi-3-small-8k-instruct",trust_remote_code=True)

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

index = VectorStoreIndex.from_documents(documents)

while True:
    query_engine = index.as_query_engine()
    query = input("query:")
    response = query_engine.query(query)
    print(response)
