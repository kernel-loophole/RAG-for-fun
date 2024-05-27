from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.document_transformers import Html2TextTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
import nest_asyncio

nest_asyncio.apply()

loader = DirectoryLoader('../rag_model/data', glob="**/*.txt")
docs = loader.load()
print(len(docs))
# Converts HTML to plain text 
html2text = Html2TextTransformer()
docs_transformed = html2text.transform_documents(docs)

# Chunk text
text_splitter = CharacterTextSplitter(chunk_size=10, 
                                      chunk_overlap=0)
chunked_documents = text_splitter.split_documents(docs_transformed)

# Load chunked documents into the FAISS index
db = FAISS.from_documents(chunked_documents, 
                          HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))


# Connect query to FAISS index using a retriever
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 2}
)
text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)

print('================================================================')
print(text_splitter)
query=input('query:')

docs = db.similarity_search(query)
print(docs[0].page_content)