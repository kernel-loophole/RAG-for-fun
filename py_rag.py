# Load Dataset

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("thenlper/gte-large")

from datasets import load_dataset
import pandas as pd

# https://huggingface.co/datasets/AIatMongoDB/embedded_movies
dataset = load_dataset("AIatMongoDB/embedded_movies")

# Convert the dataset to a pandas dataframe
dataset_df = pd.DataFrame(dataset["train"])

dataset_df.head(5)
# Data Preparation

# Remove data point where plot coloumn is missing
dataset_df = dataset_df.dropna(subset=["fullplot"])
print("\nNumber of missing values in each column after removal:")
print(dataset_df.isnull().sum())

# Remove the plot_embedding from each data point in the dataset as we are going to create new embeddings with an open source embedding model from Hugging Face
dataset_df = dataset_df.drop(columns=["plot_embedding"])
dataset_df.head(5)

def get_embedding(text: str) -> list[float]:
    if not text.strip():
        print("Attempted to get embedding for empty text.")
        return []

    embedding = embedding_model.encode(text)

    return embedding.tolist()


dataset_df["embedding"] = dataset_df["fullplot"].apply(get_embedding)

dataset_df.head()
import pymongo



def get_mongo_client(mongo_uri):
    """Establish connection to the MongoDB."""
    try:
        client = pymongo.MongoClient(mongo_uri)
        print("Connection to MongoDB successful")
        return client
    except pymongo.errors.ConnectionFailure as e:
        print(f"Connection failed: {e}")
        return None


mongo_uri ="mongodb+srv://zulqarnain:24247676@ac-08i01lv-shard-00-00.cicz70u.mongodb.net:27017/?retryWrites=true&w=majority";
if not mongo_uri:
    print("MONGO_URI not set in environment variables")

mongo_client = get_mongo_client(mongo_uri)

# Ingest data into MongoDB
db = mongo_client["movies"]
collection = db["movie_collection_2"]
documents = dataset_df.to_dict("records")
collection.insert_many(documents)
print("Data ingestion into MongoDB completed")
def vector_search(user_query, collection):
    """
    Perform a vector search in the MongoDB collection based on the user query.

    Args:
    user_query (str): The user's query string.
    collection (MongoCollection): The MongoDB collection to search.

    Returns:
    list: A list of matching documents.
    """

    # Generate embedding for the user query
    query_embedding = get_embedding(user_query)

    if query_embedding is None:
        return "Invalid query or embedding generation failed."

    # Define the vector search pipeline
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 150,  # Number of candidate matches to consider
                "limit": 4,  # Return top 4 matches
            }
        },
        {
            "$project": {
                "_id": 0,  # Exclude the _id field
                "fullplot": 1,  # Include the plot field
                "title": 1,  # Include the title field
                "genres": 1,  # Include the genres field
                "score": {"$meta": "vectorSearchScore"},  # Include the search score
            }
        },
    ]

    # Execute the search
    results = collection.aggregate(pipeline)
    return list(results)
def get_search_result(query, collection):

    get_knowledge = vector_search(query, collection)

    search_result = ""
    for result in get_knowledge:
        search_result += f"Title: {result.get('title', 'N/A')}, Plot: {result.get('fullplot', 'N/A')}\n"

    return search_result
# Conduct query with retrival of sources
query = "What is the best romantic movie to watch and why?"
source_information = get_search_result(query, collection)
combined_information = (
    f"Query: {query}\nContinue to answer the query by using the Search Results:\n{source_information}."
)

print(combined_information)