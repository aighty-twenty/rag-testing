## Script to compare query types to integrated vectorization-based index
## Compare https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/integrated-vectorization/azure-search-integrated-vectorization-sample.ipynb

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType

import re
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
AISEARCH_ENDPOINT = os.getenv('AISEARCH_ENDPOINT')
AISEARCH_INDEX = os.getenv('AISEARCH_INDEX')
AISEARCH_KEY = os.getenv('AISEARCH_KEY')

NUMBER_OF_RESULTS = 2

query = "Biggest city"

search_client = SearchClient(
    AISEARCH_ENDPOINT, AISEARCH_INDEX, AzureKeyCredential(AISEARCH_KEY)
)
vector_query = VectorizableTextQuery(
    text=query, k_nearest_neighbors=10, fields="embedding", exhaustive=True
)


# pure text search
text_results = search_client.search(
    search_text=query,
    select=["id", "content", "sourcefile", "sourcepage", "category"],
    top=NUMBER_OF_RESULTS,
)


# pure vector search
vector_results = search_client.search(
    search_text=None,
    vector_queries=[vector_query],
    select=["id", "content", "sourcefile", "sourcepage", "category"],
    top=NUMBER_OF_RESULTS,
)


# hybrid (vector + full-text) search
hybrid_results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["id", "content", "sourcefile", "sourcepage", "category"],
    top=NUMBER_OF_RESULTS,
)


# semantic hybrid search (vector + full-text + semantic reranking)
semantic_results = search_client.search(
    search_text=query,
    vector_queries=[vector_query],
    select=["id", "content", "sourcefile", "sourcepage", "category"],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="default",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=NUMBER_OF_RESULTS,
)

# Initialize DataFrame
results_df = pd.DataFrame(
    columns=["Query Type", "Search Score", "Reranker Score", "Content"]
)


# Define a helper function to process results and modify the global DataFrame
def process_results(results, query_type, df):
    for result in results:
        cleaned_content = re.sub(r"\s+", " ", result["content"])
        df.loc[len(df)] = [
            query_type,
            result["@search.score"],
            result["@search.reranker_score"],
            cleaned_content,
        ]
    return df


# Process results
process_results(text_results, "Text Search", results_df)
process_results(vector_results, "Vector Search", results_df)
process_results(hybrid_results, "Hybrid Search", results_df)
process_results(semantic_results, "Semantic Hybrid Search", results_df)

transposed_df = results_df.T

# Print the transposed DataFrame
print(transposed_df)

transposed_df.to_excel("search_results.xlsx", index=True, engine="openpyxl")
