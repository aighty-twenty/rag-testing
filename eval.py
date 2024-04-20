## Script to compare query types to integrated vectorization-based index
## Compare https://github.com/Azure/azure-search-vector-samples/blob/main/demo-python/code/integrated-vectorization/azure-search-integrated-vectorization-sample.ipynb

from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.models import (
    QueryType,
    QueryCaptionType,
    QueryAnswerType
)

from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

import re
import subprocess
import os
import asyncio
from dotenv import load_dotenv

import pandas as pd
from pydantic import BaseModel

class SearchResult(BaseModel):
    queryType: str | None
    search_score: float | None
    reranker_score: float | None
    cleaned_content: str | None
    evaluation: dict | None = None


# Env variables & global variables
load_dotenv()

AISEARCH_ENDPOINT = os.getenv('AISEARCH_ENDPOINT')
AISEARCH_INDEX = os.getenv('AISEARCH_INDEX')
AISEARCH_KEY = os.getenv('AISEARCH_KEY')

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

async def run():
    # Set Azure OpenAI as default LLM for evaluations
    azure_openai_commands = [
        f'deepeval set-azure-openai --openai-endpoint={AZURE_OPENAI_ENDPOINT} \
        --openai-api-key={AZURE_OPENAI_API_KEY} \
        --deployment-name={AZURE_OPENAI_DEPLOYMENT_NAME} \
        --openai-api-version={AZURE_OPENAI_API_VERSION}'
    ]
    for cmd in azure_openai_commands:
        result = subprocess.run(cmd, shell=True, text=True)


    # Retrieval and evaluation parameters
    NUMBER_OF_RESULTS = 2

    query = "What is the biggest city in the world?"
    expected_output = "The biggest city in the world is Tokyo, Japan."
    llm_output = "Thailand"


    # Search queries
    search_client = SearchClient(AISEARCH_ENDPOINT, AISEARCH_INDEX, AzureKeyCredential(AISEARCH_KEY))
    vector_query = VectorizableTextQuery(text=query, k_nearest_neighbors=10, fields="embedding", exhaustive=True)


    ## pure text search
    text_results = search_client.search(  
        search_text=query,
        select=["id", "content", "sourcefile", "sourcepage", "category"],
        top=NUMBER_OF_RESULTS
    )

    ## pure vector search
    vector_results = search_client.search(  
        search_text=None,
        vector_queries=[vector_query],
        select=["id", "content", "sourcefile", "sourcepage", "category"],
        top=NUMBER_OF_RESULTS
    )

    ## hybrid (vector + full-text) search
    hybrid_results = search_client.search(  
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "sourcefile", "sourcepage", "category"],
        top=NUMBER_OF_RESULTS
    )

    ## semantic hybrid search (vector + full-text + semantic reranking)
    semantic_results = search_client.search(  
        search_text=query,
        vector_queries=[vector_query],
        select=["id", "content", "sourcefile", "sourcepage", "category"],
        query_type=QueryType.SEMANTIC,
        semantic_configuration_name='default',
        query_caption=QueryCaptionType.EXTRACTIVE,
        query_answer=QueryAnswerType.EXTRACTIVE,
        top=NUMBER_OF_RESULTS
    )

    retrieval_results: list[SearchResult] = []

    for result in text_results:
        retrieval_results.append(SearchResult(
            queryType="Text", 
            search_score=result['@search.score'], 
            reranker_score=result['@search.reranker_score'], 
            cleaned_content=re.sub(r'\s+', ' ', result['content'])
        ))

    for result in vector_results:
        retrieval_results.append(SearchResult(
            queryType="Vector", 
            search_score=result['@search.score'], 
            reranker_score=result['@search.reranker_score'], 
            cleaned_content=re.sub(r'\s+', ' ', result['content'])
        ))

    for result in hybrid_results:
        retrieval_results.append(SearchResult(
            queryType="Hybrid", 
            search_score=result['@search.score'], 
            reranker_score=result['@search.reranker_score'], 
            cleaned_content=re.sub(r'\s+', ' ', result['content'])
        ))

    for result in semantic_results:
        retrieval_results.append(SearchResult(
            queryType="Hybrid Semantic", 
            search_score=result['@search.score'], 
            reranker_score=result['@search.reranker_score'], 
            cleaned_content=re.sub(r'\s+', ' ', result['content'])
        ))


    # Evaluation

    test_cases = []
    precision_metrics = []
    recall_metrics = []
    relevancy_metrics = []
    
    for result in retrieval_results:
        retrieval_context = result.cleaned_content
        test_case = LLMTestCase(
            input=query,
            actual_output=llm_output,
            expected_output=expected_output,
            retrieval_context=[retrieval_context]
        )
        test_cases.append(test_case)
        precision_metrics.append(ContextualPrecisionMetric(include_reason=True, strict_mode=False))
        recall_metrics.append(ContextualRecallMetric(include_reason=True))
        relevancy_metrics.append(ContextualRelevancyMetric(include_reason=True))

    tasks = []
    for i, test_case in enumerate(test_cases):
        tasks.append(precision_metrics[i].a_measure(test_case))
        tasks.append(recall_metrics[i].a_measure(test_case))
        tasks.append(relevancy_metrics[i].a_measure(test_case))
    
    await asyncio.gather(*tasks)

    print(precision_metrics)
    print(recall_metrics)
    print(relevancy_metrics)

    for i, _ in enumerate(retrieval_results):
        retrieval_results[i].evaluation = {
            "precision": precision_metrics[i].score,
            "precision_reason": precision_metrics[i].reason,
            "recall": recall_metrics[i].score,
            "recall_reason": recall_metrics[i].reason,
            "relevancy": relevancy_metrics[i].score,
            "relevancy_reason": relevancy_metrics[i].reason
        }

    # Bring everything together

    data_list = []

    # Loop over each result in retrieval_results
    for result in retrieval_results:
        # Create a dictionary for each result and add it to the list
        data_list.append({
            "Query Type": result.queryType,
            "Search Score": result.search_score,
            "Reranker Score": result.reranker_score,
            "Contextual Precision": result.evaluation['precision'],
            "Contextual Precision Reason": result.evaluation['precision_reason'],
            "Contextual Recall": result.evaluation['recall'],
            "Contextual Recall Reason": result.evaluation['recall_reason'],
            "Contextual Relevancy": result.evaluation['relevancy'],
            "Contextual Relevancy Reason": result.evaluation['relevancy_reason'],
            "Content": result.cleaned_content
        })

    # Create DataFrame from the list of dictionaries
    results_df = pd.DataFrame(data_list)

    # Optionally transpose the DataFrame if needed
    transposed_df = results_df.T

    # Print the transposed DataFrame
    print(transposed_df)

    # Save the transposed DataFrame to an Excel file using 'openpyxl' as the engine
    transposed_df.to_excel("search_results.xlsx", index=True, engine='openpyxl')


# Starting the asyncio event loop and running the evaluation
if __name__ == "__main__":
    asyncio.run(run())