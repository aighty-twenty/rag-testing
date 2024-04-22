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
from copy import deepcopy


import pandas as pd
from pydantic import BaseModel

class SearchResult(BaseModel):
    search_type: str | None
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
    NUMBER_OF_RESULTS = 3

    query = "What is the biggest city in the world?"
    expected_output = "The biggest city in the world is Tokyo, Japan."
    llm_output = ""


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
    
    # Evaluation, create the test cases and metrics per search_type
    results_list_1 = [deepcopy(text_results), deepcopy(vector_results), deepcopy(hybrid_results), deepcopy(semantic_results)]
    results_list_2 = [deepcopy(text_results), deepcopy(vector_results), deepcopy(hybrid_results), deepcopy(semantic_results)]

    test_cases = []
    precision_metrics = []
    recall_metrics = []
    relevancy_metrics = []

    for results in results_list_1: 
        retrieval_context = []
        for result in results:
            retrieval_context.append(re.sub(r"\s+", " ", result["content"]))
                
        test_case = LLMTestCase(
            input=query,
            actual_output=llm_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        test_cases.append(test_case)
        precision_metrics.append(ContextualPrecisionMetric(include_reason=True))
        recall_metrics.append(ContextualRecallMetric(include_reason=True))
        relevancy_metrics.append(ContextualRelevancyMetric(include_reason=True))

    tasks = []
    for i, test_case in enumerate(test_cases):
        tasks.append(precision_metrics[i].a_measure(test_case))
        tasks.append(recall_metrics[i].a_measure(test_case))
        tasks.append(relevancy_metrics[i].a_measure(test_case))
    
    await asyncio.gather(*tasks)

    # Bring everything together
    df1_list: list[dict] = []
    df2_list: list[dict] = []
    
    search_types = ["Pure Text", "Pure Vector", "Hybrid", "Semantic Hybrid"]

    # Write search results to DataFrame
    
    # Loop over each result in retrieval_results
    for i, results in enumerate(results_list_2):
        search_type = search_types[i]

        for result in results:
            df1_list.append({
                "Search Type": search_type,
                "Search Score": result["@search.score"],
                "Reranker Score": result["@search.reranker_score"],
                "Content": re.sub(r"\s+", " ", result["content"])
            })
        
        # evaluation results per search type
        df2_list.append({
            "Search Type": search_type,
            "Contextual Precision": precision_metrics[i].score,
            "Contextual Precision Reason": precision_metrics[i].reason,
            "Contextual Recall": recall_metrics[i].score,
            "Contextual Recall Reason": recall_metrics[i].reason,
            "Contextual Relevancy": relevancy_metrics[i].score,
            "Contextual Relevancy Reason": relevancy_metrics[i].reason,
        })

    # Create DataFrames from the lists of dictionaries
    results_df1 = pd.DataFrame(df1_list, index=None)
    transposed_df1 = results_df1.T

    results_df2 = pd.DataFrame(df2_list, index=None)
    transposed_df2 = results_df2.T
   
    # Write both dataframes to Excel file
    with pd.ExcelWriter("eval_results.xlsx", engine='openpyxl') as writer:
        transposed_df1.to_excel(writer, sheet_name='Search & Eval')
        startrow = len(transposed_df1) + 2  
        transposed_df2.to_excel(writer, sheet_name='Search & Eval', startrow=startrow)
    

if __name__ == "__main__":
    asyncio.run(run())