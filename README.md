# README #

## Two separate RAG applications

This repository has two folders for setting up two separate RAG applications on Microsoft Azure. Firstly, based on data ingestion via the [layout model](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout?view=doc-intel-4.0.0) of Azure AI Document Intelligence (formerly Form Recognizer) and secondly via the [integrated vectorization feature of Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization). Both folders are a clone of this [Microsoft repository](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main)


In this repository, the two folders are identical at first. However, for using the two different data ingestion methods, in the integrated-vectorization an env variable needs to be set via `azd env set USE_FEATURE_INT_VECTORIZATION true` and then the deployment should be done via `azd up` in both folders.

For cost-efficiency, some resources can in principal be shared between the two applications and also be reused from previous deployments but for some quick testing having separate resources is easier to set up. Refer to these [docs](https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/docs/deploy_existing.md) for more details on sharing resources.

## Sample PDF document

In the data folder a sample PDF document with some longer tables is provided to resemble a test case.

## query.py & eval.py

You need to set the env variables in a `.env` file in order to run the Python scripts. See `.env.example` for the necessary variables.

The `query.py` script is used to query the Azure AI Search service with integrated vectorization directly. It runs four different kinds of queries: full-text, vector, hybrid and hybrid semantic.

The `eval.py` uses DeepEval in order to assign scores to the retrieval results. 

