# README #

## Two separate RAG applications

This repository has two folders for setting up two separate RAG applications on Microsoft Azure. Firstly, based on data ingestion via the [layout model](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/concept-layout?view=doc-intel-4.0.0) of Azure AI Document Intelligence (formerly Form Recognizer) and secondly via the [integrated vectorization feature of Azure AI Search](https://learn.microsoft.com/en-us/azure/search/vector-search-integrated-vectorization). Both folders are a clone of this [Microsoft repository](https://github.com/Azure-Samples/azure-search-openai-demo/tree/main)

The files to be ingested need to be placed in the `data` folder.

In this repository, the two folders are identical at first. However, for using the two different data ingestion methods, in the integrated-vectorization an env variable needs to be set via `azd env set USE_FEATURE_INT_VECTORIZATION true` and then the deployment should be done via `azd up`. For the document intelligence one `azd up` can be run directly.

For cost-efficiency, some resources can in principal be shared between the two applications and also be reused from previous deployments but for some quick testing having separate resources is easier to set up. Refer to these [docs](https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/docs/deploy_existing.md) for more details on sharing resources.

## Sample PDF document

In the data folder a sample PDF document with some longer tables is provided to resemble a test case.

## query.py & eval.py

The `query.py` script is used to query the Azure AI Search service with integrated vectorization directly. It runs four different kinds of queries: full-text, vector, hybrid and hybrid semantic.

The `eval.py` uses DeepEval in order to assign scores to the retrieval results. 

Check the documentation for the definitions of the used evaluation metrics:
- [Precision](https://docs.confident-ai.com/docs/metrics-contextual-precision)
- [Recall](https://docs.confident-ai.com/docs/metrics-contextual-recall)
- [Relevancy](https://docs.confident-ai.com/docs/metrics-contextual-relevancy)

## Steps to run 
- Clone the repository `git clone https://github.com/aighty-twenty/rag-testing`
- Change your working directory into the repository folder `cd rag-testing`
- Set the env variables in a `.env` file. See `.env.example` for the necessary variables
- Create a Python virtual environment: `python3 -m venv .venv`
- Activate the virtual environment: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Run one of the Python scripts: `python eval.py` or `python query.py`
