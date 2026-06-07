import warnings
warnings.filterwarnings("ignore")



# Plan of Action; Modus Operandi
"""
Load a document
Split it and  convert to embeddings then store in a DB / Vector Store
Enter query---> Convert to embeddings--> send to Retriever(which sends it to the VS for similarity search)
It returns valid document chunks. Send the Chunks and query as a prompt to LLM
Fin
"""