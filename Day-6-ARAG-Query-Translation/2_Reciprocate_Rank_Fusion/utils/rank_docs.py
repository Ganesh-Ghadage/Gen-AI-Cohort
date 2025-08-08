from typing import List, Dict
from collections import defaultdict
from langchain_core.documents import Document

def rank_documents(
  docs: List[Document],
  k: int = 60
) -> List[Document]:
  """
  Apply Reciprocal Rank Fusion (RRF) to combine multiple ranked lists.

  Args:
    docs List[Document]: list of documents.
    k (int): RRF constant. Defaults to 60.

  Returns:
    List[Document]: Final ranked list of documents with highest combined RRF scores.
  """
  doc_scores = defaultdict(float)
  doc_store = {}

  for rank, doc in enumerate(docs):
    doc_id = doc.metadata.get("id") or hash(doc.page_content.strip())
    score = 1 / (k + rank)
    doc_scores[doc_id] += score
    # Store one version of the doc
    if doc_id not in doc_store:
      doc_store[doc_id] = doc

  # Sort by RRF score (descending)
  ranked_doc_ids = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

  # for doc_id, score in ranked_doc_ids:
  #   print(f"doc_id: {doc_id}, score : {score} ")

  # Return the corresponding Document objects
  return [doc_store[doc_id] for doc_id, _ in ranked_doc_ids]


# def reciprocal_rank_fusion(rankings, k=60):
#   scores = {}
#   for ranking in rankings:
#     for rank, doc_id in enumerate(ranking):
#       scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
  
#   return sorted(scores.items(), key=lambda x: x[1], reverse=True)