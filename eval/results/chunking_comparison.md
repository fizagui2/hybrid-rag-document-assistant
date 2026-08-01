# Chunking Strategy Comparison

Cases evaluated per strategy: {'fixed': 11, 'structure': 11, 'semantic': 11}

## Overall

| Metric | Fixed | Structure | Semantic | Winner |
|---|---|---|---|---|
| correctness | 0.89 | 0.81 | 0.81 | fixed |
| faithfulness | 0.90 | 0.90 | 0.89 | tie (fixed, structure) |
| retrieval_relevance | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| citation_accuracy | 0.57 | 0.57 | 0.57 | fixed |

## ambiguous

| Metric | Fixed | Structure | Semantic | Winner |
|---|---|---|---|---|
| correctness | 0.45 | 0.50 | 0.45 | structure |
| faithfulness | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| retrieval_relevance | N/A | N/A | N/A | — |
| citation_accuracy | 0.67 | 0.00 | 0.00 | fixed |

## multi_hop

| Metric | Fixed | Structure | Semantic | Winner |
|---|---|---|---|---|
| correctness | 0.96 | 0.72 | 0.75 | fixed |
| faithfulness | 0.80 | 0.77 | 0.74 | fixed |
| retrieval_relevance | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| citation_accuracy | 0.23 | 0.33 | 0.33 | tie (structure, semantic) |

## no_answer

| Metric | Fixed | Structure | Semantic | Winner |
|---|---|---|---|---|
| correctness | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| faithfulness | N/A | N/A | N/A | — |
| retrieval_relevance | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| citation_accuracy | N/A | N/A | N/A | — |

## straightforward

| Metric | Fixed | Structure | Semantic | Winner |
|---|---|---|---|---|
| correctness | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| faithfulness | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| retrieval_relevance | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
| citation_accuracy | 1.00 | 1.00 | 1.00 | tie (fixed, structure, semantic) |
