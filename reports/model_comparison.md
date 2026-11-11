# Model Evaluation Results

Held-out **test period** (newest interactions by time); evaluated on 1029 users with at least one relevant (rating >= 4) test item; already-seen items are excluded from recommendations. RMSE/MAE are computed on 9606 test ratings. Best value per column in **bold**.

| Model | precision@5 | recall@5 | ndcg@5 | coverage@5 | diversity@5 | precision@10 | recall@10 | ndcg@10 | coverage@10 | diversity@10 | rmse | mae |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Popularity (baseline) | 0.0391 | 0.0517 | 0.0556 | 0.0486 | **0.8767** | 0.0374 | 0.0883 | 0.0693 | 0.0720 | **0.8513** | 1.0558 | 0.8685 |
| ALS matrix factorisation | 0.0301 | 0.0364 | 0.0387 | 0.0318 | 0.7912 | 0.0283 | 0.0697 | 0.0506 | 0.0519 | 0.7907 | **0.9624** | **0.7848** |
| Hybrid (MF + pop + genre) | **0.0733** | **0.0928** | **0.0975** | **0.1441** | 0.6904 | **0.0623** | **0.1494** | **0.1170** | **0.1993** | 0.7139 | **0.9624** | **0.7848** |

Metric definitions: Precision@K, Recall@K, NDCG@K (binary relevance, log2 discount); Coverage@K = share of the catalogue appearing in any user's top-K; Diversity@K = mean pairwise (1 - cosine) distance between genre vectors of the K recommended items (intra-list diversity). RMSE/MAE: predicted vs actual rating (popularity uses the Bayesian item mean; hybrid reuses the MF rating head).
