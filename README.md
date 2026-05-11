# Healthcare Fraud Detection — KMeans + PCA

Unsupervised ML pipeline to detect fraudulent 
healthcare providers using clustering.

## Problem
Identify suspicious providers without labeled data
using pattern recognition in claim behavior.

## Pipeline

Provider Data → Scaling → PCA (2D) 
→ Elbow Method → KMeans → Flag Fraud Cluster


## Dataset Features
| Feature | Description |
|---|---|
| claim_amount | Total claimed amount |
| num_claims | Number of claims filed |
| num_patients | Unique patients served |
| avg_claim_per_patient | Average billing per patient |
| num_procedures | Procedures performed |
| inpatient_ratio | Inpatient claim proportion |
| unique_drugs_prescribed | Drug variety |

## Results
| Metric | Value |
|---|---|
| PCA Variance Retained | ~70% |
| Optimal Clusters (K) | 2 |
| Silhouette Score | *0.60* |
| Flagged Providers | ~200 |

## Fraud Indicators Found
- High claim amount (3x normal)
- Low unique patient count
- High procedures per patient
- High inpatient ratio

## Run
bash
pip install scikit-learn matplotlib seaborn pandas numpy
python fraud_detection.py


## Tech Stack
Python | Scikit-learn | PCA | KMeans | Matplotlib | Seaborn
