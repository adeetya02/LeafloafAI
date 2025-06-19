markdown# LeafAndLoaf AI - ML-Powered Grocery Recommendations

## Overview
SOTA personal recommendation system for grocery retail using advanced ML algorithms optimized for CPU performance.

## Architecture
- **Personal Recommendations**: 3-node system (Frequently Bought, Discovery, Seasonal)
- **Training**: GCP Vertex AI (CPU-optimized)
- **Inference**: <300ms latency target
- **Models**: BERT4Rec++, Temporal Matrix Factorization, Neural Prophet

## Project Structure
leafandloaf-ai/
├── config/          # Configuration files
├── data/            # Training data
├── models/          # Trained model artifacts
├── src/             # Source code
│   ├── models/      # Model implementations
│   └── inference/   # Inference engine
└── scripts/         # Deployment scripts

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Configure GCP: Update `config/gcp_config.yaml`
3. Generate training data: `python src/data_generator.py`
4. Train models: `./scripts/deploy_training.ps1`
5. Deploy inference: `./scripts/deploy_inference.ps1`

## Test Users
- Large Family Bulk Buyer
- Family with Young Kids
- Family with Teens
- Health Conscious Individual
- Budget Conscious Senior

## Performance Targets
- Latency: <300ms end-to-end
- Accuracy: >85% precision on frequently bought
- Discovery engagement: >40%