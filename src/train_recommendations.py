"""
Main training script for personal recommendations
Trains all four models (Frequently Bought, Discovery, Seasonal, Goes Well With)
"""

import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import json
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.frequently_bought import FrequentlyBoughtModel
from src.models.discovery import DiscoveryModel
from src.models.seasonal import SeasonalModel
from src.models.goes_well_with import GoesWellWithModel
from src.data_generator import DataGenerator


class RecommendationTrainer:
    """
    Main trainer class that orchestrates training of all recommendation models
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.base_path = Path(__file__).parent.parent
        
        # Load configuration
        with open(self.base_path / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize models
        self.frequently_bought_model = FrequentlyBoughtModel(self.config)
        self.discovery_model = DiscoveryModel(self.config)
        self.seasonal_model = SeasonalModel(self.config)
        self.goes_well_with_model = GoesWellWithModel(self.config)
        
        # Training metrics storage
        self.training_metrics = {}
        
    def load_data(self, use_generated: bool = True) -> pd.DataFrame:
        """Load training data"""
        if use_generated:
            # Use generated data
            data_path = self.base_path / "data" / "processed" / "user_interactions.csv"
            if not data_path.exists():
                print("Generated data not found. Running data generator...")
                generator = DataGenerator()
                products = generator.load_products()
                interactions = generator.generate_shopping_patterns(products)
                generator.save_data(interactions, products)
            
            return pd.read_csv(data_path)
        else:
            # Load real data (when available)
            # TODO: Implement real data loading
            raise NotImplementedError("Real data loading not implemented yet")
    
    def train_all_models(self, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Train all four recommendation models"""
        print("="*50)
        print("Starting Personal Recommendations Training")
        print(f"Data shape: {interactions_df.shape}")
        print(f"Users: {interactions_df['user_id'].nunique()}")
        print(f"Products: {interactions_df['product_id'].nunique()}")
        print("="*50)
        
        overall_metrics = {
            "training_start": datetime.now().isoformat(),
            "data_stats": {
                "total_interactions": len(interactions_df),
                "unique_users": interactions_df['user_id'].nunique(),
                "unique_products": interactions_df['product_id'].nunique(),
                "date_range": {
                    "start": interactions_df['timestamp'].min(),
                    "end": interactions_df['timestamp'].max()
                }
            }
        }
        
        # Train Frequently Bought Model
        print("\n" + "="*30)
        print("Training Frequently Bought Model...")
        print("="*30)
        fb_metrics = self.frequently_bought_model.train(
            interactions_df, 
            epochs=self.config['models']['frequently_bought']['epochs']
        )
        overall_metrics['frequently_bought'] = fb_metrics
        
        # Train Discovery Model
        print("\n" + "="*30)
        print("Training Discovery Model...")
        print("="*30)
        discovery_metrics = self.discovery_model.train(
            interactions_df,
            epochs=20  # Reduced for BERT model
        )
        overall_metrics['discovery'] = discovery_metrics
        
        # Train Seasonal Model
        print("\n" + "="*30)
        print("Training Seasonal Model...")
        print("="*30)
        seasonal_metrics = self.seasonal_model.train(interactions_df)
        overall_metrics['seasonal'] = seasonal_metrics
        
        # Train Goes Well With Model
        print("\n" + "="*30)
        print("Training Goes Well With Model...")
        print("="*30)
        gww_metrics = self.goes_well_with_model.train(
            interactions_df,
            epochs=self.config['models']['goes_well_with']['epochs']
        )
        overall_metrics['goes_well_with'] = gww_metrics
        
        overall_metrics['training_end'] = datetime.now().isoformat()
        self.training_metrics = overall_metrics
        
        return overall_metrics
    
    def save_models(self, model_dir: Optional[Path] = None) -> None:
        """Save all trained models"""
        if model_dir is None:
            model_dir = self.base_path / "models" / "saved" / datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        print("\nSaving models...")
        self.frequently_bought_model.save(model_dir / "frequently_bought")
        self.discovery_model.save(model_dir / "discovery")
        self.seasonal_model.save(model_dir / "seasonal")
        self.goes_well_with_model.save(model_dir / "goes_well_with")
        
        # Save training metrics
        with open(model_dir / "training_metrics.json", 'w') as f:
            json.dump(self.training_metrics, f, indent=2)
        
        print(f"\nAll models saved to: {model_dir}")
        return model_dir
    
    def test_recommendations(self, model_dir: Path, test_user: str = "user_001") -> None:
        """Test recommendations for a sample user"""
        print("\n" + "="*50)
        print(f"Testing Recommendations for {test_user}")
        print("="*50)
        
        # Load models
        self.frequently_bought_model.load(model_dir / "frequently_bought")
        self.discovery_model.load(model_dir / "discovery")
        self.seasonal_model.load(model_dir / "seasonal")
        self.goes_well_with_model.load(model_dir / "goes_well_with")
        
        # Get frequently bought recommendations
        print("\n1. Frequently Bought Recommendations:")
        fb_recs = self.frequently_bought_model.get_recommendations(test_user, n_recommendations=5)
        for i, rec in enumerate(fb_recs, 1):
            print(f"   {i}. {rec['product_id']} (score: {rec['score']:.3f}) - {rec['reason']}")
        
        # Get discovery recommendations (need recent items)
        print("\n2. Discovery Recommendations:")
        recent_items = ["PROD_0001", "PROD_0002", "PROD_0003"]  # Sample items
        discovery_recs = self.discovery_model.get_recommendations(
            test_user, recent_items, n_recommendations=5
        )
        for i, rec in enumerate(discovery_recs, 1):
            print(f"   {i}. {rec['product_id']} (score: {rec['score']:.3f}) - {rec['reason']}")
        
        # Get seasonal recommendations
        print("\n3. Seasonal Recommendations:")
        seasonal_recs = self.seasonal_model.get_recommendations(
            datetime.now(), n_recommendations=5
        )
        for i, rec in enumerate(seasonal_recs, 1):
            print(f"   {i}. {rec['product_id']} (score: {rec['score']:.3f}) - {rec['reason']}")
        
        # Get goes well with recommendations
        print("\n4. Goes Well With Recommendations:")
        cart_items = ["pasta", "tomatoes"]  # Sample cart
        gww_recs = self.goes_well_with_model.get_recommendations(
            cart_items, n_recommendations=5
        )
        for i, rec in enumerate(gww_recs, 1):
            print(f"   {i}. {rec['product_id']} (score: {rec['score']:.3f}) - {rec['reason']}")


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train Personal Recommendation Models")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                      help="Path to configuration file")
    parser.add_argument("--data-source", type=str, default="generated",
                      choices=["generated", "real"],
                      help="Data source to use")
    parser.add_argument("--test", action="store_true",
                      help="Run test recommendations after training")
    parser.add_argument("--model-dir", type=str, default=None,
                      help="Directory to save models")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = RecommendationTrainer(config_path=args.config)
    
    # Load data
    print("Loading training data...")
    interactions_df = trainer.load_data(use_generated=(args.data_source == "generated"))
    
    # Train models
    metrics = trainer.train_all_models(interactions_df)
    
    # Save models
    model_dir = trainer.save_models(args.model_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    print(f"Total training time: {metrics['training_end']} - {metrics['training_start']}")
    print("\nModel Metrics:")
    for model_name in ['frequently_bought', 'discovery', 'seasonal', 'goes_well_with']:
        if model_name in metrics:
            print(f"\n{model_name.title().replace('_', ' ')}:")
            for key, value in metrics[model_name].items():
                print(f"  - {key}: {value}")
    
    # Test if requested
    if args.test:
        trainer.test_recommendations(model_dir)
    
    print("\nDone! Models are ready for deployment.")


if __name__ == "__main__":
    main()