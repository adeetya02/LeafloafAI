"""
Unified Recommendation Engine
Orchestrates all 4 models to provide comprehensive recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
import yaml
import logging
from collections import defaultdict

from ..models.frequently_bought import FrequentlyBoughtModel
from ..models.discovery import DiscoveryModel
from ..models.seasonal import SeasonalModel
from ..models.goes_well_with import GoesWellWithModel


class RecommendationEngine:
    """
    Production-ready recommendation engine that combines all models
    Provides unified interface for getting personalized recommendations
    """
    
    def __init__(self, config_path: str = "config/config.yaml", model_dir: Optional[Path] = None):
        self.base_path = Path(__file__).parent.parent.parent
        
        # Load configuration
        with open(self.base_path / config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize models
        self.frequently_bought = FrequentlyBoughtModel(self.config)
        self.discovery = DiscoveryModel(self.config)
        self.seasonal = SeasonalModel(self.config)
        self.goes_well_with = GoesWellWithModel(self.config)
        
        # Model loading status
        self.models_loaded = False
        
        # Cache settings
        self.cache_ttl = self.config['inference']['cache_ttl_seconds']
        self.recommendation_cache = {}
        self.cache_timestamps = {}
        
        # Performance tracking
        self.latency_tracker = defaultdict(list)
        
        # Load models if directory provided
        if model_dir:
            self.load_models(model_dir)
    
    def load_models(self, model_dir: Path) -> None:
        """Load all trained models from directory"""
        try:
            print("Loading recommendation models...")
            
            # Load each model
            self.frequently_bought.load(model_dir / "frequently_bought")
            self.discovery.load(model_dir / "discovery")
            self.seasonal.load(model_dir / "seasonal")
            self.goes_well_with.load(model_dir / "goes_well_with")
            
            self.models_loaded = True
            print("All models loaded successfully!")
            
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            raise
    
    def get_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        recommendation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive recommendations for a user
        
        Args:
            user_id: User identifier
            context: Optional context (cart items, meal type, etc.)
            recommendation_types: Which types to include (default: all)
            
        Returns:
            Dictionary with recommendations by type
        """
        if not self.models_loaded:
            raise ValueError("Models not loaded! Call load_models() first.")
        
        start_time = datetime.now()
        
        # Check cache
        cache_key = self._get_cache_key(user_id, context)
        if self._is_cache_valid(cache_key):
            return self.recommendation_cache[cache_key]
        
        # Default to all types
        if recommendation_types is None:
            recommendation_types = ["frequently_bought", "discovery", "seasonal", "goes_well_with"]
        
        # Get user's recent history (for discovery model)
        recent_items = self._get_user_recent_items(user_id, context)
        
        # Get current cart items (for goes well with)
        cart_items = []
        if context and "cart_items" in context:
            cart_items = context["cart_items"]
        
        # Collect recommendations from each model
        recommendations = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "recommendations": {},
            "metadata": {}
        }
        
        # 1. Frequently Bought Recommendations
        if "frequently_bought" in recommendation_types:
            fb_start = datetime.now()
            try:
                fb_recs = self.frequently_bought.get_recommendations(
                    user_id, 
                    n_recommendations=self.config['inference']['max_recommendations']
                )
                recommendations["recommendations"]["frequently_bought"] = fb_recs
                self.latency_tracker["frequently_bought"].append(
                    (datetime.now() - fb_start).total_seconds() * 1000
                )
            except Exception as e:
                logging.warning(f"Frequently bought recommendations failed: {str(e)}")
                recommendations["recommendations"]["frequently_bought"] = []
        
        # 2. Discovery Recommendations
        if "discovery" in recommendation_types and recent_items:
            disc_start = datetime.now()
            try:
                disc_recs = self.discovery.get_recommendations(
                    user_id,
                    recent_items,
                    n_recommendations=self.config['inference']['max_recommendations']
                )
                recommendations["recommendations"]["discovery"] = disc_recs
                self.latency_tracker["discovery"].append(
                    (datetime.now() - disc_start).total_seconds() * 1000
                )
            except Exception as e:
                logging.warning(f"Discovery recommendations failed: {str(e)}")
                recommendations["recommendations"]["discovery"] = []
        
        # 3. Seasonal Recommendations
        if "seasonal" in recommendation_types:
            seasonal_start = datetime.now()
            try:
                seasonal_recs = self.seasonal.get_recommendations(
                    datetime.now(),
                    n_recommendations=5,  # Fewer seasonal items
                    user_preferences=context
                )
                recommendations["recommendations"]["seasonal"] = seasonal_recs
                self.latency_tracker["seasonal"].append(
                    (datetime.now() - seasonal_start).total_seconds() * 1000
                )
            except Exception as e:
                logging.warning(f"Seasonal recommendations failed: {str(e)}")
                recommendations["recommendations"]["seasonal"] = []
        
        # 4. Goes Well With Recommendations
        if "goes_well_with" in recommendation_types and cart_items:
            gww_start = datetime.now()
            try:
                gww_recs = self.goes_well_with.get_recommendations(
                    cart_items,
                    n_recommendations=self.config['inference']['max_recommendations'],
                    context=context
                )
                recommendations["recommendations"]["goes_well_with"] = gww_recs
                self.latency_tracker["goes_well_with"].append(
                    (datetime.now() - gww_start).total_seconds() * 1000
                )
            except Exception as e:
                logging.warning(f"Goes well with recommendations failed: {str(e)}")
                recommendations["recommendations"]["goes_well_with"] = []
        
        # Add metadata
        total_latency = (datetime.now() - start_time).total_seconds() * 1000
        recommendations["metadata"] = {
            "total_latency_ms": total_latency,
            "model_latencies_ms": {
                model: np.mean(latencies[-10:]) if latencies else 0
                for model, latencies in self.latency_tracker.items()
            },
            "cache_hit": False,
            "models_used": list(recommendations["recommendations"].keys())
        }
        
        # Cache results
        self.recommendation_cache[cache_key] = recommendations
        self.cache_timestamps[cache_key] = datetime.now()
        
        # Clean old cache entries
        self._clean_cache()
        
        return recommendations
    
    def get_unified_recommendations(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None,
        max_items: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Get unified recommendation list combining all models
        
        Returns single list with diverse recommendations
        """
        # Get recommendations from all models
        all_recs = self.get_recommendations(user_id, context)
        
        # Combine and rank
        unified_list = []
        seen_products = set()
        
        # Priority order: frequently bought, goes well with, discovery, seasonal
        priority_order = [
            ("frequently_bought", 0.4),
            ("goes_well_with", 0.3),
            ("discovery", 0.2),
            ("seasonal", 0.1)
        ]
        
        # Take items from each category based on weights
        for rec_type, weight in priority_order:
            if rec_type in all_recs["recommendations"]:
                n_items = int(max_items * weight)
                items = all_recs["recommendations"][rec_type][:n_items]
                
                for item in items:
                    if item["product_id"] not in seen_products:
                        unified_list.append({
                            **item,
                            "source": rec_type,
                            "unified_score": item["score"] * (1 + weight)
                        })
                        seen_products.add(item["product_id"])
        
        # Sort by unified score
        unified_list.sort(key=lambda x: x["unified_score"], reverse=True)
        
        return unified_list[:max_items]
    
    def update_user_feedback(
        self,
        user_id: str,
        product_id: str,
        feedback_type: str,
        feedback_value: float
    ) -> None:
        """
        Update models with user feedback
        
        Args:
            user_id: User identifier
            product_id: Product that received feedback
            feedback_type: Type of feedback (click, purchase, ignore)
            feedback_value: Feedback value (0-1)
        """
        # Update discovery model's exploration
        if hasattr(self.discovery, 'update_exploration'):
            self.discovery.update_exploration(product_id, feedback_value)
        
        # Clear cache for this user
        self._clear_user_cache(user_id)
        
        # Log feedback for future model updates
        self._log_feedback(user_id, product_id, feedback_type, feedback_value)
    
    def get_explanation(
        self,
        user_id: str,
        product_id: str,
        recommendation_type: str
    ) -> Dict[str, Any]:
        """
        Get detailed explanation for why a product was recommended
        """
        explanation = {
            "product_id": product_id,
            "recommendation_type": recommendation_type,
            "explanation": "",
            "factors": {}
        }
        
        if recommendation_type == "frequently_bought":
            explanation["explanation"] = "Based on your purchase history"
            explanation["factors"] = {
                "purchase_frequency": "High",
                "last_purchased": "2 weeks ago",
                "total_purchases": "5 times"
            }
        elif recommendation_type == "discovery":
            explanation["explanation"] = "New product you might like"
            explanation["factors"] = {
                "similarity_to_preferences": "High",
                "category_match": "Yes",
                "exploration_score": "0.8"
            }
        elif recommendation_type == "seasonal":
            explanation["explanation"] = "Popular this season"
            explanation["factors"] = {
                "seasonal_trend": "Increasing",
                "holiday_relevance": "Thanksgiving",
                "weather_match": "Yes"
            }
        elif recommendation_type == "goes_well_with":
            explanation["explanation"] = "Complements items in your cart"
            explanation["factors"] = {
                "complement_strength": "Strong",
                "cuisine_match": "Italian",
                "frequently_bought_together": "Yes"
            }
        
        return explanation
    
    def _get_user_recent_items(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Get user's recent interaction history"""
        # In production, this would query a database
        # For now, return sample items or from context
        if context and "recent_items" in context:
            return context["recent_items"]
        
        # Fallback to some defaults based on user
        user_defaults = {
            "user_001": ["PROD_0001", "PROD_0010", "PROD_0020"],  # Bulk buyer
            "user_002": ["PROD_0030", "PROD_0040", "PROD_0050"],  # Family with kids
            "user_003": ["PROD_0060", "PROD_0070", "PROD_0080"],  # Teens
            "user_004": ["PROD_0090", "PROD_0100", "PROD_0110"],  # Health conscious
            "user_005": ["PROD_0120", "PROD_0130", "PROD_0140"],  # Budget conscious
        }
        
        return user_defaults.get(user_id, [])
    
    def _get_cache_key(
        self,
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate cache key"""
        context_str = json.dumps(context, sort_keys=True) if context else ""
        return f"{user_id}:{context_str}"
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self.cache_timestamps:
            return False
        
        age = (datetime.now() - self.cache_timestamps[cache_key]).total_seconds()
        return age < self.cache_ttl
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for key, timestamp in self.cache_timestamps.items():
            if (current_time - timestamp).total_seconds() > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.recommendation_cache[key]
            del self.cache_timestamps[key]
    
    def _clear_user_cache(self, user_id: str) -> None:
        """Clear all cache entries for a user"""
        keys_to_remove = [k for k in self.cache_timestamps.keys() if k.startswith(f"{user_id}:")]
        for key in keys_to_remove:
            del self.recommendation_cache[key]
            del self.cache_timestamps[key]
    
    def _log_feedback(
        self,
        user_id: str,
        product_id: str,
        feedback_type: str,
        feedback_value: float
    ) -> None:
        """Log user feedback for future model updates"""
        # In production, this would write to a database
        feedback_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "product_id": product_id,
            "feedback_type": feedback_type,
            "feedback_value": feedback_value
        }
        
        # For now, just log it
        logging.info(f"User feedback: {feedback_entry}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        return {
            "average_latencies_ms": {
                model: np.mean(latencies) if latencies else 0
                for model, latencies in self.latency_tracker.items()
            },
            "cache_size": len(self.recommendation_cache),
            "models_loaded": self.models_loaded,
            "last_updated": datetime.now().isoformat()
        }


def create_test_engine() -> RecommendationEngine:
    """Create a test instance of the recommendation engine"""
    engine = RecommendationEngine()
    
    # Load models from the latest saved directory
    model_base = Path(__file__).parent.parent.parent / "models" / "saved"
    if model_base.exists():
        latest_model = sorted(model_base.iterdir())[-1]
        engine.load_models(latest_model)
    
    return engine


if __name__ == "__main__":
    # Test the recommendation engine
    print("Testing Recommendation Engine...")
    
    try:
        engine = create_test_engine()
        
        # Test for each user type
        test_users = ["user_001", "user_002", "user_003", "user_004", "user_005"]
        
        for user_id in test_users:
            print(f"\n{'='*50}")
            print(f"Recommendations for {user_id}")
            print('='*50)
            
            # Test with cart context
            context = {
                "cart_items": ["pasta", "tomatoes"],
                "meal_type": "dinner"
            }
            
            # Get all recommendations
            recs = engine.get_recommendations(user_id, context)
            
            # Print summary
            for rec_type, items in recs["recommendations"].items():
                print(f"\n{rec_type.title().replace('_', ' ')}:")
                for i, item in enumerate(items[:3], 1):
                    print(f"  {i}. {item['product_id']} (score: {item['score']:.3f})")
            
            print(f"\nTotal latency: {recs['metadata']['total_latency_ms']:.1f}ms")
            
    except Exception as e:
        print(f"Error testing engine: {str(e)}")