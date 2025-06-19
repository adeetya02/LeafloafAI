"""
Generate synthetic training data for 5 test users
Includes realistic shopping patterns for each persona
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import yaml
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

class DataGenerator:
    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.base_path: Path = Path(__file__).parent.parent
        config_file: Path = self.base_path / config_path
        
        with open(config_file, 'r') as f:
            self.config: Dict[str, Any] = yaml.safe_load(f)
        
        self.random_seed: int = self.config['training']['random_seed']
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        # User personas
        self.users: Dict[str, Dict[str, Any]] = {
            "user_001": {
                "name": "Large Family Bulk Buyer",
                "family_size": 7,
                "cart_size": (50, 70),
                "frequency": 7,
                "preferences": ["bulk", "family_size", "value"],
                "categories": ["dairy", "meat", "produce", "snacks", "frozen"]
            },
            "user_002": {
                "name": "Family with Young Kids",
                "family_size": 4,
                "cart_size": (30, 40),
                "frequency": 3,
                "preferences": ["kid_friendly", "convenient", "healthy_snacks"],
                "categories": ["dairy", "snacks", "produce", "bakery"]
            },
            "user_003": {
                "name": "Family with Teens",
                "family_size": 4,
                "cart_size": (40, 50),
                "frequency": 5,
                "preferences": ["quick_meals", "snacks", "beverages"],
                "categories": ["frozen", "snacks", "beverages", "meat"]
            },
            "user_004": {
                "name": "Health Conscious Individual",
                "family_size": 1,
                "cart_size": (20, 25),
                "frequency": 3,
                "preferences": ["organic", "fresh", "whole_foods"],
                "categories": ["produce", "dairy", "bakery"]
            },
            "user_005": {
                "name": "Budget Conscious Senior",
                "family_size": 2,
                "cart_size": (15, 20),
                "frequency": 7,
                "preferences": ["value", "store_brand", "canned"],
                "categories": ["canned", "produce", "dairy", "bakery"]
            }
        }
        
    def load_products(self, filepath: Optional[str] = None) -> pd.DataFrame:
        """Load Baldor products from CSV"""
        if filepath is None:
            filepath = str(self.base_path / "data" / "raw" / "baldor_products.csv")
        
        if not Path(filepath).exists():
            print(f"Product file not found. Generating sample products...")
            return self._generate_sample_products()
        
        return pd.read_csv(filepath)
    
    def _generate_sample_products(self) -> pd.DataFrame:
        """Generate sample products for testing"""
        categories: Dict[str, List[str]] = {
            "produce": ["Organic Tomatoes", "Bananas", "Apples", "Spinach", "Carrots"],
            "dairy": ["Whole Milk", "Greek Yogurt", "Cheddar Cheese", "Eggs", "Butter"],
            "meat": ["Chicken Breast", "Ground Beef", "Pork Chops", "Salmon", "Turkey"],
            "snacks": ["Chips", "Cookies", "Crackers", "Popcorn", "Granola Bars"],
            "frozen": ["Pizza", "Ice Cream", "Frozen Vegetables", "Chicken Nuggets", "Fries"],
            "beverages": ["Orange Juice", "Soda", "Coffee", "Tea", "Sports Drinks"],
            "bakery": ["Bread", "Bagels", "Muffins", "Croissants", "Donuts"],
            "canned": ["Soup", "Beans", "Tomato Sauce", "Tuna", "Corn"]
        }
        
        products: List[Dict[str, Any]] = []
        product_id: int = 1
        
        for category, items in categories.items():
            for item in items:
                for variant in ["Regular", "Organic", "Premium"]:
                    products.append({
                        "product_id": f"PROD_{product_id:04d}",
                        "name": f"{variant} {item}",
                        "category": category,
                        "price": round(np.random.uniform(1.99, 19.99), 2),
                        "attributes": json.dumps({
                            "organic": variant == "Organic",
                            "premium": variant == "Premium",
                            "family_size": random.choice([True, False])
                        })
                    })
                    product_id += 1
        
        return pd.DataFrame(products)
    
    def generate_shopping_patterns(self, products: pd.DataFrame) -> pd.DataFrame:
        """Generate realistic shopping patterns for each user"""
        all_interactions: List[Dict[str, Any]] = []
        
        for user_id, user_info in self.users.items():
            interactions = self._generate_user_interactions(
                user_id, user_info, products
            )
            all_interactions.extend(interactions)
        
        return pd.DataFrame(all_interactions)
    
    def _generate_user_interactions(
        self, user_id: str, user_info: Dict[str, Any], products: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate interactions for a single user"""
        interactions: List[Dict[str, Any]] = []
        
        # Filter products by user's preferred categories
        user_products = products[products['category'].isin(user_info['categories'])]
        
        # Generate 3 months of shopping history
        start_date = datetime.now() - timedelta(days=90)
        current_date = start_date
        
        while current_date < datetime.now():
            # Generate a shopping session
            session_id = f"{user_id}_{current_date.strftime('%Y%m%d%H%M%S')}"
            cart_size = random.randint(*user_info['cart_size'])
            
            # Select products for this session
            if len(user_products) > 0:
                n_products = min(cart_size, len(user_products))
                session_products = user_products.sample(n=n_products, replace=True)
                
                for _, product in session_products.iterrows():
                    # Purchase interaction
                    interaction: Dict[str, Any] = {
                        "user_id": user_id,
                        "product_id": product['product_id'],
                        "interaction_type": "purchase",
                        "timestamp": current_date.isoformat(),
                        "session_id": session_id,
                        "quantity": self._get_quantity(user_info, product),
                        "price": float(product['price'])
                    }
                    interactions.append(interaction)
                    
                    # View interaction
                    view_time = current_date - timedelta(minutes=random.randint(5, 30))
                    view_interaction: Dict[str, Any] = {
                        "user_id": user_id,
                        "product_id": product['product_id'],
                        "interaction_type": "view",
                        "timestamp": view_time.isoformat(),
                        "session_id": session_id,
                        "quantity": None,
                        "price": float(product['price'])
                    }
                    interactions.append(view_interaction)
            
            # Move to next shopping trip
            current_date += timedelta(days=user_info['frequency'] + random.randint(-1, 1))
        
        return interactions
    
    def _get_quantity(self, user_info: Dict[str, Any], product: pd.Series) -> int:
        """Get realistic quantity based on user type and product"""
        if user_info['family_size'] > 4:
            return random.randint(2, 5)
        elif user_info['family_size'] > 2:
            return random.randint(1, 3)
        else:
            return 1
    
    def save_data(self, interactions_df: pd.DataFrame, products_df: pd.DataFrame) -> Dict[str, Any]:
        """Save generated data"""
        processed_path = self.base_path / "data" / "processed"
        processed_path.mkdir(parents=True, exist_ok=True)
        
        # Save interactions
        interactions_df.to_csv(processed_path / "user_interactions.csv", index=False)
        print(f"Saved {len(interactions_df)} interactions")
        
        # Save products
        products_df.to_csv(processed_path / "products.csv", index=False)
        print(f"Saved {len(products_df)} products")
        
        # Generate summary
        summary: Dict[str, Any] = {
            "total_users": int(interactions_df['user_id'].nunique()),
            "total_products": len(products_df),
            "total_interactions": len(interactions_df),
            "interactions_by_user": interactions_df['user_id'].value_counts().to_dict()
        }
        
        with open(processed_path / "data_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary

def main() -> None:
    """Main function to generate training data"""
    print("Generating synthetic training data...")
    
    generator = DataGenerator()
    products = generator.load_products()
    print(f"Loaded {len(products)} products")
    
    interactions = generator.generate_shopping_patterns(products)
    print(f"Generated {len(interactions)} interactions")
    
    summary = generator.save_data(interactions, products)
    print(f"\nSummary: {summary['total_interactions']} total interactions")

if __name__ == "__main__":
    main()