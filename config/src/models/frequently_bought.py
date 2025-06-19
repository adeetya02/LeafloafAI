"""
Temporal Matrix Factorization++ for Frequently Bought Items
SOTA CPU-optimized implementation with temporal decay
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
from pathlib import Path


class TemporalMatrixFactorizationPlus(nn.Module):
    """
    Enhanced Matrix Factorization with temporal components
    CPU-optimized using sparse operations and caching
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 128, 
                 temporal_decay: float = 0.95):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.temporal_decay = temporal_decay
        
        # User and item embeddings
        self.user_embeddings = nn.Embedding(n_users, n_factors)
        self.item_embeddings = nn.Embedding(n_items, n_factors)
        
        # Temporal components
        self.user_temporal = nn.Embedding(n_users, n_factors)
        self.item_temporal = nn.Embedding(n_items, n_factors)
        
        # Bias terms
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.xavier_uniform_(self.user_temporal.weight)
        nn.init.xavier_uniform_(self.item_temporal.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor, 
                time_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with temporal weighting"""
        # Get embeddings
        user_embed = self.user_embeddings(user_ids)
        item_embed = self.item_embeddings(item_ids)
        
        # Add temporal components if provided
        if time_weights is not None:
            time_weights = time_weights.unsqueeze(1)
            user_temporal_embed = self.user_temporal(user_ids) * time_weights
            item_temporal_embed = self.item_temporal(item_ids) * time_weights
            user_embed = user_embed + user_temporal_embed
            item_embed = item_embed + item_temporal_embed
        
        # Compute interaction
        interaction = (user_embed * item_embed).sum(dim=1)
        
        # Add biases
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        prediction = interaction + user_b + item_b + self.global_bias
        
        return prediction


class FrequentlyBoughtModel:
    """
    Production-ready Frequently Bought recommendation model
    Combines Temporal Matrix Factorization with business rules
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['models']['frequently_bought']
        self.embedding_dim = self.model_config['embedding_dim']
        self.temporal_decay = self.model_config['temporal_decay']
        
        self.model: Optional[TemporalMatrixFactorizationPlus] = None
        self.user_encoder: Optional[LabelEncoder] = None
        self.item_encoder: Optional[LabelEncoder] = None
        self.item_popularity: Optional[Dict[str, float]] = None
        self.user_history: Optional[Dict[str, List[Tuple[str, datetime]]]] = None
        
    def prepare_training_data(self, interactions_df: pd.DataFrame) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """Prepare data for training with temporal features"""
        # Convert timestamps
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        # Filter to purchases only
        purchases = interactions_df[interactions_df['interaction_type'] == 'purchase'].copy()
        
        # Encode users and items
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        
        purchases['user_idx'] = self.user_encoder.fit_transform(purchases['user_id'])
        purchases['item_idx'] = self.item_encoder.fit_transform(purchases['product_id'])
        
        # Calculate temporal weights (more recent = higher weight)
        max_date = purchases['timestamp'].max()
        purchases['days_ago'] = (max_date - purchases['timestamp']).dt.days
        purchases['time_weight'] = self.temporal_decay ** (purchases['days_ago'] / 7)
        
        # Calculate item popularity
        item_counts = purchases['product_id'].value_counts()
        self.item_popularity = (item_counts / item_counts.max()).to_dict()
        
        # Store user history for inference
        self.user_history = {}
        for user_id, group in purchases.groupby('user_id'):
            history = [(row['product_id'], row['timestamp']) 
                      for _, row in group.iterrows()]
            self.user_history[user_id] = sorted(history, key=lambda x: x[1], reverse=True)
        
        # Convert to tensors
        user_ids = torch.LongTensor(purchases['user_idx'].values)
        item_ids = torch.LongTensor(purchases['item_idx'].values)
        time_weights = torch.FloatTensor(purchases['time_weight'].values)
        
        # Create negative samples
        n_items = len(self.item_encoder.classes_)
        negative_samples = []
        
        for idx in range(len(purchases)):
            user_idx = purchases.iloc[idx]['user_idx']
            # Sample items user hasn't bought
            user_items = set(purchases[purchases['user_idx'] == user_idx]['item_idx'])
            neg_items = list(set(range(n_items)) - user_items)
            if neg_items:
                neg_item = np.random.choice(neg_items)
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': neg_item,
                    'time_weight': purchases.iloc[idx]['time_weight']
                })
        
        neg_df = pd.DataFrame(negative_samples)
        neg_user_ids = torch.LongTensor(neg_df['user_idx'].values)
        neg_item_ids = torch.LongTensor(neg_df['item_idx'].values)
        neg_time_weights = torch.FloatTensor(neg_df['time_weight'].values)
        
        return (user_ids, item_ids, time_weights, 
                neg_user_ids, neg_item_ids, neg_time_weights)
    
    def train(self, interactions_df: pd.DataFrame, epochs: int = 50) -> Dict[str, float]:
        """Train the model on interaction data"""
        print("Training Frequently Bought Model...")
        
        # Prepare data
        (user_ids, item_ids, time_weights,
         neg_user_ids, neg_item_ids, neg_time_weights) = self.prepare_training_data(interactions_df)
        
        # Initialize model
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        self.model = TemporalMatrixFactorizationPlus(
            n_users, n_items, self.embedding_dim, self.temporal_decay
        )
        
        # Training setup
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        batch_size = self.model_config['batch_size']
        n_samples = len(user_ids)
        
        # Training loop
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            perm = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                # Get batch
                batch_idx = perm[i:i+batch_size]
                batch_users = user_ids[batch_idx]
                batch_items = item_ids[batch_idx]
                batch_times = time_weights[batch_idx]
                
                # Positive samples
                pos_pred = self.model(batch_users, batch_items, batch_times)
                pos_labels = torch.ones_like(pos_pred)
                
                # Negative samples (same batch size)
                neg_batch_users = neg_user_ids[batch_idx]
                neg_batch_items = neg_item_ids[batch_idx]
                neg_batch_times = neg_time_weights[batch_idx]
                
                neg_pred = self.model(neg_batch_users, neg_batch_items, neg_batch_times)
                neg_labels = torch.zeros_like(neg_pred)
                
                # Combined loss
                predictions = torch.cat([pos_pred, neg_pred])
                labels = torch.cat([pos_labels, neg_labels])
                
                loss = criterion(predictions, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / (n_samples // batch_size)
            losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        return {"final_loss": losses[-1], "n_users": n_users, "n_items": n_items}
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10,
                          filter_purchased: bool = True) -> List[Dict[str, Any]]:
        """Get frequently bought recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Handle new users
        if user_id not in self.user_encoder.classes_:
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.user_encoder.transform([user_id])[0]
        user_tensor = torch.LongTensor([user_idx])
        
        # Get all item predictions
        n_items = len(self.item_encoder.classes_)
        item_indices = torch.arange(n_items)
        user_indices = user_tensor.repeat(n_items)
        
        # Use current time weight = 1.0 for real-time predictions
        time_weights = torch.ones(n_items)
        
        with torch.no_grad():
            scores = self.model(user_indices, item_indices, time_weights)
            scores = torch.sigmoid(scores).numpy()
        
        # Get user's purchase history
        purchased_items = set()
        if filter_purchased and user_id in self.user_history:
            purchased_items = {item[0] for item in self.user_history[user_id]}
        
        # Create recommendations
        recommendations = []
        item_ids = self.item_encoder.classes_
        
        for idx, score in enumerate(scores):
            item_id = item_ids[idx]
            
            if item_id not in purchased_items:
                # Boost score by popularity
                popularity_boost = self.item_popularity.get(item_id, 0.1)
                final_score = float(score) * (1 + 0.2 * popularity_boost)
                
                recommendations.append({
                    "product_id": item_id,
                    "score": final_score,
                    "recommendation_type": "frequently_bought",
                    "reason": "Based on your purchase history"
                })
        
        # Sort and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:n_recommendations]
    
    def _get_popular_items(self, n_recommendations: int) -> List[Dict[str, Any]]:
        """Fallback to popular items for new users"""
        if not self.item_popularity:
            return []
        
        popular_items = sorted(self.item_popularity.items(), 
                             key=lambda x: x[1], reverse=True)
        
        recommendations = []
        for item_id, popularity in popular_items[:n_recommendations]:
            recommendations.append({
                "product_id": item_id,
                "score": popularity,
                "recommendation_type": "popular",
                "reason": "Trending with other shoppers"
            })
        
        return recommendations
    
    def save(self, model_path: Path) -> None:
        """Save model and encoders"""
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save PyTorch model
        torch.save(self.model.state_dict(), model_path / "model.pt")
        
        # Save encoders and metadata
        joblib.dump({
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'item_popularity': self.item_popularity,
            'user_history': self.user_history,
            'config': self.model_config
        }, model_path / "metadata.pkl")
        
        print(f"Model saved to {model_path}")
    
    def load(self, model_path: Path) -> None:
        """Load model and encoders"""
        # Load metadata
        metadata = joblib.load(model_path / "metadata.pkl")
        self.user_encoder = metadata['user_encoder']
        self.item_encoder = metadata['item_encoder']
        self.item_popularity = metadata['item_popularity']
        self.user_history = metadata['user_history']
        
        # Initialize and load model
        n_users = len(self.user_encoder.classes_)
        n_items = len(self.item_encoder.classes_)
        
        self.model = TemporalMatrixFactorizationPlus(
            n_users, n_items, self.embedding_dim, self.temporal_decay
        )
        self.model.load_state_dict(torch.load(model_path / "model.pt"))
        self.model.eval()
        
        print(f"Model loaded from {model_path}")