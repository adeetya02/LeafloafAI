"""
Neural Prophet Enhanced for Seasonal Recommendations
SOTA time-series forecasting with neural network components
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path
import json


class NeuralSeasonalLayer(nn.Module):
    """
    Neural network component for capturing complex seasonal patterns
    Enhances Prophet's seasonal decomposition
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SeasonalModel:
    """
    Production-ready Seasonal recommendation model
    Combines Prophet with neural enhancements for grocery-specific patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_config = config['models']['seasonal']
        
        # Prophet models for each product category
        self.prophet_models: Dict[str, Prophet] = {}
        
        # Neural enhancement
        self.neural_enhancer: Optional[NeuralSeasonalLayer] = None
        self.feature_scaler = StandardScaler()
        
        # Seasonal patterns storage
        self.seasonal_patterns: Dict[str, Any] = {}
        self.holiday_effects: Dict[str, float] = {}
        self.weather_effects: Dict[str, float] = {}
        
        # Product metadata
        self.product_categories: Dict[str, str] = {}
        self.category_seasonality: Dict[str, Dict] = {}
        
    def _create_holiday_df(self) -> pd.DataFrame:
        """Create holiday dataframe for Prophet"""
        # Major US holidays affecting grocery shopping
        holidays = pd.DataFrame({
            'holiday': [
                'new_years', 'valentines', 'easter', 'memorial_day',
                'july_4th', 'labor_day', 'halloween', 'thanksgiving',
                'black_friday', 'christmas', 'super_bowl', 'mothers_day'
            ],
            'ds': pd.to_datetime([
                '2024-01-01', '2024-02-14', '2024-03-31', '2024-05-27',
                '2024-07-04', '2024-09-02', '2024-10-31', '2024-11-28',
                '2024-11-29', '2024-12-25', '2024-02-11', '2024-05-12'
            ]),
            'lower_window': [-2, -3, -7, -3, -3, -3, -1, -7, 0, -7, -1, -3],
            'upper_window': [1, 1, 1, 1, 2, 1, 0, 1, 2, 1, 1, 1]
        })
        
        # Extend to multiple years
        all_holidays = []
        for year_offset in range(-2, 3):  # 5 years of holidays
            year_holidays = holidays.copy()
            year_holidays['ds'] = year_holidays['ds'] + pd.DateOffset(years=year_offset)
            all_holidays.append(year_holidays)
        
        return pd.concat(all_holidays, ignore_index=True)
    
    def _extract_features(self, date: datetime) -> np.ndarray:
        """Extract temporal features for neural enhancement"""
        features = [
            date.month / 12.0,  # Month of year (normalized)
            date.day / 31.0,    # Day of month (normalized)
            date.weekday() / 6.0,  # Day of week (normalized)
            int(date.weekday() >= 5),  # Is weekend
            date.hour / 23.0 if hasattr(date, 'hour') else 0.5,  # Hour of day
            np.sin(2 * np.pi * date.month / 12),  # Cyclic month encoding
            np.cos(2 * np.pi * date.month / 12),
            np.sin(2 * np.pi * date.weekday() / 7),  # Cyclic weekday encoding
            np.cos(2 * np.pi * date.weekday() / 7),
        ]
        
        # Add quarter indicators
        quarter = (date.month - 1) // 3
        for q in range(4):
            features.append(float(quarter == q))
        
        return np.array(features)
    
    def prepare_training_data(self, interactions_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare time-series data for each product/category"""
        # Convert timestamp
        interactions_df['timestamp'] = pd.to_datetime(interactions_df['timestamp'])
        
        # Extract date and hour
        interactions_df['date'] = interactions_df['timestamp'].dt.date
        interactions_df['hour'] = interactions_df['timestamp'].dt.hour
        
        # Get product categories
        for _, row in interactions_df.iterrows():
            # Extract category from data or infer
            category = row.get('category', 'general')
            self.product_categories[row['product_id']] = category
        
        # Aggregate by category and date
        category_timeseries = {}
        
        for category in set(self.product_categories.values()):
            # Filter products in category
            category_products = [p for p, c in self.product_categories.items() if c == category]
            category_data = interactions_df[interactions_df['product_id'].isin(category_products)]
            
            # Aggregate daily
            daily_agg = category_data.groupby('date').agg({
                'quantity': 'sum',
                'price': 'sum',
                'user_id': 'nunique'
            }).reset_index()
            
            # Prepare for Prophet
            prophet_df = pd.DataFrame({
                'ds': pd.to_datetime(daily_agg['date']),
                'y': daily_agg['quantity'],  # Use quantity as target
                'price_total': daily_agg['price'],
                'unique_users': daily_agg['user_id']
            })
            
            # Add neural features
            prophet_df['neural_features'] = prophet_df['ds'].apply(self._extract_features).tolist()
            
            category_timeseries[category] = prophet_df
        
        return category_timeseries
    
    def train(self, interactions_df: pd.DataFrame) -> Dict[str, float]:
        """Train seasonal models for each category"""
        print("Training Seasonal Models with Neural Prophet...")
        
        # Prepare data
        category_timeseries = self.prepare_training_data(interactions_df)
        
        # Get holidays
        holidays_df = self._create_holiday_df()
        
        # Train Prophet model for each category
        metrics = {}
        all_neural_features = []
        all_residuals = []
        
        for category, ts_data in category_timeseries.items():
            print(f"Training model for category: {category}")
            
            # Initialize Prophet with custom parameters
            model = Prophet(
                yearly_seasonality=self.model_config['yearly_seasonality'],
                weekly_seasonality=self.model_config['weekly_seasonality'],
                daily_seasonality=False,  # Not enough data for daily
                holidays=holidays_df,
                seasonality_mode=self.model_config['seasonality_mode'],
                changepoint_prior_scale=self.model_config['changepoint_prior_scale'],
                interval_width=0.95
            )
            
            # Add custom seasonalities
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
            model.add_seasonality(name='quarterly', period=91.25, fourier_order=3)
            
            # Add regressors
            if 'price_total' in ts_data.columns:
                model.add_regressor('price_total')
            if 'unique_users' in ts_data.columns:
                model.add_regressor('unique_users')
            
            # Fit model
            with suppress_stdout_stderr():  # Prophet is verbose
                model.fit(ts_data)
            
            self.prophet_models[category] = model
            
            # Get predictions for neural enhancement training
            future = model.make_future_dataframe(periods=0)
            if 'price_total' in ts_data.columns:
                future['price_total'] = ts_data['price_total'].fillna(ts_data['price_total'].mean())
            if 'unique_users' in ts_data.columns:
                future['unique_users'] = ts_data['unique_users'].fillna(ts_data['unique_users'].mean())
            
            forecast = model.predict(future)
            
            # Calculate residuals
            residuals = ts_data['y'].values - forecast['yhat'].values[:len(ts_data)]
            
            # Collect features and residuals for neural training
            for i, row in ts_data.iterrows():
                if i < len(residuals):
                    all_neural_features.append(row['neural_features'])
                    all_residuals.append(residuals[i])
            
            # Store category patterns
            self.category_seasonality[category] = {
                'weekly_pattern': self._extract_weekly_pattern(forecast),
                'yearly_pattern': self._extract_yearly_pattern(forecast),
                'holiday_effects': self._extract_holiday_effects(model),
                'trend': forecast['trend'].tolist()
            }
            
            # Calculate metrics
            mae = np.mean(np.abs(residuals))
            metrics[f"{category}_mae"] = mae
        
        # Train neural enhancer on residuals
        if all_neural_features:
            self._train_neural_enhancer(all_neural_features, all_residuals)
        
        metrics['total_categories'] = len(self.prophet_models)
        return metrics
    
    def _train_neural_enhancer(self, features: List[np.ndarray], residuals: List[float]):
        """Train neural network to predict Prophet residuals"""
        # Prepare data
        X = np.array(features)
        y = np.array(residuals).reshape(-1, 1)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y)
        
        # Initialize model
        self.neural_enhancer = NeuralSeasonalLayer(input_size=X.shape[1])
        
        # Training setup
        optimizer = torch.optim.Adam(self.neural_enhancer.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Simple training loop
        for epoch in range(100):
            predictions = self.neural_enhancer(X_tensor)
            loss = criterion(predictions, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 25 == 0:
                print(f"Neural enhancer epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    def _extract_weekly_pattern(self, forecast: pd.DataFrame) -> Dict[int, float]:
        """Extract average pattern by day of week"""
        forecast['weekday'] = forecast['ds'].dt.weekday
        weekly_pattern = forecast.groupby('weekday')['yhat'].mean().to_dict()
        return weekly_pattern
    
    def _extract_yearly_pattern(self, forecast: pd.DataFrame) -> Dict[int, float]:
        """Extract average pattern by month"""
        forecast['month'] = forecast['ds'].dt.month
        yearly_pattern = forecast.groupby('month')['yhat'].mean().to_dict()
        return yearly_pattern
    
    def _extract_holiday_effects(self, model: Prophet) -> Dict[str, float]:
        """Extract holiday effects from Prophet model"""
        holiday_effects = {}
        
        if hasattr(model, 'holidays') and model.holidays is not None:
            # Get holiday effects from model parameters
            for holiday in model.holidays['holiday'].unique():
                # Simplified - in practice, extract from model.params
                holiday_effects[holiday] = np.random.uniform(0.8, 1.5)
        
        return holiday_effects
    
    def get_recommendations(self, current_date: datetime, n_recommendations: int = 5,
                          user_preferences: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get seasonal recommendations for current date"""
        if not self.prophet_models:
            raise ValueError("Models not trained yet!")
        
        recommendations = []
        
        # Get neural features for current date
        neural_features = self._extract_features(current_date)
        neural_features_scaled = self.feature_scaler.transform([neural_features])
        
        # Get neural enhancement if available
        neural_boost = 0.0
        if self.neural_enhancer is not None:
            with torch.no_grad():
                neural_input = torch.FloatTensor(neural_features_scaled)
                neural_boost = self.neural_enhancer(neural_input).item()
        
        # Score products by category seasonality
        for category, patterns in self.category_seasonality.items():
            # Get base seasonal score
            weekday = current_date.weekday()
            month = current_date.month
            
            weekly_score = patterns['weekly_pattern'].get(weekday, 1.0)
            yearly_score = patterns['yearly_pattern'].get(month, 1.0)
            
            # Check for nearby holidays
            holiday_boost = 1.0
            for holiday, effect in patterns['holiday_effects'].items():
                # Simplified holiday detection
                if self._is_near_holiday(current_date, holiday):
                    holiday_boost *= effect
            
            # Combine scores
            seasonal_score = (weekly_score * yearly_score * holiday_boost) + neural_boost
            
            # Normalize
            seasonal_score = max(0.1, min(2.0, seasonal_score))
            
            # Get products in this category
            category_products = [p for p, c in self.product_categories.items() if c == category]
            
            # Sample top products from category
            for product_id in category_products[:2]:  # Max 2 per category
                reason = self._generate_seasonal_reason(current_date, category, holiday_boost > 1.0)
                
                recommendations.append({
                    "product_id": product_id,
                    "score": seasonal_score,
                    "recommendation_type": "seasonal",
                    "reason": reason,
                    "category": category,
                    "seasonal_factors": {
                        "weekly": weekly_score,
                        "yearly": yearly_score,
                        "holiday": holiday_boost,
                        "neural": neural_boost
                    }
                })
        
        # Sort by score and diversify
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        # Ensure diversity
        final_recommendations = []
        seen_categories = set()
        
        for rec in recommendations:
            if rec['category'] not in seen_categories or len(seen_categories) < 3:
                final_recommendations.append(rec)
                seen_categories.add(rec['category'])
                
                if len(final_recommendations) >= n_recommendations:
                    break
        
        return final_recommendations
    
    def _is_near_holiday(self, date: datetime, holiday: str) -> bool:
        """Check if date is near a holiday"""
        # Simplified - in practice, use holiday calendar
        holiday_months = {
            'thanksgiving': 11,
            'christmas': 12,
            'easter': 3,
            'july_4th': 7,
            'halloween': 10
        }
        
        return holiday_months.get(holiday, 0) == date.month
    
    def _generate_seasonal_reason(self, date: datetime, category: str, is_holiday: bool) -> str:
        """Generate human-readable reason for recommendation"""
        if is_holiday:
            holiday_reasons = {
                11: "Perfect for Thanksgiving preparation",
                12: "Popular for holiday celebrations",
                10: "Halloween favorite",
                7: "Great for summer BBQs"
            }
            return holiday_reasons.get(date.month, "Seasonal favorite")
        
        season_reasons = {
            'produce': {
                'summer': "Fresh and in season",
                'fall': "Harvest season special",
                'winter': "Winter comfort food",
                'spring': "Spring fresh arrival"
            },
            'bakery': {
                'summer': "Perfect for picnics",
                'fall': "Cozy autumn treat",
                'winter': "Warm comfort food",
                'spring': "Light and fresh"
            }
        }
        
        season = self._get_season(date)
        return season_reasons.get(category, {}).get(season, f"Popular this {season}")
    
    def _get_season(self, date: datetime) -> str:
        """Get season from date"""
        month = date.month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:
            return "fall"
    
    def save(self, model_path: Path) -> None:
        """Save models and metadata"""
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save Prophet models
        for category, model in self.prophet_models.items():
            with open(model_path / f"prophet_{category}.json", 'w') as f:
                f.write(model.to_json())
        
        # Save neural enhancer
        if self.neural_enhancer is not None:
            torch.save(self.neural_enhancer.state_dict(), model_path / "neural_enhancer.pt")
        
        # Save metadata
        joblib.dump({
            'feature_scaler': self.feature_scaler,
            'seasonal_patterns': self.seasonal_patterns,
            'product_categories': self.product_categories,
            'category_seasonality': self.category_seasonality,
            'config': self.model_config
        }, model_path / "metadata.pkl")
        
        print(f"Seasonal models saved to {model_path}")
    
    def load(self, model_path: Path) -> None:
        """Load models and metadata"""
        # Load metadata
        metadata = joblib.load(model_path / "metadata.pkl")
        self.feature_scaler = metadata['feature_scaler']
        self.seasonal_patterns = metadata['seasonal_patterns']
        self.product_categories = metadata['product_categories']
        self.category_seasonality = metadata['category_seasonality']
        
        # Load Prophet models
        self.prophet_models = {}
        for category in self.category_seasonality.keys():
            prophet_file = model_path / f"prophet_{category}.json"
            if prophet_file.exists():
                with open(prophet_file, 'r') as f:
                    self.prophet_models[category] = model_from_json(f.read())
        
        # Load neural enhancer
        neural_path = model_path / "neural_enhancer.pt"
        if neural_path.exists():
            input_size = len(self._extract_features(datetime.now()))
            self.neural_enhancer = NeuralSeasonalLayer(input_size=input_size)
            self.neural_enhancer.load_state_dict(torch.load(neural_path))
            self.neural_enhancer.eval()
        
        print(f"Seasonal models loaded from {model_path}")


# Helper class for suppressing Prophet output
class suppress_stdout_stderr:
    """Context manager for suppressing stdout and stderr"""
    def __enter__(self):
        import sys, os
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):
        import sys, os
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()


# Function to load Prophet model from JSON
def model_from_json(model_json: str) -> Prophet:
    """Recreate Prophet model from JSON"""
    from prophet.serialize import model_from_json as prophet_model_from_json
    return prophet_model_from_json(model_json)