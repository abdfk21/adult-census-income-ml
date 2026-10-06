"""
Prediction module for Adult Census Income Model.
Handles model loading, inference, base learner probability decomposition, and explainability.
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from src.preprocessing import transform_raw_to_features, NUMERICAL_COLS, CATEGORICAL_COLS


class IncomePredictionService:
    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            # Check potential locations (compressed .pkl.gz prioritized for GitHub under 100MB limit)
            candidates = [
                'model/best_adult_income_model_compressed.pkl.gz',
                'model/best_adult_income_model.pkl',
                'Data/best_adult_income_model.pkl',
                os.path.join(os.path.dirname(__file__), '..', 'model', 'best_adult_income_model_compressed.pkl.gz'),
                os.path.join(os.path.dirname(__file__), '..', 'model', 'best_adult_income_model.pkl'),
                os.path.join(os.path.dirname(__file__), '..', 'Data', 'best_adult_income_model.pkl')
            ]
            for p in candidates:
                if os.path.exists(p):
                    model_path = p
                    break
        
        if model_path is None or not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found. Checked candidate paths.")
            
        self.model_path = model_path
        self.model = joblib.load(model_path)
        self.feature_names: List[str] = list(getattr(self.model, 'feature_names_in_', []))
        
        # Extract base estimators and meta classifier
        self.named_estimators = getattr(self.model, 'named_estimators_', {})
        self.final_estimator = getattr(self.model, 'final_estimator_', None)

    def predict_single(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs inference on a single record with raw human-readable features.
        
        Parameters
        ----------
        input_data : dict
            Dictionary containing the 14 raw features.
            
        Returns
        -------
        dict
            Inference results containing prediction, label, probabilities,
            base learner breakdown, and explainability signals.
        """
        raw_df = pd.DataFrame([input_data])
        processed_df = transform_raw_to_features(raw_df, self.feature_names)
        
        # Stacking model prediction
        prob = self.model.predict_proba(processed_df)[0]
        prob_under_50k = float(prob[0])
        prob_over_50k = float(prob[1])
        pred_class = int(prob_over_50k >= 0.5)
        pred_label = ">50K" if pred_class == 1 else "<=50K"
        confidence = prob_over_50k if pred_class == 1 else prob_under_50k
        
        # Base learner breakdown
        base_learner_preds: Dict[str, float] = {}
        for name, est in self.named_estimators.items():
            try:
                p = est.predict_proba(processed_df)[0][1]
                base_learner_preds[name] = float(p)
            except Exception:
                pass
                
        # Meta-learner parameters
        meta_weights: Dict[str, float] = {}
        intercept = 0.0
        if self.final_estimator is not None and hasattr(self.final_estimator, 'coef_'):
            coefs = self.final_estimator.coef_[0]
            names = list(self.named_estimators.keys())
            for i, c in enumerate(coefs):
                n = names[i] if i < len(names) else f"estimator_{i}"
                meta_weights[n] = float(c)
            intercept = float(self.final_estimator.intercept_[0])

        return {
            "prediction": pred_class,
            "label": pred_label,
            "prob_under_50k": prob_under_50k,
            "prob_over_50k": prob_over_50k,
            "confidence": confidence,
            "base_learner_probs": base_learner_preds,
            "meta_weights": meta_weights,
            "meta_intercept": intercept,
            "processed_features": processed_df
        }

    def predict_batch(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs inference on a DataFrame of raw features and appends predictions.
        """
        processed_df = transform_raw_to_features(raw_df, self.feature_names)
        probs = self.model.predict_proba(processed_df)
        preds = (probs[:, 1] >= 0.5).astype(int)
        
        result_df = raw_df.copy()
        result_df['Predicted_Income_Class'] = preds
        result_df['Predicted_Income'] = np.where(preds == 1, '>50K', '<=50K')
        result_df['Probability_<=50K'] = np.round(probs[:, 0], 4)
        result_df['Probability_>50K'] = np.round(probs[:, 1], 4)
        result_df['Model_Confidence'] = np.round(np.maximum(probs[:, 0], probs[:, 1]), 4)
        
        return result_df

    def get_global_feature_importances(self, top_n: int = 15) -> Dict[str, pd.DataFrame]:
        """
        Extracts feature importances from constituent tree-based base learners
        (Random Forest and XGBoost) in the stacking ensemble.
        """
        results = {}
        
        # Random Forest importances
        if 'rf' in self.named_estimators and hasattr(self.named_estimators['rf'], 'feature_importances_'):
            rf = self.named_estimators['rf']
            rf_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            results['Random Forest'] = rf_df

        # XGBoost importances
        if 'xgb' in self.named_estimators and hasattr(self.named_estimators['xgb'], 'feature_importances_'):
            xgb = self.named_estimators['xgb']
            xgb_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': xgb.feature_importances_
            }).sort_values('importance', ascending=False).head(top_n)
            results['XGBoost'] = xgb_df

        return results
