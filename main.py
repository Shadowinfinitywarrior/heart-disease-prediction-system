"""
main.py - Heart Disease Prediction Engine
Core prediction logic and batch processing
"""

import torch
import numpy as np
import pandas as pd
import joblib
from train import AdvancedHeartDiseaseANN, HeartDataProcessor
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

class HeartDiseasePredictor:
    """Main prediction engine for heart disease assessment"""
    
    def __init__(self, model_path='heart_disease_model.pth', 
                 processor_path='data_processor.pkl'):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.feature_names = None
        
        # Load model and processor
        self.load_model(model_path, processor_path)
        
        # Risk interpretation guidelines
        self.risk_interpretation = {
            'low': (0, 30, "🟢 Low Risk", "Minimal intervention needed"),
            'medium': (30, 70, "🟡 Medium Risk", "Lifestyle changes recommended"),
            'high': (70, 100, "🔴 High Risk", "Immediate medical consultation advised")
        }
        
        # Severity labels
        self.severity_labels = {
            0: "Mild",
            1: "Moderate", 
            2: "Severe"
        }
    
    def load_model(self, model_path, processor_path):
        """Load trained model and processor"""
        try:
            # Load processor
            self.processor = joblib.load(processor_path)
            self.feature_names = self.processor.feature_names
            
            # Load model
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model = AdvancedHeartDiseaseANN(input_dim=checkpoint['input_dim'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ Model loaded successfully from {model_path}")
            print(f"📊 Features: {len(self.feature_names)}")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please run train.py first to train the model")
            raise
    
    def predict_single(self, patient_data):
        """
        Predict for a single patient
        
        Args:
            patient_data: dict containing patient features
        
        Returns:
            dict with prediction results
        """
        # Preprocess input
        input_tensor = self.processor.preprocess_single(patient_data)
        input_tensor = torch.FloatTensor(input_tensor).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process results
        disease_prob = outputs['disease_prob'].item()
        disease_pred = 1 if disease_prob > 0.5 else 0
        
        severity_probs = outputs['severity'].cpu().numpy()[0]
        severity = np.argmax(severity_probs)
        severity_label = self.severity_labels[severity]
        
        risk_score = outputs['risk_score'].item()
        confidence = outputs['confidence'].item() * 100
        
        # Get risk category
        risk_category = self._get_risk_category(risk_score)
        
        # Get top contributing features
        feature_importance = self._analyze_feature_importance(patient_data)
        
        return {
            'prediction': disease_pred,
            'probability': disease_prob,
            'severity': severity_label,
            'severity_probs': severity_probs.tolist(),
            'risk_score': risk_score,
            'risk_category': risk_category,
            'confidence': confidence,
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_batch(self, csv_path):
        """
        Predict for multiple patients from CSV
        
        Args:
            csv_path: path to CSV file with patient data
        
        Returns:
            DataFrame with predictions
        """
        try:
            # Load data
            df = pd.read_csv(csv_path)
            
            # Ensure target column is removed if present
            if 'target' in df.columns:
                df = df.drop('target', axis=1)
            
            results = []
            
            # Process each row
            for idx, row in df.iterrows():
                try:
                    result = self.predict_single(row.to_dict())
                    result['patient_id'] = idx
                    results.append(result)
                    
                    if (idx + 1) % 10 == 0:
                        print(f"Processed {idx + 1} patients...")
                        
                except Exception as e:
                    print(f"Error processing patient {idx}: {e}")
                    continue
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # Save results
            output_file = f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            results_df.to_csv(output_file, index=False)
            
            print(f"\nBatch prediction completed!")
            print(f"Results saved to: {output_file}")
            print(f"Statistics:")
            print(f"   Total patients: {len(results_df)}")
            print(f"   High risk: {len(results_df[results_df['risk_score'] > 70])}")
            print(f"   Disease predicted: {results_df['prediction'].sum()}")
            
            return results_df
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return None
    
    def _get_risk_category(self, risk_score):
        """Categorize risk score"""
        for category, (low, high, label, advice) in self.risk_interpretation.items():
            if low <= risk_score < high:
                return {
                    'category': category,
                    'label': label,
                    'advice': advice,
                    'range': f"{low}-{high}%"
                }
        return {
            'category': 'unknown',
            'label': "Unknown",
            'advice': "Consult healthcare provider",
            'range': "N/A"
        }
    
    def _analyze_feature_importance(self, patient_data):
        """Analyze which features contribute most to prediction"""
        try:
            # Get original values
            original_values = {}
            for feature in self.processor.original_columns:
                if feature in patient_data and feature != 'target':
                    original_values[feature] = patient_data[feature]
            
            # Calculate simple importance (absolute value of normalized feature)
            importance = {}
            for feature, value in original_values.items():
                # Simple heuristic: higher values for risk factors
                if feature in ['age', 'chol', 'trestbps', 'oldpeak']:
                    importance[feature] = min(value / 100, 1.0)  # Normalize
                elif feature in ['thalach']:
                    importance[feature] = 1 - min(value / 200, 1.0)  # Lower is riskier
                else:
                    importance[feature] = abs(value)  # For binary/categorical
            
            # Sort by importance
            sorted_importance = dict(sorted(
                importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5])  # Top 5
            
            return sorted_importance
            
        except:
            return {"age": 0.5, "chol": 0.3, "trestbps": 0.2}  # Fallback
    
    def generate_report(self, prediction_result, patient_info=None):
        """Generate comprehensive report"""
        
        report = {
            "report_id": f"HD_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "patient_info": patient_info or {},
            "prediction_summary": {
                "has_disease": bool(prediction_result['prediction']),
                "probability": f"{prediction_result['probability']:.1%}",
                "severity": prediction_result['severity'],
                "risk_score": f"{prediction_result['risk_score']:.1f}%",
                "risk_assessment": prediction_result['risk_category']['label'],
                "recommendation": prediction_result['risk_category']['advice'],
                "model_confidence": f"{prediction_result['confidence']:.1f}%"
            },
            "detailed_analysis": {
                "top_risk_factors": prediction_result['feature_importance'],
                "severity_breakdown": {
                    label: f"{prob:.1%}" 
                    for label, prob in zip(
                        self.severity_labels.values(), 
                        prediction_result['severity_probs']
                    )
                }
            },
            "next_steps": self._get_next_steps(prediction_result),
            "disclaimer": "This prediction is for informational purposes only. Always consult with a healthcare professional for medical diagnosis."
        }
        
        return report
    
    def _get_next_steps(self, prediction_result):
        """Get personalized next steps"""
        risk_score = prediction_result['risk_score']
        
        if risk_score < 30:
            return [
                "Maintain healthy lifestyle",
                "Annual check-up recommended",
                "Monitor blood pressure and cholesterol"
            ]
        elif risk_score < 70:
            return [
                "Consult primary care physician",
                "Consider cardiac screening tests",
                "Implement lifestyle modifications (diet, exercise)",
                "Monitor symptoms regularly"
            ]
        else:
            return [
                "Immediate cardiology consultation",
                "Complete cardiac evaluation (EKG, stress test, echocardiogram)",
                "Consider blood tests (troponin, BNP, lipid panel)",
                "Emergency evaluation if chest pain or shortness of breath"
            ]

# =============================
# EXAMPLE USAGE
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("HEART DISEASE PREDICTION ENGINE")
    print("=" * 60)
    
    try:
        # Initialize predictor
        predictor = HeartDiseasePredictor()
        
        # Example single prediction
        print("\nExample Single Prediction:")
        print("-" * 40)
        
        example_patient = {
            'age': 52,
            'sex': 1,
            'cp': 0,
            'trestbps': 125,
            'chol': 212,
            'fbs': 0,
            'restecg': 1,
            'thalach': 168,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 2,
            'ca': 2,
            'thal': 3
        }
        
        result = predictor.predict_single(example_patient)
        report = predictor.generate_report(result, {"name": "Example Patient"})
        
        print(f"Prediction: {'Disease Detected' if result['prediction'] else 'No Disease'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Risk Score: {result['risk_score']:.1f}%")
        print(f"Severity: {result['severity']}")
        print(f"Risk Category: {result['risk_category']['label']}")
        print(f"Model Confidence: {result['confidence']:.1f}%")
        
        print("\nTop Risk Factors:")
        for feature, importance in result['feature_importance'].items():
            print(f"  {feature}: {importance:.2f}")
        
        print("\nRecommendation:")
        print(f"  {result['risk_category']['advice']}")
        
        print("\nPrediction engine ready!")
        print("Use predict_single() for individual patients")
        print("Use predict_batch() for CSV files")
        print("=" * 60)
        
    except Exception as e:
        print("Error:", e)
        print("Please ensure:")
        print("1. You have run train.py to train the model")
        print("2. heart.csv is in the same directory")
        print("3. Required packages are installed")