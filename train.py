"""
train.py - Enhanced Heart Disease Prediction Model Trainer
Advanced ANN with 95%+ accuracy target
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, precision_score, 
                           recall_score, f1_score, roc_auc_score,
                           confusion_matrix)
import joblib
import warnings
warnings.filterwarnings('ignore')

# =============================
# ENHANCED ANN ARCHITECTURE
# =============================

class AdvancedHeartDiseaseANN(nn.Module):
    """Enhanced ANN architecture with deeper layers and attention for 95%+ accuracy"""
    
    def __init__(self, input_dim):
        super(AdvancedHeartDiseaseANN, self).__init__()
        
        # Multi-head feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Sigmoid(),
            nn.Dropout(0.15)
        )
        
        # Enhanced residual blocks
        self.residual_block1 = self._create_residual_block(input_dim, 256)
        self.residual_block2 = self._create_residual_block(256, 512)
        self.residual_block3 = self._create_residual_block(512, 256)
        self.residual_block4 = self._create_residual_block(256, 128)
        
        # Advanced risk pathway
        self.risk_pathway = nn.Sequential(
            nn.Linear(128, 96),
            nn.BatchNorm1d(96),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(96, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Multi-task outputs
        self.disease_classifier = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, 1)
        )
        self.severity_predictor = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 3)
        )
        self.risk_regressor = nn.Sequential(
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )
        
        # Enhanced confidence estimator
        self.confidence_net = nn.Sequential(
            nn.Linear(48, 32),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
    
    def _create_residual_block(self, in_dim, out_dim):
        """Enhanced residual block with normalization"""
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
            nn.Dropout(0.15)
        )
    
    def _initialize_weights(self):
        """Xavier/Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Enhanced feature attention
        attn_weights = self.feature_attention(x)
        x_attended = x * attn_weights
        
        # Deeper residual processing
        x1 = self.residual_block1(x_attended)
        x2 = self.residual_block2(x1)
        x3 = self.residual_block3(x2)
        x4 = self.residual_block4(x3)
        
        # Risk feature extraction
        risk_features = self.risk_pathway(x4)
        
        # Multi-task predictions
        disease_logits = self.disease_classifier(risk_features)
        disease_prob = torch.sigmoid(disease_logits)
        severity_prob = torch.softmax(self.severity_predictor(risk_features), dim=1)
        risk_score = torch.sigmoid(self.risk_regressor(risk_features)) * 100
        confidence = self.confidence_net(risk_features)
        
        return {
            'disease_prob': disease_prob,
            'severity': severity_prob,
            'risk_score': risk_score,
            'confidence': confidence,
            'attention': attn_weights
        }

# =============================
# DATA PROCESSOR
# =============================

class HeartDataProcessor:
    """Advanced preprocessing with novel feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.feature_names = None
        self.novel_features_added = False
        
    def create_novel_features(self, df):
        """Create novel engineered features"""
        
        # Cardiac risk ratios
        if 'chol' in df.columns and 'trestbps' in df.columns:
            df['chol_bp_ratio'] = df['chol'] / (df['trestbps'] + 1e-5)
            df['bp_chol_product'] = (df['trestbps'] * df['chol']) / 1000
        
        # Age-adjusted metrics
        if 'age' in df.columns and 'thalach' in df.columns:
            df['predicted_max_hr'] = 220 - df['age']
            df['hr_reserve_ratio'] = df['thalach'] / (df['predicted_max_hr'] + 1e-5)
        
        # Metabolic syndrome score
        metabolic_factors = 0
        if 'trestbps' in df.columns:
            metabolic_factors += (df['trestbps'] > 130).astype(int)
        if 'fbs' in df.columns:
            metabolic_factors += (df['fbs'] > 120).astype(int)
        if 'chol' in df.columns:
            metabolic_factors += (df['chol'] > 200).astype(int)
        df['metabolic_score'] = metabolic_factors
        
        # ST-depression severity
        if 'oldpeak' in df.columns:
            df['oldpeak_category'] = pd.cut(
                df['oldpeak'], 
                bins=[-1, 0.5, 1.5, 2.5, 10],
                labels=[0, 1, 2, 3]
            ).astype(float)
        
        # Combined risk index
        risk_indicators = []
        if 'age' in df.columns:
            risk_indicators.append(df['age'] / 100)
        if 'chol' in df.columns:
            risk_indicators.append(df['chol'] / 500)
        if 'trestbps' in df.columns:
            risk_indicators.append(df['trestbps'] / 200)
        
        if risk_indicators:
            df['composite_risk'] = sum(risk_indicators) / len(risk_indicators)
        
        self.novel_features_added = True
        return df
    
    def prepare_data(self, df_path='heart.csv'):
        """Load and preprocess dataset"""
        df = pd.read_csv(df_path)
        self.original_columns = df.columns.tolist()
        
        df = self.create_novel_features(df)
        
        if 'target' in df.columns:
            y = df['target'].values
            X = df.drop('target', axis=1)
        else:
            y = None
            X = df
        
        self.feature_names = X.columns.tolist()
        X_imputed = self.imputer.fit_transform(X)
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        return X_scaled, y, df
    
    def preprocess_single(self, data_dict):
        """Preprocess single patient data"""
        df = pd.DataFrame([data_dict])
        df = self.create_novel_features(df)
        
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        df = df[self.feature_names]
        df_imputed = self.imputer.transform(df)
        df_scaled = self.scaler.transform(df_imputed)
        
        return df_scaled

# =============================
# MODEL TRAINER
# =============================

class ModelTrainer:
    """Complete model training pipeline with cross-validation"""
    
    def __init__(self, model_path='heart_disease_model.pth'):
        self.model_path = model_path
        self.processor = HeartDataProcessor()
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def train(self, epochs=250, batch_size=16, lr=0.0015):
        """Train model with enhanced techniques"""
        
        print("🔄 Loading and preprocessing data...")
        X, y, df = self.processor.prepare_data()
        
        print(f"📊 Dataset: {len(X)} samples, {X.shape[1]} features")
        
        # Split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)
        y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        # Initialize model
        self.model = AdvancedHeartDiseaseANN(input_dim=X_train.shape[1])
        self.model.to(self.device)
        
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        class EnhancedLoss(nn.Module):
            def forward(self, outputs, targets):
                disease_loss = nn.BCELoss()(outputs['disease_prob'], targets)
                attention_reg = torch.mean(torch.abs(outputs['attention'] - 0.5))
                confidence_penalty = torch.mean((outputs['confidence'] - outputs['disease_prob'].detach()) ** 2)
                return disease_loss + 0.02 * attention_reg + 0.1 * confidence_penalty
        
        criterion = EnhancedLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=5e-4, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
        
        print("🚀 Training enhanced model...")
        best_val_acc = 0.0
        patience = 25
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # L1 regularization
                l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                loss = loss + 0.0005 * l1_norm
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                train_loss += loss.item()
                predictions = (outputs['disease_prob'] > 0.5).float()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_outputs = self.model(X_test_tensor.to(self.device))
                predictions = (val_outputs['disease_prob'] > 0.5).float()
                val_correct = (predictions == y_test_tensor.to(self.device)).sum().item()
                val_total = y_test_tensor.size(0)
            
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            
            scheduler.step()
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"✅ Early stopping at epoch {epoch} (Best Val Acc: {best_val_acc:.4f})")
                break
            
            if epoch % 15 == 0 or epoch < 5:
                print(f"Epoch {epoch:3d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
        
        # Load best model
        self.load_model()
        
        # Final evaluation
        print("\n📊 Final Model Evaluation:")
        print("=" * 60)
        self.evaluate(X_test_tensor, y_test)
        
        # Cross-validation
        print("\n🔄 Performing 5-fold cross-validation...")
        cv_scores = self.cross_validate(X, y, n_splits=5)
        print(f"📈 CV Accuracy: {cv_scores['mean']:.4f} (+/- {cv_scores['std']:.4f})")
        
        # Save processor
        joblib.dump(self.processor, 'data_processor.pkl')
        
        return self.model
    
    def cross_validate(self, X, y, n_splits=5):
        """K-fold cross-validation"""
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train_fold = torch.FloatTensor(X[train_idx])
            y_train_fold = torch.FloatTensor(y[train_idx]).unsqueeze(1)
            X_val_fold = torch.FloatTensor(X[val_idx])
            y_val_fold = y[val_idx]
            
            # Quick training
            fold_model = AdvancedHeartDiseaseANN(input_dim=X.shape[1])
            fold_model.to(self.device)
            optimizer = optim.AdamW(fold_model.parameters(), lr=0.001, weight_decay=1e-4)
            
            for _ in range(50):
                fold_model.train()
                optimizer.zero_grad()
                outputs = fold_model(X_train_fold.to(self.device))
                loss = nn.BCELoss()(outputs['disease_prob'], y_train_fold.to(self.device))
                loss.backward()
                optimizer.step()
            
            # Evaluate
            fold_model.eval()
            with torch.no_grad():
                outputs = fold_model(X_val_fold.to(self.device))
                predictions = (outputs['disease_prob'] > 0.5).cpu().numpy()
                acc = accuracy_score(y_val_fold, predictions)
                cv_scores.append(acc)
            
            print(f"  Fold {fold + 1}: {acc:.4f}")
        
        return {'scores': cv_scores, 'mean': np.mean(cv_scores), 'std': np.std(cv_scores)}
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_test.to(self.device))
            predictions = (outputs['disease_prob'] > 0.5).cpu().numpy()
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, zero_division=0)
        recall = recall_score(y_test, predictions, zero_division=0)
        f1 = f1_score(y_test, predictions, zero_division=0)
        roc_auc = roc_auc_score(y_test, predictions)
        
        print(f"🎯 Model Performance:")
        print(f"   Accuracy:  {accuracy:.3f}")
        print(f"   Precision: {precision:.3f}")
        print(f"   Recall:    {recall:.3f}")
        print(f"   F1-Score:  {f1:.3f}")
        print(f"   ROC-AUC:   {roc_auc:.3f}")
        
        cm = confusion_matrix(y_test, predictions)
        print(f"\n📈 Confusion Matrix:")
        print(f"   True Neg: {cm[0,0]} | False Pos: {cm[0,1]}")
        print(f"   False Neg: {cm[1,0]} | True Pos: {cm[1,1]}")
        
        return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'roc_auc': roc_auc}
    
    def save_model(self):
        """Save model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.model.residual_block1[0].in_features,
            'feature_names': self.processor.feature_names
        }, self.model_path)
    
    def load_model(self):
        """Load model"""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model = AdvancedHeartDiseaseANN(input_dim=checkpoint['input_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

# =============================
# MAIN EXECUTION
# =============================

if __name__ == "__main__":
    print("=" * 60)
    print("❤️  HEART DISEASE PREDICTION MODEL TRAINER")
    print("=" * 60)
    
    trainer = ModelTrainer()
    model = trainer.train(epochs=250, batch_size=16, lr=0.0015)
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print("Model saved as: heart_disease_model.pth")
    print("Processor saved as: data_processor.pkl")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run main.py for predictions")
    print("2. Use gui.py for graphical interface")