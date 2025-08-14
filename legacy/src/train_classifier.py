"""
Training script for Error + Confidence Classifier
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error, f1_score
from sklearn.utils.class_weight import compute_class_weight
import pickle
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for classifier training"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # Lightweight transformer
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    hidden_dim: int = 256
    dropout: float = 0.1
    logits_feature_dim: int = 64  # Dimension for logits features if available
    weight_decay: float = 0.01
    use_class_weighting: bool = True

class SelfCorrectionDataset(Dataset):
    """Dataset for self-correction failure mode classification"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 tokenizer,
                 label_encoder: LabelEncoder,
                 max_length: int = 512,
                 logits_features: Optional[np.ndarray] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.max_length = max_length
        self.logits_features = logits_features
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Create input text by combining initial and revised answers
        initial_answer = str(row['initial_answer'])
        revised_answer = str(row['revised_answer'])
        reprompt_id = str(row.get('reprompt_id', ''))
        
        # Format input text
        input_text = f"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get labels
        failure_mode = self.label_encoder.transform([row['failure_mode']])[0]
        delta_accuracy = float(row['delta_accuracy'])
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'failure_mode': torch.tensor(failure_mode, dtype=torch.long),
            'delta_accuracy': torch.tensor(delta_accuracy, dtype=torch.float),
        }
        
        # Add logits features if available
        if self.logits_features is not None:
            item['logits_features'] = torch.tensor(
                self.logits_features[idx], dtype=torch.float
            )
        
        return item

class ErrorConfidenceClassifier(nn.Module):
    """
    Multi-head classifier for failure mode detection and confidence prediction
    
    Architecture:
    1. Text encoder (lightweight transformer)
    2. Optional logits feature encoder
    3. Feature fusion layer
    4. Two prediction heads:
       - Failure mode classifier (multi-class)
       - Confidence score predictor (regression)
    """
    
    def __init__(self, 
                 config: TrainingConfig,
                 num_failure_modes: int,
                 logits_feature_dim: Optional[int] = None):
        super().__init__()
        self.config = config
        self.num_failure_modes = num_failure_modes
        
        # Text encoder (lightweight transformer)
        self.text_encoder = AutoModel.from_pretrained(config.model_name)
        self.text_encoder_dim = self.text_encoder.config.hidden_size
        
        # Optional logits feature encoder
        self.use_logits_features = logits_feature_dim is not None
        if self.use_logits_features:
            self.logits_encoder = nn.Sequential(
                nn.Linear(logits_feature_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2)
            )
            logits_output_dim = config.hidden_dim // 2
        else:
            logits_output_dim = 0
        
        # Feature fusion layer
        fusion_input_dim = self.text_encoder_dim + logits_output_dim
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Prediction heads
        self.failure_mode_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, num_failure_modes)
        )
        
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 1)
        )
    
    def forward(self, 
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                logits_features: Optional[torch.Tensor] = None):
        
        # Encode text
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use [CLS] token representation
        text_features = text_outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_dim]
        
        # Process logits features if available
        if self.use_logits_features and logits_features is not None:
            logits_encoded = self.logits_encoder(logits_features)
            # Concatenate text and logits features
            fused_features = torch.cat([text_features, logits_encoded], dim=1)
        else:
            fused_features = text_features
        
        # Feature fusion
        fused_features = self.fusion_layer(fused_features)
        
        # Prediction heads
        failure_mode_logits = self.failure_mode_head(fused_features)
        confidence_score = self.confidence_head(fused_features).squeeze(-1)
        
        return {
            'failure_mode_logits': failure_mode_logits,
            'confidence_score': confidence_score
        }

class ErrorConfidenceTrainer:
    """Trainer for the error and confidence classifier"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.label_encoder = LabelEncoder()
        self.model = None
        self.class_weights = None
        
        logger.info(f"Using device: {self.device}")
        
    def prepare_data(self, data_path: str) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare training/validation data
        
        Args:
            data_path: Path to CSV file with columns:
                      initial_answer, revised_answer, reprompt_id, failure_mode, delta_accuracy
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load data
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} samples from {data_path}")
        
        # Encode failure modes
        failure_modes = df['failure_mode'].values
        self.label_encoder.fit(failure_modes)
        num_classes = len(self.label_encoder.classes_)
        logger.info(f"Found {num_classes} failure mode classes: {self.label_encoder.classes_}")
        
        # Compute class weights for balanced loss
        if self.config.use_class_weighting:
            class_weights = compute_class_weight(
                'balanced',
                classes=np.arange(num_classes),
                y=self.label_encoder.transform(failure_modes)
            )
            self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            logger.info(f"Computed class weights: {self.class_weights}")
        
        # Split data
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=failure_modes
        )
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
        
        # Check for logits features (optional)
        logits_features_train = None
        logits_features_val = None
        if 'logits_features' in df.columns:
            # Assume logits features are stored as JSON strings or arrays
            logits_features_train = self._parse_logits_features(train_df['logits_features'])
            logits_features_val = self._parse_logits_features(val_df['logits_features'])
            logger.info(f"Loaded logits features with dimension {logits_features_train.shape[1]}")
        
        # Create datasets
        train_dataset = SelfCorrectionDataset(
            train_df.reset_index(drop=True),
            self.tokenizer,
            self.label_encoder,
            self.config.max_length,
            logits_features_train
        )
        
        val_dataset = SelfCorrectionDataset(
            val_df.reset_index(drop=True),
            self.tokenizer,
            self.label_encoder,
            self.config.max_length,
            logits_features_val
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0  # Set to 0 for compatibility
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        logits_dim = logits_features_train.shape[1] if logits_features_train is not None else None
        self.model = ErrorConfidenceClassifier(
            self.config,
            num_classes,
            logits_dim
        ).to(self.device)
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
        
        return train_loader, val_loader
    
    def _parse_logits_features(self, logits_series) -> np.ndarray:
        """Parse logits features from DataFrame column"""
        features = []
        for item in logits_series:
            if isinstance(item, str):
                # Parse JSON string
                feature_vec = json.loads(item)
            else:
                # Assume it's already a list/array
                feature_vec = item
            features.append(feature_vec)
        return np.array(features)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """
        Train the model
        
        Returns:
            Dictionary with training metrics
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Loss functions
        if self.class_weights is not None:
            failure_mode_criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            failure_mode_criterion = nn.CrossEntropyLoss()
        
        confidence_criterion = nn.MSELoss()
        
        # Training loop
        train_metrics = {'epochs': [], 'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            total_train_loss = 0
            
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    logits_features=batch.get('logits_features')
                )
                
                # Compute losses
                failure_mode_loss = failure_mode_criterion(
                    outputs['failure_mode_logits'], batch['failure_mode']
                )
                confidence_loss = confidence_criterion(
                    outputs['confidence_score'], batch['delta_accuracy']
                )
                
                # Combined loss (weighted)
                total_loss = failure_mode_loss + 0.5 * confidence_loss
                
                # Backward pass
                total_loss.backward()
                optimizer.step()
                
                total_train_loss += total_loss.item()
                
                if batch_idx % 10 == 0:
                    logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {total_loss.item():.4f}")
            
            # Validation
            val_metrics = self.evaluate(val_loader)
            
            avg_train_loss = total_train_loss / len(train_loader)
            
            logger.info(
                f"Epoch {epoch+1}/{self.config.num_epochs}: "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {val_metrics['total_loss']:.4f}, "
                f"Val F1: {val_metrics['macro_f1']:.4f}"
            )
            
            # Store metrics
            train_metrics['epochs'].append(epoch + 1)
            train_metrics['train_loss'].append(avg_train_loss)
            train_metrics['val_loss'].append(val_metrics['total_loss'])
            train_metrics['val_f1'].append(val_metrics['macro_f1'])
        
        return train_metrics
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """Evaluate model on validation set"""
        self.model.eval()
        total_loss = 0
        all_failure_preds = []
        all_failure_labels = []
        all_confidence_preds = []
        all_confidence_labels = []
        
        failure_mode_criterion = nn.CrossEntropyLoss()
        confidence_criterion = nn.MSELoss()
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    logits_features=batch.get('logits_features')
                )
                
                # Compute losses
                failure_mode_loss = failure_mode_criterion(
                    outputs['failure_mode_logits'], batch['failure_mode']
                )
                confidence_loss = confidence_criterion(
                    outputs['confidence_score'], batch['delta_accuracy']
                )
                total_loss += (failure_mode_loss + 0.5 * confidence_loss).item()
                
                # Collect predictions
                failure_preds = torch.argmax(outputs['failure_mode_logits'], dim=1)
                all_failure_preds.extend(failure_preds.cpu().numpy())
                all_failure_labels.extend(batch['failure_mode'].cpu().numpy())
                all_confidence_preds.extend(outputs['confidence_score'].cpu().numpy())
                all_confidence_labels.extend(batch['delta_accuracy'].cpu().numpy())
        
        # Compute metrics
        macro_f1 = f1_score(all_failure_labels, all_failure_preds, average='macro')
        confidence_mse = mean_squared_error(all_confidence_labels, all_confidence_preds)
        
        return {
            'total_loss': total_loss / len(data_loader),
            'macro_f1': macro_f1,
            'confidence_mse': confidence_mse
        }
    
    def save_model(self, model_path: str, vectorizer_path: str):
        """Save trained model and label encoder"""
        # Save model state dict
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'num_classes': len(self.label_encoder.classes_)
        }, model_path)
        
        # Save label encoder and tokenizer info
        with open(vectorizer_path, 'wb') as f:
            pickle.dump({
                'label_encoder': self.label_encoder,
                'tokenizer_name': self.config.model_name,
                'max_length': self.config.max_length
            }, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    
    def predict(self, 
                initial_answer: str,
                revised_answer: str,
                reprompt_id: str,
                logits_features: Optional[np.ndarray] = None) -> Dict:
        """Make prediction for a single example"""
        self.model.eval()
        
        # Prepare input
        input_text = f"Initial: {initial_answer} [SEP] Revised: {revised_answer} [SEP] Prompt: {reprompt_id}"
        
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.config.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        logits_tensor = None
        if logits_features is not None:
            logits_tensor = torch.tensor(logits_features, dtype=torch.float).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_features=logits_tensor
            )
        
        # Get predictions
        failure_mode_probs = F.softmax(outputs['failure_mode_logits'], dim=1)
        failure_mode_pred = torch.argmax(failure_mode_probs, dim=1).item()
        failure_mode_label = self.label_encoder.inverse_transform([failure_mode_pred])[0]
        
        confidence_score = outputs['confidence_score'].item()
        
        return {
            'failure_mode': failure_mode_label,
            'failure_mode_confidence': failure_mode_probs[0][failure_mode_pred].item(),
            'confidence_score': confidence_score,
            'all_failure_mode_probs': {
                label: prob.item() 
                for label, prob in zip(self.label_encoder.classes_, failure_mode_probs[0])
            }
        }

def create_sample_training_data(output_path: str = "sample_training_data.csv"):
    """Create sample training data for demonstration"""
    
    sample_data = [
        {
            'initial_answer': 'London',
            'revised_answer': 'Paris', 
            'reprompt_id': 'neutral_verification',
            'failure_mode': 'corrected',
            'delta_accuracy': 1
        },
        {
            'initial_answer': 'Paris',
            'revised_answer': 'Lyon',
            'reprompt_id': 'adversarial_challenge',
            'failure_mode': 'overcorrected', 
            'delta_accuracy': -1
        },
        {
            'initial_answer': 'Madrid',
            'revised_answer': 'Madrid',
            'reprompt_id': 'critique_revise_specific',
            'failure_mode': 'anchored',
            'delta_accuracy': 0
        },
        {
            'initial_answer': 'Paris',
            'revised_answer': 'Paris',
            'reprompt_id': 'concise_confirmation',
            'failure_mode': 'unchanged_correct',
            'delta_accuracy': 0
        },
        {
            'initial_answer': 'Berlin',
            'revised_answer': 'Munich', 
            'reprompt_id': 'step_by_step_break_down',
            'failure_mode': 'wavering',
            'delta_accuracy': 0
        },
        {
            'initial_answer': 'Paris',
            'revised_answer': 'Paris, the capital city',
            'reprompt_id': 'exhaustive_explanation',
            'failure_mode': 'perfectionism',
            'delta_accuracy': 0
        }
    ]
    
    # Replicate to create more training samples
    expanded_data = []
    for _ in range(100):  # Create 600 samples total
        for item in sample_data:
            expanded_data.append(item.copy())
    
    df = pd.DataFrame(expanded_data)
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample training data with {len(df)} samples at {output_path}")
    return output_path

if __name__ == "__main__":
    # Create sample data
    data_path = create_sample_training_data()
    
    # Initialize trainer
    config = TrainingConfig(
        batch_size=8,  # Small batch for demo
        num_epochs=3,  # Few epochs for demo
        learning_rate=5e-5
    )
    
    trainer = ErrorConfidenceTrainer(config)
    
    # Prepare data and train
    train_loader, val_loader = trainer.prepare_data(data_path)
    metrics = trainer.train(train_loader, val_loader)
    
    # Save model
    trainer.save_model("models/error_confidence_model.pt", "models/vectorizer.pkl")
    
    # Test prediction
    prediction = trainer.predict(
        initial_answer="London",
        revised_answer="Paris",
        reprompt_id="neutral_verification"
    )
    print("Sample prediction:", prediction)
