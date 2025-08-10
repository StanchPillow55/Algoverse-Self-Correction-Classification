#!/usr/bin/env python3
"""
Test script for Error + Confidence Classifier
"""

from src.train_classifier import (
    create_sample_training_data, 
    TrainingConfig, 
    ErrorConfidenceTrainer
)
import logging
import os

def main():
    """Main test function"""
    print("=" * 60)
    print("TESTING ERROR + CONFIDENCE CLASSIFIER")
    print("=" * 60)
    
    # Create sample data
    print("\n1. Creating sample training data...")
    data_path = create_sample_training_data("sample_training_data.csv")
    print(f"✓ Created training data at: {data_path}")
    
    # Initialize trainer with small config for testing
    print("\n2. Initializing trainer...")
    config = TrainingConfig(
        batch_size=4,      # Very small batch for testing
        num_epochs=2,      # Just 2 epochs for quick test
        learning_rate=1e-4,
        hidden_dim=128,    # Smaller model
        max_length=256     # Shorter sequences
    )
    
    trainer = ErrorConfidenceTrainer(config)
    print("✓ Trainer initialized")
    
    try:
        # Prepare data and train
        print("\n3. Preparing data loaders...")
        train_loader, val_loader = trainer.prepare_data(data_path)
        print(f"✓ Data prepared - Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        print("\n4. Starting training...")
        metrics = trainer.train(train_loader, val_loader)
        print("✓ Training completed!")
        
        # Display training results
        print("\nTraining Results:")
        for epoch in metrics['epochs']:
            idx = epoch - 1
            print(f"Epoch {epoch}: Train Loss: {metrics['train_loss'][idx]:.4f}, "
                  f"Val Loss: {metrics['val_loss'][idx]:.4f}, "
                  f"Val F1: {metrics['val_f1'][idx]:.4f}")
        
        # Save model
        print("\n5. Saving model...")
        model_path = "models/test_error_confidence_model.pt"
        vectorizer_path = "models/test_vectorizer.pkl"
        trainer.save_model(model_path, vectorizer_path)
        print(f"✓ Model saved to: {model_path}")
        print(f"✓ Vectorizer saved to: {vectorizer_path}")
        
        # Test predictions
        print("\n6. Testing predictions...")
        test_cases = [
            ("London", "Paris", "neutral_verification", "Should detect 'corrected'"),
            ("Paris", "Lyon", "adversarial_challenge", "Should detect 'overcorrected'"),
            ("Madrid", "Madrid", "critique_revise_specific", "Should detect 'anchored'"),
            ("Paris", "Paris", "concise_confirmation", "Should detect 'unchanged_correct'")
        ]
        
        for initial, revised, prompt, expected in test_cases:
            prediction = trainer.predict(initial, revised, prompt)
            print(f"\nInput: '{initial}' → '{revised}' (prompt: {prompt})")
            print(f"Expected: {expected}")
            print(f"Predicted: failure_mode='{prediction['failure_mode']}', "
                  f"confidence={prediction['confidence_score']:.3f}")
            print(f"Mode confidence: {prediction['failure_mode_confidence']:.3f}")
        
        print("\n" + "=" * 60)
        print("✓ ALL TESTS PASSED SUCCESSFULLY!")
        print("✓ Error + Confidence Classifier is working correctly")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        if os.path.exists("sample_training_data.csv"):
            os.remove("sample_training_data.csv")
            print("✓ Cleaned up temporary files")

if __name__ == "__main__":
    # Configure logging for the test
    logging.basicConfig(level=logging.INFO)
    
    success = main()
    exit(0 if success else 1)
